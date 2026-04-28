"""File-level StratifiedKFold splits over all fully-labeled soundscapes.

Output artifacts (under birdclef/splits/):
    folds_5_strat.parquet    : filename -> fold (0..4)   (default sweep CV)
    folds_10_strat.parquet   : filename -> fold (0..9)   (deeper finalist CV)

Both files are persisted as static splits — same seed → same folds. Use
`load_folds(n_splits=5)` (default) or `load_folds(n_splits=10)` from any
trainer / pseudo-labeler / evaluator that needs them.

Why single-label modal stratification (and NOT multi-label):

    Multi-label stratification (iterative-stratification) was tried and
    EMPIRICALLY HURT macro AUC by 0.065 (0.806 → 0.741). With 59 files and
    71 classes, the algorithm has too many constraints — it tries to spread
    rare classes across folds in arrangements that leave the SSM with too
    little train signal per class. Several sonotypes that previously hit
    AUC ≥ 0.95 collapsed to 0.04-0.16 under multi-label.

    Single-label modal stratification ignores the long tail and just
    balances the dominant species, which empirically lands the rare-class
    files in clusters the SSM can learn from. Sometimes the dumber method
    wins on small datasets.

The 11 truly-broken classes (file_support=1) cannot be helped by ANY
file-level CV scheme — by definition their only file goes into one fold's
val and that fold's train sees zero positives. The fix for those is
"pinning" (planned: assign fold=-1 to mean always-train), which is
orthogonal to the stratification choice.

What still rides on the file-level granularity:
- 12 windows of one file always go together (no within-file leakage).
- All 59+ fully-labeled soundscapes are folded; partial-label files are
  excluded from CV but still available for training-side use.

Stitched OOF macro AUC = concat each fold's val predictions, compute one
global metric. With 59 files × 12 windows = 708 evaluation rows, this is the
primary selection metric.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from birdclef.config.paths import FOLD_KINDS, SPLIT_ROOT, folds_path
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta


# Default fold counts to materialize. Both are produced by a single call to
# `_02_build_splits.py` so downstream scripts can pick either via `--n-splits`.
DEFAULT_N_SPLITS = (5, 10)


def _labeled_file_frame() -> pd.DataFrame:
    """One row per fully-labeled soundscape file with stratification key.

    Returns columns: filename, primary_label (modal species across the 12
    windows of that file), label_set (full union of labels — kept for the
    diagnostic summary), site, date.  Files with empty label_lists across
    all windows are dropped — they can't contribute to evaluation either.
    """
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    if sc.empty:
        raise RuntimeError("No fully-labeled soundscapes found.")

    rows = []
    for fname, group in sc.groupby("filename"):
        all_labels: List[str] = []
        for lbls in group["label_list"]:
            if isinstance(lbls, (list, tuple, np.ndarray)):
                all_labels.extend(list(lbls))
        if not all_labels:
            continue
        modal = Counter(all_labels).most_common(1)[0][0]
        # site/date are constant within a filename — pick the first row's value.
        first = group.iloc[0]
        rows.append({
            "filename": fname,
            "primary_label": modal,
            "label_set": tuple(sorted(set(all_labels))),
            "site": str(first.get("site", "unknown")),
            "date": str(first.get("date", "00000000")),
        })
    files = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    if files.empty:
        raise RuntimeError("No labeled soundscape files have any species labels.")
    return files


def _multilabel_matrix(files: pd.DataFrame) -> np.ndarray:
    """Build (n_files, n_classes) multi-hot matrix from `label_set` column.

    Used by `stratification_summary` for the train-presence diagnostic, NOT
    by the stratifier itself.
    """
    idx = label_to_idx()
    n_classes = len(idx)
    Y = np.zeros((len(files), n_classes), dtype=np.uint8)
    for i, lbs in enumerate(files["label_set"]):
        for lb in lbs:
            j = idx.get(lb)
            if j is not None:
                Y[i, j] = 1
    return Y


def _safe_stratify_key(files: pd.DataFrame, n_splits: int) -> pd.Series:
    """Return a stratification key that StratifiedKFold won't choke on.

    StratifiedKFold raises if any class has fewer members than n_splits. We
    fold the long tail (any modal species with < n_splits files) into a
    single "_RARE_" bucket so the dominant species drive the stratification.
    """
    counts = files["primary_label"].value_counts()
    rare = set(counts[counts < n_splits].index)
    return files["primary_label"].where(~files["primary_label"].isin(rare), other="_RARE_")


def build_folds(n_splits: int, seed: int = 42) -> pd.DataFrame:
    """Build a single fold assignment using single-label modal stratification.

    Returns a DataFrame with columns (filename, primary_label, fold).
    One row per fully-labeled soundscape file. Folds are 0..n_splits-1.

    NOTE: file_support=1 classes (their only file always lands in one fold's
    val with zero training signal in that fold) are *not* fixed here.  We
    explicitly chose visibility-of-weakness over local-AUC-cosmetics — see
    `outputs/eda/per_class_audit.csv` for the per-class breakdown that
    drives data-acquisition decisions.
    """
    files = _labeled_file_frame()
    if len(files) < n_splits:
        raise RuntimeError(
            f"Only {len(files)} labeled files; cannot make {n_splits} folds."
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    fold_of = np.full(len(files), -1, dtype=np.int8)
    key = _safe_stratify_key(files, n_splits)
    for fold, (_tr, va) in enumerate(skf.split(np.zeros(len(files)), key.values)):
        fold_of[va] = fold
    if (fold_of < 0).any():
        raise RuntimeError("Some files were not assigned to any fold (bug).")
    files = files.assign(fold=fold_of)
    return files[["filename", "primary_label", "fold"]].reset_index(drop=True)


def build_grouped_folds(n_splits: int, kind: str) -> pd.DataFrame:
    """GroupKFold by site or by (site, date).

    kind="site": groups = file's site. With ~8 sites in BirdCLEF26, fold sizes
        are highly uneven (S22 alone is ~66% of files). The S22 fold trains on
        ~20 files only — accept the noise as the price of strict site-out CV.
    kind="sitedate": groups = (site, date) tuple. ~48 groups for the labeled
        pool → balanced ~12 files/fold, but S22's individual days can land in
        train AND val (different days) so mic/ambience are still leaking.
        Practical compromise when pure site-level produces unstable folds.
    """
    if kind not in ("site", "sitedate"):
        raise ValueError(f"build_grouped_folds: kind must be site|sitedate, got {kind!r}")
    files = _labeled_file_frame()
    if len(files) < n_splits:
        raise RuntimeError(
            f"Only {len(files)} labeled files; cannot make {n_splits} folds."
        )
    if kind == "site":
        groups = files["site"].astype(str).to_numpy()
    else:
        groups = (files["site"].astype(str) + "_" + files["date"].astype(str)).to_numpy()
    n_unique_groups = int(pd.Series(groups).nunique())
    if n_unique_groups < n_splits:
        raise RuntimeError(
            f"GroupKFold(kind={kind}, n_splits={n_splits}) requires "
            f">= {n_splits} unique groups, found {n_unique_groups}."
        )
    gkf = GroupKFold(n_splits=int(n_splits))
    fold_of = np.full(len(files), -1, dtype=np.int8)
    for fold, (_tr, va) in enumerate(gkf.split(np.zeros(len(files)), groups=groups)):
        fold_of[va] = fold
    if (fold_of < 0).any():
        raise RuntimeError("Some files were not assigned to any fold (bug).")
    files = files.assign(fold=fold_of)
    return files[["filename", "primary_label", "site", "date", "fold"]].reset_index(drop=True)


def build_folds_kind(n_splits: int, kind: str = "strat", seed: int = 42) -> pd.DataFrame:
    """Build folds of the requested kind. Dispatches to the right builder."""
    if kind == "strat":
        return build_folds(n_splits=int(n_splits), seed=int(seed))
    return build_grouped_folds(n_splits=int(n_splits), kind=kind)


def build_and_persist_folds(n_splits: int, seed: int = 42, kind: str = "strat") -> pd.DataFrame:
    """Build a fold assignment and persist it to `folds_path(n_splits, kind)`."""
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    folds = build_folds_kind(n_splits=n_splits, kind=kind, seed=seed)
    out_path = folds_path(n_splits, kind=kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(out_path, index=False)
    return folds


def build_and_persist_all(seeds: int = 42, n_splits_list=DEFAULT_N_SPLITS,
                          kinds: tuple = ("strat",)) -> dict:
    """Build fold parquets for every (n_splits, kind) pair. Returns nested dict."""
    out: dict = {}
    for kind in kinds:
        out[kind] = {}
        for n in n_splits_list:
            out[kind][int(n)] = build_and_persist_folds(
                n_splits=int(n), seed=seeds, kind=kind,
            )
    return out


def load_folds(n_splits: int = 5, kind: str = "strat") -> pd.DataFrame:
    """Read the persisted fold assignment for `n_splits` folds of `kind`.

    Returns DataFrame with columns (filename, primary_label, fold), plus
    (site, date) when the parquet was built with a grouped kind. Raises
    FileNotFoundError if the parquet hasn't been built — run
    `python -m birdclef.scripts._02_build_splits --kind <kind>` first.
    """
    p = folds_path(n_splits, kind=kind)
    if not Path(p).exists():
        raise FileNotFoundError(
            f"Fold parquet missing: {p}. Run "
            f"`python -m birdclef.scripts._02_build_splits --kind {kind}` to build it."
        )
    return pd.read_parquet(p)


def stratification_summary(folds: pd.DataFrame) -> str:
    """Human-readable per-fold diagnostics: size, modal-species coverage,
    and (most importantly for the multi-label fix) per-class train-presence
    minimum across folds.

    The "min_train_pos_across_folds" line is the new bug-detector: any class
    that hits 0 here means it lives entirely in one fold's val and the
    model gets no training signal — exactly the failure mode the audit
    surfaced. With multi-label stratification this should be ≥ 1 for every
    class with ≥ n_splits total positive files.
    """
    n_splits = int(folds.loc[folds["fold"] >= 0, "fold"].max()) + 1
    files = _labeled_file_frame()
    files = files.merge(folds[["filename", "fold"]], on="filename", how="inner")
    Y = _multilabel_matrix(files)
    fold_of = files["fold"].to_numpy()

    # Per-class train-side positive count, per fold. Pinned files (fold=-1)
    # are in EVERY fold's training set, so they always contribute.
    n_classes = Y.shape[1]
    per_fold_train_pos = np.zeros((n_splits, n_classes), dtype=np.int32)
    for f in range(n_splits):
        train_mask = (fold_of != f)  # includes fold=-1 (pinned) → always train
        per_fold_train_pos[f] = Y[train_mask].sum(axis=0)
    file_supp = Y.sum(axis=0)
    seen_classes = np.where(file_supp > 0)[0]
    min_train_per_class = per_fold_train_pos[:, seen_classes].min(axis=0)

    n_with_zero_train = int((min_train_per_class == 0).sum())
    n_with_one_train = int((min_train_per_class == 1).sum())

    # Val-side coverage (pinned files contribute nothing to val)
    cv_files = files[files["fold"] >= 0]
    cv_Y = _multilabel_matrix(cv_files)
    val_support = cv_Y.sum(axis=0)
    val_seen = int((val_support > 0).sum())
    n_pinned = int((folds["fold"] == -1).sum())

    lines = [
        f"n_splits={n_splits}  pinned (always-train): {n_pinned} files",
        "  fold | n_files | n_unique_modal | top_3_modal (count)",
    ]
    for f in range(n_splits):
        sub = folds[folds["fold"] == f]
        sp_counts = sub["primary_label"].value_counts()
        top3 = ", ".join(f"{s}:{c}" for s, c in sp_counts.head(3).items())
        lines.append(f"  {f:4d} | {len(sub):7d} | {len(sp_counts):14d} | {top3}")
    lines.append("")
    lines.append(f"Total files: {len(folds)}  unique modal species: {folds['primary_label'].nunique()}")
    lines.append(f"Classes with any positive across labeled pool (file-support>0): {len(seen_classes)}")
    lines.append(f"  of those, with VAL coverage (file in some non-pinned fold)   : {val_seen}")
    lines.append(f"  of those, learnable in every fold (min train-pos >= 1)       : {len(seen_classes) - n_with_zero_train}")
    lines.append(
        f"  classes with ZERO train-positives in some fold : {n_with_zero_train}  "
        f"(should be 0 with pinning — anything >0 is a cascade leak)"
    )
    lines.append(
        f"  classes with only 1 train-positive in some fold: {n_with_one_train}  "
        f"(weak signal but learnable)"
    )
    return "\n".join(lines)
