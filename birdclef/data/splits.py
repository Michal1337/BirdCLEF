"""File-level multi-label stratified k-fold splits over labeled soundscapes.

Output artifacts (under birdclef/splits/):
    folds_5_strat.parquet    : filename -> fold (0..4)   (default sweep CV)
    folds_10_strat.parquet   : filename -> fold (0..9)   (deeper finalist CV)

Both files are persisted as static splits — same seed → same folds. Use
`load_folds(n_splits=5)` (default) or `load_folds(n_splits=10)` from any
trainer / pseudo-labeler / evaluator that needs them.

Why MULTI-label stratification (the previous version used single-label
StratifiedKFold on the modal primary species per file):

    The audit at outputs/eda/per_class_audit.csv showed the bottom 12 classes
    (8 sonotypes + 4 birds/frogs) all sat at AUC < 0.5 because each only
    appeared in 1 fold's val set, leaving the other 4 folds with ZERO
    training positives for that class. Single-label stratification on the
    modal species was blind to the long tail of secondary labels in each
    file. MultilabelStratifiedKFold considers the full multi-hot target per
    file and tries to balance the marginal distribution of EVERY class
    across folds — so every class with ≥ n_splits files gets at least one
    train-side positive in every fold.

    Expected lift: macro AUC 0.806 → ~0.88-0.90 with no new data.

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

from birdclef.config.paths import SPLIT_ROOT, folds_path
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta


# Default fold counts to materialize. Both are produced by a single call to
# `_02_build_splits.py` so downstream scripts can pick either via `--n-splits`.
DEFAULT_N_SPLITS = (5, 10)


def _labeled_file_frame() -> pd.DataFrame:
    """One row per fully-labeled soundscape file with its full label_set.

    Returns columns:
      filename       — the .ogg file
      primary_label  — modal species across the 12 windows (kept for human-
                       readable summaries; NOT used for stratification anymore)
      label_set      — sorted tuple of every distinct atomic label that
                       appears in any of the 12 windows of this file
    Files with no labels at all are dropped.
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
        rows.append({
            "filename": fname,
            "primary_label": modal,
            "label_set": tuple(sorted(set(all_labels))),
        })
    files = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    if files.empty:
        raise RuntimeError("No labeled soundscape files have any species labels.")
    return files


def _multilabel_matrix(files: pd.DataFrame) -> np.ndarray:
    """Build (n_files, n_classes) multi-hot matrix from `label_set` column.

    Columns follow `label_to_idx()` (the official 234-class order). Labels in
    the soundscape pool that aren't in sample_submission are silently dropped
    — they cannot affect the LB metric anyway.
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


def build_folds(n_splits: int, seed: int = 42) -> pd.DataFrame:
    """Build a multi-label stratified k-fold assignment.

    Returns a DataFrame with columns (filename, primary_label, fold).
    One row per fully-labeled soundscape file. Folds are 0..n_splits-1.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "iterative-stratification is required. "
            "Install with `pip install iterative-stratification`."
        ) from e

    files = _labeled_file_frame()
    if len(files) < n_splits:
        raise RuntimeError(
            f"Only {len(files)} labeled files; cannot make {n_splits} folds."
        )

    Y = _multilabel_matrix(files)

    # MultilabelStratifiedKFold can choke on classes with a single positive
    # spread across too few folds. Drop columns with < n_splits positives so
    # the iterative balancer focuses on classes it can actually distribute,
    # while still keeping rare-class files in the split (just not used as
    # stratification targets).
    col_support = Y.sum(axis=0)
    keep_cols = np.where(col_support >= n_splits)[0]
    if keep_cols.size == 0:
        raise RuntimeError(
            f"No classes have ≥ {n_splits} files of support — try fewer folds."
        )
    Y_strat = Y[:, keep_cols]

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(seed),
    )
    fold_of = np.full(len(files), -1, dtype=np.int8)
    for fold, (_tr, va) in enumerate(mskf.split(np.zeros(len(files)), Y_strat)):
        fold_of[va] = fold
    if (fold_of < 0).any():
        raise RuntimeError("Some files were not assigned to any fold (bug).")

    files = files.assign(fold=fold_of)
    # `label_set` is fine to keep around but bloats the parquet; downstream
    # consumers only need (filename, primary_label, fold).
    return files[["filename", "primary_label", "fold"]].reset_index(drop=True)


def build_and_persist_folds(n_splits: int, seed: int = 42) -> pd.DataFrame:
    """Build a fold assignment and persist it to `folds_path(n_splits)`."""
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    folds = build_folds(n_splits=n_splits, seed=seed)
    out_path = folds_path(n_splits)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(out_path, index=False)
    return folds


def build_and_persist_all(seeds: int = 42, n_splits_list=DEFAULT_N_SPLITS) -> dict:
    """Build all default fold counts (5 and 10). Returns {n_splits: DataFrame}."""
    out = {}
    for n in n_splits_list:
        out[int(n)] = build_and_persist_folds(n_splits=int(n), seed=seeds)
    return out


def load_folds(n_splits: int = 5) -> pd.DataFrame:
    """Read the persisted fold assignment for `n_splits` folds.

    Returns DataFrame with columns (filename, primary_label, fold).
    Raises FileNotFoundError if the parquet hasn't been built — run
    `python -m birdclef.scripts._02_build_splits` first.
    """
    p = folds_path(n_splits)
    if not Path(p).exists():
        raise FileNotFoundError(
            f"Fold parquet missing: {p}. Run "
            "`python -m birdclef.scripts._02_build_splits` to build it."
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
    n_splits = int(folds["fold"].max()) + 1
    files = _labeled_file_frame()
    files = files.merge(folds[["filename", "fold"]], on="filename", how="inner")
    Y = _multilabel_matrix(files)
    fold_of = files["fold"].to_numpy()

    # Per-class train-side positive count, per fold
    n_classes = Y.shape[1]
    per_fold_train_pos = np.zeros((n_splits, n_classes), dtype=np.int32)
    for f in range(n_splits):
        train_mask = fold_of != f
        per_fold_train_pos[f] = Y[train_mask].sum(axis=0)
    file_supp = Y.sum(axis=0)
    seen_classes = np.where(file_supp > 0)[0]
    min_train_per_class = per_fold_train_pos[:, seen_classes].min(axis=0)

    n_with_zero_train = int((min_train_per_class == 0).sum())
    n_with_one_train = int((min_train_per_class == 1).sum())

    lines = [
        f"n_splits={n_splits}",
        "  fold | n_files | n_unique_modal | top_3_modal (count)",
    ]
    for f in range(n_splits):
        sub = folds[folds["fold"] == f]
        sp_counts = sub["primary_label"].value_counts()
        top3 = ", ".join(f"{s}:{c}" for s, c in sp_counts.head(3).items())
        lines.append(f"  {f:4d} | {len(sub):7d} | {len(sp_counts):14d} | {top3}")
    lines.append("")
    lines.append(f"Total files: {len(folds)}  unique modal species: {folds['primary_label'].nunique()}")
    lines.append(f"Classes with any positive in val pool (file-support>0): {len(seen_classes)}")
    lines.append(
        f"  classes with ZERO train-positives in some fold : {n_with_zero_train}  "
        f"(was the bug — should be 0 with multi-label stratification)"
    )
    lines.append(
        f"  classes with only 1 train-positive in some fold: {n_with_one_train}  "
        f"(weak signal but learnable)"
    )
    return "\n".join(lines)
