"""File-level StratifiedKFold splits over all fully-labeled soundscapes.

Output artifacts (under birdclef/splits/):
    folds_5_strat.parquet    : filename -> fold (0..4)   (default sweep CV)
    folds_10_strat.parquet   : filename -> fold (0..9)   (deeper finalist CV)

Both files are persisted as static splits — same seed → same folds. Use
`load_folds(n_splits=5)` (default) or `load_folds(n_splits=10)` from any
trainer / pseudo-labeler / evaluator that needs them.

Why this design:
- All 59 fully-labeled soundscapes are folded (no permanent hold-out — V-anchor
  was abandoned because it didn't predict LB and cost 28% of train data per
  fold).
- File-level grouping is implicit (StratifiedKFold over per-file rows ensures
  the 12 windows of one file always go together).
- Stratification key = each file's modal primary species, so every fold has
  representation of common species. Rare species (those with <n_splits files)
  fall back to whichever folds StratifiedKFold can place them in.
- Site×month grouping was tried and abandoned: only ~10–20 unique groups means
  per-fold variance dominates the metric. See plan file for full reasoning.

Stitched OOF macro AUC = concat each fold's val predictions, compute one
global metric. With 59 files × 12 windows = 708 evaluation rows, this is the
new primary selection metric (replaces V-anchor).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from birdclef.config.paths import SPLIT_ROOT, folds_path
from birdclef.data.soundscapes import load_soundscape_meta


# Default fold counts to materialize. Both are produced by a single call to
# `_02_build_splits.py` so downstream scripts can pick either via `--n-splits`.
DEFAULT_N_SPLITS = (5, 10)


def _labeled_file_frame() -> pd.DataFrame:
    """One row per fully-labeled soundscape file with stratification key.

    Returns columns: filename, primary_label (modal species across the 12
    windows of that file). Files with empty label_lists across all windows
    are dropped — they can't contribute to evaluation either.
    """
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    if sc.empty:
        raise RuntimeError("No fully-labeled soundscapes found.")

    # Modal species per file: across the 12 windows, count every label that
    # appears in any window's label_list, take the most frequent one. This is
    # the simplest 1D stratification key for sklearn StratifiedKFold.
    rows = []
    for fname, group in sc.groupby("filename"):
        all_labels: List[str] = []
        for lbls in group["label_list"]:
            if isinstance(lbls, (list, tuple, np.ndarray)):
                all_labels.extend(list(lbls))
        if not all_labels:
            # No labels at all → drop. (Shouldn't happen for fully_labeled but
            # defends against partial annotations.)
            continue
        modal = Counter(all_labels).most_common(1)[0][0]
        rows.append({"filename": fname, "primary_label": modal})
    files = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    if files.empty:
        raise RuntimeError("No labeled soundscape files have any species labels.")
    return files


def _safe_stratify_key(files: pd.DataFrame, n_splits: int) -> pd.Series:
    """Return a stratification key that StratifiedKFold won't choke on.

    StratifiedKFold raises if any class has fewer members than n_splits. We
    fold the long tail (any species with < n_splits files) into a single
    "_RARE_" bucket so the ~10 dominant species drive the stratification and
    rare species are distributed wherever they fit.
    """
    counts = files["primary_label"].value_counts()
    rare = set(counts[counts < n_splits].index)
    key = files["primary_label"].where(~files["primary_label"].isin(rare), other="_RARE_")
    return key


def build_folds(n_splits: int, seed: int = 42) -> pd.DataFrame:
    """Build a single fold assignment.

    Returns a DataFrame with columns (filename, fold). One row per fully-
    labeled soundscape file. Folds are 0..n_splits-1.
    """
    files = _labeled_file_frame()
    if len(files) < n_splits:
        raise RuntimeError(
            f"Only {len(files)} labeled files; cannot make {n_splits} folds."
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(len(files), -1, dtype=np.int8)
    key = _safe_stratify_key(files, n_splits)
    for fold, (_tr, va) in enumerate(skf.split(np.zeros(len(files)), key.values)):
        fold_of[va] = fold
    files = files.assign(fold=fold_of)
    if (files["fold"] < 0).any():
        raise RuntimeError("Some files were not assigned to any fold (bug).")
    return files


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
    """Human-readable per-fold size + species coverage.

    Format:
        n_splits=5
          fold | n_files | n_unique_species | top_3_species (count)
          0    | 12      | 27               | sp_A:4, sp_B:3, sp_C:2
          ...
        Total files: 59  unique species across all folds: ~30
    """
    n_splits = int(folds["fold"].max()) + 1
    lines = [f"n_splits={n_splits}", "  fold | n_files | n_unique_species | top_3_species (count)"]
    for f in range(n_splits):
        sub = folds[folds["fold"] == f]
        sp_counts = sub["primary_label"].value_counts()
        top3 = ", ".join(f"{s}:{c}" for s, c in sp_counts.head(3).items())
        lines.append(f"  {f:4d} | {len(sub):7d} | {len(sp_counts):16d} | {top3}")
    n_unique = folds["primary_label"].nunique()
    lines.append(f"\nTotal files: {len(folds)}  unique modal species: {n_unique}")
    return "\n".join(lines)
