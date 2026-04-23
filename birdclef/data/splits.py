"""Site×date GroupKFold + permanent V-anchor split for labeled soundscapes.

Output artifacts (all under birdclef/splits/):
    folds_site_date.parquet : filename -> fold (0..n_splits-1), site_date
    v_anchor_files.txt      : one filename per line (never used for training)

The V-anchor is carved out FIRST (~15% of labeled files, stratified by site
and 6-hour bucket). Remaining files are then 5-fold GroupKFold'd on site_date
(month granularity).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from birdclef.config.paths import FOLDS_PQ, SPLIT_ROOT, VANCHOR
from birdclef.data.soundscapes import load_soundscape_meta, parse_fname


def _hour_bucket(h: int) -> str:
    if h < 0:
        return "unk"
    if h < 6:
        return "0-5"
    if h < 12:
        return "6-11"
    if h < 18:
        return "12-17"
    return "18-23"


def _site_date_key(row: pd.Series) -> str:
    # Month granularity (YYYYMM) — keeps fold populations meaningful.
    return f"{row['site']}_{row['date'][:6]}"


def _labeled_file_frame() -> pd.DataFrame:
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    files = sc.drop_duplicates("filename").copy()
    files["hour_bucket"] = files["hour_utc"].apply(_hour_bucket)
    files["site_date"] = files.apply(_site_date_key, axis=1)
    return files[["filename", "site", "date", "hour_utc", "hour_bucket", "site_date"]].reset_index(drop=True)


def build_v_anchor(
    files: pd.DataFrame,
    fraction: float = 0.15,
    seed: int = 42,
    min_per_stratum: int = 1,
) -> list:
    """Stratified hold-out by (site, hour_bucket).

    Takes ceil(fraction * count) from each stratum, capped at count-1 so a
    stratum never empties into the V-anchor.
    """
    rng = np.random.default_rng(seed)
    selected = []
    groups = list(files.groupby(["site", "hour_bucket"]))
    groups.sort(key=lambda kv: kv[0])
    for _, sub in groups:
        n = len(sub)
        take = max(min_per_stratum, int(np.ceil(n * fraction)))
        take = min(take, max(0, n - 1))
        if take <= 0:
            continue
        chosen = rng.choice(sub["filename"].values, size=take, replace=False)
        selected.extend(chosen.tolist())
    return sorted(set(selected))


def build_folds(non_anchor: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """Return a fold index (0..n_splits-1) aligned to non_anchor.index."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_of = np.zeros(len(non_anchor), dtype=np.int8)
    for fold, (_tr, va) in enumerate(
        gkf.split(non_anchor, groups=non_anchor["site_date"].values)
    ):
        fold_of[va] = fold
    return pd.Series(fold_of, index=non_anchor.index, name="fold")


def build_and_persist(
    n_splits: int = 5,
    v_anchor_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, list]:
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    files = _labeled_file_frame()
    if len(files) == 0:
        raise RuntimeError("No fully-labeled soundscapes found — cannot build splits.")

    v_anchor = build_v_anchor(files, fraction=v_anchor_fraction, seed=seed)
    mask_anchor = files["filename"].isin(v_anchor)
    non_anchor = files[~mask_anchor].reset_index(drop=True)
    non_anchor["fold"] = build_folds(non_anchor, n_splits=n_splits).values

    FOLDS_PQ.parent.mkdir(parents=True, exist_ok=True)
    non_anchor.to_parquet(FOLDS_PQ, index=False)
    VANCHOR.parent.mkdir(parents=True, exist_ok=True)
    VANCHOR.write_text("\n".join(v_anchor) + ("\n" if v_anchor else ""), encoding="utf-8")
    return non_anchor, v_anchor


def load_folds() -> pd.DataFrame:
    return pd.read_parquet(FOLDS_PQ)


def load_v_anchor() -> list:
    if not Path(VANCHOR).exists():
        return []
    return [ln.strip() for ln in Path(VANCHOR).read_text(encoding="utf-8").splitlines() if ln.strip()]


def stratification_summary(non_anchor: pd.DataFrame, v_anchor: list) -> str:
    """Human-readable summary of counts per (site, hour_bucket) in each split."""
    from collections import Counter

    files_all = _labeled_file_frame().set_index("filename")
    anchor_rows = files_all.loc[[f for f in v_anchor if f in files_all.index]]
    rows_a = Counter(zip(anchor_rows["site"], anchor_rows["hour_bucket"]))
    rows_n = Counter(zip(non_anchor["site"], non_anchor["hour_bucket"]))
    lines = ["site | hour  | non_anchor | v_anchor"]
    for key in sorted(set(rows_a) | set(rows_n)):
        s, h = key
        lines.append(f"{s:5s}| {h:5s}| {rows_n.get(key, 0):10d} | {rows_a.get(key, 0):8d}")
    fold_counts = Counter(non_anchor["fold"].values)
    lines.append("")
    lines.append("fold | count")
    for k in sorted(fold_counts):
        lines.append(f"{k:4d} | {fold_counts[k]}")
    lines.append(f"\nTotal non_anchor={len(non_anchor)}   v_anchor={len(v_anchor)}")
    return "\n".join(lines)
