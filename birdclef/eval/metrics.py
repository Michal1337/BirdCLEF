"""Macro-AUC + subgroup metrics.

`compute_stage_metrics` is the single entry point. It returns a dict the sweep
writer can split into lean-CSV columns (top-level) vs per-config JSON (nested).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    if not keep.any():
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    n_classes = y_true.shape[1]
    out = np.full(n_classes, np.nan, dtype=np.float32)
    for c in range(n_classes):
        yc = y_true[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        try:
            out[c] = float(roc_auc_score(yc, y_score[:, c]))
        except ValueError:
            pass
    return out


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


def split_rare_frequent(
    class_support: np.ndarray, rare_q: float = 0.2, freq_q: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rare_idx, frequent_idx) by support quintile."""
    s = np.asarray(class_support)
    lo = np.quantile(s, rare_q)
    hi = np.quantile(s, freq_q)
    rare = np.where(s <= lo)[0]
    freq = np.where(s >= hi)[0]
    return rare.astype(np.int32), freq.astype(np.int32)


def compute_stage_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    meta: pd.DataFrame,
    rare_idx: Optional[np.ndarray] = None,
    frequent_idx: Optional[np.ndarray] = None,
) -> Dict:
    """meta must have columns: site, hour_utc aligned to y_true rows."""
    out: Dict = {"macro_auc": macro_auc(y_true, y_score)}

    if rare_idx is not None and len(rare_idx) > 0:
        out["rare_auc"] = macro_auc(y_true[:, rare_idx], y_score[:, rare_idx])
    if frequent_idx is not None and len(frequent_idx) > 0:
        out["frequent_auc"] = macro_auc(y_true[:, frequent_idx], y_score[:, frequent_idx])

    per_site = {}
    for site, sub in meta.groupby("site"):
        rows = sub.index.to_numpy()
        if len(rows) == 0:
            continue
        try:
            per_site[str(site)] = macro_auc(y_true[rows], y_score[rows])
        except ValueError:
            pass
    out["per_site_auc"] = per_site
    site_vals = [v for v in per_site.values() if not np.isnan(v)]
    out["site_auc_std"] = float(np.std(site_vals)) if site_vals else float("nan")

    per_hour = {}
    meta = meta.copy()
    meta["_hb"] = meta["hour_utc"].apply(_hour_bucket)
    for hb, sub in meta.groupby("_hb"):
        rows = sub.index.to_numpy()
        if len(rows) == 0:
            continue
        try:
            per_hour[str(hb)] = macro_auc(y_true[rows], y_score[rows])
        except ValueError:
            pass
    out["per_hour_auc"] = per_hour
    return out


def primary_score(m: Dict, std_penalty: float = 1.0) -> float:
    """Ranking scalar: macro_auc - λ·site_auc_std."""
    mauc = m.get("macro_auc", float("nan"))
    std = m.get("site_auc_std", 0.0)
    if np.isnan(mauc):
        return float("-inf")
    return float(mauc - std_penalty * (0.0 if np.isnan(std) else std))
