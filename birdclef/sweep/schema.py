"""Declares which fields become lean-CSV columns vs per-config JSON.

Lean CSV (one row per config, sorted by `primary` desc):
    rank, config_name, primary, macro_auc, v_anchor_auc, site_auc_std,
    mean_oof_auc, rare_auc, frequent_auc, runtime_min, stage_metrics_path
"""
from __future__ import annotations

SUMMARY_COLUMNS = [
    "rank",
    "config_name",
    "primary",
    "macro_auc",
    "v_anchor_auc",
    "site_auc_std",
    "mean_oof_auc",
    "rare_auc",
    "frequent_auc",
    "runtime_min",
    "stage_metrics_path",
]

FLOAT_COLUMNS = {
    "primary",
    "macro_auc",
    "v_anchor_auc",
    "site_auc_std",
    "mean_oof_auc",
    "rare_auc",
    "frequent_auc",
    "runtime_min",
}


def round_summary_row(row: dict, decimals: int = 4) -> dict:
    out = dict(row)
    for k in FLOAT_COLUMNS:
        if k in out and out[k] is not None:
            try:
                out[k] = round(float(out[k]), decimals)
            except (TypeError, ValueError):
                pass
    return out
