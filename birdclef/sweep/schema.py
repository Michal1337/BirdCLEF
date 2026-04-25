"""Lean-CSV column declaration for sweep summaries.

Lean CSV (one row per config, sorted by `primary` desc):
    rank, config_name, primary, mean_oof_auc, macro_auc, first_pass_auc,
    site_auc_std, rare_auc, frequent_auc, runtime_min, stage_metrics_path

`primary == mean_oof_auc` — mean of per-fold macro AUCs. Best proxy for LB
on pipelines that include per-fold calibration steps (per-class thresholds,
isotonic regression). Each fold scores its own self-consistent predictions,
mirroring how a deployed model fits its calibration once on all data.

`macro_auc` is the stitched-OOF macro AUC (concat all fold val probs, one
global AUC). DIAGNOSTIC: a large `mean_oof_auc - macro_auc` gap (>0.02)
signals fold-local calibration drift. The deployed model wouldn't have
this drift (one fit on all data), so the gap penalises CV but not LB.

`first_pass_auc` is the pre-post-processing macro AUC. Compared against
`macro_auc` it tells you whether post-processing is helping or hurting.

`site_auc_std` is informational — std of per-site AUC across the val set.
High = uneven across sites. Not in the ranking formula since it's not the
official metric and adds variance at our data scale.
"""
from __future__ import annotations

SUMMARY_COLUMNS = [
    "rank",
    "config_name",
    "primary",
    "mean_oof_auc",
    "macro_auc",
    "first_pass_auc",
    "site_auc_std",
    "rare_auc",
    "frequent_auc",
    "runtime_min",
    "stage_metrics_path",
]

FLOAT_COLUMNS = {
    "primary",
    "mean_oof_auc",
    "macro_auc",
    "first_pass_auc",
    "site_auc_std",
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
