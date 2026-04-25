"""Declares which fields become lean-CSV columns vs per-config JSON.

Lean CSV (one row per config, sorted by `primary` desc):
    rank, config_name, primary, macro_auc (=stitched OOF, final stage),
    first_pass_auc, site_auc_std, mean_oof_auc, rare_auc, frequent_auc,
    runtime_min, stage_metrics_path

`primary == macro_auc` — the official BirdCLEF metric. Top-team writeups
(2024 / 2025) rank configs by stitched OOF macro AUC directly; we match
that. The previous formula `macro_auc − site_auc_std` was abandoned after
research showed (a) it isn't the official metric, (b) it adds noise at our
data scale, (c) no top team uses it. See `eval/metrics.py:primary_score`.

`macro_auc` is the **stitched 5-fold OOF macro AUC** — concatenate each
fold's val predictions, compute one global AUC. Replaces the old V-anchor
metric, which was abandoned after the 0.747 SED LB result confirmed it
didn't predict LB. See plan file for rationale.

`site_auc_std` stays as a **diagnostic** column — useful for spotting
fragile models (high std = uneven across sites) even though it's no longer
in the ranking formula.

`mean_oof_auc` is the unweighted mean of per-fold AUCs (informational —
higher variance than `macro_auc`, included for inspection).

`first_pass_auc` columns are informational only. A positive (first_pass −
final) gap on a given config means post-processing is hurting
generalization for that config; investigate before promoting it to
pseudo-label teacher.
"""
from __future__ import annotations

SUMMARY_COLUMNS = [
    "rank",
    "config_name",
    "primary",
    "macro_auc",
    "first_pass_auc",
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
    "first_pass_auc",
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
