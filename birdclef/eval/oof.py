"""Fold-safe OOF orchestration using the persisted site×date folds.

The OOF runner is generic: you pass a `train_predict_fn(train_idx, val_idx,
train_ctx) -> val_probs[len(val_idx), n_classes]` and this module handles
loading the fold assignment, iterating folds, stitching OOF, and calling the
metrics module.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from birdclef.data.splits import load_folds
from birdclef.eval.metrics import compute_stage_metrics


def oof_predict(
    X_meta: pd.DataFrame,
    y_true: np.ndarray,
    train_predict_fn: Callable,
    rare_idx=None,
    frequent_idx=None,
    n_splits: int = 5,
) -> Dict:
    """
    X_meta must have `filename` column — fold assignment is looked up by
    filename against `folds_site_date.parquet`.
    y_true: (N_rows, n_classes) ground-truth multi-hot.
    train_predict_fn(train_idx, val_idx) -> val_probs (N_val, n_classes).
    """
    folds = load_folds()
    fold_of = dict(zip(folds["filename"], folds["fold"].astype(int)))
    row_fold = X_meta["filename"].map(fold_of).fillna(-1).astype(int).to_numpy()

    oof = np.zeros_like(y_true, dtype=np.float32)
    per_fold = {}
    for f in range(n_splits):
        tr = np.where(row_fold != f)[0]
        va = np.where(row_fold == f)[0]
        if len(va) == 0:
            continue
        oof[va] = train_predict_fn(tr, va)
        m_fold = compute_stage_metrics(y_true[va], oof[va], X_meta.iloc[va].reset_index(drop=True),
                                       rare_idx=rare_idx, frequent_idx=frequent_idx)
        per_fold[f] = m_fold
        print(f"[oof] fold {f}  macro_auc={m_fold['macro_auc']:.4f}  "
              f"site_std={m_fold['site_auc_std']:.4f}")

    # Only score rows that were assigned a fold (-1 = outside folds, eg anchor).
    keep = row_fold >= 0
    global_metrics = compute_stage_metrics(
        y_true[keep], oof[keep], X_meta.iloc[keep].reset_index(drop=True),
        rare_idx=rare_idx, frequent_idx=frequent_idx,
    )
    return {"oof": oof, "global": global_metrics, "per_fold": per_fold}
