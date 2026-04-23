"""Score a pipeline on the permanent V-anchor split.

The V-anchor is a ~15 % site/hour-stratified hold-out of labeled soundscapes
that no training code may see. Scoring here is the PRIMARY selection metric.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from birdclef.data.splits import load_v_anchor
from birdclef.eval.metrics import compute_stage_metrics


def v_anchor_predict(
    X_meta: pd.DataFrame,
    y_true: np.ndarray,
    train_predict_fn: Callable,
    rare_idx=None,
    frequent_idx=None,
) -> Dict:
    """train_predict_fn(train_idx, val_idx) -> val_probs.

    train_idx = every row whose filename is not in V-anchor AND is labeled.
    val_idx   = every row whose filename is in V-anchor.
    """
    anchor = set(load_v_anchor())
    if not anchor:
        raise RuntimeError("V-anchor file is empty. Run scripts/02_build_splits.py first.")
    is_anchor = X_meta["filename"].isin(anchor).to_numpy()
    tr = np.where(~is_anchor)[0]
    va = np.where(is_anchor)[0]
    if len(va) == 0:
        raise RuntimeError("No rows matched the V-anchor filenames. Did cache rebuild?")
    probs = train_predict_fn(tr, va)
    scored_meta = X_meta.iloc[va].reset_index(drop=True)
    metrics = compute_stage_metrics(
        y_true[va], probs, scored_meta, rare_idx=rare_idx, frequent_idx=frequent_idx
    )
    return {"probs": probs, "metrics": metrics, "val_idx": va}
