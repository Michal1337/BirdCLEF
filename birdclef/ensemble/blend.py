"""Ensemble blending helpers.

Inputs: one or more probability arrays aligned to a common (N_rows, C) grid.
Exports a JSON "recipe" consumed by the submission inference template.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

from birdclef.eval.metrics import compute_stage_metrics


def sigmoid_mean(probs_list: List[np.ndarray], weights: List[float] | None = None) -> np.ndarray:
    arr = np.stack(probs_list, axis=0).astype(np.float32)
    if weights is None:
        return arr.mean(axis=0)
    w = np.asarray(weights, dtype=np.float32)
    w = w / max(w.sum(), 1e-8)
    return (arr * w[:, None, None]).sum(axis=0)


def rank_mean(probs_list: List[np.ndarray], weights: List[float] | None = None) -> np.ndarray:
    ranks = []
    for p in probs_list:
        # Per-column rank-normalize to [0,1]
        r = np.empty_like(p)
        for c in range(p.shape[1]):
            r[:, c] = rankdata(p[:, c], method="average") / p.shape[0]
        ranks.append(r.astype(np.float32))
    return sigmoid_mean(ranks, weights=weights)


def member_correlation(probs_list: List[np.ndarray]) -> np.ndarray:
    flat = [p.ravel() for p in probs_list]
    from scipy.stats import spearmanr

    K = len(flat)
    M = np.eye(K, dtype=np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            r, _ = spearmanr(flat[i][::100], flat[j][::100])
            M[i, j] = M[j, i] = float(r)
    return M


def weight_search_grid(
    members: List[np.ndarray],
    y_true: np.ndarray,
    meta,
    step: float = 0.1,
    blend: str = "sigmoid",
) -> Dict:
    """Brute-force weight search (for up to 4 members)."""
    K = len(members)
    if K == 1:
        return {"weights": [1.0], "metrics": compute_stage_metrics(y_true, members[0], meta)}
    if K > 4:
        raise NotImplementedError("Brute-force weight search only for K<=4 members.")
    grid_axis = np.arange(0.0, 1.0 + step / 2, step)
    best = (float("-inf"), None, None)
    blend_fn = sigmoid_mean if blend == "sigmoid" else rank_mean
    # Sample all simplex weight tuples summing to 1 on this grid.
    def _tuples(depth, remaining):
        if depth == 1:
            yield (remaining,)
            return
        for w in grid_axis:
            if w <= remaining + 1e-9:
                for tail in _tuples(depth - 1, round(remaining - w, 8)):
                    yield (w,) + tail

    for weights in _tuples(K, 1.0):
        if any(w < -1e-9 for w in weights):
            continue
        blended = blend_fn(members, list(weights))
        m = compute_stage_metrics(y_true, blended, meta)
        mauc = m.get("macro_auc", float("nan"))
        if np.isnan(mauc):
            continue
        primary = mauc - (m.get("site_auc_std", 0.0) or 0.0)
        if primary > best[0]:
            best = (primary, list(weights), m)
    return {"weights": best[1], "metrics": best[2]}


def save_recipe(
    path: Path,
    member_paths: List[str],
    weights: List[float],
    blend: str = "sigmoid",
    extra: dict | None = None,
) -> None:
    payload = {
        "blend": blend,
        "weights": list(weights),
        "members": list(member_paths),
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
