"""Soundscape-wide probability boost and related file-level scalers.

If a species fires confidently in *any* window of a 60 s file, lift its
probability in *all* windows of that file. Universal in 2024 winners.
"""
from __future__ import annotations

import numpy as np


def hard_soundscape_boost(
    probs: np.ndarray,
    n_windows: int = 12,
    threshold: float = 0.5,
    lift_weight: float = 0.25,
) -> np.ndarray:
    """If max_over_windows(p_c) > threshold, shift every window's p_c toward
    that max by lift_weight."""
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C).astype(np.float32)
    file_max = view.max(axis=1, keepdims=True)
    fire = (file_max > threshold).astype(np.float32)
    lift = fire * lift_weight
    out = (1.0 - lift) * view + lift * file_max
    return out.reshape(N, C)


def file_confidence_scale(
    probs: np.ndarray, n_windows: int = 12, top_k: int = 2, power: float = 0.4
) -> np.ndarray:
    """Legacy scaler — multiplies windows by (top-k file mean)^power."""
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C).astype(np.float32)
    srt = np.sort(view, axis=1)
    top_k_mean = srt[:, -top_k:, :].mean(axis=1, keepdims=True)
    scale = np.power(top_k_mean, power)
    return (view * scale).reshape(N, C)


def rank_aware_scaling(
    probs: np.ndarray, n_windows: int = 12, power: float = 0.4
) -> np.ndarray:
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C).astype(np.float32)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(file_max, power)
    return (view * scale).reshape(N, C)
