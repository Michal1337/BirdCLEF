"""Temporal smoothing across 12 windows per soundscape file.

Two variants:
    gaussian_smooth      : convolve with [0.1, 0.2, 0.4, 0.2, 0.1] (2024 #1/#3 universal)
    adaptive_delta_smooth: original pipeline's adaptive-alpha variant (legacy)
"""
from __future__ import annotations

import numpy as np


def gaussian_smooth(
    probs: np.ndarray, n_windows: int = 12, kernel=(0.1, 0.2, 0.4, 0.2, 0.1)
) -> np.ndarray:
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"
    k = np.asarray(kernel, dtype=np.float32)
    k = k / k.sum()
    p = probs.reshape(-1, n_windows, C).astype(np.float32)
    # Pad edges (reflect)
    pad = len(k) // 2
    padded = np.pad(p, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(p)
    for i, w in enumerate(k):
        out += w * padded[:, i : i + n_windows, :]
    return out.reshape(N, C)


def adaptive_delta_smooth(
    probs: np.ndarray, n_windows: int = 12, base_alpha: float = 0.20
) -> np.ndarray:
    """Per-window alpha shrinks when local confidence is high (legacy)."""
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C).astype(np.float32)
    out = view.copy()
    for t in range(n_windows):
        conf = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        if t == 0:
            nb = (view[:, t] + view[:, t + 1]) * 0.5
        elif t == n_windows - 1:
            nb = (view[:, t - 1] + view[:, t]) * 0.5
        else:
            nb = (view[:, t - 1] + view[:, t + 1]) * 0.5
        out[:, t] = (1.0 - alpha) * view[:, t] + alpha * nb
    return out.reshape(N, C)
