"""Test-time augmentation helpers.

waveform_shift_tta: shifts the raw 60s waveform by ±samples BEFORE Perch
                    inference, then aligns windows so position_t stays semantic.
                    Matches 2025 #1 recipe (+0.012 LB).
window_roll_tta   : legacy — circular-rolls the (file, 12, C) prediction grid
                    inside Perch-logit space.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np


def _align_shift(scores_flat: np.ndarray, shift_samples: int, n_windows: int, win_samples: int) -> np.ndarray:
    """Roll window-level predictions back by `-shift_samples` worth of windows.

    scores_flat: (n_files*12, C). shift_samples is in audio samples.
    """
    shift_windows = shift_samples / float(win_samples)
    # If the shift is a multiple of the window stride, rotate by that many
    # windows. Otherwise we interpolate between floor/ceil window shifts.
    C = scores_flat.shape[1]
    s = scores_flat.reshape(-1, n_windows, C)
    f = int(np.floor(shift_windows))
    c = int(np.ceil(shift_windows))
    alpha = shift_windows - f
    lo = np.roll(s, shift=-f, axis=1)
    hi = np.roll(s, shift=-c, axis=1)
    return ((1.0 - alpha) * lo + alpha * hi).reshape(-1, C)


def waveform_shift_tta(
    scores_flat: np.ndarray,
    predict_shifted_fn: Callable[[int], np.ndarray],
    shift_secs: List[float] = (0.0, 1.25, -1.25, 2.5, -2.5),
    n_windows: int = 12,
    sr: int = 32000,
    window_secs: int = 5,
) -> np.ndarray:
    """Average predictions over several waveform-level shifts.

    `predict_shifted_fn(shift_samples)` must return a flat (n_files*12, C)
    array of Perch (or SED) logits/probs for the soundscape waveform
    cyclically shifted by `shift_samples`. The caller owns I/O.
    """
    win_samples = sr * window_secs
    stack = []
    for dt in shift_secs:
        shift_samples = int(round(dt * sr))
        if shift_samples == 0:
            stack.append(scores_flat)
            continue
        shifted = predict_shifted_fn(shift_samples)
        aligned = _align_shift(shifted, shift_samples, n_windows, win_samples)
        stack.append(aligned)
    return np.mean(np.stack(stack, axis=0), axis=0)


def window_roll_tta(
    predict_fn: Callable[[int], np.ndarray],
    shifts: List[int] = (0, 1, -1, 2, -2),
) -> np.ndarray:
    """Legacy integer-window-roll TTA (averages over 12-window circular shifts)."""
    outs = []
    for s in shifts:
        outs.append(predict_fn(s))
    return np.mean(np.stack(outs, axis=0), axis=0)
