"""Waveform and spectrogram augmentations for SED training.

Designed to be used on GPU where possible; avoids torchaudio biquad kernels
because they've been unstable under autocast on some CUDA versions.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpecAugment(nn.Module):
    """Standard time+freq masking on log-mel."""

    def __init__(
        self,
        freq_mask_param: int = 24,
        time_mask_param: int = 32,
        num_masks: int = 2,
        spec_noise_std: float = 0.01,
    ):
        super().__init__()
        self.f = int(freq_mask_param)
        self.t = int(time_mask_param)
        self.n = int(num_masks)
        self.std = float(spec_noise_std)

    def forward(self, spec: Tensor) -> Tensor:
        if spec.ndim != 4:
            raise ValueError(f"SpecAugment expects [B,1,F,T], got {spec.shape}")
        if self.std > 0:
            spec = spec + torch.randn_like(spec) * self.std
        if self.n <= 0:
            return spec
        B, _, F_, T = spec.shape
        for _ in range(self.n):
            if self.f > 0 and F_ > 1:
                w = int(torch.randint(0, min(self.f, F_ - 1) + 1, (1,)).item())
                if w > 0:
                    s = int(torch.randint(0, F_ - w + 1, (1,)).item())
                    spec[:, :, s : s + w, :] = spec.mean()
            if self.t > 0 and T > 1:
                w = int(torch.randint(0, min(self.t, T - 1) + 1, (1,)).item())
                if w > 0:
                    s = int(torch.randint(0, T - w + 1, (1,)).item())
                    spec[:, :, :, s : s + w] = spec.mean()
        return spec


class WaveformAug(nn.Module):
    """Gain, noise, optional bandlimit smoothing."""

    def __init__(self, gain_prob: float = 0.7, noise_prob: float = 0.5, filter_prob: float = 0.2):
        super().__init__()
        self.gp = float(gain_prob); self.np_ = float(noise_prob); self.fp = float(filter_prob)

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        for i in range(B):
            xi = x[i : i + 1]
            if torch.rand(1, device=x.device).item() < self.gp:
                g = float(torch.empty(1, device=x.device).uniform_(0.7, 1.3).item())
                xi = xi * g
            if torch.rand(1, device=x.device).item() < self.np_:
                base = xi.std().clamp(min=1e-4)
                amp = float(torch.empty(1, device=x.device).uniform_(0.002, 0.02).item())
                xi = xi + torch.randn_like(xi) * (base * amp)
            if torch.rand(1, device=x.device).item() < self.fp:
                k = int(torch.randint(5, 33, (1,), device=x.device).item()) | 1
                lo = F.avg_pool1d(xi, kernel_size=k, stride=1, padding=k // 2)
                xi = lo if torch.rand(1, device=x.device).item() < 0.5 else xi - lo
            x[i : i + 1] = xi
        return x


def mixup(
    x: Tensor, y: Tensor, alpha: float = 0.5, mode: str = "max",
) -> tuple[Tensor, Tensor]:
    """Waveform-level mixup.

    mode="max": label = max(y, y[perm])  — additive recipe used by 2024 winners.
    mode="convex": label = lam*y + (1-lam)*y[perm].
    """
    if alpha <= 0:
        return x, y
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = lam * x + (1.0 - lam) * x[perm]
    if mode == "max":
        y2 = torch.maximum(y, y[perm])
    else:
        y2 = lam * y + (1.0 - lam) * y[perm]
    return x2, y2


def background_mix(
    x: Tensor, bg_bank: Tensor, prob: float = 0.5, snr_db_range: tuple[float, float] = (3.0, 20.0),
) -> Tensor:
    """Mix a random background sample into x at a random SNR."""
    B = x.size(0)
    if bg_bank is None or bg_bank.size(0) == 0 or prob <= 0:
        return x
    for i in range(B):
        if torch.rand(1).item() >= prob:
            continue
        j = int(torch.randint(0, bg_bank.size(0), (1,)).item())
        bg = bg_bank[j]
        if bg.size(-1) < x.size(-1):
            bg = bg.repeat(int(x.size(-1) // bg.size(-1) + 1))[: x.size(-1)]
        else:
            start = int(torch.randint(0, bg.size(-1) - x.size(-1) + 1, (1,)).item())
            bg = bg[start : start + x.size(-1)]
        snr_db = float(torch.empty(1).uniform_(*snr_db_range).item())
        sig_pwr = x[i].pow(2).mean().clamp(min=1e-8)
        bg_pwr = bg.pow(2).mean().clamp(min=1e-8)
        k = torch.sqrt(sig_pwr / (bg_pwr * (10.0 ** (snr_db / 10.0))))
        x[i] = x[i] + k * bg.to(x.device)
    return x
