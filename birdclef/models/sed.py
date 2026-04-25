"""SED model: timm backbone + attention-pool head + spectrogram frontend.

Produces per-clip multilabel logits for 5 s windows (one prediction per clip).
For 60 s soundscape inference, caller chops waveform into 12 windows and runs
in batches of 12.

Two front-end implementations:
  - MelFrontend:  torchaudio.transforms.MelSpectrogram. Used at train time.
  - ConvMelSpectrogram + SEDExportWrapper: Conv1d-based mel-spec that uses
    only ONNX-exportable ops (no complex STFT). Used at export time only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SEDConfig:
    backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    n_classes: int = 234
    dropout: float = 0.30
    sample_rate: int = 32000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 320
    f_min: int = 20
    f_max: int = 16000
    freq_mask_param: int = 24
    time_mask_param: int = 32
    specaugment_masks: int = 2
    spec_noise_std: float = 0.01


class MelFrontend(nn.Module):
    def __init__(self, cfg: SEDConfig):
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] or [B, 1, T]
        if x.ndim == 3:
            x = x.squeeze(1)
        # FFT under fp16 autocast is a well-known source of NaN/Inf. Force
        # the entire mel front-end to run in fp32 regardless of outer AMP
        # context; the backbone+head still enjoy autocast outside this block.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            s = self.mel(x)
            s = self.to_db(s)
            # Normalize per-sample
            m = s.mean(dim=(-2, -1), keepdim=True)
            v = s.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            s = (s - m) / v
        return s.unsqueeze(1)  # [B, 1, n_mels, T]


class AttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(self.norm(x)).squeeze(-1)
        w = torch.softmax(w, dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class SED(nn.Module):
    def __init__(self, cfg: SEDConfig):
        super().__init__()
        import timm
        from birdclef.data.augment import SpecAugment

        self.cfg = cfg
        self.frontend = MelFrontend(cfg)
        self.spec_aug = SpecAugment(
            freq_mask_param=cfg.freq_mask_param,
            time_mask_param=cfg.time_mask_param,
            num_masks=cfg.specaugment_masks,
            spec_noise_std=cfg.spec_noise_std,
        )
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True, in_chans=1, num_classes=0, global_pool="",
        )
        feat_dim = int(self.backbone.num_features)
        self.pool = AttentionPool(feat_dim, dropout=cfg.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, cfg.n_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.frontend(waveform)
        if self.training:
            x = self.spec_aug(x)
        feats = self.backbone(x)
        if feats.ndim == 4:
            # [B, C, H, W] -> treat W as time steps
            feats = feats.mean(dim=2).transpose(1, 2)  # [B, W, C]
        elif feats.ndim > 2:
            feats = feats.flatten(1).unsqueeze(1)
        else:
            feats = feats.unsqueeze(1)
        pooled = self.pool(feats)
        return self.head(pooled)


# ────────────────────────────────────────────────────────────────────────────
# ONNX-friendly mel front-end (export only)
# ────────────────────────────────────────────────────────────────────────────
class ConvMelSpectrogram(nn.Module):
    """Mel power-spectrogram → log-mel using only Conv1d / MatMul / Log.

    Numerically equivalent to torchaudio's
    `MelSpectrogram(power=2.0) + AmplitudeToDB(stype='power', top_db=80)`
    modulo:
      - constant-zero padding (vs torchaudio's reflect) at clip boundaries.
      - float-precision drift in the DFT basis matrix.

    Used only when exporting SED to ONNX, since `torch.stft` returns complex
    tensors which the legacy ONNX exporter (opset ≤17) cannot serialize.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.top_db = float(top_db)

        # Hann window × DFT basis, packed as Conv1d weights.
        window = torch.hann_window(self.n_fft, periodic=True).float()
        n_idx = torch.arange(self.n_fft, dtype=torch.float32)
        k_idx = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32)
        # e^{-i 2π k n / N}  → real and imag parts
        angle = 2.0 * math.pi * k_idx.unsqueeze(1) * n_idx.unsqueeze(0) / float(self.n_fft)
        basis_real = torch.cos(angle) * window.unsqueeze(0)
        basis_imag = -torch.sin(angle) * window.unsqueeze(0)
        # Stack: (2*(n_fft//2+1), 1, n_fft)
        weight = torch.cat([basis_real, basis_imag], dim=0).unsqueeze(1)
        self.register_buffer("stft_weight", weight)

        # Mel filterbank from torchaudio for exact training-time match.
        from torchaudio.functional import melscale_fbanks
        mel_fb = melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=float(f_min),
            f_max=float(f_max),
            n_mels=int(n_mels),
            sample_rate=int(sample_rate),
        )  # (n_freqs, n_mels)
        # Keep transposed copy so forward only uses MatMul on a fixed tensor.
        self.register_buffer("mel_fb_t", mel_fb.t().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) waveform → (B, n_mels, frames) log-mel power spectrogram
        if x.ndim == 3:
            x = x.squeeze(1)
        # Center=True: pad n_fft//2 on each side. Constant pad (not reflect)
        # so the export graph stays simple. Boundary samples differ by at most
        # ~n_fft/2 / WINDOW_SAMPLES = 1 % of the clip; macro-AUC unaffected.
        pad = self.n_fft // 2
        x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode="constant", value=0.0)
        stft = F.conv1d(x_padded, self.stft_weight, stride=self.hop_length)
        n_freq = self.n_fft // 2 + 1
        real = stft[:, :n_freq, :]
        imag = stft[:, n_freq:, :]
        power = real * real + imag * imag                        # (B, n_freq, frames)
        # mel_fb_t @ power : (B, n_mels, frames)
        mel = torch.matmul(self.mel_fb_t, power)
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        # top_db clamp to match torchaudio.transforms.AmplitudeToDB
        log_mel_max = log_mel.amax(dim=(-2, -1), keepdim=True)
        log_mel = torch.maximum(log_mel, log_mel_max - self.top_db)
        return log_mel


class SEDExportWrapper(nn.Module):
    """Drop-in replacement for SED at ONNX export time.

    Same forward signature (waveform → logits), same trained weights, but
    swaps the torchaudio mel front-end for the ONNX-safe `ConvMelSpectrogram`.
    Skips SpecAugment (eval-mode only).
    """

    def __init__(self, sed: SED):
        super().__init__()
        cfg = sed.cfg
        self.mel = ConvMelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max,
        )
        self.backbone = sed.backbone
        self.pool = sed.pool
        self.head = sed.head

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        log_mel = self.mel(waveform)                              # (B, n_mels, frames)

        # Per-sample standardization, equivalent to MelFrontend's
        # `(x - x.mean()) / x.std().clamp(min=1e-6)` but written with only
        # ReduceMean/Mul/Sqrt/Clamp so the ONNX graph stays NaN-safe even on
        # constant inputs (where torch.std emits sqrt(~0) and ORT silently
        # returns NaN that clamp() cannot repair).
        m = log_mel.mean(dim=(-2, -1), keepdim=True)
        diff = log_mel - m
        var = (diff * diff).mean(dim=(-2, -1), keepdim=True)
        # Clamp variance BEFORE sqrt so we never feed sqrt a negative number
        # (variance can drift slightly negative through fp accumulation), then
        # clamp std again as a guard for very low-variance inputs.
        std = torch.sqrt(var.clamp(min=1e-12)).clamp(min=1e-6)
        x = diff / std
        x = x.unsqueeze(1)                                        # (B, 1, n_mels, frames)
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = feats.mean(dim=2).transpose(1, 2)
        elif feats.ndim > 2:
            feats = feats.flatten(1).unsqueeze(1)
        else:
            feats = feats.unsqueeze(1)
        pooled = self.pool(feats)
        return self.head(pooled)
