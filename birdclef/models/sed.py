"""SED model: timm backbone + attention-pool head + spectrogram frontend.

Produces per-clip multilabel logits for 5 s windows (one prediction per clip).
For 60 s soundscape inference, caller chops waveform into 12 windows and runs
in batches of 12.
"""
from __future__ import annotations

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
