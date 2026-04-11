import math
import inspect
from typing import Any, Dict

import torch
from torch import Tensor, nn

from birdclef_example.data import SpectrogramTransform


def build_sincos_position_encoding(
    seq_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    encoding = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding.unsqueeze(0).to(dtype=dtype)


class AttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.score(self.norm(x)).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class HybridPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.attention_pool = AttentionPool(dim=dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        attention_pooled = self.attention_pool(x)
        mean_pooled = x.mean(dim=1)
        return 0.5 * (attention_pooled + mean_pooled)


class SpecAugment(nn.Module):
    """Apply time/frequency masking on log-mel spectrograms."""

    def __init__(
        self,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        num_masks: int = 0,
        spec_noise_std: float = 0.0,
    ):
        super().__init__()
        self.freq_mask_param = max(0, int(freq_mask_param))
        self.time_mask_param = max(0, int(time_mask_param))
        self.num_masks = max(0, int(num_masks))
        self.spec_noise_std = max(0.0, float(spec_noise_std))

    def _mask_axis(self, spec: Tensor, axis: int, mask_param: int) -> Tensor:
        if mask_param <= 0:
            return spec
        dim_size = spec.size(axis)
        if dim_size <= 1:
            return spec

        max_width = min(mask_param, dim_size - 1)
        if max_width < 1:
            return spec

        for _ in range(self.num_masks):
            width = int(torch.randint(0, max_width + 1, (1,), device=spec.device).item())
            if width == 0:
                continue
            start_max = dim_size - width
            start = int(torch.randint(0, start_max + 1, (1,), device=spec.device).item())
            if axis == 2:
                spec[:, :, start : start + width, :] = 0.0
            elif axis == 3:
                spec[:, :, :, start : start + width] = 0.0
        return spec

    def forward(self, spec: Tensor) -> Tensor:
        if self.num_masks <= 0:
            if self.spec_noise_std > 0:
                return spec + torch.randn_like(spec) * self.spec_noise_std
            return spec

        spec = spec.clone()
        spec = self._mask_axis(spec, axis=2, mask_param=self.freq_mask_param)
        spec = self._mask_axis(spec, axis=3, mask_param=self.time_mask_param)
        if self.spec_noise_std > 0:
            spec = spec + torch.randn_like(spec) * self.spec_noise_std
        return spec


class SimpleCNN(nn.Module):
    """Compact spectrogram transformer baseline for BirdCLEF."""

    def __init__(
        self,
        n_classes: int,
        dropout: float = 0.3,
        sample_rate: int = 32000,
        n_mels: int = 160,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: int = 20,
        f_max: int | None = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        token_grid_size: int = 22,
        pooling: str = "hybrid",
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        specaugment_masks: int = 0,
        spec_noise_std: float = 0.0,
    ):
        super().__init__()
        if pooling not in {"attention", "hybrid"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        self.spectrogram = SpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        )
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_masks=specaugment_masks,
            spec_noise_std=spec_noise_std,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if pooling == "hybrid":
            self.pool = HybridPool(embed_dim, dropout=dropout)
        else:
            self.pool = AttentionPool(embed_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectrogram(x)
        if self.training:
            x = self.spec_augment(x)
        x = self.stem(x)
        x = self.token_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + build_sincos_position_encoding(
            seq_len=x.size(1),
            dim=x.size(2),
            device=x.device,
            dtype=x.dtype,
        )
        x = self.encoder(x)
        x = self.pool(x)
        return self.head(x)


class ConvNetClassifier(nn.Module):
    """Pure convolutional spectrogram encoder baseline."""

    def __init__(
        self,
        n_classes: int,
        dropout: float = 0.3,
        sample_rate: int = 32000,
        n_mels: int = 160,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: int = 20,
        f_max: int | None = None,
        channels: int = 48,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        specaugment_masks: int = 0,
        spec_noise_std: float = 0.0,
    ):
        super().__init__()
        self.spectrogram = SpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        )
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_masks=specaugment_masks,
            spec_noise_std=spec_noise_std,
        )
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 4, channels * 6, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 6),
            nn.SiLU(inplace=True),
        )
        hidden = channels * 6
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectrogram(x)
        if self.training:
            x = self.spec_augment(x)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def _filter_model_kwargs(model_cls: type[nn.Module], model_config: Dict[str, Any]) -> Dict[str, Any]:
    valid = set(inspect.signature(model_cls.__init__).parameters.keys())
    valid.discard("self")
    return {key: value for key, value in model_config.items() if key in valid}


def build_model(architecture: str, model_config: Dict[str, Any]) -> nn.Module:
    architecture = architecture.lower()
    model_registry: Dict[str, type[nn.Module]] = {
        "spec_transformer": SimpleCNN,
        "simple_transformer": SimpleCNN,
        "simplecnn": SimpleCNN,
        "convnet": ConvNetClassifier,
    }
    model_cls = model_registry.get(architecture)
    if model_cls is None:
        supported = ", ".join(sorted(model_registry.keys()))
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {supported}")
    return model_cls(**_filter_model_kwargs(model_cls, model_config))
