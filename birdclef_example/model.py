import math

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
    def __init__(
        self,
        freq_mask_param: int = 12,
        time_mask_param: int = 24,
        num_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = max(0, freq_mask_param)
        self.time_mask_param = max(0, time_mask_param)
        self.num_masks = max(0, num_masks)

    def _sample_width(self, max_width: int, limit: int, device: torch.device) -> int:
        width_limit = min(max_width, limit)
        if width_limit <= 0:
            return 0
        return int(torch.randint(0, width_limit + 1, (1,), device=device).item())

    def _apply_mask(self, x: Tensor, dim: int, max_width: int) -> Tensor:
        size = x.size(dim)
        width = self._sample_width(max_width=max_width, limit=size, device=x.device)
        if width <= 0 or width >= size:
            return x
        start = int(torch.randint(0, size - width + 1, (1,), device=x.device).item())
        masked = x.clone()
        if dim == -2:
            masked[:, :, start : start + width, :] = 0
        else:
            masked[:, :, :, start : start + width] = 0
        return masked

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.num_masks == 0:
            return x
        for _ in range(self.num_masks):
            x = self._apply_mask(x, dim=-2, max_width=self.freq_mask_param)
            x = self._apply_mask(x, dim=-1, max_width=self.time_mask_param)
        return x


class SimpleCNN(nn.Module):
    """Compact spectrogram transformer for BirdCLEF."""

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
        freq_mask_param: int = 12,
        time_mask_param: int = 24,
        specaugment_masks: int = 2,
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
        self.specaugment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_masks=specaugment_masks,
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
        x = self.specaugment(x)
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
