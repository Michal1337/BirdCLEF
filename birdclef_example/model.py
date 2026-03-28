from torch import Tensor, nn

from birdclef_example.data import SpectrogramTransform


class SimpleCNN(nn.Module):
    """Lightweight CNN to transform a waveform into species logits."""

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.3,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: int = 20,
        f_max: int | None = None,
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
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, base_channels),
            self._conv_block(base_channels, base_channels * 2),
            self._conv_block(base_channels * 2, base_channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(base_channels * 4, n_classes),
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectrogram(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)
