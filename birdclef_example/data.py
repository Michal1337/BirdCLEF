import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class SpectrogramTransform(nn.Module):
    """Convert raw audio to a normalized log-mel spectrogram."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: int = 20,
        f_max: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.normalize = normalize

    def forward(self, waveform: Tensor) -> Tensor:
        spec = self.melspec(waveform)
        spec = self.to_db(spec)
        if self.normalize:
            mean = spec.mean(dim=[-2, -1], keepdim=True)
            std = spec.std(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
            spec = (spec - mean) / std
        return spec


def _split_secondary_labels(raw_value: Optional[str]) -> List[str]:
    if not raw_value or not isinstance(raw_value, str):
        return []
    labels = [label for label in re.split(r"[,;\s]+", raw_value.strip()) if label]
    return labels


def _read_audio(filepath: Path) -> Tuple[Tensor, int]:
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T.copy())
    return waveform, sr


def _time_to_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def build_label_map(
    metadata: pd.DataFrame, taxonomy_csv: Optional[Path] = None
) -> Dict[str, int]:
    label_order: List[str] = []

    if taxonomy_csv and taxonomy_csv.exists():
        taxonomy = pd.read_csv(taxonomy_csv)
        for col in ("primary_label", "ebird_code", "species_id"):
            if col in taxonomy.columns:
                for label in taxonomy[col].dropna().astype(str).tolist():
                    if label not in label_order:
                        label_order.append(label)
                break

    if "primary_label" in metadata.columns:
        for label in metadata["primary_label"].dropna().astype(str).tolist():
            if label not in label_order:
                label_order.append(label)

    if "secondary_labels" in metadata.columns:
        secondary_series = metadata["secondary_labels"].dropna()
        for raw in secondary_series.astype(str):
            for label in _split_secondary_labels(raw):
                if label not in label_order:
                    label_order.append(label)

    if not label_order:
        raise ValueError("No labels found in metadata or taxonomy.")

    return {label: idx for idx, label in enumerate(label_order)}


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        audio_dir: Path,
        metadata: pd.DataFrame,
        label_map: Dict[str, int],
        sample_rate: int = 32000,
        duration: float = 5.0,
        transform: Optional[SpectrogramTransform] = None,
        training: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        self.metadata = metadata.reset_index(drop=True)
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.segment_samples = int(duration * sample_rate)
        self.training = training
        self.transform = transform or SpectrogramTransform(sample_rate=sample_rate)
        self._cache_path: Optional[Path] = None
        self._cache_waveform: Optional[Tensor] = None
        self._cache_sr: Optional[int] = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.metadata.iloc[idx]
        waveform = self._load_waveform(row)
        spec = self.transform(waveform)
        target = self._encode_target(row)
        return spec, target

    def _load_waveform(self, row: pd.Series) -> Tensor:
        filename = str(row.get("filename") or row.get("recording_id"))
        filepath = self.audio_dir / filename
        if not filepath.exists():
            species_folder = row.get("primary_label") or row.get("common_name")
            filepath = self.audio_dir / species_folder / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing audio file {filename} in {self.audio_dir}")

        waveform, sr = self._load_cached_audio(filepath)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        start_time = row.get("start")
        end_time = row.get("end")
        if isinstance(start_time, str) and isinstance(end_time, str):
            return self._slice_by_times(waveform, start_time, end_time)
        num_samples = waveform.size(1)
        if num_samples < self.segment_samples:
            pad_len = self.segment_samples - num_samples
            waveform = F.pad(waveform, (0, pad_len))
            return waveform

        max_start = num_samples - self.segment_samples
        if self.training:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2
        end = start + self.segment_samples
        return waveform[:, start:end]

    def _load_cached_audio(self, filepath: Path) -> Tuple[Tensor, int]:
        if self._cache_path == filepath and self._cache_waveform is not None and self._cache_sr is not None:
            return self._cache_waveform, self._cache_sr
        waveform, sr = _read_audio(filepath)
        self._cache_path = filepath
        self._cache_waveform = waveform
        self._cache_sr = sr
        return waveform, sr

    def _slice_by_times(self, waveform: Tensor, start_time: str, end_time: str) -> Tensor:
        total = waveform.size(1)
        start_sample = min(total, max(0, int(_time_to_seconds(start_time) * self.sample_rate)))
        end_sample = min(
            total, max(start_sample + 1, int(_time_to_seconds(end_time) * self.sample_rate))
        )
        chunk = waveform[:, start_sample:end_sample]
        length = chunk.size(1)
        if length < self.segment_samples:
            pad_len = self.segment_samples - length
            chunk = F.pad(chunk, (0, pad_len))
        elif length > self.segment_samples:
            chunk = chunk[:, : self.segment_samples]
        return chunk

    def _encode_target(self, row: pd.Series) -> Tensor:
        target = torch.zeros(len(self.label_map), dtype=torch.float32)
        primary = str(row.get("primary_label", ""))
        primary_idx = self.label_map.get(primary)
        if primary_idx is not None:
            target[primary_idx] = 1.0

        secondary = _split_secondary_labels(row.get("secondary_labels"))
        for label in secondary:
            idx = self.label_map.get(label)
            if idx is not None:
                target[idx] = 1.0
        return target


class SoundscapeSampler:
    """Yield fixed-length segments for each soundscape for inference."""

    def __init__(
        self,
        sample_rate: int = 32000,
        duration: float = 5.0,
        hop: float = 2.5,
        transform: Optional[SpectrogramTransform] = None,
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(duration * sample_rate)
        self.hop_samples = int(hop * sample_rate)
        self.transform = transform or SpectrogramTransform(sample_rate=sample_rate)

    def iterate_segments(self, soundscape_path: Path) -> Iterable[Tensor]:
        waveform, sr = _read_audio(soundscape_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        total = waveform.size(1)
        start = 0
        while start + self.segment_samples <= total:
            chunk = waveform[:, start : start + self.segment_samples]
            yield self.transform(chunk)
            start += self.hop_samples
        if start < total:
            remainder = waveform[:, start:]
            pad = self.segment_samples - remainder.size(1)
            remainder = F.pad(remainder, (0, pad))
            yield self.transform(remainder)


import re
