import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


PRIMARY_LABEL_SPLIT_RE = re.compile(r"[;,\s]+")


def _is_missing(raw_value: object) -> bool:
    if raw_value is None:
        return True
    try:
        return bool(pd.isna(raw_value))
    except TypeError:
        return False


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


def parse_primary_labels(raw_value: object) -> List[str]:
    if _is_missing(raw_value):
        return []
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text == "[]":
        return []
    return [label for label in PRIMARY_LABEL_SPLIT_RE.split(raw_text) if label]


def parse_secondary_labels(raw_value: object) -> List[str]:
    if _is_missing(raw_value):
        return []
    if isinstance(raw_value, (list, tuple, set)):
        return [str(label).strip() for label in raw_value if str(label).strip()]

    raw_text = str(raw_value).strip()
    if not raw_text or raw_text == "[]":
        return []

    try:
        parsed = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        labels = []
        for value in parsed:
            label = str(value).strip()
            if label:
                labels.append(label)
        return labels

    cleaned = raw_text.strip("[]")
    labels = []
    for value in cleaned.split(","):
        label = value.strip().strip("'").strip('"')
        if label:
            labels.append(label)
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
        if "primary_label" in taxonomy.columns:
            for label in taxonomy["primary_label"].dropna().astype(str).tolist():
                if label not in label_order:
                    label_order.append(label)

    if "primary_label" in metadata.columns:
        for raw_value in metadata["primary_label"].tolist():
            for label in parse_primary_labels(raw_value):
                if label not in label_order:
                    label_order.append(label)

    if not label_order:
        raise ValueError("No labels found in metadata or taxonomy.")

    return {label: idx for idx, label in enumerate(label_order)}


def prepare_train_audio_metadata(metadata: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    required_cols = {"filename", "primary_label"}
    missing = required_cols.difference(metadata.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"train.csv is missing required columns: {missing_list}")

    prepared = metadata.copy()
    prepared = prepared.dropna(subset=["filename", "primary_label"]).reset_index(drop=True)
    prepared["filename"] = prepared["filename"].astype(str)
    prepared["primary_label"] = prepared["primary_label"].astype(str)
    if "secondary_labels" not in prepared.columns:
        prepared["secondary_labels"] = "[]"
    prepared["secondary_labels"] = prepared["secondary_labels"].fillna("[]").astype(str)
    prepared["audio_filepath"] = prepared["filename"].map(lambda name: str(audio_dir / name))
    prepared["source"] = "train_audio"
    return prepared


def prepare_soundscape_metadata(metadata: pd.DataFrame, soundscape_dir: Path) -> pd.DataFrame:
    required_cols = {"filename", "start", "end", "primary_label"}
    missing = required_cols.difference(metadata.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"train_soundscapes_labels.csv is missing required columns: {missing_list}")

    prepared = metadata.copy()
    prepared = prepared.dropna(subset=["filename", "start", "end", "primary_label"]).reset_index(drop=True)
    # train_soundscapes_labels.csv ships every row twice; without this, every
    # soundscape window is fed to training duplicated, which inflates support
    # for already-frequent labels and skews per-class loss weights.
    prepared = prepared.drop_duplicates(
        subset=["filename", "start", "end", "primary_label"]
    ).reset_index(drop=True)
    prepared["filename"] = prepared["filename"].astype(str)
    prepared["primary_label"] = prepared["primary_label"].astype(str)
    prepared["audio_filepath"] = prepared["filename"].map(lambda name: str(soundscape_dir / name))
    prepared["source"] = "train_soundscapes"
    return prepared


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        label_map: Dict[str, int],
        sample_rate: int = 32000,
        duration: float = 5.0,
        training: bool = True,
        preload_audio: bool = True,
        max_cached_files: int = 100,
        preload_workers: int = 1,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.segment_samples = int(duration * sample_rate)
        self.training = training
        self.preload_audio = preload_audio
        self.max_cached_files = max_cached_files
        self.preload_workers = max(1, preload_workers)
        self._audio_cache: OrderedDict[Path, Tensor] = OrderedDict()

        if self.preload_audio:
            self._preload_all_audio()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.metadata.iloc[idx]
        waveform = self._load_waveform(row)
        target = self._encode_target(row)
        return waveform, target

    def _preload_all_audio(self) -> None:
        unique_paths = [
            Path(path_str)
            for path_str in self.metadata["audio_filepath"].dropna().astype(str).drop_duplicates().tolist()
        ]
        if not unique_paths:
            return

        desc = "Preloading train audio" if self.training else "Preloading val audio"
        if self.preload_workers == 1:
            for filepath in tqdm(unique_paths, desc=desc, leave=False):
                self._audio_cache[filepath] = self._load_audio_file(filepath)
            return

        with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            futures = {
                executor.submit(self._load_audio_file, filepath): filepath for filepath in unique_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                filepath = futures[future]
                self._audio_cache[filepath] = future.result()

    def _load_waveform(self, row: pd.Series) -> Tensor:
        filepath = Path(str(row["audio_filepath"]))
        if not filepath.exists():
            raise FileNotFoundError(f"Missing audio file {filepath}")

        waveform = self._load_cached_audio(filepath)

        start_time = row.get("start")
        end_time = row.get("end")
        if isinstance(start_time, str) and isinstance(end_time, str):
            return self._slice_by_times(waveform, start_time, end_time)

        return self._sample_fixed_length_chunk(waveform)

    def _load_audio_file(self, filepath: Path) -> Tensor:
        waveform, sr = _read_audio(filepath)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        return waveform.contiguous()

    def _load_cached_audio(self, filepath: Path) -> Tensor:
        cached = self._audio_cache.get(filepath)
        if cached is not None:
            self._audio_cache.move_to_end(filepath)
            return cached

        waveform = self._load_audio_file(filepath)
        self._audio_cache[filepath] = waveform
        if not self.preload_audio and len(self._audio_cache) > self.max_cached_files:
            self._audio_cache.popitem(last=False)
        return waveform

    def _sample_fixed_length_chunk(self, waveform: Tensor) -> Tensor:
        num_samples = waveform.size(1)
        if num_samples < self.segment_samples:
            return F.pad(waveform, (0, self.segment_samples - num_samples))

        max_start = num_samples - self.segment_samples
        start = torch.randint(0, max_start + 1, (1,)).item() if self.training else max_start // 2
        end = start + self.segment_samples
        return waveform[:, start:end]

    def _slice_by_times(self, waveform: Tensor, start_time: str, end_time: str) -> Tensor:
        total = waveform.size(1)
        start_sample = min(total, max(0, int(_time_to_seconds(start_time) * self.sample_rate)))
        end_sample = min(
            total, max(start_sample + 1, int(_time_to_seconds(end_time) * self.sample_rate))
        )
        chunk = waveform[:, start_sample:end_sample]
        length = chunk.size(1)
        if length < self.segment_samples:
            chunk = F.pad(chunk, (0, self.segment_samples - length))
        elif length > self.segment_samples:
            chunk = chunk[:, : self.segment_samples]
        return chunk

    def _encode_target(self, row: pd.Series) -> Tensor:
        target = torch.zeros(len(self.label_map), dtype=torch.float32)
        labels = parse_primary_labels(row.get("primary_label"))
        if str(row.get("source", "")) == "train_audio":
            labels.extend(parse_secondary_labels(row.get("secondary_labels")))
        for label in labels:
            idx = self.label_map.get(label)
            if idx is not None:
                target[idx] = 1.0
        return target


class SoundscapeSampler:
    """Yield fixed-length raw waveform segments for inference."""

    def __init__(
        self,
        sample_rate: int = 32000,
        duration: float = 5.0,
        hop: float = 5.0,
        preload_audio: bool = True,
        max_cached_files: int = 100,
        preload_workers: int = 1,
    ):
        if hop <= 0:
            raise ValueError(f"hop must be positive, got {hop}")
        self.sample_rate = sample_rate
        self.duration = duration
        self.segment_samples = int(duration * sample_rate)
        self.hop_samples = int(hop * sample_rate)
        self.preload_audio = preload_audio
        self.max_cached_files = max_cached_files
        self.preload_workers = max(1, preload_workers)
        self._audio_cache: OrderedDict[Path, Tensor] = OrderedDict()

    def _load_audio_file(self, soundscape_path: Path) -> Tensor:
        waveform, sr = _read_audio(soundscape_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
        return waveform.contiguous()

    def _load_cached_audio(self, soundscape_path: Path) -> Tensor:
        cached = self._audio_cache.get(soundscape_path)
        if cached is not None:
            self._audio_cache.move_to_end(soundscape_path)
            return cached

        waveform = self._load_audio_file(soundscape_path)
        self._audio_cache[soundscape_path] = waveform
        if not self.preload_audio and len(self._audio_cache) > self.max_cached_files:
            self._audio_cache.popitem(last=False)
        return waveform

    def iterate_segment_items(self, soundscape_path: Path) -> Iterator[Tuple[int, Tensor]]:
        waveform = self._load_cached_audio(soundscape_path)

        total = waveform.size(1)
        if total == 0:
            return

        start = 0
        while start < total:
            end = min(total, start + self.segment_samples)
            chunk = waveform[:, start:end]
            if chunk.size(1) < self.segment_samples:
                chunk = F.pad(chunk, (0, self.segment_samples - chunk.size(1)))
            segment_end_seconds = int(
                round((start + self.segment_samples) / self.sample_rate)
            )
            yield segment_end_seconds, chunk
            start += self.hop_samples
