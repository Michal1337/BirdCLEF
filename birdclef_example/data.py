import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


N_WINDOWS_PER_FILE = 12       # 12 × 5 s = 60 s soundscape file
WINDOW_SECONDS = 5
SOUNDSCAPE_SOURCES = {"pseudo_soundscape", "train_soundscapes"}


def _try_load_soundscape_memmap():
    """Best-effort load of birdclef/cache/soundscapes/soundscapes_f16_32k.npy.

    Returns (memmap[n_files, FILE_SAMPLES], filename → row_idx dict, sr) or
    (None, {}, 0) if the cache isn't built. Lets BirdCLEFDataset short-circuit
    soundscape OGG decodes — a 100x speedup over per-call sf.read at random
    access patterns when ~10k soundscapes are mixed into the train pool.
    """
    try:
        from birdclef.config.paths import (
            FILE_SAMPLES, SOUNDSCAPE_NPY, SOUNDSCAPE_INDEX, SR,
        )
    except Exception:
        return None, {}, 0
    if not SOUNDSCAPE_NPY.exists() or not SOUNDSCAPE_INDEX.exists():
        return None, {}, 0
    try:
        store = np.load(SOUNDSCAPE_NPY, mmap_mode="r")
        idx = pd.read_parquet(SOUNDSCAPE_INDEX)
        if "ok" in idx.columns:
            idx = idx[idx["ok"] == 1]
        row_by_file = dict(
            zip(idx["filename"].astype(str), idx["row_idx"].astype(int))
        )
        return store, row_by_file, int(SR)
    except Exception:
        return None, {}, 0


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


def _format_seconds(s: int) -> str:
    """Seconds-int → "0:00:SS" / "0:00:00"-style time string the dataset slices on."""
    return f"0:{(s // 60):02d}:{(s % 60):02d}"


def load_pseudo_round(pseudo_dir: Path, rnd: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Read birdclef/cache/pseudo/round{N}/.

    Returns (probs[N_rows, n_classes] float32,
             keep_mask[N_rows, n_classes] uint8,
             meta DataFrame with columns row_id/filename/window/(is_labeled),
             info dict from info.json — empty dict if missing).
    """
    rdir = Path(pseudo_dir) / f"round{int(rnd)}"
    meta_path = rdir / "meta.parquet"
    probs_path = rdir / "probs.npz"
    if not meta_path.exists() or not probs_path.exists():
        raise FileNotFoundError(
            f"Pseudo round {rnd} not found under {rdir}. "
            "Run `python -m birdclef.scripts._05_pseudo_label --round N` first."
        )
    meta = pd.read_parquet(meta_path)
    arrs = np.load(probs_path)
    # _05_pseudo_label writes either {'probs', 'keep_mask'} (SED teacher) or
    # {'final', 'first_pass', 'keep_mask'} (SSM teacher). Prefer 'final'/'probs'.
    if "final" in arrs.files:
        probs = arrs["final"].astype(np.float32)
    elif "probs" in arrs.files:
        probs = arrs["probs"].astype(np.float32)
    else:
        raise ValueError(f"{probs_path}: no 'probs' or 'final' array found")
    keep = arrs["keep_mask"].astype(np.uint8) if "keep_mask" in arrs.files else np.ones_like(probs, dtype=np.uint8)
    info_path = rdir / "info.json"
    info: dict = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return probs, keep, meta, info


def prepare_pseudo_soundscape_metadata(
    pseudo_meta: pd.DataFrame,
    keep_mask: np.ndarray,
    soundscape_dir: Path,
    require_kept_positive: bool = True,
    exclude_filenames: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build BirdCLEFDataset-compatible rows for every (file, window) in the
    pseudo cache, with `pseudo_row` indexing into the (probs, keep_mask) arrays.

    require_kept_positive: drop rows whose keep_mask is entirely zero — those
    contribute no gradient and just bloat the loader.

    exclude_filenames: drop pseudo rows whose filename is in this set. CRITICAL
    when the val pool overlaps with the pseudo cache — pass the set of
    val-pool filenames to prevent training on val audio (the teacher
    over-fits its own training rows; using those as pseudo-targets and
    evaluating on the same audio is a leak that inflates val AUC by
    +0.10-0.15).
    """
    if "filename" not in pseudo_meta.columns or "window" not in pseudo_meta.columns:
        raise ValueError(
            "pseudo meta must have columns 'filename' and 'window'. "
            f"Found: {list(pseudo_meta.columns)}"
        )
    df = pseudo_meta.copy().reset_index(drop=True)
    df["pseudo_row"] = df.index.astype(np.int64)
    if require_kept_positive:
        any_kept = keep_mask.any(axis=1)
        df = df[any_kept[df["pseudo_row"].to_numpy()]].reset_index(drop=True)

    df["filename"] = df["filename"].astype(str)
    if exclude_filenames:
        before = len(df)
        df = df[~df["filename"].isin(set(exclude_filenames))].reset_index(drop=True)
        # Caller log: how many rows we cut. The kept count should equal
        # (eligible_files - val_files) × N_WINDOWS_PER_FILE before any
        # require_kept_positive filtering trims further.
        print(f"[pseudo] excluded {before - len(df)} pseudo rows whose "
              f"filename was in val pool (leak prevention)")
    df["window"] = df["window"].astype(int).clip(0, N_WINDOWS_PER_FILE - 1)
    df["start"] = df["window"].map(lambda w: _format_seconds(int(w) * WINDOW_SECONDS))
    df["end"] = df["window"].map(lambda w: _format_seconds(int(w + 1) * WINDOW_SECONDS))
    df["audio_filepath"] = df["filename"].map(lambda name: str(soundscape_dir / name))
    df["primary_label"] = ""           # ignored — target comes from pseudo_row
    df["secondary_labels"] = "[]"
    df["source"] = "pseudo_soundscape"
    return df[["filename", "start", "end", "audio_filepath", "primary_label",
               "secondary_labels", "source", "pseudo_row"]]


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
        pseudo_probs: Optional["np.ndarray"] = None,
        pseudo_keep: Optional["np.ndarray"] = None,
    ):
        """
        pseudo_probs: optional (N, n_classes) float32 — soft pseudo-targets
            from `_05_pseudo_label`. Rows whose `metadata['source'] ==
            'pseudo_soundscape'` look up their target via `metadata['pseudo_row']`.
        pseudo_keep: optional (N, n_classes) uint8 — per-cell loss mask for the
            pseudo rows. Cells with keep=0 contribute zero gradient.

        When BOTH pseudo arrays are provided, `__getitem__` returns a 3-tuple
        `(waveform, target, loss_mask)`. Otherwise it returns the legacy 2-tuple
        `(waveform, target)` so existing trainers / val loaders keep working
        unchanged.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.segment_samples = int(duration * sample_rate)
        self.training = training
        self.preload_audio = preload_audio
        self.max_cached_files = max_cached_files
        self.preload_workers = max(1, preload_workers)
        self._audio_cache: OrderedDict[Path, Tensor] = OrderedDict()

        # Fast path for soundscape rows (pseudo or labeled): an mmap'd
        # float16 32 kHz array of every decoded soundscape. Avoids the
        # OGG-decode-per-getitem cost that murders throughput when ~10k
        # pseudo soundscapes are mixed into the train pool. Built by
        # `python -m birdclef.cache.build_soundscape_cache`. If the cache
        # isn't present, _sc_store stays None and we fall back to OGG decode.
        self._sc_store, self._sc_row_by_file, self._sc_store_sr = _try_load_soundscape_memmap()
        if self._sc_store is not None:
            n_sc_in_meta = int(self.metadata["source"].isin(SOUNDSCAPE_SOURCES).sum()) \
                if "source" in self.metadata.columns else 0
            print(f"[BirdCLEFDataset] soundscape memmap fast path active "
                  f"({len(self._sc_row_by_file)} files indexed; "
                  f"{n_sc_in_meta} soundscape rows in this dataset's metadata)")

        # Pseudo plumbing. Both arrays must be set together — any None disables
        # the pseudo path and reverts to legacy 2-tuple output.
        self._pseudo_probs = pseudo_probs
        self._pseudo_keep = pseudo_keep
        self._pseudo_active = (pseudo_probs is not None) and (pseudo_keep is not None)
        if self._pseudo_active:
            if pseudo_probs.shape != pseudo_keep.shape:
                raise ValueError(
                    f"pseudo_probs {pseudo_probs.shape} and pseudo_keep "
                    f"{pseudo_keep.shape} must have matching shape."
                )
            if "source" not in self.metadata.columns:
                raise ValueError(
                    "metadata needs a 'source' column to identify pseudo rows "
                    "(values like 'pseudo_soundscape', 'train_audio', "
                    "'train_soundscapes')."
                )
            if (
                "pseudo_row" not in self.metadata.columns
                and (self.metadata["source"] == "pseudo_soundscape").any()
            ):
                raise ValueError(
                    "metadata has 'pseudo_soundscape' rows but no 'pseudo_row' "
                    "column to index them — build with "
                    "`prepare_pseudo_soundscape_metadata`."
                )
            self._n_classes_pseudo = int(pseudo_probs.shape[1])
            if self._n_classes_pseudo != len(label_map):
                raise ValueError(
                    f"pseudo array width {self._n_classes_pseudo} mismatches "
                    f"label_map ({len(label_map)})."
                )

        if self.preload_audio:
            self._preload_all_audio()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        waveform = self._load_waveform(row)
        if not self._pseudo_active:
            target = self._encode_target(row)
            return waveform, target

        # Pseudo path: 3-tuple with per-cell loss mask.
        source = str(row.get("source", ""))
        if source == "pseudo_soundscape":
            pr = int(row["pseudo_row"])
            target = torch.from_numpy(self._pseudo_probs[pr].astype("float32"))
            mask = torch.from_numpy(self._pseudo_keep[pr].astype("float32"))
        else:
            target = self._encode_target(row)
            mask = torch.ones_like(target)
        return waveform, target, mask

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
        # Fast path for soundscape rows when the memmap cache is loaded.
        # Bypasses OGG decode entirely — slices a (1, win_samples) view from
        # the (n_files, FILE_SAMPLES) float16 mmap. Each soundscape sample
        # becomes O(1) memcpy instead of an O(60s) sf.read + decode.
        if (
            self._sc_store is not None
            and "source" in row.index
            and str(row["source"]) in SOUNDSCAPE_SOURCES
            and self.sample_rate == self._sc_store_sr
        ):
            filename = str(row["filename"])
            sc_row = self._sc_row_by_file.get(filename)
            if sc_row is not None:
                start_time = row.get("start")
                end_time = row.get("end")
                if isinstance(start_time, str) and isinstance(end_time, str):
                    s_off = int(_time_to_seconds(start_time) * self._sc_store_sr)
                    e_off = int(_time_to_seconds(end_time)   * self._sc_store_sr)
                else:
                    # No explicit window range — fall back to a uniform random
                    # 5s window inside the 60s file. Matches the OGG path's
                    # `_sample_fixed_length_chunk` semantics.
                    max_start = max(1, self._sc_store.shape[1] - self.segment_samples)
                    s_off = int(torch.randint(0, max_start + 1, (1,)).item()
                                if self.training else max_start // 2)
                    e_off = s_off + self.segment_samples
                # Clamp to memmap bounds + pad/truncate to segment_samples.
                s_off = max(0, min(s_off, self._sc_store.shape[1]))
                e_off = max(s_off + 1, min(e_off, self._sc_store.shape[1]))
                slc = np.asarray(
                    self._sc_store[sc_row, s_off:e_off], dtype=np.float32,
                )
                wave = torch.from_numpy(slc).unsqueeze(0)   # (1, T)
                length = wave.size(1)
                if length < self.segment_samples:
                    wave = F.pad(wave, (0, self.segment_samples - length))
                elif length > self.segment_samples:
                    wave = wave[:, : self.segment_samples]
                return wave.contiguous()

        # Slow path: OGG decode + LRU cache. Used for focal recordings and
        # any soundscape file the memmap doesn't index.
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
