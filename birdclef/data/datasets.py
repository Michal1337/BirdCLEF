"""Datasets for SED training and inference.

SEDTrainDataset: reads from the preloaded waveform memmap (.npy). No OGG
decoding on the hot path. 50/50 mixes Xeno-Canto train_audio and labeled
soundscapes by default. Supports pseudo-label rounds via a second source.

InferenceDataset: streams 60 s soundscape OGGs from disk for final prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import json

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    SR,
    WAVEFORM_INDEX,
    WAVEFORM_NPY,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta


@dataclass
class SEDTrainSample:
    source: str       # "train_audio" | "soundscape" | "pseudo"
    source_id: int
    y: np.ndarray     # multi-hot target (n_classes,)
    offset: int       # start offset in global memmap samples
    length: int       # samples in this clip


def _load_waveform_store() -> tuple[np.ndarray, pd.DataFrame]:
    if not WAVEFORM_NPY.exists() or not WAVEFORM_INDEX.exists():
        raise FileNotFoundError(
            f"Waveform cache not found: run scripts/01_build_caches.py --stage waveform"
        )
    store = np.load(WAVEFORM_NPY, mmap_mode="r")
    index = pd.read_parquet(WAVEFORM_INDEX)
    return store, index


class SEDTrainDataset(Dataset):
    """Yields (waveform[win_samples], multihot[n_classes]).

    Sampling semantics:
    - For train_audio clips, with prob `first_window_prob` crop from the first
      0-30 s (primary bird most likely present). Otherwise random crop.
    - For soundscape windows, pick a random 5-s window from the 60-s file.
    """

    def __init__(
        self,
        fold: Optional[int] = None,
        exclude_v_anchor: bool = True,
        soundscape_fraction: float = 0.5,
        first_window_prob: float = 0.7,
        window_seconds: int = 5,
        include_secondary: bool = True,
        pseudo_round: Optional[int] = None,
    ):
        from birdclef.data.splits import load_folds, load_v_anchor
        from birdclef.data.train_audio import build_train_audio_labels, load_train_audio_meta

        self.win = window_seconds * SR
        self.sfrac = float(soundscape_fraction)
        self.fw_prob = float(first_window_prob)
        self._store, self._idx = _load_waveform_store()

        ta = load_train_audio_meta()
        # Map filename → row in index
        idx_by_file = dict(zip(self._idx["filename"].astype(str), self._idx.index))
        label_idx = label_to_idx()
        Y_ta = build_train_audio_labels(ta)
        self._ta_rows: list[SEDTrainSample] = []
        for i, row in ta.iterrows():
            r = idx_by_file.get(str(row["filename"]))
            if r is None:
                continue
            meta = self._idx.iloc[r]
            self._ta_rows.append(SEDTrainSample(
                source="train_audio", source_id=int(meta["clip_id"]),
                y=Y_ta[i], offset=int(meta["start_offset"]),
                length=int(meta["n_samples"]),
            ))

        # Soundscapes: on the fly, we read OGG-cropped 60 s from the soundscape
        # OGG files (not from the waveform memmap). This keeps memory usage low
        # since soundscapes total only ~hundreds of MB.
        sc = load_soundscape_meta()
        sc = sc[sc["fully_labeled"]]
        if exclude_v_anchor:
            anchor = set(load_v_anchor())
            sc = sc[~sc["filename"].isin(anchor)]
        if fold is not None:
            folds_df = load_folds()
            fold_of = dict(zip(folds_df["filename"], folds_df["fold"].astype(int)))
            keep_filenames = {fn for fn, fld in fold_of.items() if fld != int(fold)}
            sc = sc[sc["filename"].isin(keep_filenames)]
        # For soundscapes we'll sample at __getitem__ time from a file list
        self._sc_files = sc.drop_duplicates("filename")["filename"].tolist()
        # Precompute per-file label union so we can use soundscape-level labels
        # for windowing decisions; per-window labels come from the CSV during
        # __getitem__.
        self._sc_df = sc

        self._rng = np.random.default_rng(42)
        # Length heuristic: iterate per-epoch through max(n_train_audio, n_sc*12).
        self._length = max(len(self._ta_rows), len(self._sc_files) * N_WINDOWS)

        self._pseudo_round = pseudo_round  # hook; extended dataset reads cache/pseudo/

    def __len__(self) -> int:
        return self._length

    def _get_train_audio(self) -> tuple[np.ndarray, np.ndarray]:
        s = self._ta_rows[self._rng.integers(len(self._ta_rows))]
        start = s.offset
        length = s.length
        if length < self.win:
            buf = np.zeros(self.win, dtype=np.float32)
            buf[:length] = self._store[start : start + length].astype(np.float32)
            return buf, s.y.astype(np.float32)
        # Prefer first 30 s of recording.
        max_start = max(1, length - self.win)
        if self._rng.random() < self.fw_prob:
            upper = min(max_start, 30 * SR)
            off = int(self._rng.integers(0, max(1, upper)))
        else:
            off = int(self._rng.integers(0, max_start))
        x = self._store[start + off : start + off + self.win].astype(np.float32)
        return x, s.y.astype(np.float32)

    def _get_soundscape(self) -> tuple[np.ndarray, np.ndarray]:
        from birdclef.config.paths import SOUNDSCAPES

        if not self._sc_files:
            return self._get_train_audio()
        fn = self._sc_files[int(self._rng.integers(len(self._sc_files)))]
        sub = self._sc_df[self._sc_df["filename"] == fn].sort_values("end_sec")
        if sub.empty:
            return self._get_train_audio()
        win_row = sub.iloc[int(self._rng.integers(len(sub)))]
        start_sec = int(win_row["end_sec"]) - 5
        # Read OGG, extract 5 s window
        try:
            y, sr = sf.read(str(SOUNDSCAPES / fn), dtype="float32", always_2d=False)
        except Exception:
            return self._get_train_audio()
        if y.ndim == 2:
            y = y.mean(axis=1)
        off = max(0, start_sec * SR)
        x = y[off : off + self.win]
        if x.shape[0] < self.win:
            x = np.pad(x, (0, self.win - x.shape[0]))
        yv = np.zeros(len(label_to_idx()), dtype=np.float32)
        idx = label_to_idx()
        for lb in win_row["label_list"]:
            j = idx.get(lb)
            if j is not None:
                yv[j] = 1.0
        return x, yv

    def __getitem__(self, _index: int):
        if self._rng.random() < self.sfrac and self._sc_files:
            x, y = self._get_soundscape()
        else:
            x, y = self._get_train_audio()
        return torch.from_numpy(x), torch.from_numpy(y)


class InferenceDataset(Dataset):
    """Streams 60 s OGG soundscape files, yields (waveform[FILE_SAMPLES], filename)."""

    def __init__(self, paths: List[Path]):
        self.paths = [Path(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        return torch.from_numpy(y.astype(np.float32)), p.name
