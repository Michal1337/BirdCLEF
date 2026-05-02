"""Datasets for SED training and inference.

SEDTrainDataset: reads train_audio from the preloaded waveform memmap (.npy),
soundscapes via on-the-fly OGG decode. Every `__getitem__` returns
(waveform, target, loss_mask) so the trainer can apply a per-position loss
mask without special-casing pseudo vs GT samples.

When `pseudo_round` is set, cache/pseudo/round{N}/ is opened and its soft
targets replace (or merge with) the ground-truth soundscape labels:
    - labeled soundscape (non-val-fold) → target = max(GT, pseudo)
    - unlabeled soundscape              → target = pseudo,  mask = keep_mask
    - train_audio                       → unchanged (GT multi-hot, mask = all ones)
The val fold (if `fold` is provided) is held out from training.

InferenceDataset: streams 60 s soundscape OGGs from disk for final prediction.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    PSEUDO_DIR,
    SOUNDSCAPE_INDEX,
    SOUNDSCAPE_NPY,
    SOUNDSCAPES,
    SR,
    WAVEFORM_INDEX,
    WAVEFORM_NPY,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta


def _preload_into_ram() -> bool:
    """If BIRDCLEF_PRELOAD_CACHE is truthy, the waveform + soundscape memmaps
    are read fully into RAM at dataset construction time. Eliminates the
    page-fault cost on every random window read — useful when you have
    plenty of RAM (~50 GB total for both caches) and a slow filesystem.
    """
    return str(os.environ.get("BIRDCLEF_PRELOAD_CACHE", "")).lower() in {
        "1", "true", "yes", "ram"
    }


def _load_soundscape_store():
    """Open the soundscape memmap (n_files, FILE_SAMPLES) f16 if present.

    Returns (memmap | ndarray, {filename: row_idx}) or (None, {}) when the
    cache hasn't been built. The dataset falls back to on-the-fly OGG decode
    in that case. With BIRDCLEF_PRELOAD_CACHE=1 the array is loaded fully
    into RAM (~40 GB for the full soundscape pool).
    """
    if not SOUNDSCAPE_NPY.exists() or not SOUNDSCAPE_INDEX.exists():
        return None, {}
    try:
        if _preload_into_ram():
            t0 = time.time()
            store = np.load(SOUNDSCAPE_NPY)         # full read into RAM
            print(f"[SEDTrainDataset] preloaded soundscape store into RAM: "
                  f"shape={store.shape} dtype={store.dtype} "
                  f"size={store.nbytes / 1024**3:.1f} GB "
                  f"({time.time() - t0:.1f}s)")
        else:
            store = np.load(SOUNDSCAPE_NPY, mmap_mode="r")
    except Exception as exc:
        print(f"[SEDTrainDataset] soundscape store load failed ({exc}); fallback to OGG decode")
        return None, {}
    idx = pd.read_parquet(SOUNDSCAPE_INDEX)
    if "ok" in idx.columns:
        idx = idx[idx["ok"] == 1]
    row_by_file = dict(zip(idx["filename"].astype(str), idx["row_idx"].astype(int)))
    return store, row_by_file


@dataclass
class SEDTrainSample:
    source: str       # "train_audio" | "soundscape_gt" | "soundscape_pseudo"
    source_id: int
    y: np.ndarray     # multi-hot target (n_classes,)
    offset: int       # start offset in global memmap samples
    length: int       # samples in this clip


def _load_waveform_store() -> tuple[np.ndarray, pd.DataFrame]:
    if not WAVEFORM_NPY.exists() or not WAVEFORM_INDEX.exists():
        raise FileNotFoundError(
            f"Waveform cache not found: run scripts/01_build_caches.py --stage waveform"
        )
    if _preload_into_ram():
        t0 = time.time()
        store = np.load(WAVEFORM_NPY)              # full read into RAM
        print(f"[SEDTrainDataset] preloaded waveform store into RAM: "
              f"shape={store.shape} dtype={store.dtype} "
              f"size={store.nbytes / 1024**3:.1f} GB "
              f"({time.time() - t0:.1f}s)")
    else:
        store = np.load(WAVEFORM_NPY, mmap_mode="r")
    index = pd.read_parquet(WAVEFORM_INDEX)
    return store, index


def _load_pseudo_round(rnd: int):
    """Load cache/pseudo/round{rnd}/ artifacts.

    Returns (probs: (N, C) float32, keep_mask: (N, C) uint8,
             file_to_start: dict[filename -> row offset in probs]).
    """
    rdir = PSEUDO_DIR / f"round{int(rnd)}"
    meta_path = rdir / "meta.parquet"
    probs_path = rdir / "probs.npz"
    if not meta_path.exists() or not probs_path.exists():
        raise FileNotFoundError(
            f"Pseudo round {rnd} not found under {rdir}. "
            "Run scripts/_05_pseudo_label.py --round N first."
        )
    meta = pd.read_parquet(meta_path)
    arrs = np.load(probs_path)
    # Schema varies by builder:
    #   - SSM teacher / SSM+SED blend teacher (pseudo_label.py): saves
    #     `final` (post-pipeline probs) + optional `first_pass`, `ssm_final`,
    #     `sed_final`.
    #   - SED-checkpoint teacher: saves a single `probs` key.
    if "final" in arrs.files:
        probs = arrs["final"].astype(np.float32)
    elif "probs" in arrs.files:
        probs = arrs["probs"].astype(np.float32)
    else:
        raise KeyError(
            f"Pseudo round {rnd} npz at {probs_path} has neither 'final' nor "
            f"'probs' arrays. Available keys: {arrs.files}. Rebuild via "
            "scripts/_05_pseudo_label.py."
        )
    keep = arrs["keep_mask"].astype(np.uint8) if "keep_mask" in arrs.files else np.ones_like(probs, dtype=np.uint8)
    # Rows come in blocks of N_WINDOWS per filename, ordered by appearance.
    file_start = (
        meta.reset_index()
        .drop_duplicates("filename")[["filename", "index"]]
        .rename(columns={"index": "start_row"})
    )
    file_to_start = dict(zip(file_start["filename"].astype(str), file_start["start_row"].astype(int)))
    info_path = rdir / "info.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return probs, keep, file_to_start, meta, info


class SEDTrainDataset(Dataset):
    """Yields (waveform[win_samples], target[n_classes], loss_mask[n_classes]).

    Every returned tuple has the same shape regardless of source; if a sample
    has no pseudo-label, `loss_mask` is all ones. This lets the trainer pass
    a single `loss_mask` to FocalBCE without branching.
    """

    def __init__(
        self,
        fold: Optional[int] = None,
        soundscape_fraction: float = 0.5,
        first_window_prob: float = 0.7,
        window_seconds: int = 5,
        include_secondary: bool = True,
        pseudo_round: Optional[int] = None,
        n_splits: int = 5,
        seed: int = 42,
        use_train_audio: bool = True,
    ):
        from birdclef.data.splits import load_folds
        from birdclef.data.train_audio import build_train_audio_labels, load_train_audio_meta

        self.win = window_seconds * SR
        self.sfrac = float(soundscape_fraction)
        self.fw_prob = float(first_window_prob)
        # When False, skip loading train_audio entirely — every sample comes
        # from the soundscape pool (labeled GT and/or pseudo). Useful for
        # ablations that train SED only on soundscapes with proper fold
        # holdout (no focal recordings, no domain bridging).
        self._use_train_audio = bool(use_train_audio)
        self._store, self._idx = (None, None)
        self._ta_rows: list[SEDTrainSample] = []
        if self._use_train_audio:
            self._store, self._idx = _load_waveform_store()
            ta = load_train_audio_meta()
            idx_by_file = dict(zip(self._idx["filename"].astype(str), self._idx.index))
            Y_ta = build_train_audio_labels(ta)
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

        self._n_classes = len(label_to_idx())

        # Labeled soundscape pool (window-level GT). With V-anchor abandoned,
        # the only filtering is by fold (if provided): hold out the val fold,
        # train on the other n-1 folds.
        sc = load_soundscape_meta()
        sc = sc[sc["fully_labeled"]]
        # Val-fold filenames must be excluded from BOTH the labeled GT pool
        # AND the pseudo-only pool below — otherwise a val file would slip
        # into training as a pseudo-only sample (teacher has seen it; its
        # pseudo target effectively encodes the true labels) and inflate the
        # fold-val AUC measurement.
        val_filenames: set[str] = set()
        if fold is not None:
            folds_df = load_folds(n_splits=int(n_splits))
            fold_of = dict(zip(folds_df["filename"], folds_df["fold"].astype(int)))
            keep_filenames = {fn for fn, fld in fold_of.items() if fld != int(fold)}
            sc = sc[sc["filename"].isin(keep_filenames)]
            val_filenames = {fn for fn, fld in fold_of.items() if fld == int(fold)}
        self._val_filenames = val_filenames
        self._sc_df = sc
        labeled_sc_files = sc.drop_duplicates("filename")["filename"].astype(str).tolist()

        # Pseudo-labels (optional)
        self._pseudo_round = int(pseudo_round) if pseudo_round is not None else None
        self._pseudo_probs = None
        self._pseudo_keep = None
        self._pseudo_file_start: dict = {}
        self._pseudo_info: dict = {}
        unlabeled_sc_files: list[str] = []

        if self._pseudo_round is not None:
            probs, keep, file_to_start, meta, info = _load_pseudo_round(self._pseudo_round)
            self._pseudo_probs = probs
            self._pseudo_keep = keep
            self._pseudo_file_start = file_to_start
            self._pseudo_info = info

            # Soundscape pool grows to include every file covered by the
            # pseudo-label cache. Every file in the pseudo cache that isn't
            # in the labeled GT pool AND isn't in the held-out val fold is
            # eligible. Val-fold exclusion matters because the teacher has
            # likely seen those files (built on all labeled rows), so a
            # pseudo-only sample of a val file would leak its true labels
            # into training.
            eligible_files = sorted(file_to_start.keys())
            labeled_set = set(labeled_sc_files)
            unlabeled_sc_files = [
                f for f in eligible_files
                if f not in labeled_set and f not in val_filenames
            ]

        # Final soundscape file pool (labeled GT files + pseudo-only files)
        self._sc_labeled_files = labeled_sc_files
        self._sc_pseudo_only_files = unlabeled_sc_files
        self._sc_files = labeled_sc_files + unlabeled_sc_files
        self._labeled_set = set(labeled_sc_files)

        # Precompute (filename, window_idx) -> GT multihot for fast lookup
        idx_map = label_to_idx()
        self._gt_by_key: dict[tuple[str, int], np.ndarray] = {}
        for _, row in sc.iterrows():
            wi = int(row["end_sec"]) // 5 - 1
            if wi < 0 or wi >= N_WINDOWS:
                continue
            y = np.zeros(self._n_classes, dtype=np.float32)
            for lb in row["label_list"]:
                j = idx_map.get(lb)
                if j is not None:
                    y[j] = 1.0
            self._gt_by_key[(str(row["filename"]), wi)] = y

        # Soundscape memmap cache (n_files, FILE_SAMPLES) f16. If the build
        # script hasn't run yet, _sc_store is None and we fall back to per-call
        # OGG decode in _get_soundscape — slower, but the dataset still works.
        self._sc_store, self._sc_row_by_file = _load_soundscape_store()
        if self._sc_store is not None:
            print(f"[SEDTrainDataset] soundscape memmap cache active "
                  f"({len(self._sc_row_by_file)} files indexed)")

        self._rng = np.random.default_rng(int(seed))
        # Length heuristic: enough iterations per epoch to sweep both pools once.
        self._length = max(
            len(self._ta_rows),
            max(1, len(self._sc_files)) * N_WINDOWS,
        )

    def __len__(self) -> int:
        return self._length

    def _pseudo_row_for(self, filename: str, window_idx: int) -> Optional[int]:
        """Return the row offset in pseudo_probs for (filename, window), or None."""
        if self._pseudo_probs is None:
            return None
        start = self._pseudo_file_start.get(str(filename))
        if start is None:
            return None
        return int(start) + int(window_idx)

    def _get_train_audio(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self._ta_rows[self._rng.integers(len(self._ta_rows))]
        start = s.offset
        length = s.length
        if length < self.win:
            buf = np.zeros(self.win, dtype=np.float32)
            buf[:length] = self._store[start : start + length].astype(np.float32)
            x = buf
        else:
            max_start = max(1, length - self.win)
            if self._rng.random() < self.fw_prob:
                upper = min(max_start, 30 * SR)
                off = int(self._rng.integers(0, max(1, upper)))
            else:
                off = int(self._rng.integers(0, max_start))
            x = self._store[start + off : start + off + self.win].astype(np.float32)
        y = s.y.astype(np.float32)
        mask = np.ones(self._n_classes, dtype=np.float32)
        return x, y, mask

    def _get_soundscape(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._sc_files:
            if self._use_train_audio:
                return self._get_train_audio()
            raise RuntimeError(
                "SEDTrainDataset has no soundscape files AND use_train_audio=False; "
                "training pool is empty. Re-enable train_audio or build pseudo cache."
            )
        fn = self._sc_files[int(self._rng.integers(len(self._sc_files)))]
        is_labeled = fn in self._labeled_set

        if is_labeled:
            sub = self._sc_df[self._sc_df["filename"] == fn].sort_values("end_sec")
            if sub.empty:
                # Soundscape file with no GT row that survived the fold filter →
                # pick another file. With use_train_audio=False this is the only
                # safe escape; with True, fall back to focal.
                if self._use_train_audio:
                    return self._get_train_audio()
                window_idx = int(self._rng.integers(N_WINDOWS))
            else:
                pick = int(self._rng.integers(len(sub)))
                win_row = sub.iloc[pick]
                window_idx = int(win_row["end_sec"]) // 5 - 1
        else:
            # Pseudo-only file: sample a random window uniformly.
            window_idx = int(self._rng.integers(N_WINDOWS))

        start_sec = max(0, window_idx * 5)
        off = start_sec * SR
        # Fast path: if the soundscape memmap cache is loaded and contains
        # this file, slice directly instead of decoding the OGG.
        sc_row = self._sc_row_by_file.get(fn) if self._sc_store is not None else None
        if sc_row is not None:
            x = np.asarray(self._sc_store[sc_row, off : off + self.win], dtype=np.float32)
            if x.shape[0] < self.win:
                x = np.pad(x, (0, self.win - x.shape[0]))
        else:
            # Slow fallback: per-call OGG decode (used when the soundscape
            # cache hasn't been built yet).
            try:
                y_wav, _sr = sf.read(str(SOUNDSCAPES / fn), dtype="float32", always_2d=False)
            except Exception:
                return self._get_train_audio()
            if y_wav.ndim == 2:
                y_wav = y_wav.mean(axis=1)
            x = y_wav[off : off + self.win]
            if x.shape[0] < self.win:
                x = np.pad(x, (0, self.win - x.shape[0]))

        gt = self._gt_by_key.get((fn, window_idx))
        pseudo_row = self._pseudo_row_for(fn, window_idx)
        if pseudo_row is not None:
            pseudo_y = self._pseudo_probs[pseudo_row]
            pseudo_mask = self._pseudo_keep[pseudo_row].astype(np.float32)
            if gt is not None:
                # max(GT, pseudo): GT positives stay 1.0, pseudo covers the rest
                target = np.maximum(gt, pseudo_y).astype(np.float32)
                # GT-labeled positions are always supervised, plus pseudo-kept ones
                mask = np.maximum((gt > 0).astype(np.float32), pseudo_mask)
            else:
                target = pseudo_y.astype(np.float32)
                mask = pseudo_mask
        elif gt is not None:
            target = gt.astype(np.float32)
            mask = np.ones(self._n_classes, dtype=np.float32)
        else:
            # No GT, no pseudo — should not happen once pseudo round is built.
            target = np.zeros(self._n_classes, dtype=np.float32)
            mask = np.zeros(self._n_classes, dtype=np.float32)
        return x, target, mask

    def __getitem__(self, _index: int):
        # Sampling priority:
        #   - if use_train_audio=False: always soundscape (focal pool empty)
        #   - else: flip biased coin per soundscape_fraction
        if not self._use_train_audio:
            x, y, m = self._get_soundscape()
        elif self._rng.random() < self.sfrac and self._sc_files:
            x, y, m = self._get_soundscape()
        else:
            x, y, m = self._get_train_audio()
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


class InferenceDataset(Dataset):
    """Streams 60 s OGG soundscape files, yields (waveform[FILE_SAMPLES], filename)."""

    def __init__(self, paths: List[Path]):
        self.paths = [Path(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        y, _sr = sf.read(str(p), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        return torch.from_numpy(y.astype(np.float32)), p.name
