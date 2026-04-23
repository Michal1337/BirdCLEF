"""One-time decode of train_audio/*.ogg into a single memmapped .npy file.

Layout:
    WAVEFORM_NPY    : float16 array of shape (total_samples,)
    WAVEFORM_INDEX  : parquet of (clip_id, filename, start_offset, n_samples,
                                  primary_labels_str, secondary_labels_str)

All DDP ranks open the memmap read-only at train time; the OS page cache
handles sharing across processes on the same node.

Memory sizing guard: aborts if projected size exceeds 80 % of RAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from birdclef.config.paths import (
    SR,
    TRAIN_AUDIO,
    WAVEFORM_DIR,
    WAVEFORM_INDEX,
    WAVEFORM_NPY,
    ensure_dirs,
)
from birdclef.data.train_audio import load_train_audio_meta


def _estimate_samples(paths: list[Path]) -> int:
    total = 0
    print(f"[waveform_cache] probing durations for {len(paths)} files...")
    for p in tqdm(paths, desc="probe"):
        try:
            info = sf.info(str(p))
            frames = int(info.frames)
            tgt = int(round(frames * SR / max(info.samplerate, 1)))
            total += tgt
        except Exception as exc:
            print(f"[waveform_cache] skip {p.name}: {exc}")
    return total


def _ram_guard(total_samples: int, dtype_bytes: int = 2) -> None:
    try:
        import psutil

        total_ram = psutil.virtual_memory().total
    except ImportError:
        total_ram = 16 * 1024 ** 3  # assume 16 GB if psutil missing
    projected = total_samples * dtype_bytes
    if projected > 0.80 * total_ram:
        raise RuntimeError(
            f"Projected cache {projected/1e9:.1f} GB exceeds 80% of RAM "
            f"{total_ram/1e9:.1f} GB. Use sharded cache or reduce dataset."
        )
    print(f"[waveform_cache] projected size={projected/1e9:.2f} GB (RAM={total_ram/1e9:.1f} GB)")


def _read_mono_32k(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        # Lightweight polyphase resample via torchaudio to avoid librosa dep on hot path.
        import torchaudio.functional as AF
        import torch

        y = AF.resample(torch.from_numpy(y), sr, SR).numpy()
    return y.astype(np.float16, copy=False)


def main(dry_run_files: int = 0, resume: bool = True) -> None:
    ensure_dirs()
    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train_audio_meta()
    if dry_run_files:
        df = df.iloc[:dry_run_files].copy()

    paths = [Path(p) for p in df["abspath"]]
    if WAVEFORM_INDEX.exists() and WAVEFORM_NPY.exists() and resume:
        prev = pd.read_parquet(WAVEFORM_INDEX)
        print(f"[waveform_cache] found existing cache with {len(prev)} rows")
        done = set(prev["filename"].tolist())
        todo = df[~df["filename"].astype(str).isin(done)].reset_index(drop=True)
        if len(todo) == 0:
            print("[waveform_cache] nothing to do.")
            return
        print(f"[waveform_cache] resuming: {len(todo)}/{len(df)} new files")
        df = todo
        paths = [Path(p) for p in df["abspath"]]

    total = _estimate_samples(paths)
    _ram_guard(total, dtype_bytes=2)

    # We can't easily grow a memmap; write to a temporary buffer then flush.
    buf = np.empty(total, dtype=np.float16)
    records = []
    offset = 0
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="decode")):
        p = Path(row["abspath"])
        try:
            y = _read_mono_32k(p)
        except Exception as exc:
            print(f"[waveform_cache] skip {p.name}: {exc}")
            continue
        n = int(y.shape[0])
        if offset + n > total:
            # Grow the buffer on the rare off-by-one case
            extra = np.empty(offset + n - total, dtype=np.float16)
            buf = np.concatenate([buf, extra])
            total = buf.shape[0]
        buf[offset : offset + n] = y
        records.append({
            "clip_id": i,
            "filename": str(row["filename"]),
            "start_offset": int(offset),
            "n_samples": n,
            "primary_labels": json.dumps(list(row["primary_labels"])),
            "secondary_labels": json.dumps(list(row["secondary_labels"])),
        })
        offset += n

    buf = buf[:offset]
    # Append to existing cache if resuming
    if WAVEFORM_NPY.exists() and resume:
        prev = np.load(WAVEFORM_NPY, mmap_mode="r")
        merged = np.empty(prev.shape[0] + buf.shape[0], dtype=np.float16)
        merged[: prev.shape[0]] = prev[:]
        merged[prev.shape[0] :] = buf
        # rewrite offsets in new records
        shift = int(prev.shape[0])
        for r in records:
            r["start_offset"] += shift
        np.save(WAVEFORM_NPY, merged)
        prev_idx = pd.read_parquet(WAVEFORM_INDEX)
        idx_df = pd.concat([prev_idx, pd.DataFrame(records)], ignore_index=True)
    else:
        np.save(WAVEFORM_NPY, buf)
        idx_df = pd.DataFrame(records)

    idx_df.to_parquet(WAVEFORM_INDEX, index=False)
    print(
        f"[waveform_cache] DONE. rows={len(idx_df)}, size={buf.nbytes/1e9:.2f} GB, "
        f"path={WAVEFORM_NPY}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run-files", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    main(dry_run_files=args.dry_run_files, resume=not args.no_resume)
