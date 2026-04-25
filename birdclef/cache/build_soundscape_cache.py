"""One-time decode of train_soundscapes/*.ogg into a single fixed-shape memmap.

Layout:
    SOUNDSCAPE_NPY    : float16 np.lib.format memmap of shape (n_files, FILE_SAMPLES)
                        — every soundscape padded/truncated to exactly 60 s @ 32 kHz mono.
                        Created via np.lib.format.open_memmap so np.load(..., mmap_mode="r")
                        works without specifying shape.
    SOUNDSCAPE_INDEX  : parquet (filename, row_idx, ok) — row_idx maps a filename
                        to its row in the memmap.

The dataset's `_get_soundscape` slices (row_idx, off:off+win) directly from the
memmap, replacing the per-sample sf.read() that decodes a full 60 s OGG. Once
the OS page cache warms up (epoch 1), subsequent epochs read at near-RAM
speed instead of paying full libvorbis decode cost every time.

Sizing:
    n_files × FILE_SAMPLES × 2 bytes ≈ 10_658 × 1_920_000 × 2 ≈ 41 GB on disk.

Build is sequential; OGG decode dominates (parallelizes via ThreadPoolExecutor
since libsndfile releases the GIL).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from birdclef.config.paths import (
    FILE_SAMPLES,
    SOUNDSCAPE_CACHE_DIR,
    SOUNDSCAPE_INDEX,
    SOUNDSCAPE_NPY,
    SOUNDSCAPES,
    SR,
    ensure_dirs,
)


def _read_60s_resample(path: Path) -> np.ndarray:
    """Decode a single OGG to mono float16 32 kHz, padded/truncated to FILE_SAMPLES."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF
        y = AF.resample(torch.from_numpy(y), sr, SR).numpy()
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    elif y.shape[0] > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float16, copy=False)


def _decode(args):
    i, p = args
    try:
        return i, p.name, _read_60s_resample(p), None
    except Exception as exc:
        return i, p.name, None, str(exc)


def main(dry_run_files: int = 0, num_threads: int = 8) -> None:
    ensure_dirs()
    SOUNDSCAPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(SOUNDSCAPES.glob("*.ogg"))
    if dry_run_files:
        paths = paths[:dry_run_files]
    n = len(paths)
    if n == 0:
        raise SystemExit(f"No .ogg files under {SOUNDSCAPES}")

    projected_bytes = n * FILE_SAMPLES * 2
    print(f"[soundscape_cache] {n} files × {FILE_SAMPLES} samples × 2 B "
          f"= {projected_bytes / 1e9:.2f} GB target memmap")

    # Pre-allocate as a proper .npy file (np.load mmap_mode='r' will work).
    out = np.lib.format.open_memmap(
        str(SOUNDSCAPE_NPY), mode="w+",
        dtype=np.float16, shape=(n, FILE_SAMPLES),
    )

    records: list[dict] = [{"filename": p.name, "row_idx": i, "ok": 0} for i, p in enumerate(paths)]

    n_ok = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for i, fn, y, err in tqdm(ex.map(_decode, list(enumerate(paths))),
                                   total=n, desc="decode"):
            if y is None:
                print(f"[soundscape_cache] skip {fn}: {err}")
                n_err += 1
                continue
            out[i] = y
            records[i]["ok"] = 1
            n_ok += 1

    out.flush()
    del out  # release the memmap before pandas writes alongside

    pd.DataFrame(records).to_parquet(SOUNDSCAPE_INDEX, index=False)
    print(
        f"[soundscape_cache] DONE  ok={n_ok}  err={n_err}  "
        f"path={SOUNDSCAPE_NPY}  index={SOUNDSCAPE_INDEX}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run-files", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()
    main(dry_run_files=args.dry_run_files, num_threads=args.num_threads)
