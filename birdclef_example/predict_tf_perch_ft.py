"""
Kaggle inference pipeline for Perch v2 embeddings + a fine-tuned PyTorch classifier head.

This script mirrors train_tf_perch_ft.py:
- TensorFlow Perch v2 is used only for embedding extraction.
- A PyTorch MLP head predicts BirdCLEF class probabilities.

The script reads Kaggle test metadata (row_id-based) and writes a wide submission CSV.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import build_label_map


# Kaggle notebook/script configuration.
# Update these paths to match your dataset/notebook layout.
MODEL_DIR = Path("/kaggle/input/perch-v2-model/1")
HEAD_PATH = Path("/kaggle/input/perch-head/best_model.pt")
TAXONOMY_CSV = Path("/kaggle/input/birdclef-2026/taxonomy.csv")
TEST_METADATA = Path("/kaggle/input/birdclef-2026/test.csv")
SOUNDSCAPE_DIR = Path("/kaggle/input/birdclef-2026/test_soundscapes")
OUTPUT_CSV = Path("/kaggle/working/submission.csv")
LABEL_MAP_PATH: Path | None = Path("/kaggle/input/perch-head/label_map.json")
SAMPLE_RATE = 32000
DURATION = 5.0
BATCH_SIZE = 128


def _read_audio(filepath: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def _resample_if_needed(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return waveform
    waveform_t = torch.from_numpy(waveform.reshape(1, -1))
    return AF.resample(waveform_t, orig_freq=source_sr, new_freq=target_sr).numpy()[0]


def _chunk_at_end_second(
    waveform: np.ndarray,
    sample_rate: int,
    segment_samples: int,
    end_second: int,
) -> np.ndarray:
    end_sample = max(1, int(round(end_second * sample_rate)))
    start_sample = max(0, end_sample - segment_samples)

    if start_sample >= len(waveform):
        chunk = np.zeros(segment_samples, dtype=np.float32)
    else:
        end_sample = min(len(waveform), end_sample)
        chunk = waveform[start_sample:end_sample]
        if len(chunk) < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - len(chunk)))

    if len(chunk) > segment_samples:
        chunk = chunk[:segment_samples]
    return chunk.astype(np.float32, copy=False)


def _build_head(input_dim: int, num_classes: int) -> torch.nn.Module:
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )


def load_label_map(label_map_path: Path | None, taxonomy_csv: Path) -> Dict[str, int]:
    if label_map_path and label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return {str(k): int(v) for k, v in loaded.items()}

    # Build from taxonomy only for deterministic class order on Kaggle.
    metadata_stub = pd.DataFrame({"primary_label": []})
    return build_label_map(metadata_stub, taxonomy_csv)


def parse_test_windows(metadata: pd.DataFrame) -> Dict[str, List[int]]:
    windows: Dict[str, List[int]] = defaultdict(list)

    if "row_id" in metadata.columns:
        for row_id in metadata["row_id"].dropna().astype(str):
            stem, sep, sec_txt = row_id.rpartition("_")
            if not sep:
                raise ValueError(f"Could not parse row_id '{row_id}'")
            try:
                sec = int(sec_txt)
            except ValueError as exc:
                raise ValueError(f"Could not parse second from row_id '{row_id}'") from exc
            windows[f"{stem}.ogg"].append(sec)
        return {name: sorted(set(secs)) for name, secs in windows.items()}

    required_cols = {"filename", "seconds"}
    if not required_cols.issubset(metadata.columns):
        missing = ", ".join(sorted(required_cols.difference(metadata.columns)))
        raise ValueError(
            "test metadata must contain either row_id or filename+seconds columns; "
            f"missing: {missing}"
        )

    for _, row in metadata.iterrows():
        windows[str(row["filename"])].append(int(row["seconds"]))
    return {name: sorted(set(secs)) for name, secs in windows.items()}


def extract_embeddings_for_chunks(
    infer_fn,
    chunks: List[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1536), dtype=np.float32)

    outputs: List[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = np.stack(chunks[start : start + batch_size], axis=0)
        tf_out = infer_fn(inputs=batch)
        outputs.append(tf_out["embedding"].numpy())
    return np.vstack(outputs)


def build_submission_rows(
    test_metadata: pd.DataFrame,
    per_row_probs: Dict[str, np.ndarray],
    ordered_labels: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    if "row_id" in test_metadata.columns:
        row_ids = test_metadata["row_id"].astype(str).tolist()
    else:
        row_ids = [f"{Path(str(name)).stem}_{int(sec)}" for name, sec in zip(test_metadata["filename"], test_metadata["seconds"])]

    for row_id in row_ids:
        probs = per_row_probs.get(row_id)
        if probs is None:
            raise RuntimeError(f"Missing prediction for row_id '{row_id}'")
        row = {"row_id": row_id}
        row.update({label: float(probs[i]) for i, label in enumerate(ordered_labels)})
        rows.append(row)

    return pd.DataFrame(rows, columns=["row_id"] + ordered_labels)


def main() -> None:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    test_metadata = pd.read_csv(TEST_METADATA)
    windows_by_file = parse_test_windows(test_metadata)

    label_map = load_label_map(LABEL_MAP_PATH, TAXONOMY_CSV)
    ordered_labels = sorted(label_map.keys(), key=lambda label: label_map[label])

    perch = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = perch.signatures["serving_default"]

    checkpoint = torch.load(HEAD_PATH, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", 1536))
    num_classes = len(ordered_labels)
    head = _build_head(input_dim=input_dim, num_classes=num_classes)
    head.load_state_dict(checkpoint["model_state"])
    device = torch.device("cpu")
    head.to(device)
    head.eval()

    segment_samples = int(SAMPLE_RATE * DURATION)
    per_row_probs: Dict[str, np.ndarray] = {}

    for filename in tqdm(sorted(windows_by_file.keys()), desc="Soundscapes"):
        filepath = SOUNDSCAPE_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing soundscape file: {filepath}")

        waveform, sr = _read_audio(filepath)
        waveform = _resample_if_needed(waveform, source_sr=sr, target_sr=SAMPLE_RATE)

        seconds = windows_by_file[filename]
        chunks = [
            _chunk_at_end_second(
                waveform=waveform,
                sample_rate=SAMPLE_RATE,
                segment_samples=segment_samples,
                end_second=sec,
            )
            for sec in seconds
        ]

        embeddings = extract_embeddings_for_chunks(
            infer_fn=infer_fn,
            chunks=chunks,
            batch_size=BATCH_SIZE,
        )
        xb = torch.from_numpy(embeddings).float().to(device)
        with torch.no_grad():
            probs = torch.sigmoid(head(xb)).cpu().numpy()

        stem = Path(filename).stem
        for sec, prob in zip(seconds, probs):
            per_row_probs[f"{stem}_{sec}"] = prob

    submission = build_submission_rows(test_metadata, per_row_probs, ordered_labels)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
