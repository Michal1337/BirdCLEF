"""
Fine-tune a Perch v2 model on BirdCLEF using the same data split as other experiments.

Notes:
- This script keeps the validation split identical to train_ddp_sota.py:
  train_audio + soundscape_train -> training, soundscape_val -> validation.
- Perch runs in TensorFlow only to extract embeddings; the classifier is trained in PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio.functional as AF
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (
    build_label_map,
    parse_primary_labels,
    parse_secondary_labels,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_audio(filepath: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def _time_to_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def slice_or_sample(
    waveform: np.ndarray,
    sample_rate: int,
    segment_samples: int,
    start_time: str | None,
    end_time: str | None,
    training: bool,
) -> np.ndarray:
    if start_time is not None and end_time is not None:
        start_sample = max(0, int(_time_to_seconds(start_time) * sample_rate))
        end_sample = max(start_sample + 1, int(_time_to_seconds(end_time) * sample_rate))
        end_sample = min(len(waveform), end_sample)
        chunk = waveform[start_sample:end_sample]
    else:
        if len(waveform) < segment_samples:
            chunk = waveform
        else:
            if training:
                max_start = len(waveform) - segment_samples
                start = random.randint(0, max_start)
            else:
                start = max(0, (len(waveform) - segment_samples) // 2)
            chunk = waveform[start : start + segment_samples]

    if len(chunk) < segment_samples:
        chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
    elif len(chunk) > segment_samples:
        chunk = chunk[:segment_samples]
    return chunk.astype(np.float32, copy=False)


def build_label_vector(
    row: pd.Series,
    label_map: Dict[str, int],
) -> np.ndarray:
    target = np.zeros(len(label_map), dtype=np.float32)
    labels = parse_primary_labels(row.get("primary_label"))
    if str(row.get("source", "")) == "train_audio":
        labels.extend(parse_secondary_labels(row.get("secondary_labels")))
    for label in labels:
        idx = label_map.get(label)
        if idx is not None:
            target[idx] = 1.0
    return target


def iter_dataset(
    metadata: pd.DataFrame,
    label_map: Dict[str, int],
    sample_rate: int,
    duration: float,
    training: bool,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    segment_samples = int(sample_rate * duration)
    rows = metadata.sample(frac=1.0, random_state=42).iterrows() if training else metadata.iterrows()
    for _, row in rows:
        filepath = Path(str(row["audio_filepath"]))
        if not filepath.exists():
            continue
        waveform, sr = _read_audio(filepath)
        if sr != sample_rate:
            waveform_t = torch.from_numpy(waveform.reshape(1, -1))
            resampled = AF.resample(
                waveform_t, orig_freq=sr, new_freq=sample_rate
            ).numpy()[0]
        else:
            resampled = waveform
        chunk = slice_or_sample(
            resampled,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            start_time=row.get("start") if isinstance(row.get("start"), str) else None,
            end_time=row.get("end") if isinstance(row.get("end"), str) else None,
            training=training,
        )
        target = build_label_vector(row, label_map)
        yield chunk, target


def extract_perch_embeddings(
    perch_dir: Path,
    metadata: pd.DataFrame,
    label_map: Dict[str, int],
    sample_rate: int,
    duration: float,
    batch_size: int,
    training: bool,
    disable_tf_gpu: bool,
    require_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if disable_tf_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf  # local import to control GPU visibility

    try:
        if disable_tf_gpu:
            tf.config.set_visible_devices([], "GPU")
        else:
            gpus = tf.config.list_physical_devices("GPU")
            if require_gpu and not gpus:
                raise RuntimeError("No TensorFlow GPU devices found, but --require-gpu was set.")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        intra = int(os.environ.get("TF_NUM_INTRAOP_THREADS", "0"))
        inter = int(os.environ.get("TF_NUM_INTEROP_THREADS", "0"))
        if intra > 0:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
        if inter > 0:
            tf.config.threading.set_inter_op_parallelism_threads(inter)
    except Exception:
        pass

    perch = tf.saved_model.load(str(perch_dir))
    infer_fn = perch.signatures["serving_default"]

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    batch: List[np.ndarray] = []

    def _flush(batch_arr: List[np.ndarray]) -> np.ndarray:
        inp = np.stack(batch_arr, axis=0)
        outputs = infer_fn(inputs=tf.convert_to_tensor(inp))
        return outputs["embedding"].numpy()

    total = len(metadata)
    with tqdm(total=total, desc="Perch embeddings", leave=False) as pbar:
        for chunk, target in iter_dataset(
            metadata,
            label_map=label_map,
            sample_rate=sample_rate,
            duration=duration,
            training=training,
        ):
            batch.append(chunk)
            Y_chunks.append(target)
            if len(batch) >= batch_size:
                X_chunks.append(_flush(batch))
                batch = []
            pbar.update(1)

    if batch:
        X_chunks.append(_flush(batch))

    X_arr = np.vstack(X_chunks) if X_chunks else np.zeros((0, 1536), dtype=np.float32)
    Y_arr = np.vstack(Y_chunks) if Y_chunks else np.zeros((0, len(label_map)), dtype=np.float32)
    return X_arr, Y_arr


def train_torch_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, float]:
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, y_train.shape[1]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = float("-inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch:02d} train", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch:02d} val", leave=False):
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                targets.append(yb.numpy())
        y_pred = np.vstack(preds) if preds else np.zeros((0, y_train.shape[1]))
        y_true = np.vstack(targets) if targets else np.zeros((0, y_train.shape[1]))
        val_auc = birdclef_roc_auc(y_true, y_pred)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auc": float(val_auc),
            }
        )
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "input_dim": X_train.shape[1]},
                output_dir / "best_model.pt",
            )

    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "history": history,
    }


def birdclef_roc_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """Exact equivalent of Kaggle macro ROC-AUC (skip classes with no positives)."""

    if targets.shape != preds.shape:
        raise ValueError(
            f"Targets and predictions must have the same shape, got {targets.shape} and {preds.shape}."
        )
    if targets.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got targets.ndim={targets.ndim}.")

    # Select classes with at least one positive
    class_sums = targets.sum(axis=0)
    valid_classes = np.where(class_sums > 0)[0]

    if len(valid_classes) == 0:
        raise ValueError("No valid classes with positive samples.")

    # Slice only valid classes
    targets_filtered = targets[:, valid_classes]
    preds_filtered = preds[:, valid_classes]

    # Compute macro ROC-AUC (same as Kaggle)
    return roc_auc_score(targets_filtered, preds_filtered, average="macro")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Perch v2 on BirdCLEF.")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/evafs/groups/re-com/mgromadzki/data"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("birdclef_example/outputs/perch_ft"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--intra-threads", type=int, default=0)
    parser.add_argument("--inter-threads", type=int, default=0)
    parser.add_argument(
        "--disable-tf-gpu",
        action="store_true",
        help="Force TensorFlow to run on CPU to avoid CUDA binary mismatches.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if no TensorFlow GPU is available.",
    )
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="Save Perch embeddings to disk for reuse.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.intra_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.intra_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.intra_threads)
    if args.inter_threads > 0:
        os.environ["TF_NUM_INTEROP_THREADS"] = str(args.inter_threads)

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_audio_metadata_path = data_dir / "train.csv"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    taxonomy_path = data_dir / "taxonomy.csv"
    train_audio_dir = data_dir / "train_audio"
    train_soundscape_dir = data_dir / "train_soundscapes"

    train_audio_df = pd.read_csv(train_audio_metadata_path)
    soundscape_df = pd.read_csv(soundscape_labels_path)

    train_audio_meta = prepare_train_audio_metadata(train_audio_df, train_audio_dir)
    soundscape_meta = prepare_soundscape_metadata(soundscape_df, train_soundscape_dir)

    unique_soundscapes = sorted(soundscape_meta["filename"].unique().tolist())
    soundscape_train_files, soundscape_val_files = train_test_split(
        unique_soundscapes,
        test_size=args.val_split,
        random_state=args.seed,
    )
    soundscape_train_meta = soundscape_meta[
        soundscape_meta["filename"].isin(soundscape_train_files)
    ].reset_index(drop=True)
    soundscape_val_meta = soundscape_meta[
        soundscape_meta["filename"].isin(soundscape_val_files)
    ].reset_index(drop=True)

    train_meta = pd.concat([train_audio_meta, soundscape_train_meta], ignore_index=True)
    val_meta = soundscape_val_meta.copy()

    label_source = pd.concat([train_meta, val_meta], ignore_index=True)
    label_map = build_label_map(label_source, taxonomy_path)
    expected_num_classes = 234
    if len(label_map) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes from taxonomy, got {len(label_map)}."
        )

    if args.require_gpu and args.disable_tf_gpu:
        raise ValueError("--require-gpu and --disable-tf-gpu cannot be used together.")

    print("Extracting Perch embeddings for training set...")
    X_train, y_train = extract_perch_embeddings(
        perch_dir=args.model_dir,
        metadata=train_meta,
        label_map=label_map,
        sample_rate=args.sample_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        training=True,
        disable_tf_gpu=args.disable_tf_gpu,
        require_gpu=args.require_gpu,
    )
    print("Extracting Perch embeddings for validation set...")
    X_val, y_val = extract_perch_embeddings(
        perch_dir=args.model_dir,
        metadata=val_meta,
        label_map=label_map,
        sample_rate=args.sample_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        training=False,
        disable_tf_gpu=args.disable_tf_gpu,
        require_gpu=args.require_gpu,
    )

    if args.cache_embeddings:
        np.savez_compressed(
            output_dir / "perch_embeddings.npz",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    summary = train_torch_head(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    summary.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "model_dir": str(args.model_dir),
            "output_dir": str(output_dir),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
