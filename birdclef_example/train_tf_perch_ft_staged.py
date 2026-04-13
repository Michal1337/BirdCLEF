#!/usr/bin/env python3
"""
Pure TensorFlow fine-tuning for Perch v2 with staged unfreezing:
1) Train a new classification head on frozen Perch embeddings.
2) Unfreeze the last N Perch variables and continue training.

Uses the same data split as train_ddp_sota.py:
  train_audio + soundscape_train -> training, soundscape_val -> validation.
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
import tensorflow as tf
import torchaudio.functional as AF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    tf.random.set_seed(seed)


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


def build_label_vector(row: pd.Series, label_map: Dict[str, int]) -> np.ndarray:
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
            waveform_t = tf.convert_to_tensor(waveform.reshape(1, -1))
            resampled = AF.resample(waveform_t, orig_freq=sr, new_freq=sample_rate).numpy()[0]
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


class PerchHead(tf.keras.Model):
    def __init__(self, perch_dir: Path, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.perch = tf.saved_model.load(str(perch_dir))
        self.infer = self.perch.signatures["serving_default"]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head = tf.keras.layers.Dense(n_classes)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        outputs = self.infer(inputs=tf.convert_to_tensor(inputs))
        emb = outputs["embedding"]
        if training:
            emb = self.dropout(emb, training=training)
        return self.head(emb)


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred, average="macro"))
    except ValueError:
        return float("nan")


def train_epoch(
    model: PerchHead,
    train_ds: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    train_vars: List[tf.Variable],
    epoch: int,
) -> float:
    epoch_loss = tf.keras.metrics.Mean()
    for batch_x, batch_y in tqdm(train_ds, desc=f"Epoch {epoch:02d} train", leave=False):
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=logits)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if not pairs:
            raise RuntimeError("No gradients to apply. Are trainable variables initialized?")
        optimizer.apply_gradients(pairs)
        epoch_loss.update_state(loss)
    return float(epoch_loss.result().numpy())


def eval_epoch(model: PerchHead, val_ds: tf.data.Dataset, epoch: int) -> float:
    all_preds = []
    all_targets = []
    for batch_x, batch_y in tqdm(val_ds, desc=f"Epoch {epoch:02d} val", leave=False):
        logits = model(batch_x, training=False)
        probs = tf.nn.sigmoid(logits).numpy()
        all_preds.append(probs)
        all_targets.append(batch_y.numpy())
    y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, 1))
    y_true = np.vstack(all_targets) if all_targets else np.zeros((0, 1))
    return compute_auc(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged TF fine-tuning for Perch v2.")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/evafs/groups/re-com/mgromadzki/data"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("birdclef_example/outputs/perch_ft_tf"))
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-unfreeze", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-unfreeze", type=float, default=3e-4)
    parser.add_argument("--unfreeze-last-n", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--intra-threads", type=int, default=0)
    parser.add_argument("--inter-threads", type=int, default=0)
    parser.add_argument("--disable-tf-gpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    if args.disable_tf_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if args.require_gpu and not gpus:
            raise RuntimeError("No TensorFlow GPU devices found, but --require-gpu was set.")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    if args.intra_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.intra_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.intra_threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
    if args.inter_threads > 0:
        os.environ["TF_NUM_INTEROP_THREADS"] = str(args.inter_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)

    set_seed(args.seed)

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_audio_df = pd.read_csv(data_dir / "train.csv")
    soundscape_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")

    train_audio_meta = prepare_train_audio_metadata(train_audio_df, data_dir / "train_audio")
    soundscape_meta = prepare_soundscape_metadata(soundscape_df, data_dir / "train_soundscapes")

    unique_soundscapes = sorted(soundscape_meta["filename"].unique().tolist())
    soundscape_train_files, soundscape_val_files = train_test_split(
        unique_soundscapes, test_size=args.val_split, random_state=args.seed
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
    label_map = build_label_map(label_source, data_dir / "taxonomy.csv")
    if len(label_map) != 234:
        raise ValueError(f"Expected 234 classes, got {len(label_map)}.")

    def make_dataset(meta: pd.DataFrame, training: bool) -> tf.data.Dataset:
        output_sig = (
            tf.TensorSpec(shape=(int(args.sample_rate * args.duration),), dtype=tf.float32),
            tf.TensorSpec(shape=(len(label_map),), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(
            lambda: iter_dataset(
                meta,
                label_map=label_map,
                sample_rate=args.sample_rate,
                duration=args.duration,
                training=training,
            ),
            output_signature=output_sig,
        )
        if training:
            ds = ds.shuffle(2048, seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_meta, training=True)
    val_ds = make_dataset(val_meta, training=False)

    model = PerchHead(args.model_dir, n_classes=len(label_map), dropout=args.dropout)
    # Build variables by running a single batch through the model.
    for batch_x, _ in train_ds.take(1):
        _ = model(batch_x, training=False)
        break

    best_val_auc = float("-inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []

    # Phase 1: head only
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_head)
    head_vars = model.head.trainable_variables
    for epoch in range(1, args.epochs_head + 1):
        train_loss = train_epoch(model, train_ds, optimizer, head_vars, epoch)
        val_auc = eval_epoch(model, val_ds, epoch)
        history.append({"stage": "head", "epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        print(f"[head] Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            tf.saved_model.save(model, str(output_dir / "best_model"))

    # Phase 2: unfreeze last N Perch variables
    if args.epochs_unfreeze > 0 and args.unfreeze_last_n > 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_unfreeze)
        perch_vars = list(getattr(model.perch, "variables", []))
        train_vars = head_vars + perch_vars[-args.unfreeze_last_n :]
        for epoch in range(1, args.epochs_unfreeze + 1):
            train_loss = train_epoch(model, train_ds, optimizer, train_vars, epoch)
            val_auc = eval_epoch(model, val_ds, epoch)
            history.append(
                {"stage": "unfreeze", "epoch": epoch, "train_loss": train_loss, "val_auc": val_auc}
            )
            print(
                f"[unfreeze] Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}"
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = args.epochs_head + epoch
                tf.saved_model.save(model, str(output_dir / "best_model"))

    summary = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "epochs_head": args.epochs_head,
        "epochs_unfreeze": args.epochs_unfreeze,
        "batch_size": args.batch_size,
        "lr_head": args.lr_head,
        "lr_unfreeze": args.lr_unfreeze,
        "unfreeze_last_n": args.unfreeze_last_n,
        "model_dir": str(args.model_dir),
        "output_dir": str(output_dir),
        "history": history,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
