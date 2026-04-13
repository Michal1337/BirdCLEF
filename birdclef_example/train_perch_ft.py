"""
Fine-tune a Perch v2 model on BirdCLEF using the same data split as other experiments.

This script mirrors train_tf_perch_ft.py but uses a PyTorch Perch backbone
(checkpoint: perch_backbone.pt) to extract embeddings.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio.functional as AF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (
    SpectrogramTransform,
    build_label_map,
    parse_primary_labels,
    parse_secondary_labels,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class PerchTorchEmbedder(torch.nn.Module):
    def __init__(self, backbone_path: Path, image_size: int, sample_rate: int):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required to load perch_backbone.pt. Install with: pip install timm"
            ) from exc

        self.spectrogram = SpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=160,
            n_fft=2048,
            hop_length=512,
            f_min=20,
            f_max=None,
            normalize=True,
        )
        self.image_size = int(image_size)
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )

        state = torch.load(str(backbone_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"Warning: backbone checkpoint load mismatch (missing={len(missing)}, unexpected={len(unexpected)})"
            )

    def _wave_to_image(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(waveform)
        x = x.clamp(min=-4.0, max=4.0)
        x = (x + 4.0) / 8.0
        x = torch.nn.functional.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        images = self._wave_to_image(waveform)
        feats = self.backbone(images)
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim > 2:
            feats = feats.flatten(1)
        return feats


def extract_perch_embeddings_torch(
    backbone_path: Path,
    metadata: pd.DataFrame,
    label_map: Dict[str, int],
    sample_rate: int,
    duration: float,
    batch_size: int,
    training: bool,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PerchTorchEmbedder(
        backbone_path=backbone_path,
        image_size=image_size,
        sample_rate=sample_rate,
    ).to(device)
    model.eval()

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    batch: List[np.ndarray] = []

    def _flush(batch_arr: List[np.ndarray]) -> np.ndarray:
        inp = torch.from_numpy(np.stack(batch_arr, axis=0)).float().unsqueeze(1).to(device)
        with torch.no_grad():
            embeddings = model(inp).float().cpu().numpy()
        return embeddings

    total = len(metadata)
    with tqdm(total=total, desc="Perch embeddings (torch)", leave=False) as pbar:
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

    if X_chunks:
        X_arr = np.vstack(X_chunks)
    else:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, int(sample_rate * duration), device=device)
            emb_dim = int(model(dummy).shape[1])
        X_arr = np.zeros((0, emb_dim), dtype=np.float32)

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
                {
                    "model_state": model.state_dict(),
                    "input_dim": X_train.shape[1],
                },
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

    class_sums = targets.sum(axis=0)
    valid_classes = np.where(class_sums > 0)[0]

    if len(valid_classes) == 0:
        raise ValueError("No valid classes with positive samples.")

    targets_filtered = targets[:, valid_classes]
    preds_filtered = preds[:, valid_classes]
    return roc_auc_score(targets_filtered, preds_filtered, average="macro")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Perch v2 on BirdCLEF with torch backbone.")
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/evafs/groups/re-com/mgromadzki/data"))
    parser.add_argument("--backbone-path", type=Path, default=Path("perch_backbone.pt"))
    parser.add_argument("--model-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("birdclef_example/outputs/perch_ft"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="Save Perch embeddings to disk for reuse.",
    )
    args = parser.parse_args()

    if args.model_dir is not None:
        args.backbone_path = args.model_dir

    set_seed(args.seed)

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

    if not args.backbone_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_path}")

    print("Extracting Perch embeddings (torch) for training set...")
    X_train, y_train = extract_perch_embeddings_torch(
        backbone_path=args.backbone_path,
        metadata=train_meta,
        label_map=label_map,
        sample_rate=args.sample_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        training=True,
        image_size=args.image_size,
    )
    print("Extracting Perch embeddings (torch) for validation set...")
    X_val, y_val = extract_perch_embeddings_torch(
        backbone_path=args.backbone_path,
        metadata=val_meta,
        label_map=label_map,
        sample_rate=args.sample_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        training=False,
        image_size=args.image_size,
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
            "backbone_path": str(args.backbone_path),
            "output_dir": str(output_dir),
            "image_size": args.image_size,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
