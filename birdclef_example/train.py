import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef_example.data import (
    BirdCLEFDataset,
    SpectrogramTransform,
    build_label_map,
)
from birdclef_example.model import SimpleCNN
from birdclef_example.utils import evaluate_model, save_model, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BirdCLEF multi-label classifier")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    if "primary_label" not in metadata.columns:
        raise ValueError("metadata CSV must contain a primary_label column")
    metadata = metadata.dropna(subset=["primary_label"]).reset_index(drop=True)

    label_map = build_label_map(metadata, args.taxonomy)
    spec_transform = SpectrogramTransform(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    stratify = metadata["primary_label"] if args.val_split > 0 else None
    if args.val_split > 0:
        train_meta, val_meta = train_test_split(
            metadata,
            test_size=args.val_split,
            random_state=args.seed,
            stratify=stratify,
        )
    else:
        train_meta = metadata
        val_meta = metadata.iloc[0:0].copy()

    train_dataset = BirdCLEFDataset(
        audio_dir=args.audio_dir,
        metadata=train_meta,
        label_map=label_map,
        sample_rate=args.sample_rate,
        duration=args.segment_duration,
        transform=spec_transform,
        training=True,
    )

    val_dataset = BirdCLEFDataset(
        audio_dir=args.audio_dir,
        metadata=val_meta,
        label_map=label_map,
        sample_rate=args.sample_rate,
        duration=args.segment_duration,
        transform=spec_transform,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        if len(val_dataset)
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(
        n_classes=len(label_map),
        in_channels=1,
        base_channels=64,
        dropout=args.dropout,
    ).to(device)

    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])  # type: ignore

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            progress.set_postfix({"batch_loss": running_loss / batch_idx})
        scheduler.step()

        val_loss = float("nan")
        val_auc = float("nan")
        if val_loader is not None:
            val_loss, val_auc = evaluate_model(model, val_loader, criterion, device)
        avg_train_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f}"
        )

        if val_loader is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = args.output_dir / "best_model.pt"
            save_model(
                {
                    "model_state": model.state_dict(),
                    "label_map": label_map,
                    "epoch": epoch,
                },
                model_path,
            )
            with open(args.output_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(label_map, f)

    print(f"Training complete, best checkpoint at {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
