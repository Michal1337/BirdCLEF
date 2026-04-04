import json
import inspect
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef_example.data import (
    BirdCLEFDataset,
    build_label_map,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)
from birdclef_example.model import SimpleCNN
from birdclef_example.utils import evaluate_model, is_better_score, save_model, set_seed


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, bool]:
    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
    fused_enabled = False
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        optimizer_kwargs["fused"] = True
        fused_enabled = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    return optimizer, fused_enabled


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler._LRScheduler:
    warmup_epochs = max(0, min(warmup_epochs, epochs))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float,
    epoch_label: str,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(train_loader, desc=epoch_label, leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix({"batch_loss": running_loss / batch_idx})

    return running_loss / max(len(train_loader), 1)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = repo_root / "birdclef_example" / "outputs"

    train_audio_metadata_path = data_dir / "train.csv"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    taxonomy_path = data_dir / "taxonomy.csv"
    train_audio_dir = data_dir / "train_audio"
    train_soundscape_dir = data_dir / "train_soundscapes"

    epochs = 15
    batch_size = 32
    lr = 1e-3
    min_lr_ratio = 0.05
    warmup_epochs = 2
    weight_decay = 2e-4
    dropout = 0.35
    sample_rate = 32000
    segment_duration = 5.0
    n_mels = 160
    n_fft = 2048
    hop_length = 512
    val_split = 0.30
    seed = 42
    preload_audio = True
    preload_workers = 16
    num_workers = 0 if preload_audio else 4
    use_bf16 = True
    grad_clip_norm = 1.0
    resume = None

    if not train_audio_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_audio_dir}")
    if not train_soundscape_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_soundscape_dir}")

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_audio_df = pd.read_csv(train_audio_metadata_path)
    soundscape_df = pd.read_csv(soundscape_labels_path)

    train_audio_meta = prepare_train_audio_metadata(train_audio_df, train_audio_dir)
    soundscape_meta = prepare_soundscape_metadata(soundscape_df, train_soundscape_dir)

    unique_soundscapes = sorted(soundscape_meta["filename"].unique().tolist())
    soundscape_train_files, soundscape_val_files = train_test_split(
        unique_soundscapes,
        test_size=val_split,
        random_state=seed,
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

    train_dataset = BirdCLEFDataset(
        metadata=train_meta,
        label_map=label_map,
        sample_rate=sample_rate,
        duration=segment_duration,
        training=True,
        preload_audio=preload_audio,
        preload_workers=preload_workers,
    )
    val_dataset = BirdCLEFDataset(
        metadata=val_meta,
        label_map=label_map,
        sample_rate=sample_rate,
        duration=segment_duration,
        training=False,
        preload_audio=preload_audio,
        preload_workers=preload_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "n_classes": len(label_map),
        "dropout": dropout,
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "embed_dim": 320,
        "num_heads": 10,
        "num_layers": 6,
        "token_grid_size": 22,
        "pooling": "hybrid",
        "freq_mask_param": 16,
        "time_mask_param": 32,
        "specaugment_masks": 3,
    }
    model = SimpleCNN(**model_config).to(device)

    if resume is not None and resume.exists():
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer, fused_enabled = build_optimizer(model, lr=lr, weight_decay=weight_decay, device=device)
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
    )
    amp_enabled = use_bf16 and device.type == "cuda"

    print(
        f"Training rows: {len(train_meta)} "
        f"(train_audio={len(train_audio_meta)}, train_soundscapes={len(soundscape_train_meta)})"
    )
    print(
        "Training schedule: single-stage mixed training for "
        f"{epochs} epochs @ lr={lr} with {warmup_epochs}-epoch warmup "
        f"and cosine decay to {lr * min_lr_ratio:.2e}"
    )
    print(
        f"Validation rows: {len(val_meta)} "
        f"from {len(soundscape_val_files)} held-out soundscapes"
    )
    print(f"Num labels: {len(label_map)}")
    print(f"Preload audio to RAM: {preload_audio}")
    print(f"Preload workers: {preload_workers if preload_audio else 0}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"BF16 autocast: {amp_enabled}")
    print(f"Fused AdamW: {fused_enabled}")

    best_val_auc = float("nan")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        avg_train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch_label=f"epoch {epoch}",
        )
        scheduler.step()

        val_loss, val_auc = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            use_bf16=use_bf16,
        )
        print(
            f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_birdclef_roc_auc={val_auc:.4f}"
        )

        should_save = False
        if is_better_score(val_auc, best_val_auc):
            best_val_auc = val_auc
            should_save = True
        elif pd.isna(val_auc) and val_loss < best_val_loss:
            should_save = True

        if should_save:
            best_val_loss = val_loss
            model_path = output_dir / "best_model.pt"
            save_model(
                {
                    "model_state": model.state_dict(),
                    "label_map": label_map,
                    "model_config": model_config,
                    "epoch": epoch,
                    "best_val_auc": val_auc,
                    "best_val_loss": val_loss,
                    "use_bf16": amp_enabled,
                    "fused_adamw": fused_enabled,
                    "warmup_epochs": warmup_epochs,
                    "min_lr_ratio": min_lr_ratio,
                },
                model_path,
            )
            with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(label_map, f, indent=2, sort_keys=True)

    print(f"Training complete, best checkpoint at {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
