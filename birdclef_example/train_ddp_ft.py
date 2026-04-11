import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import timm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (  # noqa: E402
    BirdCLEFDataset,
    SpectrogramTransform,
    build_label_map,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)
from birdclef_example.model import SpecAugment  # noqa: E402
from birdclef_example.utils import evaluate_model, is_better_score, save_model, set_seed  # noqa: E402


def _get_distributed_context() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return distributed, world_size, rank, local_rank


def _setup_distributed(distributed: bool) -> None:
    if not distributed:
        return
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def _cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _print_main(rank: int, message: str) -> None:
    if _is_main_process(rank):
        print(message)


def build_rank_metadata_shard(metadata: pd.DataFrame, world_size: int, rank: int) -> pd.DataFrame:
    if world_size <= 1:
        return metadata.reset_index(drop=True)
    if metadata.empty:
        return metadata.copy()

    shard_size = (len(metadata) + world_size - 1) // world_size
    start = rank * shard_size
    end = min(start + shard_size, len(metadata))
    shard = metadata.iloc[start:end].copy()
    missing = shard_size - len(shard)
    if missing > 0:
        shard = pd.concat([shard, metadata.iloc[:missing].copy()], ignore_index=True)
    return shard.reset_index(drop=True)


def apply_mixup(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or probability <= 0.0 or inputs.size(0) < 2:
        return inputs, targets
    if torch.rand(1, device=inputs.device).item() > probability:
        return inputs, targets

    lam = torch.distributions.Beta(alpha, alpha).sample((1,)).to(inputs.device).item()
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[indices]
    mixed_targets = lam * targets + (1.0 - lam) * targets[indices]
    return mixed_inputs, mixed_targets


class TimmSpectrogramClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        backbone_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        dropout: float = 0.3,
        sample_rate: int = 32000,
        n_mels: int = 160,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: int = 20,
        f_max: int | None = None,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        specaugment_masks: int = 0,
        spec_noise_std: float = 0.0,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.spectrogram = SpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        )
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_masks=specaugment_masks,
            spec_noise_std=spec_noise_std,
        )

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=3,
            num_classes=n_classes,
            drop_rate=dropout,
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        classifier = self.backbone.get_classifier()
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _wave_to_image(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(waveform)
        if self.training:
            x = self.spec_augment(x)

        # Robust normalization for image backbones pretrained on RGB inputs.
        x = x.clamp(min=-4.0, max=4.0)
        x = (x + 4.0) / 8.0
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._wave_to_image(waveform)
        return self.backbone(x)


def build_optimizer(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, bool]:
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name and "classifier" not in name and "head" not in name and "fc" not in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
    ]

    fused_enabled = False
    optimizer_kwargs = {}
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        optimizer_kwargs["fused"] = True
        fused_enabled = True

    optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
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
            eta_min=min(group["lr"] for group in optimizer.param_groups) * min_lr_ratio,
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
        eta_min=min(group["lr"] for group in optimizer.param_groups) * min_lr_ratio,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def train_one_epoch_ddp(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float,
    epoch_label: str,
    rank: int,
    mixup_alpha: float,
    mixup_probability: float,
) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0
    progress = tqdm(
        train_loader,
        desc=epoch_label,
        leave=False,
        disable=not _is_main_process(rank),
    )
    for batch_idx, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs, targets = apply_mixup(
            inputs=inputs,
            targets=targets,
            alpha=mixup_alpha,
            probability=mixup_probability,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        running_loss += loss.item()
        batch_count = batch_idx
        if _is_main_process(rank):
            progress.set_postfix({"batch_loss": running_loss / batch_idx})

    local_avg = running_loss / max(batch_count, 1)
    loss_tensor = torch.tensor(local_avg, device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor = loss_tensor / dist.get_world_size()
    return float(loss_tensor.item())


def build_experiment_configs() -> List[Dict[str, Any]]:
    # Model families are inspired by strong BirdCLEF pipelines using timm image backbones on log-mels.
    return [
        {
            "name": "ft01_effb0_anchor",
            "backbone_name": "efficientnet_b0",
            "image_size": 224,
            "freq_mask_param": 12,
            "time_mask_param": 24,
            "specaugment_masks": 2,
            "spec_noise_std": 0.01,
            "mixup_alpha": 0.35,
            "mixup_probability": 1.0,
            "freeze_backbone_epochs": 2,
            "lr_head": 1e-3,
            "lr_backbone": 2e-4,
            "weight_decay": 2e-4,
            "warmup_epochs": 2,
            "min_lr_ratio": 0.03,
        },
        {
            "name": "ft02_effb2",
            "backbone_name": "efficientnet_b2",
            "image_size": 260,
            "freq_mask_param": 12,
            "time_mask_param": 24,
            "specaugment_masks": 2,
            "spec_noise_std": 0.01,
            "mixup_alpha": 0.35,
            "mixup_probability": 1.0,
            "freeze_backbone_epochs": 2,
            "lr_head": 1e-3,
            "lr_backbone": 1.5e-4,
            "weight_decay": 2e-4,
            "warmup_epochs": 2,
            "min_lr_ratio": 0.03,
        },
        {
            "name": "ft03_convnext_tiny",
            "backbone_name": "convnext_tiny",
            "image_size": 224,
            "freq_mask_param": 10,
            "time_mask_param": 20,
            "specaugment_masks": 2,
            "spec_noise_std": 0.0,
            "mixup_alpha": 0.3,
            "mixup_probability": 1.0,
            "freeze_backbone_epochs": 3,
            "lr_head": 8e-4,
            "lr_backbone": 1e-4,
            "weight_decay": 4e-4,
            "warmup_epochs": 3,
            "min_lr_ratio": 0.02,
        },
        {
            "name": "ft04_mobilenetv3",
            "backbone_name": "mobilenetv3_large_100",
            "image_size": 224,
            "freq_mask_param": 12,
            "time_mask_param": 24,
            "specaugment_masks": 2,
            "spec_noise_std": 0.01,
            "mixup_alpha": 0.4,
            "mixup_probability": 1.0,
            "freeze_backbone_epochs": 1,
            "lr_head": 1.2e-3,
            "lr_backbone": 3e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "min_lr_ratio": 0.05,
        },
        {
            "name": "ft05_resnet50",
            "backbone_name": "resnet50",
            "image_size": 224,
            "freq_mask_param": 8,
            "time_mask_param": 16,
            "specaugment_masks": 1,
            "spec_noise_std": 0.0,
            "mixup_alpha": 0.2,
            "mixup_probability": 0.8,
            "freeze_backbone_epochs": 2,
            "lr_head": 1e-3,
            "lr_backbone": 2e-4,
            "weight_decay": 3e-4,
            "warmup_epochs": 2,
            "min_lr_ratio": 0.03,
        },
    ]


def save_experiment_summaries(output_dir: Path, all_summaries: List[Dict[str, Any]], rank: int) -> None:
    results_json_path = output_dir / "experiments_summary.json"
    results_csv_path = output_dir / "experiments_summary.csv"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, sort_keys=True)
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df = summary_df.sort_values(by=["best_val_auc", "best_val_loss"], ascending=[False, True])
        summary_df.to_csv(results_csv_path, index=False)
    _print_main(rank, f"Saved experiment summary: {results_json_path}")
    _print_main(rank, f"Saved experiment summary: {results_csv_path}")


def run_experiment(
    experiment: Dict[str, Any],
    train_dataset: BirdCLEFDataset,
    val_dataset: Optional[BirdCLEFDataset],
    label_map: Dict[str, int],
    base_model_config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    rank: int,
    local_rank: int,
    distributed: bool,
    use_bf16: bool,
    train_meta_len: int,
    train_rank_meta_len: int,
    val_meta_len: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    grad_clip_norm: float,
    early_stop_patience: int,
) -> Dict[str, Any]:
    exp_name = experiment["name"]
    exp_output_dir = output_dir / exp_name
    if _is_main_process(rank):
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model_config = base_model_config.copy()
    model_config.update(
        {
            "n_classes": len(label_map),
            "backbone_name": experiment["backbone_name"],
            "image_size": int(experiment["image_size"]),
            "freq_mask_param": int(experiment["freq_mask_param"]),
            "time_mask_param": int(experiment["time_mask_param"]),
            "specaugment_masks": int(experiment["specaugment_masks"]),
            "spec_noise_std": float(experiment["spec_noise_std"]),
        }
    )

    model = TimmSpectrogramClassifier(
        **model_config,
        pretrained=True,
    ).to(device)

    freeze_backbone_epochs = int(experiment.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        model.freeze_backbone()

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer, fused_enabled = build_optimizer(
        model=model,
        lr_backbone=float(experiment["lr_backbone"]),
        lr_head=float(experiment["lr_head"]),
        weight_decay=float(experiment["weight_decay"]),
        device=device,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        warmup_epochs=int(experiment["warmup_epochs"]),
        min_lr_ratio=float(experiment["min_lr_ratio"]),
    )
    amp_enabled = use_bf16 and device.type == "cuda"

    _print_main(rank, "")
    _print_main(rank, f"=== {exp_name} ===")
    _print_main(rank, f"Backbone: {experiment['backbone_name']} | image_size={experiment['image_size']}")
    _print_main(rank, f"DDP enabled: {distributed} (world_size={dist.get_world_size() if dist.is_initialized() else 1})")
    _print_main(rank, f"Training rows: {train_meta_len}")
    _print_main(rank, f"Rows per-rank training shard: {train_rank_meta_len}")
    _print_main(rank, f"Validation rows: {val_meta_len}")
    _print_main(rank, f"Num labels: {len(label_map)}")
    _print_main(
        rank,
        (
            f"mixup(alpha={experiment['mixup_alpha']}, p={experiment['mixup_probability']}) | "
            f"freeze_backbone_epochs={freeze_backbone_epochs}"
        ),
    )

    best_val_auc = float("nan")
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            live_model = model.module if isinstance(model, DDP) else model
            live_model.unfreeze_backbone()

        avg_train_loss = train_one_epoch_ddp(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch_label=f"{exp_name} | epoch {epoch}",
            rank=rank,
            mixup_alpha=float(experiment["mixup_alpha"]),
            mixup_probability=float(experiment["mixup_probability"]),
        )
        scheduler.step()

        if dist.is_initialized():
            dist.barrier()

        stop_flag = torch.tensor([0], device=device, dtype=torch.int32)
        if _is_main_process(rank):
            eval_model = model.module if isinstance(model, DDP) else model
            if val_loader is None:
                raise RuntimeError("Validation loader is not initialized on main rank.")
            val_loss, val_auc = evaluate_model(
                eval_model,
                val_loader,
                criterion,
                device,
                use_bf16=use_bf16,
            )

            row = {
                "experiment": exp_name,
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "backbone_name": experiment["backbone_name"],
                "mixup_alpha": float(experiment["mixup_alpha"]),
                "mixup_probability": float(experiment["mixup_probability"]),
                "lr_head": float(experiment["lr_head"]),
                "lr_backbone": float(experiment["lr_backbone"]),
                "freeze_backbone_epochs": freeze_backbone_epochs,
            }
            history.append(row)
            print(
                f"{exp_name} | Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_birdclef_roc_auc={val_auc:.4f}"
            )

            should_save = False
            if is_better_score(val_auc, best_val_auc):
                best_val_auc = val_auc
                should_save = True
                no_improve_epochs = 0
            elif pd.isna(val_auc) and val_loss < best_val_loss:
                should_save = True
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if should_save:
                best_val_loss = val_loss
                best_epoch = epoch
                model_path = exp_output_dir / "best_model.pt"
                save_model(
                    {
                        "model_state": eval_model.state_dict(),
                        "label_map": label_map,
                        "model_config": model_config,
                        "epoch": epoch,
                        "best_val_auc": val_auc,
                        "best_val_loss": val_loss,
                        "use_bf16": amp_enabled,
                        "fused_adamw": fused_enabled,
                        "backbone_name": experiment["backbone_name"],
                        "freeze_backbone_epochs": freeze_backbone_epochs,
                        "lr_head": float(experiment["lr_head"]),
                        "lr_backbone": float(experiment["lr_backbone"]),
                    },
                    model_path,
                )
                with open(exp_output_dir / "label_map.json", "w", encoding="utf-8") as f:
                    json.dump(label_map, f, indent=2, sort_keys=True)

            save_model(
                {
                    "model_state": eval_model.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "experiment": experiment,
                    "model_config": model_config,
                },
                exp_output_dir / "last_model.pt",
            )

            if no_improve_epochs >= early_stop_patience:
                stop_flag[0] = 1

        if dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
            dist.barrier()

        if int(stop_flag.item()) == 1:
            _print_main(rank, f"Early stopping triggered for {exp_name} at epoch {epoch}.")
            break

    summary: Dict[str, Any] = {
        "experiment": exp_name,
        "architecture": "timm_spectrogram",
        "backbone_name": experiment["backbone_name"],
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "mixup_alpha": float(experiment["mixup_alpha"]),
        "mixup_probability": float(experiment["mixup_probability"]),
        "lr_head": float(experiment["lr_head"]),
        "lr_backbone": float(experiment["lr_backbone"]),
        "weight_decay": float(experiment["weight_decay"]),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "output_dir": str(exp_output_dir),
        "model_path": str(exp_output_dir / "best_model.pt"),
        "model_config": model_config,
        "history": history,
    }

    if _is_main_process(rank):
        if history:
            pd.DataFrame(history).to_csv(exp_output_dir / "metrics.csv", index=False)
        with open(exp_output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()

    return summary


def main() -> None:
    distributed, world_size, rank, local_rank = _get_distributed_context()
    _setup_distributed(distributed)

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = repo_root / "birdclef_example" / "outputs" / "experiments_ft2"

    train_audio_metadata_path = data_dir / "train.csv"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    taxonomy_path = data_dir / "taxonomy.csv"
    train_audio_dir = data_dir / "train_audio"
    train_soundscape_dir = data_dir / "train_soundscapes"

    epochs = 15
    batch_size = 32
    dropout = 0.35
    sample_rate = 32000
    segment_duration = 5.0
    n_mels = 160
    n_fft = 2048
    hop_length = 512
    val_split = 0.30
    seed = 42
    preload_audio = True
    preload_workers = 8
    num_workers = 4
    use_bf16 = True
    grad_clip_norm = 1.0
    early_stop_patience = 4

    if not train_audio_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_audio_dir}")
    if not train_soundscape_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_soundscape_dir}")

    set_seed(seed + rank)
    if _is_main_process(rank):
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

    train_rank_meta = build_rank_metadata_shard(train_meta, world_size=world_size, rank=rank)
    train_dataset = BirdCLEFDataset(
        metadata=train_rank_meta,
        label_map=label_map,
        sample_rate=sample_rate,
        duration=segment_duration,
        training=True,
        preload_audio=preload_audio,
        preload_workers=preload_workers,
    )

    val_dataset = None
    if _is_main_process(rank):
        val_dataset = BirdCLEFDataset(
            metadata=val_meta,
            label_map=label_map,
            sample_rate=sample_rate,
            duration=segment_duration,
            training=False,
            preload_audio=preload_audio,
            preload_workers=preload_workers,
        )

    base_model_config = {
        "n_classes": len(label_map),
        "dropout": dropout,
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
    }

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    experiments = build_experiment_configs()
    all_summaries: List[Dict[str, Any]] = []

    for experiment in experiments:
        summary: Dict[str, Any]
        try:
            summary = run_experiment(
                experiment=experiment,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                label_map=label_map,
                base_model_config=base_model_config,
                output_dir=output_dir,
                device=device,
                rank=rank,
                local_rank=local_rank,
                distributed=distributed,
                use_bf16=use_bf16,
                train_meta_len=len(train_meta),
                train_rank_meta_len=len(train_rank_meta),
                val_meta_len=len(val_meta),
                batch_size=batch_size,
                num_workers=num_workers,
                epochs=epochs,
                grad_clip_norm=grad_clip_norm,
                early_stop_patience=early_stop_patience,
            )
        except Exception as ex:
            if _is_main_process(rank):
                failure = {
                    "experiment": experiment.get("name", "unknown"),
                    "architecture": "timm_spectrogram",
                    "backbone_name": experiment.get("backbone_name", "unknown"),
                    "status": "failed",
                    "error": str(ex),
                }
                all_summaries.append(failure)
                save_experiment_summaries(output_dir=output_dir, all_summaries=all_summaries, rank=rank)
                print(f"[WARN] Experiment failed: {failure['experiment']} -> {failure['error']}")
            if dist.is_initialized():
                dist.barrier()
            continue

        if _is_main_process(rank):
            all_summaries.append({key: value for key, value in summary.items() if key != "history"})
            save_experiment_summaries(output_dir=output_dir, all_summaries=all_summaries, rank=rank)

    _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
