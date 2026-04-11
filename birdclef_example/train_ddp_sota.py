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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (  # noqa: E402
    BirdCLEFDataset,
    SpectrogramTransform,
    build_label_map,
    parse_primary_labels,
    parse_secondary_labels,
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


class AudioWaveAugment(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        gain_prob: float = 0.7,
        noise_prob: float = 0.5,
        filter_prob: float = 0.35,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain_prob = gain_prob
        self.noise_prob = noise_prob
        self.filter_prob = filter_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        input_dtype = x.dtype
        # torchaudio IIR CUDA kernels are sensitive to reduced-precision dtypes (e.g. bf16 under autocast).
        # Keep this augmentation path in float32 for numerical/runtime stability, then cast back.
        x = x.float()
        bsz = x.size(0)
        for i in range(bsz):
            xi = x[i : i + 1]
            if torch.rand(1, device=x.device).item() < self.gain_prob:
                gain = float(torch.empty(1, device=x.device).uniform_(0.7, 1.3).item())
                xi = xi * gain
            if torch.rand(1, device=x.device).item() < self.noise_prob:
                base = xi.std().detach().clamp(min=1e-4)
                noise_scale = float(torch.empty(1, device=x.device).uniform_(0.002, 0.02).item())
                xi = xi + torch.randn_like(xi) * (base * noise_scale)
            if torch.rand(1, device=x.device).item() < self.filter_prob:
                # Use pure PyTorch smoothing/differencing instead of torchaudio biquad kernels
                # to avoid CUDA IIR kernel instability on some environments.
                kernel = int(torch.randint(5, 33, (1,), device=x.device).item())
                if kernel % 2 == 0:
                    kernel += 1
                low = F.avg_pool1d(xi, kernel_size=kernel, stride=1, padding=kernel // 2)
                if torch.rand(1, device=x.device).item() < 0.5:
                    xi = low
                else:
                    xi = xi - low
            x[i : i + 1] = xi
        return x.to(dtype=input_dtype)


class BCEFocalComboLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.03,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + (1.0 - targets) * self.label_smoothing

        bce_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = bce_elem.mean()

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = ((1.0 - pt).pow(self.focal_gamma) * bce_elem).mean()

        return self.bce_weight * bce + self.focal_weight * focal


class TimmSpectrogramClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        backbone_name: str,
        pretrained: bool,
        image_size: int,
        dropout: float,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        f_min: int = 20,
        f_max: int | None = None,
        freq_mask_param: int = 0,
        time_mask_param: int = 0,
        specaugment_masks: int = 0,
        spec_noise_std: float = 0.0,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.wave_aug = AudioWaveAugment(sample_rate=sample_rate)
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
            pretrained=pretrained,
            in_chans=3,
            num_classes=n_classes,
            drop_rate=dropout,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform
        if self.training:
            x = self.wave_aug(x)
        x = self.spectrogram(x)
        if self.training:
            x = self.spec_augment(x)
        x = x.clamp(min=-4.0, max=4.0)
        x = (x + 4.0) / 8.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def _collect_row_labels(row: pd.Series) -> List[str]:
    labels = parse_primary_labels(row.get("primary_label"))
    if str(row.get("source", "")) == "train_audio":
        labels.extend(parse_secondary_labels(row.get("secondary_labels")))
    return list(dict.fromkeys([label for label in labels if label]))


def build_sample_weights(train_meta: pd.DataFrame, label_map: Dict[str, int], power: float = 0.5) -> torch.Tensor:
    class_counts = torch.zeros(len(label_map), dtype=torch.float64)
    rows_labels: List[List[str]] = []
    for _, row in train_meta.iterrows():
        labels = _collect_row_labels(row)
        rows_labels.append(labels)
        for label in labels:
            idx = label_map.get(label)
            if idx is not None:
                class_counts[idx] += 1.0

    class_counts = class_counts.clamp(min=1.0)
    inv = (1.0 / class_counts).pow(power)

    weights = torch.ones(len(train_meta), dtype=torch.float32)
    for i, labels in enumerate(rows_labels):
        if not labels:
            continue
        vals = []
        for label in labels:
            idx = label_map.get(label)
            if idx is not None:
                vals.append(float(inv[idx].item()))
        if vals:
            weights[i] = float(sum(vals) / len(vals))
    return weights


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
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    return lam * inputs + (1 - lam) * inputs[idx], lam * targets + (1 - lam) * targets[idx]


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
        if any(k in name for k in ["classifier", "head", "fc"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
    ]

    fused_enabled = False
    opt_kwargs: Dict[str, Any] = {}
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        opt_kwargs["fused"] = True
        fused_enabled = True
    optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
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
            eta_min=min(g["lr"] for g in optimizer.param_groups) * min_lr_ratio,
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
        eta_min=min(g["lr"] for g in optimizer.param_groups) * min_lr_ratio,
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])


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
    progress = tqdm(train_loader, desc=epoch_label, leave=False, disable=not _is_main_process(rank))
    for batch_idx, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs, targets = apply_mixup(inputs, targets, mixup_alpha, mixup_probability)

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
    experiments: List[Dict[str, Any]] = []

    # Focused sweep on eca_nfnet_l0 only.
    nfnet_base = {
        "backbone_name": "eca_nfnet_l0",
        "image_size": 288,
        "n_mels": 160,
        "n_fft": 2048,
        "hop_length": 512,
        "freq_mask_param": 12,
        "time_mask_param": 24,
        "specaugment_masks": 2,
        "spec_noise_std": 0.0,
        "mixup_probability": 1.0,
        "weight_decay": 4e-4,
        "warmup_epochs": 3,
        "min_lr_ratio": 0.02,
        "bce_weight": 0.7,
        "focal_weight": 0.3,
        "focal_gamma": 2.0,
        "sampling_power": 0.5,
    }
    idx = 1
    for lr_backbone in [6e-5, 8e-5, 1.0e-4, 1.2e-4]:
        for lr_head in [6e-4, 7e-4, 8e-4, 9e-4]:
            exp = nfnet_base.copy()
            exp.update(
                {
                    "name": f"sota{idx:02d}_nfnet_lr{lr_head:.0e}_{lr_backbone:.0e}",
                    "lr_head": lr_head,
                    "lr_backbone": lr_backbone,
                    "mixup_alpha": 0.2,
                    "label_smoothing": 0.03,
                }
            )
            experiments.append(exp)
            idx += 1

    for mixup_alpha, smoothing in [(0.10, 0.02), (0.15, 0.02), (0.25, 0.02), (0.25, 0.03)]:
        exp = nfnet_base.copy()
        exp.update(
            {
                "name": f"sota{idx:02d}_nfnet_mix{int(mixup_alpha*100)}_ls{int(smoothing*100)}",
                "lr_head": 8e-4,
                "lr_backbone": 1.0e-4,
                "mixup_alpha": mixup_alpha,
                "label_smoothing": smoothing,
            }
        )
        experiments.append(exp)
        idx += 1
    for focal_gamma, focal_weight in [(1.5, 0.25), (2.0, 0.35), (2.5, 0.35)]:
        exp = nfnet_base.copy()
        exp.update(
            {
                "name": f"sota{idx:02d}_nfnet_fg{str(focal_gamma).replace('.', '')}_fw{int(focal_weight*100)}",
                "lr_head": 8e-4,
                "lr_backbone": 1.0e-4,
                "mixup_alpha": 0.2,
                "label_smoothing": 0.03,
                "focal_gamma": focal_gamma,
                "focal_weight": focal_weight,
                "bce_weight": max(0.5, 1.0 - focal_weight),
            }
        )
        experiments.append(exp)
        idx += 1

    return experiments


def save_experiment_summaries(output_dir: Path, all_summaries: List[Dict[str, Any]], rank: int) -> None:
    json_path = output_dir / "experiments_summary.json"
    csv_path = output_dir / "experiments_summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, sort_keys=True)
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        if "best_val_auc" in df.columns:
            df = df.sort_values(by=["best_val_auc", "best_val_loss"], ascending=[False, True])
        df.to_csv(csv_path, index=False)
    _print_main(rank, f"Saved experiment summary: {json_path}")
    _print_main(rank, f"Saved experiment summary: {csv_path}")


def run_experiment(
    experiment: Dict[str, Any],
    train_rank_meta: pd.DataFrame,
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
    preload_audio: bool,
    preload_workers: int,
) -> Dict[str, Any]:
    exp_name = experiment["name"]
    exp_output_dir = output_dir / exp_name
    if _is_main_process(rank):
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    exp_train_meta = train_rank_meta.copy().reset_index(drop=True)
    train_dataset = BirdCLEFDataset(
        metadata=exp_train_meta,
        label_map=label_map,
        sample_rate=int(base_model_config["sample_rate"]),
        duration=float(base_model_config["duration"]),
        training=True,
        preload_audio=preload_audio,
        preload_workers=preload_workers,
    )

    sample_weights = build_sample_weights(
        exp_train_meta,
        label_map,
        power=float(experiment.get("sampling_power", 0.5)),
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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

    model_config = {
        "n_classes": len(label_map),
        "backbone_name": experiment["backbone_name"],
        "pretrained": True,
        "image_size": int(experiment["image_size"]),
        "dropout": float(base_model_config["dropout"]),
        "sample_rate": int(base_model_config["sample_rate"]),
        "n_mels": int(experiment["n_mels"]),
        "n_fft": int(experiment["n_fft"]),
        "hop_length": int(experiment["hop_length"]),
        "freq_mask_param": int(experiment["freq_mask_param"]),
        "time_mask_param": int(experiment["time_mask_param"]),
        "specaugment_masks": int(experiment["specaugment_masks"]),
        "spec_noise_std": float(experiment["spec_noise_std"]),
    }
    model = TimmSpectrogramClassifier(**model_config).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    criterion = BCEFocalComboLoss(
        bce_weight=float(experiment["bce_weight"]),
        focal_weight=float(experiment["focal_weight"]),
        focal_gamma=float(experiment["focal_gamma"]),
        label_smoothing=float(experiment["label_smoothing"]),
    )

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

    best_val_auc = float("nan")
    best_val_loss = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
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

        if _is_main_process(rank):
            eval_model = model.module if isinstance(model, DDP) else model
            if val_loader is None:
                raise RuntimeError("Validation loader is not initialized on main rank.")
            # Evaluate with plain BCE for consistent monitoring.
            val_loss, val_auc = evaluate_model(
                eval_model,
                val_loader,
                torch.nn.BCEWithLogitsLoss(),
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
                "bce_weight": float(experiment["bce_weight"]),
                "focal_weight": float(experiment["focal_weight"]),
                "focal_gamma": float(experiment["focal_gamma"]),
                "label_smoothing": float(experiment["label_smoothing"]),
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
            elif pd.isna(val_auc) and val_loss < best_val_loss:
                should_save = True

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
                        "fused_adamw": fused_enabled,
                        "experiment": experiment,
                    },
                    model_path,
                )
                with open(exp_output_dir / "label_map.json", "w", encoding="utf-8") as f:
                    json.dump(label_map, f, indent=2, sort_keys=True)

        if dist.is_initialized():
            dist.barrier()

    summary: Dict[str, Any] = {
        "experiment": exp_name,
        "architecture": "sota_timm_spectrogram",
        "backbone_name": experiment["backbone_name"],
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "mixup_alpha": float(experiment["mixup_alpha"]),
        "mixup_probability": float(experiment["mixup_probability"]),
        "label_smoothing": float(experiment["label_smoothing"]),
        "bce_weight": float(experiment["bce_weight"]),
        "focal_weight": float(experiment["focal_weight"]),
        "focal_gamma": float(experiment["focal_gamma"]),
        "sampling_power": float(experiment["sampling_power"]),
        "lr_head": float(experiment["lr_head"]),
        "lr_backbone": float(experiment["lr_backbone"]),
        "weight_decay": float(experiment["weight_decay"]),
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
    del train_dataset
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
    output_dir = repo_root / "birdclef_example" / "outputs" / "experiments_sota3"

    train_audio_metadata_path = data_dir / "train.csv"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    taxonomy_path = data_dir / "taxonomy.csv"
    train_audio_dir = data_dir / "train_audio"
    train_soundscape_dir = data_dir / "train_soundscapes"

    epochs = 15
    batch_size = 24
    dropout = 0.35
    sample_rate = 32000
    segment_duration = 5.0
    val_split = 0.30
    seed = 42
    preload_audio = True
    preload_workers = 8
    num_workers = 4
    use_bf16 = True
    grad_clip_norm = 1.0

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

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    base_model_config = {
        "dropout": dropout,
        "sample_rate": sample_rate,
        "duration": segment_duration,
    }

    experiments = build_experiment_configs()
    all_summaries: List[Dict[str, Any]] = []

    for experiment in experiments[10:]:
        summary: Dict[str, Any]
        try:
            summary = run_experiment(
                experiment=experiment,
                train_rank_meta=train_rank_meta,
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
                preload_audio=preload_audio,
                preload_workers=preload_workers,
            )
        except Exception as ex:
            if _is_main_process(rank):
                failure = {
                    "experiment": experiment.get("name", "unknown"),
                    "architecture": "sota_timm_spectrogram",
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
