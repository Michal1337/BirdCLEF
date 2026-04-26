"""SOTA timm-spectrogram trainer — focal-recordings variant.

Trains on the entire train.csv (≈35.5k focal recordings, 206 classes) and
evaluates on the entire labeled-soundscape pool (66 files = 59 fully + 7
partially labeled = 739 windows, 75 atomic classes). No fold loop — one model
per experiment, single held-out val set covering as much soundscape signal as
exists.

Why this exists alongside `train_ddp_sota_5fold.py`:
    The 5-fold trainer trains on the 59 labeled soundscapes only — 708 windows
    of training data is starvation for a deep CNN. This file flips that:
    35.5k focal recordings train, 739 soundscape windows validate. Domain
    mismatch (focal vs soundscape) is real and is the main reason every
    BirdCLEF top solution heavily augments + mixes up.

Class coverage caveat:
    Soundscape labels cover only 75 of 234 classes. 25 of those 75 are
    `47158sonNN` sonotypes that have ZERO focal recordings in train.csv —
    a CNN trained only on train.csv cannot learn them. This script reports
    two metrics so the picture stays honest:

      val_auc_seen        macro AUC over the ~75 val classes with positives.
                          Mirrors the official LB metric (which skips
                          zero-positive classes). Selection metric.
      val_auc_focal_seen  macro AUC over the ~48 classes that are BOTH in
                          train.csv AND have val positives. Diagnoses
                          "are we learning what we trained on?"

    Expect val_auc_focal_seen to be substantially higher than val_auc_seen —
    the gap is the sonotype + the 27 other unlearnable classes pulling the
    average down. That gap is structural for this trainer; closing it needs
    a soundscape-trained head (see `train_ddp_sota_5fold.py` or SED).

Invocation pattern matches `train_ddp_sota_5fold.py`:

    torchrun --standalone --nproc_per_node=2 \\
        -m birdclef_example.train_ddp_focal --exp sota11_nfnet_lr8e-04_1e-04

No-arg run sweeps every experiment from `build_experiment_configs()`.

Outputs:
    birdclef_example/outputs/focal/<config_name>/
        best_model.pt      # selected on val_auc_seen
        label_map.json
        metrics.csv        # per-epoch history (val_loss, val_auc_seen, val_auc_focal_seen)
        summary.json

CSV summary at `birdclef_example/outputs/focal/experiments_summary.csv`:
    sorted by `primary` (= best val_auc_seen) desc.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (
    BirdCLEFDataset,
    build_label_map,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)
from birdclef_example.train_ddp_sota import (  # noqa: E402
    BCEFocalComboLoss,
    TimmSpectrogramClassifier,
    _cleanup_distributed,
    _get_distributed_context,
    _is_main_process,
    _print_main,
    _setup_distributed,
    build_experiment_configs,
    build_optimizer,
    build_rank_metadata_shard,
    build_sample_weights,
    build_scheduler,
    evaluate_validation_metrics,
    train_one_epoch_ddp,
)
from birdclef_example.utils import birdclef_roc_auc, is_better_score, save_model, set_seed  # noqa: E402


def build_focal_experiment_configs() -> List[Dict[str, Any]]:
    """Extended config grid for the focal trainer.

    Includes the existing 23 nfnet variants from `build_experiment_configs()`
    plus focal-specific extensions:

      - **Backbone diversity**: efficientnet_v2_s, convnext_pico, regnety_032,
        eca_nfnet_l1 (one notch up). Beats Perch on focal classes likely
        requires architecture diversity, not just LR tuning.
      - **Longer training**: 30 / 50 epochs for the most promising bases —
        focal→soundscape domain transfer benefits from more updates.
      - **Heavier augmentation**: mixup α=1.0 + spec_noise_std=0.05–0.10
        + larger SpecAug masks. Reduces train→val domain gap.

    Each config can override CLI `--epochs` via the `epochs` key (read by
    `run_one_experiment`). Configs without an `epochs` key inherit from CLI.
    """
    base_configs = list(build_experiment_configs())

    # Common defaults shared across the new backbone variants
    common = {
        "image_size": 288,
        "n_mels": 160,
        "n_fft": 2048,
        "hop_length": 512,
        "freq_mask_param": 12,
        "time_mask_param": 24,
        "specaugment_masks": 2,
        "spec_noise_std": 0.0,
        "mixup_alpha": 0.2,
        "mixup_probability": 1.0,
        "label_smoothing": 0.03,
        "weight_decay": 4e-4,
        "warmup_epochs": 3,
        "min_lr_ratio": 0.02,
        "bce_weight": 0.7,
        "focal_weight": 0.3,
        "focal_gamma": 2.0,
        "sampling_power": 0.5,
        "lr_head": 8e-4,
        "lr_backbone": 1e-4,
    }

    extensions: List[Dict[str, Any]] = []

    # --- Backbone diversity (each at default 30 epochs) ---
    for backbone, lr_b, lr_h, img in [
        ("eca_nfnet_l1",     8e-5,  6e-4, 320),
        ("efficientnet_v2_s", 1e-4, 8e-4, 288),
        ("convnext_pico",    1e-4,  8e-4, 288),
        ("regnety_032",      1e-4,  8e-4, 288),
    ]:
        cfg = dict(common)
        cfg.update({
            "name": f"focal_{backbone}_e30",
            "backbone_name": backbone,
            "image_size": img,
            "lr_backbone": lr_b,
            "lr_head": lr_h,
            "epochs": 30,
        })
        extensions.append(cfg)

    # --- Longer training on the strongest base (eca_nfnet_l0, sota02 lr) ---
    for n_epochs in (30, 50):
        cfg = dict(common)
        cfg.update({
            "name": f"focal_nfnet_l0_e{n_epochs}",
            "backbone_name": "eca_nfnet_l0",
            "lr_backbone": 6e-5,  # sota02
            "lr_head": 7e-4,      # sota02
            "epochs": n_epochs,
        })
        extensions.append(cfg)

    # --- Heavy aug variants (longer training to let the model exploit it) ---
    for tag, overrides in [
        ("mixupA10",   {"mixup_alpha": 1.0}),
        ("specnoise",  {"spec_noise_std": 0.08, "freq_mask_param": 16, "time_mask_param": 32}),
        ("heavyaug",   {"mixup_alpha": 0.6, "spec_noise_std": 0.05,
                        "freq_mask_param": 14, "time_mask_param": 28, "specaugment_masks": 3}),
    ]:
        cfg = dict(common)
        cfg.update({
            "name": f"focal_nfnet_l0_e30_{tag}",
            "backbone_name": "eca_nfnet_l0",
            "lr_backbone": 6e-5,
            "lr_head": 7e-4,
            "epochs": 30,
        })
        cfg.update(overrides)
        extensions.append(cfg)

    return base_configs + extensions


def _select_experiments(name_filter: Optional[str]) -> List[Dict[str, Any]]:
    all_exp = build_focal_experiment_configs()
    if not name_filter:
        return all_exp
    matches = [e for e in all_exp if e["name"] == name_filter]
    if not matches:
        sub = [e for e in all_exp if name_filter in e["name"]]
        if not sub:
            available = ", ".join(e["name"] for e in all_exp[:5])
            raise SystemExit(
                f"No experiment matched --exp '{name_filter}'. "
                f"First 5 available: {available} ..."
            )
        return sub
    return matches


def _build_focal_class_indices(
    train_meta: pd.DataFrame, label_map: Dict[str, int],
) -> np.ndarray:
    """Indices of classes that have at least one focal recording in train.csv.

    Used as the `mapped_label_indices` argument to evaluate_validation_metrics
    so the secondary `val_auc_focal_seen` metric is the AUC restricted to
    classes the CNN actually trained on.
    """
    seen: set[str] = set()
    if "primary_label" in train_meta.columns:
        for raw in train_meta["primary_label"].astype(str).tolist():
            for lb in raw.replace(";", " ").replace(",", " ").split():
                if lb in label_map:
                    seen.add(lb)
    return np.array(sorted(label_map[lb] for lb in seen), dtype=np.int64)


def _build_taxa_indices(label_map: Dict[str, int],
                        taxonomy_path: Path) -> Dict[str, np.ndarray]:
    """Per-taxonomic-class label indices (Aves / Amphibia / Insecta /
    Mammalia / Reptilia). Lets us split macro AUC by taxon and surface
    where the CNN actually moves the needle (e.g. Amphibia after XC,
    Mammalia/Reptilia where Perch is blind).
    """
    tax = pd.read_csv(taxonomy_path)
    label_to_taxon = dict(zip(
        tax["primary_label"].astype(str), tax["class_name"].astype(str),
    ))
    out: Dict[str, list[int]] = {}
    for lb, idx in label_map.items():
        taxon = label_to_taxon.get(str(lb), "Unknown")
        out.setdefault(taxon, []).append(int(idx))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


def _r4(x) -> float | None:
    """Round-or-None: round to 4 decimal places, propagate NaN as None for
    JSON-friendliness. Pandas handles None as NaN cleanly in DataFrames.
    """
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


@torch.no_grad()
def _evaluate_full(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    focal_class_indices: np.ndarray,
    taxa_indices: Dict[str, np.ndarray],
    use_bf16: bool = False,
) -> Dict[str, Any]:
    """One forward pass over the val loader → loss, val_auc_seen,
    val_auc_focal_seen, and per-taxon macro AUCs. All AUCs use
    `birdclef_roc_auc` which auto-skips classes with zero positives.
    """
    model.eval()
    losses: List[float] = []
    targets_chunks: List[torch.Tensor] = []
    preds_chunks: List[torch.Tensor] = []
    amp_enabled = use_bf16 and device.type == "cuda"

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)
        losses.append(loss.item())
        targets_chunks.append(targets.float().cpu())
        preds_chunks.append(torch.sigmoid(logits).float().cpu())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    if not targets_chunks:
        return {"val_loss": avg_loss}

    Y = torch.cat(targets_chunks, dim=0).numpy()
    P = torch.cat(preds_chunks, dim=0).numpy()

    def _auc(idx: np.ndarray) -> float:
        if len(idx) == 0:
            return float("nan")
        try:
            return float(birdclef_roc_auc(Y[:, idx], P[:, idx]))
        except ValueError:
            return float("nan")

    out: Dict[str, Any] = {
        "val_loss":           avg_loss,
        "val_auc_seen":       _auc(np.arange(Y.shape[1], dtype=np.int64)),
        "val_auc_focal_seen": _auc(focal_class_indices),
    }
    for taxon, idx in taxa_indices.items():
        # Stable lowercase keys: "val_auc_aves", "val_auc_amphibia", etc.
        out[f"val_auc_{taxon.lower()}"] = _auc(idx)
    return out


def run_one_experiment(
    experiment: Dict[str, Any],
    train_dataset: BirdCLEFDataset,
    val_dataset: Optional[BirdCLEFDataset],
    train_rank_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    label_map: Dict[str, int],
    focal_class_indices: np.ndarray,
    taxa_indices: Dict[str, np.ndarray],
    base_model_config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    rank: int,
    local_rank: int,
    distributed: bool,
    use_bf16: bool,
    train_meta_len: int,
    train_rank_meta_len: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    grad_clip_norm: float,
) -> Dict[str, Any]:
    """One full training run for one experiment.

    `train_dataset`/`val_dataset` are built once in main() and reused across
    experiments — preloading 35k focal recordings once instead of per-config.

    Selection metric: `val_auc_seen` (= birdclef_roc_auc over the ~75 val
    classes with positives — same shape as LB). The diagnostic
    `val_auc_focal_seen` (~48-class AUC) comes for free from
    evaluate_validation_metrics's mapped_auc slot.

    Per-experiment overrides:
      - `experiment["epochs"]` — overrides CLI `--epochs` (lets the extended
        sweep mix 15-epoch quick runs with 50-epoch deep dives in one
        invocation). Falls back to the `epochs` arg if not present.
    """
    epochs = int(experiment.get("epochs", epochs))
    exp_name = experiment["name"]
    exp_output_dir = output_dir / exp_name
    if _is_main_process(rank):
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    sample_weights = build_sample_weights(
        train_rank_meta,
        label_map,
        power=float(experiment.get("sampling_power", 0.5)),
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if _is_main_process(rank) and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
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
        optimizer=optimizer, epochs=epochs,
        warmup_epochs=int(experiment["warmup_epochs"]),
        min_lr_ratio=float(experiment["min_lr_ratio"]),
    )
    amp_enabled = use_bf16 and device.type == "cuda"

    _print_main(rank, "")
    _print_main(rank, f"=== {exp_name} ===")
    _print_main(rank, f"Backbone: {experiment['backbone_name']} | image_size={experiment['image_size']}")
    _print_main(rank, f"DDP enabled: {distributed} (world_size="
                      f"{dist.get_world_size() if dist.is_initialized() else 1})")
    _print_main(rank, f"Train rows (focal): {train_meta_len}  (rank shard: {train_rank_meta_len})")
    _print_main(rank, f"Val rows (soundscape): {len(val_meta)}")
    _print_main(rank, f"Total label-map classes: {len(label_map)}  "
                      f"focal-trained classes: {len(focal_class_indices)}")

    # Selection metric is val_auc_focal_seen (NOT val_auc_seen). The 28 val
    # classes the CNN can't possibly learn (sonotypes etc., absent from
    # train.csv) sit at ~0.5 in val_auc_seen and add only noise to the
    # ranking. val_auc_focal_seen restricts to classes the CNN actually
    # trained on — cleaner ranker, same shape as the LB-relevant subset.
    best_metrics: Dict[str, Any] = {}
    best_val_loss = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        avg_train_loss = train_one_epoch_ddp(
            model=model, train_loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch_label=f"{exp_name} | ep {epoch}",
            rank=rank,
            mixup_alpha=float(experiment["mixup_alpha"]),
            mixup_probability=float(experiment["mixup_probability"]),
        )
        scheduler.step()

        if dist.is_initialized():
            dist.barrier()

        if _is_main_process(rank):
            eval_model = model.module if isinstance(model, DDP) else model
            assert val_loader is not None
            metrics = _evaluate_full(
                eval_model, val_loader, torch.nn.BCEWithLogitsLoss(),
                device, focal_class_indices=focal_class_indices,
                taxa_indices=taxa_indices, use_bf16=use_bf16,
            )

            # All floats rounded to 4 decimals on write; NaNs → None
            row = {
                "config_name": exp_name, "epoch": epoch,
                "train_loss": _r4(avg_train_loss),
                **{k: _r4(v) for k, v in metrics.items()},
            }
            history.append(row)
            taxa_log = "  ".join(
                f"{taxon[:3].lower()}={row.get(f'val_auc_{taxon.lower()}'):.4f}"
                for taxon in sorted(taxa_indices)
                if row.get(f"val_auc_{taxon.lower()}") is not None
            )
            print(
                f"{exp_name} | ep {epoch:02d} | "
                f"train={row['train_loss']} | "
                f"loss={row['val_loss']} "
                f"focal_seen={row['val_auc_focal_seen']} "
                f"seen={row['val_auc_seen']} | {taxa_log}"
            )

            # PRIMARY = val_auc_focal_seen
            current_primary = metrics.get("val_auc_focal_seen", float("nan"))
            best_primary = best_metrics.get("val_auc_focal_seen", float("nan"))
            should_save = False
            if is_better_score(current_primary, best_primary):
                best_metrics = dict(metrics)
                best_val_loss = float(metrics.get("val_loss", float("inf")))
                should_save = True
            elif pd.isna(current_primary) and metrics.get("val_loss", float("inf")) < best_val_loss:
                # Pathological: no val positives in focal subset. Tie-break on loss.
                best_val_loss = float(metrics.get("val_loss", float("inf")))
                should_save = True

            if should_save:
                best_epoch = epoch
                model_path = exp_output_dir / "best_model.pt"
                save_model(
                    {
                        "model_state": eval_model.state_dict(),
                        "label_map": label_map,
                        "model_config": model_config,
                        "epoch": epoch,
                        **{f"best_{k}": _r4(v) for k, v in metrics.items()},
                        "fused_adamw": fused_enabled,
                        "experiment": experiment,
                    },
                    model_path,
                )
                with open(exp_output_dir / "label_map.json", "w", encoding="utf-8") as f:
                    json.dump(label_map, f, indent=2, sort_keys=True)

        if dist.is_initialized():
            dist.barrier()

    # Build final summary — `primary` = best val_auc_focal_seen (same ranker
    # used for selection above). All floats rounded to 4 decimals.
    summary: Dict[str, Any] = {
        "_kind": "focal_run",
        "config_name": exp_name,
        "architecture": "sota_timm_spectrogram_focal",
        "backbone_name": experiment["backbone_name"],
        "best_epoch": best_epoch,
        "primary": _r4(best_metrics.get("val_auc_focal_seen")),
        **{f"best_{k}": _r4(v) for k, v in best_metrics.items()},
        "n_focal_classes": int(len(focal_class_indices)),
        "experiment_config": experiment,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None,
                    help="Experiment name from build_experiment_configs(). "
                         "Substring match supported. Default: sweep all configs.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--preload-workers", type=int, default=8)
    ap.add_argument("--preload-train-audio", action="store_true", default=False,
                    help="Preload all 35k train.csv files into RAM. Off by default — "
                         "needs ~130 GB; with `False` the dataset uses an LRU cache.")
    ap.add_argument("--preload-val-audio", action="store_true", default=True,
                    help="Preload all 66 labeled soundscapes (~14 MB total). Default on.")
    ap.add_argument("--no-preload-val-audio", dest="preload_val_audio", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-bf16", action="store_true", default=True)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir. Default: birdclef_example/outputs/focal")
    ap.add_argument("--include-xc", action="store_true",
                    help="Also load Xeno-Canto recordings (data/train_audio_xc.csv + "
                         "data/train_audio_xc/) on top of train.csv. Targets the weak "
                         "Amphibia/Aves classes from the per-class audit. Build via "
                         "`python -m birdclef.scripts._xc_fetch`.")
    args = ap.parse_args()

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
    output_dir = Path(args.out_dir) if args.out_dir else \
        repo_root / "birdclef_example" / "outputs" / "focal"

    train_csv_path = data_dir / "train.csv"
    train_audio_dir = data_dir / "train_audio"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    train_soundscape_dir = data_dir / "train_soundscapes"
    taxonomy_path = data_dir / "taxonomy.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Missing train.csv: {train_csv_path}")
    if not train_audio_dir.exists():
        raise FileNotFoundError(f"Missing train_audio/: {train_audio_dir}")
    if not train_soundscape_dir.exists():
        raise FileNotFoundError(f"Missing train_soundscapes/: {train_soundscape_dir}")

    set_seed(int(args.seed) + rank)
    if _is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Train: train.csv (focal recordings, ~35.5k rows × 206 classes)
    train_csv_df = pd.read_csv(train_csv_path)
    train_meta = prepare_train_audio_metadata(train_csv_df, train_audio_dir)

    # Optional: extend with Xeno-Canto recordings for the weak-class targets.
    # XC rows live in a parallel data/train_audio_xc/ tree; we rewrite
    # audio_filepath against that root so they coexist with train.csv rows
    # downstream without changing anything else (source="train_audio" is
    # preserved so secondary-label handling stays consistent).
    if args.include_xc:
        xc_csv = data_dir / "train_audio_xc.csv"
        xc_audio_dir = data_dir / "train_audio_xc"
        if not xc_csv.exists():
            raise FileNotFoundError(
                f"--include-xc requested but {xc_csv} missing. "
                f"Build it: python -m birdclef.scripts._xc_fetch"
            )
        xc_df = pd.read_csv(xc_csv)
        xc_meta = prepare_train_audio_metadata(xc_df, xc_audio_dir)
        _print_main(rank, f"[setup] including Xeno-Canto: +{len(xc_meta)} rows from {xc_csv}")
        train_meta = pd.concat([train_meta, xc_meta], ignore_index=True)

    # Drop rows whose audio file is missing on disk (the cache build step
    # might have been partial, or XC downloads may have failed).
    # Avoids hard-fail in BirdCLEFDataset.__getitem__.
    train_meta = train_meta[train_meta["audio_filepath"].apply(lambda p: Path(p).exists())]
    train_meta = train_meta.reset_index(drop=True)

    # Val: ALL labeled soundscape windows (fully + partially labeled = 66 files).
    # No fold split — the entire labeled pool is the val set. prepare_soundscape_metadata
    # already drops the duplicated rows that ship in the raw CSV.
    soundscape_df = pd.read_csv(soundscape_labels_path)
    val_meta = prepare_soundscape_metadata(soundscape_df, train_soundscape_dir)

    label_map = build_label_map(train_meta, taxonomy_path)
    expected_num_classes = 234
    if len(label_map) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes from taxonomy, got {len(label_map)}."
        )

    focal_class_indices = _build_focal_class_indices(train_meta, label_map)
    taxa_indices = _build_taxa_indices(label_map, taxonomy_path)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    base_model_config = {"dropout": 0.35, "sample_rate": 32000, "duration": 5.0}

    selected_exps = _select_experiments(args.exp)
    _print_main(rank, "=== focal-train sweep plan ===")
    _print_main(rank, f"  experiments        : {len(selected_exps)}  (--exp filter: {args.exp or '<all>'})")
    _print_main(rank, f"  train rows (focal) : {len(train_meta)}")
    _print_main(rank, f"  val rows (sscape)  : {len(val_meta)}")
    _print_main(rank, f"  focal-trained cls  : {len(focal_class_indices)}  / {len(label_map)}")
    _print_main(rank, f"  preload_train      : {args.preload_train_audio}")
    _print_main(rank, f"  preload_val        : {args.preload_val_audio}")
    _print_main(rank, f"  output root        : {output_dir}")
    _print_main(rank, f"  per-run dir        : {{exp_name}}/")
    _print_main(rank, f"  summary CSV        : {output_dir / 'experiments_summary.csv'}")
    _print_main(rank, "")

    # Build the train + val datasets ONCE and reuse across all experiments —
    # preloading 35k focal recordings per experiment would burn hours of
    # disk-read time. Sampler / loader / loss / optimizer still rebuild
    # per-config since they depend on experiment hyperparameters.
    train_rank_meta = build_rank_metadata_shard(train_meta, world_size=world_size, rank=rank)
    _print_main(rank, f"[setup] building train dataset (rank shard: {len(train_rank_meta)} rows)...")
    train_dataset = BirdCLEFDataset(
        metadata=train_rank_meta,
        label_map=label_map,
        sample_rate=int(base_model_config["sample_rate"]),
        duration=float(base_model_config["duration"]),
        training=True,
        preload_audio=bool(args.preload_train_audio),
        preload_workers=int(args.preload_workers),
    )
    val_dataset = None
    if _is_main_process(rank):
        _print_main(rank, f"[setup] building val dataset ({len(val_meta)} rows)...")
        val_dataset = BirdCLEFDataset(
            metadata=val_meta, label_map=label_map,
            sample_rate=int(base_model_config["sample_rate"]),
            duration=float(base_model_config["duration"]),
            training=False, preload_audio=bool(args.preload_val_audio),
            preload_workers=int(args.preload_workers),
        )
    if dist.is_initialized():
        dist.barrier()

    all_summaries: List[Dict[str, Any]] = []
    for experiment in selected_exps:
        try:
            summary = run_one_experiment(
                experiment=experiment,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_rank_meta=train_rank_meta,
                val_meta=val_meta,
                label_map=label_map,
                focal_class_indices=focal_class_indices,
                taxa_indices=taxa_indices,
                base_model_config=base_model_config,
                output_dir=output_dir, device=device,
                rank=rank, local_rank=local_rank, distributed=distributed,
                use_bf16=args.use_bf16,
                train_meta_len=len(train_meta),
                train_rank_meta_len=len(train_rank_meta),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                epochs=int(args.epochs),
                grad_clip_norm=float(args.grad_clip_norm),
            )
        except Exception as ex:
            if _is_main_process(rank):
                failure = {
                    "experiment": experiment.get("name", "unknown"),
                    "status": "failed", "error": str(ex),
                }
                all_summaries.append(failure)
                print(f"[WARN] {failure['experiment']} FAILED: {ex}")
            if dist.is_initialized():
                dist.barrier()
            continue

        if _is_main_process(rank):
            lean = {k: v for k, v in summary.items() if k != "history"}
            all_summaries.append(lean)
            with open(output_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, sort_keys=True, default=str)
            df = pd.DataFrame(all_summaries)
            if "primary" in df.columns:
                df = df.sort_values(by=["primary"], ascending=[False],
                                    na_position="last")
            df.to_csv(output_dir / "experiments_summary.csv", index=False)

    _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
