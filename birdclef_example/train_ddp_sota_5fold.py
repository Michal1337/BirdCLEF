"""SOTA timm-spectrogram trainer — file-level 5/10-fold variant.

This is `train_ddp_sota.py` with one surgical change:

  The leaky GroupShuffleSplit-by-filename for the train/val cut is replaced
  by `birdclef.data.splits.load_folds(n_splits)` — file-level StratifiedKFold
  by primary species over all 59 fully-labeled soundscapes (no V-anchor).
  Per-fold val is the only validation signal; the cross-fold stitched OOF
  metric is computed at the end of an `--all-folds` run and appended to the
  experiments_summary.csv as the new ranking column.

Everything else (architecture, loss, augmentations, optimizer, sweep grid)
is imported from `train_ddp_sota.py` verbatim. If a hyperparameter changes
upstream, this file picks it up.

V-anchor was dropped after the 0.747 SED LB result confirmed it didn't
predict LB — see plan file. Selection metric per fold = `val_mapped_auc`.
For ranking ACROSS configs, use the appended stitched-OOF column.

Single-(exp, fold) invocation pattern matches `birdclef.scripts._03_train_sed`:

    torchrun --standalone --nproc_per_node=2 \\
        -m birdclef_example.train_ddp_sota_5fold \\
        --exp sota11_nfnet_lr8e-04_1e-04 --fold 0

Wrapping for all 5 folds (the deep-dive after a no-arg sweep finds a winner):

    torchrun --standalone --nproc_per_node=2 \\
        -m birdclef_example.train_ddp_sota_5fold \\
        --exp sota11_nfnet_lr8e-04_1e-04 --all-folds

Outputs:
    birdclef_example/outputs/sota_5fold/<exp_name>/fold{f}/
        best_model.pt      # selected on per-fold val_mapped_auc
        label_map.json
        metrics.csv        # per-epoch history (val_* columns only)
        summary.json
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

# Reuse every primitive from the existing SOTA trainer — keeps this file
# small and means upstream tweaks (loss, augs, scheduler, sweep grid) flow
# through automatically.
from birdclef_example.data import (
    BirdCLEFDataset,
    build_label_map,
    prepare_soundscape_metadata,
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
    build_perch_mapped_label_indices,
    build_rank_metadata_shard,
    build_sample_weights,
    build_scheduler,
    evaluate_validation_metrics,
    train_one_epoch_ddp,
)
from birdclef_example.utils import birdclef_roc_auc, is_better_score, save_model, set_seed  # noqa: E402

# New-package splits — the whole point of this file.
from birdclef.data.splits import load_folds


def _select_experiments(name_filter: Optional[str]) -> List[Dict[str, Any]]:
    all_exp = build_experiment_configs()
    if not name_filter:
        return all_exp
    matches = [e for e in all_exp if e["name"] == name_filter]
    if not matches:
        # Fall back to substring match for ergonomic CLI use
        sub = [e for e in all_exp if name_filter in e["name"]]
        if not sub:
            available = ", ".join(e["name"] for e in all_exp[:5])
            raise SystemExit(
                f"No experiment matched --exp '{name_filter}'. "
                f"First 5 available: {available} ..."
            )
        return sub
    return matches


def _split_meta_by_fold(
    soundscape_meta: pd.DataFrame,
    folds_df: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve soundscape_meta into (train, val) row sets.

    train = labeled rows where folds[filename] != fold
    val   = labeled rows where folds[filename] == fold

    Files not in folds_df are dropped (unlabeled / not fully labeled).
    With V-anchor abandoned, every labeled file is in exactly one fold's val.
    """
    fold_of = dict(zip(folds_df["filename"].astype(str), folds_df["fold"].astype(int)))

    fnames = soundscape_meta["filename"].astype(str)
    fold_assign = fnames.map(fold_of)
    in_split = fold_assign.notna()

    train_mask = in_split & (fold_assign != fold)
    val_mask = in_split & (fold_assign == fold)

    train = soundscape_meta[train_mask].reset_index(drop=True)
    val = soundscape_meta[val_mask].reset_index(drop=True)
    return train, val


def run_one_fold(
    experiment: Dict[str, Any],
    fold: int,
    n_splits: int,
    train_rank_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    label_map: Dict[str, int],
    mapped_label_indices: np.ndarray,
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
    preload_audio: bool,
    preload_workers: int,
) -> Dict[str, Any]:
    """One (experiment, fold) training run.

    Selects the best checkpoint by per-fold `val_mapped_auc`. (V-anchor was
    abandoned — the cross-fold stitched-OOF metric is computed separately
    after all folds finish, not here.)
    """
    exp_name = experiment["name"]
    exp_output_dir = output_dir / exp_name / f"fold{fold}"
    if _is_main_process(rank):
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BirdCLEFDataset(
        metadata=train_rank_meta,
        label_map=label_map,
        sample_rate=int(base_model_config["sample_rate"]),
        duration=float(base_model_config["duration"]),
        training=True,
        preload_audio=preload_audio,
        preload_workers=preload_workers,
    )

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
    val_dataset = None
    if _is_main_process(rank):
        val_dataset = BirdCLEFDataset(
            metadata=val_meta, label_map=label_map,
            sample_rate=int(base_model_config["sample_rate"]),
            duration=float(base_model_config["duration"]),
            training=False, preload_audio=preload_audio,
            preload_workers=preload_workers,
        )
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
    _print_main(rank, f"=== {exp_name} | fold {fold}/{n_splits-1} ===")
    _print_main(rank, f"Backbone: {experiment['backbone_name']} | image_size={experiment['image_size']}")
    _print_main(rank, f"DDP enabled: {distributed} (world_size="
                      f"{dist.get_world_size() if dist.is_initialized() else 1})")
    _print_main(rank, f"Train rows: {train_meta_len}  (rank shard: {train_rank_meta_len})")
    _print_main(rank, f"Val rows (fold {fold}): {len(val_meta)}")
    _print_main(rank, f"Num labels: {len(label_map)}  Mapped val labels: {len(mapped_label_indices)}")

    best_val_auc = float("nan")
    best_val_full_auc = float("nan")
    best_val_loss = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        avg_train_loss = train_one_epoch_ddp(
            model=model, train_loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch_label=f"{exp_name}/f{fold} | ep {epoch}",
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
            val_loss, val_mapped_auc, val_full_auc = evaluate_validation_metrics(
                eval_model, val_loader, torch.nn.BCEWithLogitsLoss(),
                device, mapped_label_indices=mapped_label_indices, use_bf16=use_bf16,
            )

            row = {
                "experiment": exp_name, "fold": fold, "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss, "val_mapped_auc": val_mapped_auc,
                "val_full_auc": val_full_auc,
            }
            history.append(row)
            print(
                f"{exp_name}/f{fold} | ep {epoch:02d} | "
                f"train={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} val_auc={val_mapped_auc:.4f} val_full={val_full_auc:.4f}"
            )

            should_save = False
            if is_better_score(val_mapped_auc, best_val_auc):
                best_val_auc = val_mapped_auc
                best_val_full_auc = val_full_auc
                best_val_loss = val_loss
                should_save = True
            elif pd.isna(val_mapped_auc) and val_loss < best_val_loss:
                # Pathological case: val mapped AUC undefined. Fall back to
                # val_loss to break ties.
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
                        "fold": fold,
                        "n_splits": n_splits,
                        "best_val_mapped_auc": val_mapped_auc,
                        "best_val_full_auc": val_full_auc,
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
        "experiment": exp_name, "fold": fold, "n_splits": n_splits,
        "architecture": "sota_timm_spectrogram_5fold",
        "backbone_name": experiment["backbone_name"],
        "best_epoch": best_epoch,
        "best_val_mapped_auc": best_val_auc,
        "best_val_full_auc": best_val_full_auc,
        "best_val_loss": best_val_loss,
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
    del train_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return summary


def _cross_fold_oof_for_exp(
    exp_name: str,
    fold_list: List[int],
    output_dir: Path,
    soundscape_meta: pd.DataFrame,
    folds_df: pd.DataFrame,
    label_map: Dict[str, int],
    mapped_label_indices: np.ndarray,
    base_model_config: Dict[str, Any],
    device: torch.device,
    use_bf16: bool,
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    """Cross-fold OOF metrics for one experiment.

    Loads each fold's `best_model.pt`, predicts on its val rows, and
    reports BOTH:
      - mean_oof_mapped_auc / mean_oof_full_auc (mean of per-fold AUCs) —
        the LB-proxy primary metric (each fold = one deployed-style model
        on one test set, no cross-fold calibration mixing).
      - stitched_oof_mapped_auc / stitched_oof_full_auc (single AUC over
        concatenated val predictions) — diagnostic; large mean-vs-stitched
        gap signals fold-local calibration drift.
    Only computed when --all-folds was passed.
    """
    if len(fold_list) <= 1:
        return {}
    exp_dir = output_dir / exp_name
    targets_chunks: List[np.ndarray] = []
    preds_chunks: List[np.ndarray] = []
    fold_mapped: List[float] = []
    fold_full: List[float] = []

    for fold in fold_list:
        ckpt = exp_dir / f"fold{fold}" / "best_model.pt"
        if not ckpt.exists():
            print(f"[oof] WARN: missing {ckpt}; skipping fold {fold}")
            continue
        state = torch.load(ckpt, map_location="cpu")
        model_config = state["model_config"]
        model = TimmSpectrogramClassifier(**model_config).to(device)
        sd = state.get("model_state") or state.get("state_dict")
        model.load_state_dict(sd, strict=False)
        model.eval()

        _train, val_meta = _split_meta_by_fold(soundscape_meta, folds_df, fold)
        if val_meta.empty:
            continue
        val_dataset = BirdCLEFDataset(
            metadata=val_meta, label_map=label_map,
            sample_rate=int(base_model_config["sample_rate"]),
            duration=float(base_model_config["duration"]),
            training=False, preload_audio=True, preload_workers=8,
        )
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
        ft_chunks: List[np.ndarray] = []
        fp_chunks: List[np.ndarray] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16 and device.type == "cuda"):
                    logits = model(inputs)
                p = torch.sigmoid(logits).float().cpu().numpy()
                t = targets.float().cpu().numpy()
                fp_chunks.append(p); ft_chunks.append(t)
                preds_chunks.append(p); targets_chunks.append(t)

        if ft_chunks:
            Yf = np.concatenate(ft_chunks, axis=0)
            Pf = np.concatenate(fp_chunks, axis=0)
            try:
                fold_full.append(float(birdclef_roc_auc(Yf, Pf)))
            except ValueError:
                fold_full.append(float("nan"))
            if len(mapped_label_indices) > 0:
                try:
                    fold_mapped.append(float(birdclef_roc_auc(
                        Yf[:, mapped_label_indices], Pf[:, mapped_label_indices])))
                except ValueError:
                    fold_mapped.append(float("nan"))
            else:
                fold_mapped.append(float("nan"))
            print(f"[oof]   fold {fold}: mapped={fold_mapped[-1]:.4f}  full={fold_full[-1]:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not targets_chunks:
        return {}
    Y = np.concatenate(targets_chunks, axis=0)
    P = np.concatenate(preds_chunks, axis=0)
    try:
        stitched_full = float(birdclef_roc_auc(Y, P))
    except ValueError:
        stitched_full = float("nan")
    if len(mapped_label_indices) > 0:
        try:
            stitched_mapped = float(birdclef_roc_auc(
                Y[:, mapped_label_indices], P[:, mapped_label_indices]))
        except ValueError:
            stitched_mapped = float("nan")
    else:
        stitched_mapped = float("nan")

    valid_mapped = [a for a in fold_mapped if not np.isnan(a)]
    valid_full = [a for a in fold_full if not np.isnan(a)]
    mean_mapped = float(np.mean(valid_mapped)) if valid_mapped else float("nan")
    mean_full = float(np.mean(valid_full)) if valid_full else float("nan")

    return {
        # Primary ranker: mean of per-fold AUCs (LB-shape proxy).
        "mean_oof_mapped_auc": mean_mapped,
        "mean_oof_full_auc": mean_full,
        # Diagnostic: stitched AUC. Large `mean - stitched` gap (>0.02) =
        # fold-local calibration drift; deployed model wouldn't have it.
        "stitched_oof_mapped_auc": stitched_mapped,
        "stitched_oof_full_auc": stitched_full,
        "calibration_drift_mapped": mean_mapped - stitched_mapped if not (
            np.isnan(mean_mapped) or np.isnan(stitched_mapped)) else float("nan"),
        "stitched_oof_n_rows": int(Y.shape[0]),
        "n_folds_seen": int(len(valid_mapped)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None,
                    help="Experiment name from build_experiment_configs(). "
                         "Substring match supported. Default: sweep all 23 configs.")
    ap.add_argument("--fold", type=int, default=0,
                    help="Single fold (0..n_splits-1). Default: 0. Use --all-folds to loop. "
                         "A no-arg run = all 23 experiments × fold 0 (= same compute "
                         "as old leaky-split sweep). After a winner is picked, re-run "
                         "with --exp <winner> --all-folds for the deep dive.")
    ap.add_argument("--all-folds", action="store_true",
                    help="Loop fold 0..n_splits-1 (overrides --fold). Triggers stitched-OOF "
                         "metric computation at the end. Use only for the winning config.")
    ap.add_argument("--n-splits", type=int, default=5, choices=[5, 10],
                    help="Static fold parquet to use. Default 5. "
                         "Build them via `python -m birdclef.scripts._02_build_splits`.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--preload-workers", type=int, default=8)
    ap.add_argument("--preload-audio", action="store_true", default=True,
                    help="Preload all OGGs into RAM before training (default on; "
                         "needs ~30 GB RAM for the full labeled-soundscape pool).")
    ap.add_argument("--no-preload-audio", dest="preload_audio", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-bf16", action="store_true", default=True)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir. Default: birdclef_example/outputs/sota_5fold")
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
        repo_root / "birdclef_example" / "outputs" / "sota_5fold"

    # Hard guard against accidentally clobbering the old leaky-split sweep
    # results. The legacy script writes to `experiments_sota/`; the
    # vanchor-era script wrote to `sota_5fold_vanchor/`. Refuse if pointed
    # at either.
    legacy_roots = {
        (repo_root / "birdclef_example" / "outputs" / "experiments_sota").resolve(),
        (repo_root / "birdclef_example" / "outputs" / "sota_5fold_vanchor").resolve(),
    }
    if output_dir.resolve() in legacy_roots:
        raise SystemExit(
            f"--out-dir points at a legacy SOTA sweep directory:\n  {output_dir}\n"
            "Refusing to overwrite. Pick a different --out-dir (default is "
            "birdclef_example/outputs/sota_5fold)."
        )

    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    taxonomy_path = data_dir / "taxonomy.csv"
    train_soundscape_dir = data_dir / "train_soundscapes"
    if not train_soundscape_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_soundscape_dir}")

    set_seed(int(args.seed) + rank)
    if _is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    soundscape_df = pd.read_csv(soundscape_labels_path)
    soundscape_meta = prepare_soundscape_metadata(soundscape_df, train_soundscape_dir)

    folds_df = load_folds(n_splits=int(args.n_splits))
    if folds_df.empty:
        raise SystemExit(
            f"Folds parquet is empty. Run `python -m birdclef.scripts._02_build_splits` "
            f"to build the {args.n_splits}-fold split first."
        )
    n_folds = int(folds_df["fold"].max()) + 1

    label_source = soundscape_meta.copy()
    label_map = build_label_map(label_source, taxonomy_path)
    expected_num_classes = 234
    if len(label_map) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes from taxonomy, got {len(label_map)}."
        )

    perch_labels_path = repo_root / "models" / "perch_v2_cpu" / "1" / "assets" / "labels.csv"
    if not perch_labels_path.exists():
        raise FileNotFoundError(f"Missing Perch labels file: {perch_labels_path}")
    mapped_label_indices = build_perch_mapped_label_indices(taxonomy_path, perch_labels_path, label_map)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    base_model_config = {"dropout": 0.35, "sample_rate": 32000, "duration": 5.0}

    selected_exps = _select_experiments(args.exp)
    fold_list = list(range(n_folds)) if args.all_folds else [int(args.fold)]
    _print_main(rank, f"=== sweep plan ===")
    _print_main(rank, f"  experiments     : {len(selected_exps)}  (--exp filter: {args.exp or '<all>'})")
    _print_main(rank, f"  n_splits        : {args.n_splits}")
    _print_main(rank, f"  folds per exp   : {len(fold_list)}  (folds={fold_list})")
    _print_main(rank, f"  total runs      : {len(selected_exps) * len(fold_list)}")
    _print_main(rank, f"  output root     : {output_dir}")
    _print_main(rank, f"  per-run dir     : {{exp_name}}/fold{{f}}/")
    _print_main(rank, f"  summary CSV     : {output_dir / 'experiments_summary.csv'}")
    _print_main(rank, "")

    all_summaries: List[Dict[str, Any]] = []
    for experiment in selected_exps:
        for fold in fold_list:
            train_meta, val_meta = _split_meta_by_fold(soundscape_meta, folds_df, fold)
            train_rank_meta = build_rank_metadata_shard(train_meta, world_size=world_size, rank=rank)
            try:
                summary = run_one_fold(
                    experiment=experiment, fold=fold, n_splits=int(args.n_splits),
                    train_rank_meta=train_rank_meta,
                    val_meta=val_meta,
                    label_map=label_map,
                    mapped_label_indices=mapped_label_indices,
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
                    preload_audio=bool(args.preload_audio),
                    preload_workers=int(args.preload_workers),
                )
            except Exception as ex:
                if _is_main_process(rank):
                    failure = {
                        "experiment": experiment.get("name", "unknown"),
                        "fold": fold, "status": "failed", "error": str(ex),
                    }
                    all_summaries.append(failure)
                    print(f"[WARN] {failure['experiment']}/f{fold} FAILED: {ex}")
                if dist.is_initialized():
                    dist.barrier()
                continue

            if _is_main_process(rank):
                lean = {k: v for k, v in summary.items() if k != "history"}
                all_summaries.append(lean)
                # Persist the running summary after every fold so a crash
                # never loses prior runs' results.
                with open(output_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
                    json.dump(all_summaries, f, indent=2, sort_keys=True, default=str)
                df = pd.DataFrame(all_summaries)
                if "best_val_mapped_auc" in df.columns:
                    df = df.sort_values(by=["best_val_mapped_auc"], ascending=[False])
                df.to_csv(output_dir / "experiments_summary.csv", index=False)

        # Cross-fold OOF (only meaningful when all folds of this experiment
        # completed in this run — i.e. --all-folds was passed and rank 0).
        # Reports BOTH mean of per-fold AUCs (primary, LB-shape proxy) AND
        # stitched AUC (diagnostic — large mean-vs-stitched gap = fold-local
        # calibration drift).
        if args.all_folds and _is_main_process(rank):
            try:
                oof = _cross_fold_oof_for_exp(
                    exp_name=experiment["name"],
                    fold_list=fold_list,
                    output_dir=output_dir,
                    soundscape_meta=soundscape_meta,
                    folds_df=folds_df,
                    label_map=label_map,
                    mapped_label_indices=mapped_label_indices,
                    base_model_config=base_model_config,
                    device=device,
                    use_bf16=args.use_bf16,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
            except Exception as ex:
                print(f"[WARN] cross-fold OOF for {experiment['name']} FAILED: {ex}")
                oof = {}
            if oof:
                # Append a single cross-fold OOF row per experiment to the
                # summary CSV / JSON so it ranks alongside per-fold entries.
                all_summaries.append({"_kind": "cross_fold_oof",
                                      "experiment": experiment["name"], **oof})
                with open(output_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
                    json.dump(all_summaries, f, indent=2, sort_keys=True, default=str)
                df = pd.DataFrame(all_summaries)
                if "mean_oof_mapped_auc" in df.columns:
                    df = df.sort_values(by=["mean_oof_mapped_auc"],
                                        ascending=[False], na_position="last")
                df.to_csv(output_dir / "experiments_summary.csv", index=False)
                drift = oof.get("calibration_drift_mapped", float("nan"))
                drift_flag = "OK" if (not np.isnan(drift) and abs(drift) < 0.02) else "investigate"
                print(f"[oof] {experiment['name']}  "
                      f"mean_mapped={oof.get('mean_oof_mapped_auc'):.4f}  "
                      f"stitched_mapped={oof.get('stitched_oof_mapped_auc'):.4f}  "
                      f"drift={drift:+.4f} ({drift_flag})  "
                      f"rows={oof.get('stitched_oof_n_rows')}")

    _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
