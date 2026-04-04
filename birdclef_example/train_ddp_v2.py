import json
import inspect
import os
import math
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.distributed as dist
from sklearn.model_selection import GroupKFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from birdclef_example.data import (
    BirdCLEFDataset,
    SoundscapeSampler,
    build_label_map,
    parse_primary_labels,
    parse_secondary_labels,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)
from birdclef_example.model import SimpleCNN
from birdclef_example.utils import (
    birdclef_roc_auc,
    evaluate_model,
    is_better_score,
    save_model,
    set_seed,
)


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


def _sync(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def _clear_dataset_cache(dataset: Optional[BirdCLEFDataset]) -> None:
    if dataset is None:
        return
    cache = getattr(dataset, "_audio_cache", None)
    if cache is not None:
        cache.clear()


def build_rank_metadata_shard(
    metadata: pd.DataFrame,
    world_size: int,
    rank: int,
) -> pd.DataFrame:
    if world_size <= 1:
        return metadata.reset_index(drop=True)
    if metadata.empty:
        return metadata.copy()

    if "audio_filepath" not in metadata.columns:
        # Fallback for unexpected metadata schema.
        shard_size = (len(metadata) + world_size - 1) // world_size
        start = rank * shard_size
        end = min(start + shard_size, len(metadata))
        shard = metadata.iloc[start:end].copy()
        missing = shard_size - len(shard)
        if missing > 0:
            shard = pd.concat([shard, metadata.iloc[:missing].copy()], ignore_index=True)
        return shard.reset_index(drop=True)

    # Balance by audio files first (important for preload cost), while keeping row counts close.
    file_groups = (
        metadata.groupby("audio_filepath", sort=False)
        .indices
    )
    file_items = [
        (str(filepath), np.array(indices, dtype=np.int64))
        for filepath, indices in file_groups.items()
    ]
    file_items.sort(key=lambda item: len(item[1]), reverse=True)

    rank_indices: list[list[int]] = [[] for _ in range(world_size)]
    rank_row_counts = [0 for _ in range(world_size)]
    rank_file_counts = [0 for _ in range(world_size)]

    for _, indices in file_items:
        target_rank = min(
            range(world_size),
            key=lambda r: (rank_row_counts[r], rank_file_counts[r]),
        )
        rank_indices[target_rank].extend(indices.tolist())
        rank_row_counts[target_rank] += int(len(indices))
        rank_file_counts[target_rank] += 1

    shard_indices = rank_indices[rank]
    if not shard_indices:
        return metadata.iloc[:0].copy().reset_index(drop=True)

    shard = metadata.iloc[np.array(shard_indices, dtype=np.int64)].copy()
    return shard.reset_index(drop=True)


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


class BCEFocalLoss(torch.nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.focal_alpha = max(0.0, min(1.0, focal_alpha))
        self.focal_gamma = max(0.0, focal_gamma)
        self.bce_weight = max(0.0, bce_weight)
        self.focal_weight = max(0.0, focal_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal_term = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0), self.focal_gamma)
        focal_loss = focal_term * bce
        return self.bce_weight * bce.mean() + self.focal_weight * focal_loss.mean()


def _time_to_seconds(value: object) -> float:
    if isinstance(value, str):
        parts = [float(part) for part in value.split(":")]
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(value)


def row_id_from_metadata(filename: str, end_value: object) -> str:
    end_seconds = _time_to_seconds(end_value)
    end_token = "nan" if math.isnan(end_seconds) else str(int(round(end_seconds)))
    return f"{Path(filename).stem}_{end_token}"


def _extract_row_label_indices(row: pd.Series, label_map: dict[str, int]) -> list[int]:
    indices: list[int] = []
    labels = parse_primary_labels(row.get("primary_label"))
    if str(row.get("source", "")) == "train_audio":
        labels.extend(parse_secondary_labels(row.get("secondary_labels")))
    for label in labels:
        idx = label_map.get(label)
        if idx is not None:
            indices.append(idx)
    return list(dict.fromkeys(indices))


def compute_sample_weights(metadata: pd.DataFrame, label_map: dict[str, int]) -> torch.Tensor:
    if metadata.empty:
        return torch.tensor([], dtype=torch.double)

    class_counts = np.zeros(len(label_map), dtype=np.float64)
    sample_label_indices: list[list[int]] = []

    for _, row in metadata.iterrows():
        label_indices = _extract_row_label_indices(row, label_map)
        sample_label_indices.append(label_indices)
        for idx in label_indices:
            class_counts[idx] += 1.0

    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / np.sqrt(class_counts)

    sample_weights = np.ones(len(metadata), dtype=np.float64)
    for i, label_indices in enumerate(sample_label_indices):
        if label_indices:
            sample_weights[i] = float(np.mean(class_weights[label_indices]))

    sample_weights = sample_weights / np.mean(sample_weights)
    return torch.tensor(sample_weights, dtype=torch.double)


def train_one_epoch_ddp(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float,
    epoch_label: str,
    rank: int,
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
        batch_count = batch_idx
        if _is_main_process(rank):
            progress.set_postfix({"batch_loss": running_loss / batch_idx})

    local_avg = running_loss / max(batch_count, 1)
    loss_tensor = torch.tensor(local_avg, device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor = loss_tensor / dist.get_world_size()
    return float(loss_tensor.item())


def predict_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> np.ndarray:
    model.eval()
    amp_enabled = use_bf16 and device.type == "cuda"
    all_preds: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(inputs)
            all_preds.append(torch.sigmoid(logits).float().cpu())

    if not all_preds:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(all_preds, dim=0).numpy()


def _seconds_to_hms(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _load_unlabeled_segments(
    metadata_path: Optional[Path],
    soundscape_dir: Path,
    segment_duration: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if metadata_path is not None and metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "row_id" in metadata.columns:
            stem_to_filename: dict[str, str] = {}
            for path in sorted(soundscape_dir.glob("*")):
                if path.is_file() and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3"}:
                    stem_to_filename[path.stem] = path.name

            for row_id in metadata["row_id"].dropna().astype(str).tolist():
                if "_" not in row_id:
                    continue
                stem, end_token = row_id.rsplit("_", 1)
                if stem not in stem_to_filename:
                    continue
                try:
                    end_seconds = int(float(end_token))
                except ValueError:
                    continue
                start_seconds = max(0, end_seconds - int(round(segment_duration)))
                rows.append(
                    {
                        "row_id": row_id,
                        "filename": stem_to_filename[stem],
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                    }
                )

    if not rows:
        # Fallback: build pseudo-label targets from all fixed windows in soundscape files.
        hop_seconds = int(round(segment_duration))
        for path in sorted(soundscape_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".ogg", ".wav", ".flac", ".mp3"}:
                continue
            info = sf.info(str(path))
            if info.samplerate <= 0:
                continue
            duration_seconds = info.frames / float(info.samplerate)
            num_segments = max(1, int(math.ceil(duration_seconds / segment_duration)))
            for seg_idx in range(num_segments):
                end_seconds = (seg_idx + 1) * hop_seconds
                start_seconds = max(0, end_seconds - hop_seconds)
                rows.append(
                    {
                        "row_id": f"{path.stem}_{end_seconds}",
                        "filename": path.name,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                    }
                )

    if not rows:
        raise RuntimeError(f"No audio segments found for pseudo-labeling in {soundscape_dir}.")
    return pd.DataFrame(rows)


def generate_pseudo_labels_v1(
    fold_model_paths: list[Path],
    model_config: dict,
    label_map: dict[str, int],
    device: torch.device,
    use_bf16: bool,
    unlabeled_metadata_path: Optional[Path],
    unlabeled_soundscape_dir: Path,
    pseudo_conf_threshold: float,
    pseudo_max_labels_per_clip: int,
    pseudo_top1_fallback_threshold: float,
    sample_rate: int,
    segment_duration: float,
) -> pd.DataFrame:
    segments_df = _load_unlabeled_segments(
        metadata_path=unlabeled_metadata_path,
        soundscape_dir=unlabeled_soundscape_dir,
        segment_duration=segment_duration,
    )

    reverse_labels = sorted(label_map.keys(), key=lambda label: label_map[label])
    num_classes = len(reverse_labels)
    row_to_index = {row_id: idx for idx, row_id in enumerate(segments_df["row_id"].tolist())}
    accum = np.zeros((len(segments_df), num_classes), dtype=np.float32)
    counts = np.zeros((len(segments_df), 1), dtype=np.float32)

    grouped = segments_df.groupby("filename", sort=False)
    sampler = SoundscapeSampler(
        sample_rate=sample_rate,
        duration=segment_duration,
        hop=segment_duration,
        preload_audio=False,
        preload_workers=1,
    )
    amp_enabled = use_bf16 and device.type == "cuda"

    for fold_path in fold_model_paths:
        if not fold_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint for pseudo-labeling: {fold_path}")
        checkpoint = torch.load(fold_path, map_location=device)
        model = SimpleCNN(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        with torch.no_grad():
            for filename, group_df in tqdm(grouped, desc=f"Pseudo infer {fold_path.name}", leave=False):
                target_by_end = {
                    int(end_s): row_to_index[row_id]
                    for end_s, row_id in zip(group_df["end_seconds"].tolist(), group_df["row_id"].tolist())
                }
                soundscape_path = unlabeled_soundscape_dir / filename
                if not soundscape_path.exists():
                    continue
                for end_seconds, chunk in sampler.iterate_segment_items(soundscape_path):
                    row_idx = target_by_end.get(int(end_seconds))
                    if row_idx is None:
                        continue
                    batch = chunk.unsqueeze(0).to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        logits = model(batch)
                    probs = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()
                    accum[row_idx] += probs
                    counts[row_idx, 0] += 1.0

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    counts = np.maximum(counts, 1.0)
    avg_probs = accum / counts

    pseudo_rows: list[dict[str, object]] = []
    top1_scores: list[float] = []
    selected_by_main_threshold = 0
    selected_by_top1_fallback = 0
    for idx, row in segments_df.iterrows():
        probs = avg_probs[idx]
        top1_idx = int(np.argmax(probs))
        top1_score = float(probs[top1_idx])
        top1_scores.append(top1_score)
        positive_indices = np.where(probs >= pseudo_conf_threshold)[0].tolist()
        if positive_indices:
            selected_by_main_threshold += 1
        elif top1_score >= pseudo_top1_fallback_threshold:
            positive_indices = [top1_idx]
            selected_by_top1_fallback += 1
        else:
            continue
        positive_indices.sort(key=lambda i: float(probs[i]), reverse=True)
        if pseudo_max_labels_per_clip > 0:
            positive_indices = positive_indices[:pseudo_max_labels_per_clip]
        labels = [reverse_labels[i] for i in positive_indices]
        pseudo_rows.append(
            {
                "row_id": row["row_id"],
                "filename": row["filename"],
                "start": _seconds_to_hms(int(row["start_seconds"])),
                "end": _seconds_to_hms(int(row["end_seconds"])),
                "primary_label": " ".join(labels),
                "secondary_labels": "[]",
                "source": "pseudo_labels_v1",
                "audio_filepath": str(unlabeled_soundscape_dir / row["filename"]),
                "pseudo_confidence_max": float(np.max(probs[positive_indices])),
                "pseudo_num_labels": int(len(labels)),
            }
        )

    if top1_scores:
        quantiles = np.quantile(np.array(top1_scores, dtype=np.float32), [0.5, 0.9, 0.95, 0.99]).tolist()
        print(
            "Pseudo confidence diagnostics | "
            f"top1 median={quantiles[0]:.4f}, p90={quantiles[1]:.4f}, "
            f"p95={quantiles[2]:.4f}, p99={quantiles[3]:.4f}, "
            f"selected_main={selected_by_main_threshold}, "
            f"selected_top1_fallback={selected_by_top1_fallback}, "
            f"total_selected={len(pseudo_rows)}"
        )

    columns = [
        "row_id",
        "filename",
        "start",
        "end",
        "primary_label",
        "secondary_labels",
        "source",
        "audio_filepath",
        "pseudo_confidence_max",
        "pseudo_num_labels",
    ]
    return pd.DataFrame(pseudo_rows, columns=columns)


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
    output_dir = repo_root / "birdclef_example" / "outputs_v2"

    train_audio_metadata_path = data_dir / "train.csv"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    taxonomy_path = data_dir / "taxonomy.csv"
    train_audio_dir = data_dir / "train_audio"
    train_soundscape_dir = data_dir / "train_soundscapes"
    test_soundscape_dir = data_dir / "test_soundscapes"
    sample_submission_path = data_dir / "sample_submission.csv"

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
    seed = 42
    num_folds = 5
    preload_audio = True
    preload_workers = 8
    num_workers = 4
    use_bf16 = True
    grad_clip_norm = 1.0
    balanced_sampling = True

    focal_alpha = 0.25
    focal_gamma = 2.0
    bce_weight = 1.0
    focal_weight = 1.0

    waveform_aug_prob = 0.9
    gain_prob = 0.6
    gain_db_limit = 6.0
    noise_prob = 0.5
    noise_snr_db_min = 8.0
    noise_snr_db_max = 30.0
    time_shift_prob = 0.7
    time_shift_max_frac = 0.2
    drop_segment_prob = 0.2
    drop_segment_max_frac = 0.15

    pseudo_labeling_v2_enabled = True
    pseudo_labels_path = output_dir / "pseudo_labels_v2.csv"
    # Default to train_soundscapes because many local setups don't include test soundscapes files.
    pseudo_unlabeled_soundscape_dir = train_soundscape_dir
    pseudo_unlabeled_metadata_path: Optional[Path] = None
    if test_soundscape_dir.exists() and any(test_soundscape_dir.glob("*.ogg")):
        pseudo_unlabeled_soundscape_dir = test_soundscape_dir
        if sample_submission_path.exists():
            pseudo_unlabeled_metadata_path = sample_submission_path
    pseudo_conf_threshold = 0.88
    pseudo_max_labels_per_clip = 2
    pseudo_top1_fallback_threshold = 0.80
    pseudo_mix_ratio = 0.25
    pseudo_confidence_weighted_sampling = True

    resume: Optional[Path] = None

    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2 for GroupKFold, got {num_folds}")

    if not train_audio_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_audio_dir}")
    if not train_soundscape_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_soundscape_dir}")

    if preload_audio and num_workers > 0:
        _print_main(
            rank,
            f"preload_audio=True: setting DataLoader num_workers={num_workers} -> 0 to avoid worker cache duplication/crashes.",
        )
        num_workers = 0

    set_seed(seed + rank)
    if _is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    train_audio_df = pd.read_csv(train_audio_metadata_path)
    soundscape_df = pd.read_csv(soundscape_labels_path)

    train_audio_meta = prepare_train_audio_metadata(train_audio_df, train_audio_dir)
    soundscape_meta = prepare_soundscape_metadata(soundscape_df, train_soundscape_dir)

    pseudo_meta = pd.DataFrame()
    pseudo_labels_ready = False
    if pseudo_labeling_v2_enabled and pseudo_labels_path.exists():
        try:
            loaded_pseudo = pd.read_csv(pseudo_labels_path)
        except pd.errors.EmptyDataError:
            loaded_pseudo = pd.DataFrame()

        if not loaded_pseudo.empty:
            required_pseudo_cols = {"audio_filepath", "primary_label"}
            missing_pseudo = required_pseudo_cols.difference(loaded_pseudo.columns)
            if missing_pseudo:
                raise ValueError(
                    f"Pseudo labels file missing columns {sorted(missing_pseudo)}: {pseudo_labels_path}"
                )
            pseudo_meta = loaded_pseudo.copy()
            pseudo_meta["source"] = "pseudo_labels_v1"
            if "secondary_labels" not in pseudo_meta.columns:
                pseudo_meta["secondary_labels"] = "[]"
            pseudo_meta["secondary_labels"] = pseudo_meta["secondary_labels"].fillna("[]").astype(str)
            if "filename" not in pseudo_meta.columns:
                pseudo_meta["filename"] = pseudo_meta["audio_filepath"].map(
                    lambda p: Path(str(p)).name
                )
            pseudo_labels_ready = True

    label_source = pd.concat([train_audio_meta, soundscape_meta], ignore_index=True)
    label_map = build_label_map(label_source, taxonomy_path)
    expected_num_classes = 234
    if len(label_map) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes from taxonomy, got {len(label_map)}."
        )

    fold_generator = GroupKFold(n_splits=num_folds)
    fold_splits = list(
        fold_generator.split(
            soundscape_meta,
            groups=soundscape_meta["filename"].astype(str),
        )
    )

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
    amp_enabled = use_bf16 and device.type == "cuda"

    _print_main(rank, f"DDP enabled: {distributed} (world_size={world_size})")
    _print_main(rank, f"Num labels: {len(label_map)}")
    _print_main(rank, f"GroupKFold folds: {num_folds}")
    _print_main(rank, f"Balanced sampler: {balanced_sampling}")
    _print_main(
        rank,
        "Loss: BCE + Focal "
        f"(alpha={focal_alpha}, gamma={focal_gamma}, "
        f"bce_weight={bce_weight}, focal_weight={focal_weight})",
    )
    _print_main(
        rank,
        "Waveform augs: "
        f"aug_prob={waveform_aug_prob}, gain={gain_prob}, noise={noise_prob}, "
        f"shift={time_shift_prob}, drop={drop_segment_prob}",
    )
    _print_main(rank, f"Pseudo-label v2 enabled: {pseudo_labeling_v2_enabled}")
    _print_main(rank, f"Pseudo labels path: {pseudo_labels_path}")
    _print_main(rank, f"Pseudo labels loaded rows: {len(pseudo_meta)}")
    _print_main(rank, f"Pseudo mix ratio: {pseudo_mix_ratio:.2f}")
    _print_main(rank, f"Pseudo thresholds: multi={pseudo_conf_threshold:.2f}, top1_fallback={pseudo_top1_fallback_threshold:.2f}")
    _print_main(rank, f"Pseudo weighted sampling: {pseudo_confidence_weighted_sampling}")
    _print_main(rank, f"Pseudo source dir: {pseudo_unlabeled_soundscape_dir}")
    _print_main(rank, f"Pseudo source metadata: {pseudo_unlabeled_metadata_path}")

    existing_fold_model_paths = [output_dir / f"best_model_fold{i}.pt" for i in range(1, num_folds + 1)]
    if not all(path.exists() for path in existing_fold_model_paths):
        fallback_dir = repo_root / "birdclef_example" / "outputs"
        fallback_paths = [fallback_dir / f"best_model_fold{i}.pt" for i in range(1, num_folds + 1)]
        if all(path.exists() for path in fallback_paths):
            existing_fold_model_paths = fallback_paths
            _print_main(rank, f"Using fallback fold checkpoints for pseudo-v2 from {fallback_dir}")
    can_run_pseudo_only = (
        pseudo_labeling_v2_enabled
        and not pseudo_labels_ready
        and all(path.exists() for path in existing_fold_model_paths)
    )
    if can_run_pseudo_only:
        _print_main(
            rank,
            "Found all fold checkpoints and no pseudo CSV yet; "
            "running pseudo-label v2 generation only (skipping fold retraining).",
        )
        # Leave DDP before pseudo generation to avoid barrier timeout on non-main ranks.
        _cleanup_distributed(distributed)
        if not _is_main_process(rank):
            return

        distributed = False
        if _is_main_process(rank):
            pseudo_df = generate_pseudo_labels_v1(
                fold_model_paths=existing_fold_model_paths,
                model_config=model_config,
                label_map=label_map,
                device=device,
                use_bf16=use_bf16,
                unlabeled_metadata_path=pseudo_unlabeled_metadata_path,
                unlabeled_soundscape_dir=pseudo_unlabeled_soundscape_dir,
                pseudo_conf_threshold=pseudo_conf_threshold,
                pseudo_max_labels_per_clip=pseudo_max_labels_per_clip,
                pseudo_top1_fallback_threshold=pseudo_top1_fallback_threshold,
                sample_rate=sample_rate,
                segment_duration=segment_duration,
            )
            pseudo_df.to_csv(pseudo_labels_path, index=False)
            print(f"Pseudo-label v2 generated: {len(pseudo_df)} rows at {pseudo_labels_path}")
        return

    all_oof_preds: list[np.ndarray] = []
    all_oof_targets: list[np.ndarray] = []
    oof_rows: list[pd.DataFrame] = []
    fold_auc_scores: list[float] = []

    for fold_idx, (soundscape_train_idx, soundscape_val_idx) in enumerate(fold_splits, start=1):
        fold_seed = seed + 1000 * fold_idx
        set_seed(fold_seed + rank)

        soundscape_train_meta = soundscape_meta.iloc[soundscape_train_idx].reset_index(drop=True)
        val_meta = soundscape_meta.iloc[soundscape_val_idx].reset_index(drop=True)
        train_meta = pd.concat([train_audio_meta, soundscape_train_meta], ignore_index=True)
        if not pseudo_meta.empty and pseudo_mix_ratio > 0:
            pseudo_pool = pseudo_meta
            # If pseudo labels were generated from train_soundscapes fallback, prevent fold leakage:
            # do not train on pseudo rows that belong to this fold's validation filenames.
            if pseudo_unlabeled_soundscape_dir.resolve() == train_soundscape_dir.resolve():
                val_filenames = set(val_meta["filename"].astype(str).tolist())
                pseudo_pool = pseudo_pool[~pseudo_pool["filename"].astype(str).isin(val_filenames)]

            if pseudo_pool.empty:
                _print_main(rank, f"Fold {fold_idx}: pseudo pool empty after leakage filter.")
            pseudo_sample_size = max(1, int(round(len(train_meta) * pseudo_mix_ratio)))
            pseudo_sample_size = min(pseudo_sample_size, len(pseudo_pool))
            sample_weights = None
            if pseudo_confidence_weighted_sampling and "pseudo_confidence_max" in pseudo_pool.columns:
                conf = pseudo_pool["pseudo_confidence_max"].astype(float).clip(lower=0.0, upper=1.0)
                # Upweight cleaner pseudo labels while preserving diversity.
                sample_weights = (conf ** 2) + 1e-6
            fold_pseudo = pseudo_pool.sample(
                n=pseudo_sample_size,
                random_state=fold_seed,
                replace=False,
                weights=sample_weights,
            ).reset_index(drop=True)
            if not fold_pseudo.empty:
                train_meta = pd.concat([train_meta, fold_pseudo], ignore_index=True)

        train_rank_meta = build_rank_metadata_shard(train_meta, world_size=world_size, rank=rank)

        train_dataset = BirdCLEFDataset(
            metadata=train_rank_meta,
            label_map=label_map,
            sample_rate=sample_rate,
            duration=segment_duration,
            training=True,
            preload_audio=preload_audio,
            preload_workers=preload_workers,
            waveform_aug_prob=waveform_aug_prob,
            gain_prob=gain_prob,
            gain_db_limit=gain_db_limit,
            noise_prob=noise_prob,
            noise_snr_db_min=noise_snr_db_min,
            noise_snr_db_max=noise_snr_db_max,
            time_shift_prob=time_shift_prob,
            time_shift_max_frac=time_shift_max_frac,
            drop_segment_prob=drop_segment_prob,
            drop_segment_max_frac=drop_segment_max_frac,
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

        train_sampler = None
        shuffle = True
        if balanced_sampling:
            sample_weights = compute_sample_weights(train_rank_meta, label_map)
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
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

        model = SimpleCNN(**model_config).to(device)
        if resume is not None and resume.exists():
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint["model_state"])

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        train_criterion = BCEFocalLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            bce_weight=bce_weight,
            focal_weight=focal_weight,
        )
        eval_criterion = torch.nn.BCEWithLogitsLoss()

        optimizer, fused_enabled = build_optimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        )

        _print_main(
            rank,
            f"Fold {fold_idx}/{num_folds}: train_rows={len(train_meta)} "
            f"(rank_rows={len(train_rank_meta)}), val_rows={len(val_meta)}, "
            f"train_batches={len(train_loader)}",
        )
        local_unique_files = int(train_rank_meta["audio_filepath"].nunique())
        if dist.is_initialized():
            unique_tensor = torch.tensor(float(local_unique_files), device=device)
            gathered_unique = [torch.zeros_like(unique_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_unique, unique_tensor)
            if _is_main_process(rank):
                unique_counts = [int(x.item()) for x in gathered_unique]
                _print_main(rank, f"Fold {fold_idx}: unique audio files per rank={unique_counts}")
        else:
            _print_main(rank, f"Fold {fold_idx}: unique audio files per rank={[local_unique_files]}")
        _print_main(rank, f"Fold {fold_idx}: BF16 autocast={amp_enabled}, fused AdamW={fused_enabled}")

        best_val_auc = float("nan")
        best_val_loss = float("inf")
        fold_model_path = output_dir / f"best_model_fold{fold_idx}.pt"

        for epoch in range(1, epochs + 1):
            avg_train_loss = train_one_epoch_ddp(
                model=model,
                train_loader=train_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                device=device,
                amp_enabled=amp_enabled,
                grad_clip_norm=grad_clip_norm,
                epoch_label=f"fold {fold_idx} epoch {epoch}",
                rank=rank,
            )
            scheduler.step()

            _sync(distributed)

            if _is_main_process(rank):
                eval_model = model.module if isinstance(model, DDP) else model
                if val_loader is None:
                    raise RuntimeError("Validation loader is not initialized on main rank.")

                val_loss, val_auc = evaluate_model(
                    eval_model,
                    val_loader,
                    eval_criterion,
                    device,
                    use_bf16=use_bf16,
                )
                print(
                    f"Fold {fold_idx} | Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | "
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
                    save_model(
                        {
                            "model_state": eval_model.state_dict(),
                            "label_map": label_map,
                            "model_config": model_config,
                            "fold": fold_idx,
                            "epoch": epoch,
                            "best_val_auc": val_auc,
                            "best_val_loss": val_loss,
                            "use_bf16": amp_enabled,
                            "fused_adamw": fused_enabled,
                            "ddp_world_size": world_size,
                            "warmup_epochs": warmup_epochs,
                            "min_lr_ratio": min_lr_ratio,
                            "loss_name": "bce_focal",
                            "focal_alpha": focal_alpha,
                            "focal_gamma": focal_gamma,
                            "bce_weight": bce_weight,
                            "focal_weight": focal_weight,
                            "balanced_sampling": balanced_sampling,
                            "waveform_aug_prob": waveform_aug_prob,
                            "gain_prob": gain_prob,
                            "gain_db_limit": gain_db_limit,
                            "noise_prob": noise_prob,
                            "noise_snr_db_min": noise_snr_db_min,
                            "noise_snr_db_max": noise_snr_db_max,
                            "time_shift_prob": time_shift_prob,
                            "time_shift_max_frac": time_shift_max_frac,
                            "drop_segment_prob": drop_segment_prob,
                            "drop_segment_max_frac": drop_segment_max_frac,
                        },
                        fold_model_path,
                    )

            _sync(distributed)

        if _is_main_process(rank):
            if not fold_model_path.exists():
                raise RuntimeError(f"Fold model checkpoint not found: {fold_model_path}")

            fold_checkpoint = torch.load(fold_model_path, map_location=device)
            eval_model = SimpleCNN(**model_config).to(device)
            eval_model.load_state_dict(fold_checkpoint["model_state"])

            if val_loader is None:
                raise RuntimeError("Validation loader is not initialized on main rank.")

            _, fold_auc = evaluate_model(
                eval_model,
                val_loader,
                eval_criterion,
                device,
                use_bf16=use_bf16,
            )
            fold_preds = predict_dataset(
                eval_model,
                val_loader,
                device,
                use_bf16=use_bf16,
            )

            val_dataset_targets = np.stack(
                [val_dataset._encode_target(val_meta.iloc[i]).numpy() for i in range(len(val_meta))]
            ).astype(np.float32)

            if fold_preds.shape != val_dataset_targets.shape:
                raise RuntimeError(
                    f"Fold {fold_idx} prediction shape mismatch: "
                    f"preds={fold_preds.shape}, targets={val_dataset_targets.shape}"
                )

            fold_auc_scores.append(float(fold_auc))
            all_oof_preds.append(fold_preds)
            all_oof_targets.append(val_dataset_targets)

            reverse_labels = sorted(label_map.keys(), key=lambda label: label_map[label])
            fold_rows = pd.DataFrame(
                {
                    "fold": fold_idx,
                    "filename": val_meta["filename"].astype(str).tolist(),
                    "start": val_meta["start"].tolist(),
                    "end": val_meta["end"].tolist(),
                    "row_id": [
                        row_id_from_metadata(fname, end_val)
                        for fname, end_val in zip(
                            val_meta["filename"].astype(str).tolist(),
                            val_meta["end"].tolist(),
                        )
                    ],
                }
            )
            for class_idx, label in enumerate(reverse_labels):
                fold_rows[label] = fold_preds[:, class_idx]
            oof_rows.append(fold_rows)

            _print_main(rank, f"Fold {fold_idx} complete | best checkpoint: {fold_model_path}")

        _sync(distributed)

        # Fold teardown to avoid RAM/VRAM accumulation across folds.
        _clear_dataset_cache(train_dataset)
        _clear_dataset_cache(val_dataset)
        del train_loader
        del train_dataset
        if val_loader is not None:
            del val_loader
        if val_dataset is not None:
            del val_dataset
        del model
        del optimizer
        del scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _sync(distributed)

    if _is_main_process(rank):
        reverse_labels = sorted(label_map.keys(), key=lambda label: label_map[label])

        oof_targets = np.concatenate(all_oof_targets, axis=0)
        oof_preds = np.concatenate(all_oof_preds, axis=0)
        oof_auc = birdclef_roc_auc(oof_targets, oof_preds)

        oof_probs_df = pd.concat(oof_rows, ignore_index=True)
        oof_probs_path = output_dir / "oof_predictions.csv"
        oof_probs_df.to_csv(oof_probs_path, index=False)

        npz_path = output_dir / "oof_predictions.npz"
        np.savez(
            npz_path,
            targets=oof_targets.astype(np.float32),
            preds=oof_preds.astype(np.float32),
            labels=np.array(reverse_labels),
        )

        summary = {
            "num_folds": num_folds,
            "fold_auc_scores": fold_auc_scores,
            "oof_auc": float(oof_auc),
            "oof_rows": int(oof_targets.shape[0]),
            "num_labels": len(label_map),
            "oof_predictions_csv": str(oof_probs_path),
            "oof_predictions_npz": str(npz_path),
        }
        with open(output_dir / "oof_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2, sort_keys=True)

        print(
            f"OOF complete | folds={num_folds} | mean_fold_auc={np.mean(fold_auc_scores):.4f} | "
            f"oof_birdclef_roc_auc={oof_auc:.4f}"
        )
        print(f"Saved OOF artifacts to: {oof_probs_path}, {npz_path}, {output_dir / 'oof_summary.json'}")

        if pseudo_labeling_v2_enabled and not pseudo_labels_ready:
            fold_model_paths = [output_dir / f"best_model_fold{i}.pt" for i in range(1, num_folds + 1)]
            if not pseudo_unlabeled_soundscape_dir.exists():
                raise FileNotFoundError(
                    f"Pseudo unlabeled soundscape dir not found: {pseudo_unlabeled_soundscape_dir}"
                )
            pseudo_df = generate_pseudo_labels_v1(
                fold_model_paths=fold_model_paths,
                model_config=model_config,
                label_map=label_map,
                device=device,
                use_bf16=use_bf16,
                unlabeled_metadata_path=pseudo_unlabeled_metadata_path,
                unlabeled_soundscape_dir=pseudo_unlabeled_soundscape_dir,
                pseudo_conf_threshold=pseudo_conf_threshold,
                pseudo_max_labels_per_clip=pseudo_max_labels_per_clip,
                pseudo_top1_fallback_threshold=pseudo_top1_fallback_threshold,
                sample_rate=sample_rate,
                segment_duration=segment_duration,
            )
            pseudo_df.to_csv(pseudo_labels_path, index=False)
            print(
                f"Pseudo-label v2 generated: {len(pseudo_df)} rows at {pseudo_labels_path}. "
                "Re-run v2 training to consume pseudo labels."
            )

    _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
