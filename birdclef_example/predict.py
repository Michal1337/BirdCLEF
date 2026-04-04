import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from birdclef_example.data import SoundscapeSampler
from birdclef_example.model import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer BirdCLEF predictions on soundscape clips")
    parser.add_argument("--model-path", type=Path, help="Single checkpoint path (legacy mode)")
    parser.add_argument(
        "--model-paths",
        type=Path,
        nargs="+",
        help="Optional list of checkpoint paths for ensemble inference",
    )
    parser.add_argument("--soundscape-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="test metadata listing soundscapes")
    parser.add_argument("--output-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--label-map", type=Path, help="Optional label_map.json stored at training time")
    parser.add_argument("--calibration-json", type=Path, help="Optional calibration JSON from OOF fitting")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--hop-duration", type=float, default=5.0)
    parser.add_argument("--tta-offsets", type=str, default="0.0", help="Comma-separated offsets in seconds")
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--use-bf16", action="store_true", help="Enable bf16 autocast on CUDA")
    return parser.parse_args()


def get_model_paths(args: argparse.Namespace) -> List[Path]:
    if args.model_paths:
        return [Path(p) for p in args.model_paths]
    if args.model_path:
        return [Path(args.model_path)]

    default_candidates = [Path("birdclef_example/outputs") / f"best_model_fold{i}.pt" for i in range(1, 6)]
    existing = [p for p in default_candidates if p.exists()]
    if existing:
        return existing
    raise ValueError("No model checkpoint provided. Use --model-path or --model-paths.")


def parse_tta_offsets(raw_offsets: str) -> List[float]:
    offsets: List[float] = []
    for token in raw_offsets.split(","):
        token = token.strip()
        if not token:
            continue
        offsets.append(float(token))
    if not offsets:
        return [0.0]
    # Preserve order, remove exact duplicates.
    unique_offsets: List[float] = []
    for value in offsets:
        if value not in unique_offsets:
            unique_offsets.append(value)
    return unique_offsets


def load_label_map(label_map_path: Optional[Path], checkpoint: Dict) -> Dict[str, int]:
    if label_map_path and label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as fd:
            return json.load(fd)
    return checkpoint.get("label_map", {})


def load_calibration(calibration_path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    if calibration_path is None:
        return None
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    with open(calibration_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "labels" not in raw or "a" not in raw or "b" not in raw:
        raise ValueError(
            f"Calibration JSON must contain keys: labels, a, b. Got: {sorted(raw.keys())}"
        )

    labels = raw["labels"]
    a_values = raw["a"]
    b_values = raw["b"]
    if len(labels) != len(a_values) or len(labels) != len(b_values):
        raise ValueError("Calibration arrays labels/a/b must have the same length.")

    return {
        str(label): {"a": float(a), "b": float(b)}
        for label, a, b in zip(labels, a_values, b_values)
    }


def iterate_batches(
    segments: Iterable[Tuple[int, torch.Tensor]], batch_size: int
) -> Iterable[Tuple[List[int], torch.Tensor]]:
    time_buffer: List[int] = []
    waveform_buffer: List[torch.Tensor] = []
    for segment_end_seconds, waveform in segments:
        time_buffer.append(segment_end_seconds)
        waveform_buffer.append(waveform)
        if len(waveform_buffer) == batch_size:
            yield time_buffer, torch.stack(waveform_buffer)
            time_buffer = []
            waveform_buffer = []
    if waveform_buffer:
        yield time_buffer, torch.stack(waveform_buffer)


def iterate_aligned_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_duration: float,
    hop_duration: float,
    offset_seconds: float,
) -> Iterable[Tuple[int, torch.Tensor]]:
    segment_samples = int(round(segment_duration * sample_rate))
    hop_samples = int(round(hop_duration * sample_rate))
    offset_samples = int(round(offset_seconds * sample_rate))
    total = waveform.size(1)

    if total == 0:
        return

    start = 0
    max_start_for_full = max(0, total - segment_samples)
    while start < total:
        shifted_start = start + offset_samples
        if shifted_start < 0:
            shifted_start = 0
        if shifted_start > max_start_for_full:
            shifted_start = max_start_for_full

        end = min(total, shifted_start + segment_samples)
        chunk = waveform[:, shifted_start:end]
        if chunk.size(1) < segment_samples:
            chunk = F.pad(chunk, (0, segment_samples - chunk.size(1)))

        segment_end_seconds = int(round((start + segment_samples) / sample_rate))
        yield segment_end_seconds, chunk
        start += hop_samples


def apply_calibration(
    probs: torch.Tensor,
    reverse_labels: List[str],
    calibration: Optional[Dict[str, Dict[str, float]]],
) -> torch.Tensor:
    if calibration is None:
        return probs

    calibrated = probs.clone()
    eps = 1e-6
    for idx, label in enumerate(reverse_labels):
        params = calibration.get(label)
        if params is None:
            continue
        p = calibrated[idx].clamp(min=eps, max=1.0 - eps)
        logit = torch.log(p / (1.0 - p))
        logit = params["a"] * logit + params["b"]
        calibrated[idx] = torch.sigmoid(logit)
    return calibrated


def row_id_from_filename(filename: str, end_seconds: int) -> str:
    return f"{Path(filename).stem}_{end_seconds}"


def build_model_from_checkpoint(
    checkpoint: Dict,
    label_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    model_config = checkpoint.get(
        "model_config",
        {
            "n_classes": label_count,
            "dropout": 0.3,
            "sample_rate": args.sample_rate,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "token_grid_size": 22,
            "pooling": "hybrid",
            "freq_mask_param": 12,
            "time_mask_param": 24,
            "specaugment_masks": 2,
        },
    )
    model = SimpleCNN(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_soundscape(
    models: List[torch.nn.Module],
    sampler: SoundscapeSampler,
    soundscape_path: Path,
    device: torch.device,
    batch_size: int,
    tta_offsets: List[float],
    use_bf16: bool,
) -> List[Tuple[int, torch.Tensor]]:
    waveform = sampler._load_cached_audio(soundscape_path)
    end_to_sum: Dict[int, torch.Tensor] = {}
    end_to_count: Dict[int, int] = {}
    amp_enabled = use_bf16 and device.type == "cuda"

    with torch.no_grad():
        for model in models:
            for offset in tta_offsets:
                segments = iterate_aligned_segments(
                    waveform=waveform,
                    sample_rate=sampler.sample_rate,
                    segment_duration=sampler.duration,
                    hop_duration=sampler.hop_samples / sampler.sample_rate,
                    offset_seconds=offset,
                )
                for end_seconds, batch in iterate_batches(segments, batch_size):
                    batch = batch.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        probs = torch.sigmoid(model(batch)).float().cpu()
                    for i, end_second in enumerate(end_seconds):
                        if end_second not in end_to_sum:
                            end_to_sum[end_second] = probs[i].clone()
                            end_to_count[end_second] = 1
                        else:
                            end_to_sum[end_second] += probs[i]
                            end_to_count[end_second] += 1

    outputs: List[Tuple[int, torch.Tensor]] = []
    for end_second in sorted(end_to_sum.keys()):
        outputs.append((end_second, end_to_sum[end_second] / float(end_to_count[end_second])))
    return outputs


def main() -> None:
    args = parse_args()
    model_paths = get_model_paths(args)

    checkpoints = [torch.load(path, map_location="cpu") for path in model_paths]
    label_map = load_label_map(args.label_map, checkpoints[0])
    if not label_map:
        raise ValueError("label map is required either via --label-map or saved checkpoint")

    reverse_labels = sorted(label_map.keys(), key=lambda label: label_map[label])
    tta_offsets = parse_tta_offsets(args.tta_offsets)
    calibration = load_calibration(args.calibration_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        build_model_from_checkpoint(
            checkpoint=checkpoint,
            label_count=len(label_map),
            args=args,
            device=device,
        )
        for checkpoint in checkpoints
    ]

    sampler = SoundscapeSampler(
        sample_rate=args.sample_rate,
        duration=args.segment_duration,
        hop=args.hop_duration,
        preload_audio=True,
        preload_workers=args.preload_workers,
    )

    metadata = pd.read_csv(args.metadata)
    if "filename" in metadata.columns:
        filenames = metadata["filename"].dropna().astype(str).drop_duplicates().tolist()
    elif "row_id" in metadata.columns:
        filenames = (
            metadata["row_id"]
            .dropna()
            .astype(str)
            .str.replace(r"_[0-9]+$", "", regex=True)
            .drop_duplicates()
            .map(lambda stem: f"{stem}.ogg")
            .tolist()
        )
    else:
        raise ValueError(f"{args.metadata} must contain either a filename column or a row_id column")

    print(f"Using {len(models)} model(s): {[str(p) for p in model_paths]}")
    print(f"Using TTA offsets (sec): {tta_offsets}")
    if calibration is not None:
        print(f"Using calibration from: {args.calibration_json}")

    predictions = []

    for filename in tqdm(filenames, desc="Predict"):
        filepath = args.soundscape_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing soundscape {filepath}")

        for end_seconds, probs in predict_soundscape(
            models=models,
            sampler=sampler,
            soundscape_path=filepath,
            device=device,
            batch_size=args.batch_size,
            tta_offsets=tta_offsets,
            use_bf16=args.use_bf16,
        ):
            probs = apply_calibration(probs, reverse_labels, calibration)
            row_dict = {"row_id": row_id_from_filename(filename, end_seconds)}
            row_dict.update(
                {label: float(probs[idx].item()) for idx, label in enumerate(reverse_labels)}
            )
            predictions.append(row_dict)

    if not predictions:
        raise RuntimeError("No soundscapes were scored; check input metadata and files.")

    submission = pd.DataFrame(predictions)
    if "row_id" in metadata.columns:
        submission = metadata[["row_id"]].merge(submission, on="row_id", how="left")
    submission = submission[["row_id"] + reverse_labels]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
