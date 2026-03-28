import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from birdclef_example.data import SoundscapeSampler
from birdclef_example.model import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer BirdCLEF predictions on soundscape clips")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--soundscape-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="test metadata listing soundscapes")
    parser.add_argument("--output-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--label-map", type=Path, help="Optional label_map.json stored at training time")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--hop-duration", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_label_map(args: argparse.Namespace, checkpoint: Dict) -> Dict[str, int]:
    if args.label_map and args.label_map.exists():
        with open(args.label_map, "r", encoding="utf-8") as fd:
            return json.load(fd)
    return checkpoint.get("label_map", {})


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


def predict_soundscape(
    model: torch.nn.Module,
    sampler: SoundscapeSampler,
    soundscape_path: Path,
    device: torch.device,
    batch_size: int,
) -> List[Tuple[int, torch.Tensor]]:
    outputs: List[Tuple[int, torch.Tensor]] = []
    model.eval()
    with torch.no_grad():
        for end_seconds, batch in iterate_batches(
            sampler.iterate_segment_items(soundscape_path), batch_size
        ):
            batch = batch.to(device, non_blocking=True)
            probs = torch.sigmoid(model(batch)).cpu()
            outputs.extend(zip(end_seconds, probs))
    return outputs


def row_id_from_filename(filename: str, end_seconds: int) -> str:
    return f"{Path(filename).stem}_{end_seconds}"


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    label_map = load_label_map(args, checkpoint)
    if not label_map:
        raise ValueError("label map is required either via --label-map or saved checkpoint")
    reverse_labels = sorted(label_map.keys(), key=lambda label: label_map[label])

    model_config = checkpoint.get(
        "model_config",
        {
            "n_classes": len(label_map),
            "in_channels": 1,
            "base_channels": 64,
            "dropout": 0.3,
        },
    )
    model = SimpleCNN(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sampler = SoundscapeSampler(
        sample_rate=args.sample_rate,
        duration=args.segment_duration,
        hop=args.hop_duration,
        preload_audio=True,
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

    predictions = []

    for filename in tqdm(filenames, desc="Predict"):
        filepath = args.soundscape_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing soundscape {filepath}")

        for end_seconds, probs in predict_soundscape(
            model=model,
            sampler=sampler,
            soundscape_path=filepath,
            device=device,
            batch_size=args.batch_size,
        ):
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
