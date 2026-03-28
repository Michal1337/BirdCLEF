import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch
from tqdm import tqdm

from birdclef_example.data import SoundscapeSampler, SpectrogramTransform
from birdclef_example.model import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer BirdCLEF predictions on soundscape test clips")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--soundscape-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="test_soundscapes metadata")
    parser.add_argument("--output-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--label-map", type=Path, help="Optional label_map.json stored at training time")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--hop-duration", type=float, default=2.5)
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


def iterate_batches(segments: Iterable[torch.Tensor], batch_size: int) -> Iterable[torch.Tensor]:
    buffer = []
    for spec in segments:
        buffer.append(spec)
        if len(buffer) == batch_size:
            yield torch.stack(buffer)
            buffer = []
    if buffer:
        yield torch.stack(buffer)


def predict_soundscape(
    model: torch.nn.Module,
    sampler: SoundscapeSampler,
    soundscape_path: Path,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    segments = sampler.iterate_segments(soundscape_path)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in iterate_batches(segments, batch_size):
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu()
            outputs.append(probs)
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty(0, num_classes)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    label_map = load_label_map(args, checkpoint)
    if not label_map:
        raise ValueError("label map is required either via --label-map or saved checkpoint")
    reverse_labels = sorted(label_map.keys(), key=lambda label: label_map[label])

    model = SimpleCNN(n_classes=len(label_map), in_channels=1, base_channels=64)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    spec_transform = SpectrogramTransform(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    sampler = SoundscapeSampler(
        sample_rate=args.sample_rate,
        duration=args.segment_duration,
        hop=args.hop_duration,
        transform=spec_transform,
    )

    metadata = pd.read_csv(args.metadata)
    predictions = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Predict" ):
        filepath = args.soundscape_dir / str(row["filename"])
        if not filepath.exists():
            raise FileNotFoundError(f"Missing soundscape {filepath}")
        probs = predict_soundscape(
            model, sampler, filepath, device, args.batch_size, num_classes=len(label_map)
        )
        if probs.numel() == 0:
            continue
        aggregated = probs.mean(dim=0).numpy()
        row_dict = {"row_id": row.get("row_id", f"{row['filename']}")}
        row_dict.update({label: float(aggregated[idx]) for idx, label in enumerate(reverse_labels)})
        predictions.append(row_dict)

    if not predictions:
        raise RuntimeError("No soundscapes were scored; check input metadata and files.")

    submission = pd.DataFrame(predictions)
    submission = submission[["row_id"] + reverse_labels]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
