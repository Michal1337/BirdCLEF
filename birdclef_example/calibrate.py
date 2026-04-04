import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit per-class Platt calibration from OOF NPZ")
    parser.add_argument("--oof-npz", type=Path, required=True, help="Path to oof_predictions.npz")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("birdclef_example/outputs/calibration_v1.json"),
        help="Output calibration JSON",
    )
    parser.add_argument("--min-positives", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--clip-eps", type=float, default=1e-6)
    return parser.parse_args()


def logit(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def main() -> None:
    args = parse_args()
    if not args.oof_npz.exists():
        raise FileNotFoundError(f"OOF NPZ not found: {args.oof_npz}")

    data = np.load(args.oof_npz, allow_pickle=True)
    targets = data["targets"].astype(np.float32)
    preds = data["preds"].astype(np.float32)
    labels = [str(x) for x in data["labels"].tolist()]

    if targets.shape != preds.shape:
        raise ValueError(f"targets shape {targets.shape} != preds shape {preds.shape}")
    if targets.shape[1] != len(labels):
        raise ValueError("labels length does not match number of classes")

    a_values = np.ones(len(labels), dtype=np.float32)
    b_values = np.zeros(len(labels), dtype=np.float32)

    fitted = 0
    skipped = 0

    for class_idx in range(len(labels)):
        y = targets[:, class_idx]
        positives = int(y.sum())
        negatives = int((1.0 - y).sum())
        if positives < args.min_positives or negatives < args.min_positives:
            skipped += 1
            continue

        x = logit(preds[:, class_idx], eps=args.clip_eps).reshape(-1, 1)

        # Platt scaling: sigmoid(a * logit(p) + b)
        clf = LogisticRegression(max_iter=args.max_iter, solver="lbfgs")
        clf.fit(x, y)
        a_values[class_idx] = float(clf.coef_[0, 0])
        b_values[class_idx] = float(clf.intercept_[0])
        fitted += 1

    payload = {
        "labels": labels,
        "a": a_values.tolist(),
        "b": b_values.tolist(),
        "meta": {
            "fitted_classes": fitted,
            "skipped_classes": skipped,
            "min_positives": args.min_positives,
            "max_iter": args.max_iter,
            "clip_eps": args.clip_eps,
            "source_npz": str(args.oof_npz),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Saved calibration to {args.output_json} | fitted_classes={fitted}, skipped_classes={skipped}"
    )


if __name__ == "__main__":
    main()
