#!/usr/bin/env python3

"""Export a fine-tuned Perch head checkpoint back into an ONNX model.

This utility takes a checkpoint produced by the fine-tuning scripts and writes
an ONNX file with the updated head initializers injected into the base Perch
model.

Typical usage:
    python birdclef_example/export_perch_finetuned_onnx.py \
        --checkpoint outputs/experiments_ft/perch_onnx_head_ft_frozen_w_best.pt \
        --output models/perch_onnx/perch_v2_finetuned.onnx

Then point predict_ported.py at the exported model:
    PERCH_ONNX_PATH=models/perch_onnx/perch_v2_finetuned.onnx python birdclef_example/predict_ported.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    import onnx
    from onnx import numpy_helper
except Exception as exc:
    raise RuntimeError(
        "This script requires the 'onnx' package to read and write ONNX models. "
        "Install it with: pip install onnx"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ONNX = REPO_ROOT / "models" / "perch_onnx" / "perch_v2.onnx"
DEFAULT_OUTPUT_ONNX = REPO_ROOT / "models" / "perch_onnx" / "perch_v2_finetuned_partial.onnx"
HEAD_W_NAME = "jit(infer_fn)/MultiHeadClassifier/MultiHeadClassifier._call_model/heads_protopnet_logits/dot_general6_reshaped_0"
HEAD_ALPHA_NAME = "arith.constant62"
HEAD_BIAS_NAME = "arith.constant61"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fine-tuned Perch head weights back into ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned checkpoint from perch_onnx_head_ft*.py")
    parser.add_argument("--base-onnx", type=Path, default=DEFAULT_BASE_ONNX, help="Base Perch ONNX model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ONNX, help="Output ONNX path")
    parser.add_argument("--strict", action="store_true", help="Fail if tensor shapes do not match exactly")
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(data)!r}")
    missing = [key for key in ("weight", "alpha", "bias") if key not in data]
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")
    return data


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.base_onnx.exists():
        raise FileNotFoundError(f"Base ONNX model not found: {args.base_onnx}")

    ckpt = _load_checkpoint(args.checkpoint)
    weight = ckpt["weight"].detach().cpu().numpy().astype(np.float32, copy=False)
    alpha = ckpt["alpha"].detach().cpu().numpy().astype(np.float32, copy=False)
    bias = ckpt["bias"].detach().cpu().numpy().astype(np.float32, copy=False)

    model = onnx.load(str(args.base_onnx))
    init_map = {tensor.name: tensor for tensor in model.graph.initializer}
    for name in (HEAD_W_NAME, HEAD_ALPHA_NAME, HEAD_BIAS_NAME):
        if name not in init_map:
            raise RuntimeError(f"Missing expected initializer in ONNX graph: {name}")

    expected_shapes = {
        HEAD_W_NAME: tuple(init_map[HEAD_W_NAME].dims),
        HEAD_ALPHA_NAME: tuple(init_map[HEAD_ALPHA_NAME].dims),
        HEAD_BIAS_NAME: tuple(init_map[HEAD_BIAS_NAME].dims),
    }
    actual_shapes = {
        HEAD_W_NAME: tuple(weight.shape),
        HEAD_ALPHA_NAME: tuple(alpha.shape),
        HEAD_BIAS_NAME: tuple(bias.shape),
    }
    for name in expected_shapes:
        if expected_shapes[name] != actual_shapes[name]:
            msg = f"Shape mismatch for {name}: expected {expected_shapes[name]}, got {actual_shapes[name]}"
            if args.strict:
                raise ValueError(msg)
            print("Warning:", msg)

    init_map[HEAD_W_NAME].CopyFrom(numpy_helper.from_array(weight, name=HEAD_W_NAME))
    init_map[HEAD_ALPHA_NAME].CopyFrom(numpy_helper.from_array(alpha, name=HEAD_ALPHA_NAME))
    init_map[HEAD_BIAS_NAME].CopyFrom(numpy_helper.from_array(bias, name=HEAD_BIAS_NAME))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Saved fine-tuned ONNX model to: {args.output}")


if __name__ == "__main__":
    main()
