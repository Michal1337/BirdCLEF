#!/usr/bin/env python3
"""Load the Perch model and print model.summary()."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import tensorflow as tf


def build_wrapper_with_tfsm_layer(model_dir: Path) -> tf.keras.Model:
    if not hasattr(tf.keras.layers, "TFSMLayer"):
        raise RuntimeError(
            "tf.keras.layers.TFSMLayer is unavailable in this TensorFlow/Keras version."
        )

    class PerchWrapper(tf.keras.Model):
        def __init__(self, path: Path) -> None:
            super().__init__(name="perch_v2_wrapper")
            self.perch = tf.keras.layers.TFSMLayer(
                str(path), call_endpoint="serving_default"
            )

        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            outputs = self.perch(inputs)
            if isinstance(outputs, dict) and "embedding" in outputs:
                return outputs["embedding"]
            return outputs

    model = PerchWrapper(model_dir)
    # Perch expects 5-second audio chunks at 32kHz -> 160000 samples.
    model.build((None, 160000))
    return model


def load_perch_as_keras_model(model_dir: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(str(model_dir), compile=False)
    except Exception as err:
        print(f"load_model failed, trying TFSMLayer fallback: {err}")
        return build_wrapper_with_tfsm_layer(model_dir)


def print_variable_table(model: tf.keras.Model) -> None:
    print("\n=== Variables ===")
    if not model.weights:
        print("No variables found.")
        return

    total_params = 0
    for i, var in enumerate(model.weights, start=1):
        n_params = int(tf.size(var).numpy())
        total_params += n_params
        print(
            f"{i:04d} | {var.name:<80} | shape={tuple(int(d) for d in var.shape)} | params={n_params}"
        )
    print(f"Total variable params: {total_params}")


def print_savedmodel_structure(model_dir: Path, max_ops: int = 0) -> None:
    print("\n=== SavedModel structure (serving_default) ===")
    sm = tf.saved_model.load(str(model_dir))
    if "serving_default" not in sm.signatures:
        print("No serving_default signature found.")
        return

    fn = sm.signatures["serving_default"]
    ops = fn.graph.get_operations()
    print(f"Total graph operations: {len(ops)}")

    scope_counts = Counter(op.name.split("/")[0] for op in ops)
    print("\nTop-level scopes (approx layer blocks):")
    for scope, count in scope_counts.most_common():
        print(f"- {scope}: {count} ops")

    print("\nGraph ops (index | name | type):")
    op_list = ops if max_ops <= 0 else ops[:max_ops]
    for i, op in enumerate(op_list, start=1):
        print(f"{i:05d} | {op.name} | {op.type}")

    if max_ops > 0 and len(ops) > max_ops:
        print(f"... truncated {len(ops) - max_ops} ops (use --max-ops 0 for full dump)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Perch model and print model.summary()."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/perch_v2/1"),
        help="Path to Perch SavedModel directory.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full SavedModel structure (scopes, variables, ops) in addition to Keras summary.",
    )
    parser.add_argument(
        "--max-ops",
        type=int,
        default=0,
        help="Limit number of printed ops in --full mode; 0 means print all ops.",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    model = load_perch_as_keras_model(args.model_dir)
    print("=== Keras summary ===")
    model.summary(expand_nested=True)
    if args.full:
        print_variable_table(model)
        print_savedmodel_structure(args.model_dir, max_ops=args.max_ops)


if __name__ == "__main__":
    main()
