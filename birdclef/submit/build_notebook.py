"""Assemble a Kaggle-ready CPU inference notebook from inference_template.py.

Output: a .ipynb file that:
  1) Sets BIRDCLEF_PATH_MODE=kaggle
  2) Inlines the inference template
  3) Inlines the blend recipe directly as a Python dict (no separate JSON file
     to upload).
  4) Calls run_submission(...) with artifact paths rewritten to
     /kaggle/input/<dataset>/...

The generated notebook has no pip installs — onnxruntime + soundfile + pandas
must already be in Kaggle's default Python image (they are).
"""
from __future__ import annotations

import argparse
import json
import pprint
from pathlib import Path


def _py_literal(obj) -> str:
    """Format `obj` as a Python expression that eval()s back to it.

    Handles None correctly (json.dumps would emit "null" which Python rejects).
    """
    return pprint.pformat(obj, width=120, sort_dicts=False)


def _nbcell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def build(out_ipynb: Path, recipe_json: Path, sed_onnx_paths: list[str],
          perch_onnx_path: str | None = None, variant: str = "bold") -> None:
    tpl = (Path(__file__).parent / "inference_template.py").read_text(encoding="utf-8")

    # Inline the recipe so the notebook needs no separate JSON file uploaded
    # alongside. If recipe_json doesn't exist, fall back to a uniform blend
    # over whatever sed_onnx_paths the user provided.
    recipe = None
    if recipe_json and Path(recipe_json).exists():
        try:
            recipe = json.loads(Path(recipe_json).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[submit] warning: could not parse {recipe_json}: {exc}")
            recipe = None
    if recipe is None:
        n = max(1, len(sed_onnx_paths))
        recipe = {"blend": "sigmoid", "weights": [1.0 / n] * n,
                  "members": list(sed_onnx_paths) or ["<single-fold default>"],
                  "extra": {"description": "uniform fallback recipe"}}

    header = """import os, sys
os.environ['BIRDCLEF_PATH_MODE'] = 'kaggle'
# --- inference template ---
"""
    caller = f"""
from pathlib import Path

# Recipe inlined so no extra file needs to be uploaded to Kaggle.
recipe = {_py_literal(recipe)}

run_submission(
    test_dir=Path('/kaggle/input/birdclef-2026/test_soundscapes'),
    sample_sub_csv=Path('/kaggle/input/birdclef-2026/sample_submission.csv'),
    perch_onnx={_py_literal(perch_onnx_path)},
    sed_onnx_paths={_py_literal(list(sed_onnx_paths))},
    recipe=recipe,
    output_csv=Path('/kaggle/working/submission.csv'),
)
"""
    cells = [
        _nbcell(f"# BirdCLEF 2026 — {variant} submission\n\n"
                f"Members: {len(sed_onnx_paths)} SED ONNX file(s)" +
                (f" + Perch ONNX" if perch_onnx_path else ""),
                cell_type="markdown"),
        _nbcell(header + tpl),
        _nbcell(caller),
    ]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(out_ipynb).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ipynb).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[submit] wrote notebook {out_ipynb}")
    print(f"[submit] sed_onnx_paths in notebook:")
    for p in sed_onnx_paths:
        print(f"  - {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--sed-onnx", nargs="+", default=[])
    ap.add_argument("--perch-onnx", default=None)
    ap.add_argument("--variant", default="bold")
    args = ap.parse_args()
    build(Path(args.out), Path(args.recipe), args.sed_onnx, args.perch_onnx, args.variant)


if __name__ == "__main__":
    main()
