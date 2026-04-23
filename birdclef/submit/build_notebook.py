"""Assemble a Kaggle-ready CPU inference notebook from inference_template.py.

Output: a .ipynb file that:
  1) Sets BIRDCLEF_PATH_MODE=kaggle
  2) Inlines the inference template
  3) Calls run_submission(...) with artifact paths rewritten to /kaggle/input/...

The generated notebook has no pip installs — all wheels come from a Kaggle
dataset you upload separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    header = f"""import os, sys
os.environ['BIRDCLEF_PATH_MODE'] = 'kaggle'
# --- inference template ---
"""
    caller = f"""
from pathlib import Path
run_submission(
    test_dir=Path('/kaggle/input/birdclef-2026/test_soundscapes'),
    sample_sub_csv=Path('/kaggle/input/birdclef-2026/sample_submission.csv'),
    perch_onnx={json.dumps(perch_onnx_path)},
    sed_onnx_paths={json.dumps(sed_onnx_paths)},
    recipe_json={json.dumps(str(recipe_json))},
    output_csv=Path('/kaggle/working/submission.csv'),
)
"""
    cells = [
        _nbcell(f"# BirdCLEF 2026 — {variant} submission", cell_type="markdown"),
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
