"""Assemble Kaggle inference notebook(s) from a recipe.

Produces:
    SUBMIT_DIR/submission_<variant>.ipynb
"""
from __future__ import annotations

import argparse
from pathlib import Path

from birdclef.config.paths import SUBMIT_DIR
from birdclef.submit.build_notebook import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--sed-onnx", nargs="+", default=[])
    ap.add_argument("--perch-onnx", default=None)
    ap.add_argument("--variant", default="bold", choices=["bold", "safe"])
    ap.add_argument("--out-dir", default=str(SUBMIT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir) / f"submission_{args.variant}.ipynb"
    build(out, Path(args.recipe), args.sed_onnx, args.perch_onnx, args.variant)


if __name__ == "__main__":
    main()
