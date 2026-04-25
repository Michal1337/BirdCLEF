"""Assemble Kaggle inference notebook(s) from a recipe.

Produces:
    SUBMIT_DIR/submission_<variant>.ipynb
"""
from __future__ import annotations

import argparse
from pathlib import Path

from birdclef.config.paths import SUBMIT_DIR
from birdclef.submit.build_notebook import DEFAULT_SAMPLE_SUB, DEFAULT_TEST_DIR, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--sed-onnx", nargs="+", default=[])
    ap.add_argument("--perch-onnx", default=None)
    ap.add_argument("--variant", default="bold", choices=["bold", "safe"])
    ap.add_argument("--out-dir", default=str(SUBMIT_DIR))
    ap.add_argument("--test-dir", default=DEFAULT_TEST_DIR,
                    help=f"Kaggle path to test soundscapes. Default: {DEFAULT_TEST_DIR}")
    ap.add_argument("--sample-sub", default=DEFAULT_SAMPLE_SUB,
                    help=f"Kaggle path to sample_submission.csv. Default: {DEFAULT_SAMPLE_SUB}")
    args = ap.parse_args()
    out = Path(args.out_dir) / f"submission_{args.variant}.ipynb"
    build(
        out, Path(args.recipe), args.sed_onnx,
        perch_onnx_path=args.perch_onnx,
        variant=args.variant,
        test_dir=args.test_dir,
        sample_sub_csv=args.sample_sub,
    )


if __name__ == "__main__":
    main()
