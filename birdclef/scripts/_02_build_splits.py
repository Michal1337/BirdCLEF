"""Build static file-level StratifiedKFold splits over labeled soundscapes.

Default behavior: build BOTH 5-fold and 10-fold parquets in one run, so the
new SOTA / SED / SSM trainers can pick either via their `--n-splits` flag
without re-running this script.

V-anchor was removed — see plan file. All 59 fully-labeled soundscapes
contribute to every fold's train + val (no permanent hold-out).
"""
from __future__ import annotations

import argparse

from birdclef.data.splits import (
    DEFAULT_N_SPLITS,
    build_and_persist_folds,
    stratification_summary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-splits", type=int, nargs="+", default=list(DEFAULT_N_SPLITS),
        help=f"Fold counts to build. Default: {list(DEFAULT_N_SPLITS)}",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for n in args.n_splits:
        print(f"\n=== Building {n}-fold split ===")
        folds = build_and_persist_folds(n_splits=int(n), seed=int(args.seed))
        print(stratification_summary(folds))


if __name__ == "__main__":
    main()
