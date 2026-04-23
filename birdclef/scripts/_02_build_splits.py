"""Entry point: build site×date folds + V-anchor hold-out."""
from __future__ import annotations

import argparse

from birdclef.data.splits import build_and_persist, stratification_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--v-anchor-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    non_anchor, v_anchor = build_and_persist(
        n_splits=args.n_splits, v_anchor_fraction=args.v_anchor_fraction, seed=args.seed,
    )
    print(stratification_summary(non_anchor, v_anchor))


if __name__ == "__main__":
    main()
