"""Build static fold parquets over labeled soundscapes.

Default: build (5-fold, 10-fold) × kind="strat" (filename StratifiedKFold) for
backwards compat. Use `--kind site|sitedate` to build site-aware GroupKFold
parquets for stricter LB-correlated CV.

Examples:
    # Default (filename stratified)
    python -m birdclef.scripts._02_build_splits

    # Site-only GroupKFold (strict but unbalanced — S22 dominates)
    python -m birdclef.scripts._02_build_splits --kind site --n-splits 5

    # Site+date GroupKFold (balanced site-aware compromise)
    python -m birdclef.scripts._02_build_splits --kind sitedate --n-splits 5 10

    # All three kinds at once
    python -m birdclef.scripts._02_build_splits --kind strat site sitedate
"""
from __future__ import annotations

import argparse

from birdclef.config.paths import FOLD_KINDS
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
    ap.add_argument(
        "--kind", nargs="+", default=["strat"], choices=list(FOLD_KINDS),
        help="Which CV grouping(s) to build. Default: strat (filename "
             "stratified). site = GroupKFold on site (strict, unbalanced). "
             "sitedate = GroupKFold on (site, date) (balanced site-aware).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for kind in args.kind:
        for n in args.n_splits:
            print(f"\n=== Building {n}-fold split (kind={kind}) ===")
            folds = build_and_persist_folds(
                n_splits=int(n), seed=int(args.seed), kind=kind,
            )
            print(stratification_summary(folds))


if __name__ == "__main__":
    main()
