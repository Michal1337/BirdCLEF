"""Launch SED training with --override pseudo_round=<N> pre-populated.

This is just _03_train_sed.py with an opinionated override to keep the run-
order in the README clear.
"""
from __future__ import annotations

import argparse

from birdclef.train.train_sed_ddp import _build_cfg, parse_overrides, train_one_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pseudo-round", type=int, required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    overrides = parse_overrides(args.override)
    overrides["pseudo_round"] = int(args.pseudo_round)
    cfg = _build_cfg(args.config, overrides)
    train_one_fold(cfg, fold=args.fold)


if __name__ == "__main__":
    main()
