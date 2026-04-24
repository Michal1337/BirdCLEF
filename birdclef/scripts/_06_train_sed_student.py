"""Launch SED training with --pseudo-round pre-populated into the config.

Thin wrapper around `birdclef.train.train_sed_ddp.train_one_fold`. Accepts the
same CLI surface as `_03_train_sed.py` (config / fold / override /
dry-run-steps) plus one required flag (`--pseudo-round N`) which gets stamped
onto the cfg dict before training.
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

from birdclef.train.train_sed_ddp import _build_cfg, parse_overrides, train_one_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pseudo-round", type=int, required=True)
    ap.add_argument("--dry-run-steps", type=int, default=0,
                    help="Smoke-test cap. 0 (default) = full training.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Extra cfg overrides as k=v pairs (JSON-parsed).")
    args = ap.parse_args()
    overrides = parse_overrides(args.override)
    overrides["pseudo_round"] = int(args.pseudo_round)
    cfg = _build_cfg(args.config, overrides)
    train_one_fold(cfg, fold=args.fold, dry_run_steps=int(args.dry_run_steps))


if __name__ == "__main__":
    main()
