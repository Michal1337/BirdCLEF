"""Launch SED DDP training.

Prefer calling via torchrun:
    torchrun --standalone --nproc_per_node=<N> -m birdclef.train.train_sed_ddp \
        --config sed_v2s --fold 0

This wrapper is here so the README run-order uses a uniform `python -m
birdclef.scripts._03_train_sed` pattern when you only have 1 GPU (or want CPU
dry-run).
"""
from __future__ import annotations

import argparse

from birdclef.train.train_sed_ddp import _build_cfg, parse_overrides, train_one_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dry-run-steps", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = _build_cfg(args.config, parse_overrides(args.override))
    train_one_fold(cfg, fold=args.fold, dry_run_steps=args.dry_run_steps)


if __name__ == "__main__":
    main()
