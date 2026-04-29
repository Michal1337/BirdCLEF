"""Run a preset sweep of Perch+SSM configs and write lean CSV + per-config JSON."""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

from birdclef.config.paths import FOLD_KINDS, OUTPUT_ROOT
from birdclef.config.ssm_configs import (
    SWEEP_BEST_SSM,
    SWEEP_CHEAP_WINS,
    SWEEP_NOISE_FLOOR,
)
from birdclef.sweep.runner import run_sweep
from birdclef.train.train_ssm_head import run_full_evaluation


PRESETS = {
    "cheap_wins": SWEEP_CHEAP_WINS,
    "best_ssm": SWEEP_BEST_SSM,
    "noise_floor": SWEEP_NOISE_FLOOR,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="cheap_wins", choices=list(PRESETS))
    ap.add_argument("--n-splits", type=int, default=5, choices=[5, 10],
                    help="Static fold parquet to use. Default 5. Build via "
                         "`python -m birdclef.scripts._02_build_splits`.")
    ap.add_argument("--fold-kind", default="strat", choices=list(FOLD_KINDS),
                    help="CV grouping. strat=filename stratified (default, "
                         "leakiest, biggest val pool). site=GroupKFold on site "
                         "(strict, unbalanced — one fold = all of S22). "
                         "sitedate=GroupKFold on (site, date) (site-aware, "
                         "balanced — recommended for LB-correlated ranking).")
    args = ap.parse_args()

    # Stamp n_splits + fold_kind onto every config in the preset so
    # train_ssm_head picks the right fold parquet. Output dir suffixes the
    # fold count + kind so different sweeps don't clobber each other.
    configs = [
        {**c, "n_splits": int(args.n_splits), "fold_kind": str(args.fold_kind)}
        for c in PRESETS[args.sweep]
    ]
    sweep_name = args.sweep
    if args.n_splits != 5:
        sweep_name += f"_{args.n_splits}fold"
    if args.fold_kind != "strat":
        sweep_name += f"_{args.fold_kind}"
    run_sweep(
        name=sweep_name,
        configs=configs,
        stage_fn=run_full_evaluation,
        output_root=OUTPUT_ROOT / "sweep",
    )


if __name__ == "__main__":
    main()
