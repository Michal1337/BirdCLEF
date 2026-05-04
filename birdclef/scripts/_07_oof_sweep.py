"""Run a preset sweep of Perch+SSM configs and write lean CSV + per-config JSON."""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

from birdclef.config.paths import FOLD_KINDS, OUTPUT_ROOT
from birdclef.config.ssm_configs import (
    SWEEP_BEST_SSM,
    SWEEP_CHEAP_WINS,
    SWEEP_LARGER_SSM,
    SWEEP_NOISE_FLOOR,
)
from birdclef.sweep.runner import run_sweep
from birdclef.train.train_ssm_head import run_full_evaluation


PRESETS = {
    "cheap_wins": SWEEP_CHEAP_WINS,
    "best_ssm": SWEEP_BEST_SSM,
    "noise_floor": SWEEP_NOISE_FLOOR,
    "larger_ssm": SWEEP_LARGER_SSM,
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
    ap.add_argument("--pseudo-round", type=int, default=None,
                    help="If set, augment training pool with pseudo-labeled "
                         "rows from cache/pseudo/round{N}/. Val measurement is "
                         "unaffected (pseudo rows have no fold → train-only). "
                         "Default: no pseudo augmentation.")
    ap.add_argument("--pseudo-tau", type=float, default=0.5,
                    help="Probability threshold for hard pseudo-positive labels. "
                         "Only consulted when --pseudo-round is set. Default 0.5.")
    args = ap.parse_args()

    # Stamp n_splits + fold_kind (and optional pseudo-round) onto every config
    # in the preset so train_ssm_head picks the right fold parquet and pseudo
    # cache. Output dir suffixes the fold count + kind + pseudo round so
    # different sweeps don't clobber each other.
    extra: dict = {
        "n_splits": int(args.n_splits),
        "fold_kind": str(args.fold_kind),
    }
    if args.pseudo_round is not None:
        extra["pseudo_round"] = int(args.pseudo_round)
        extra["pseudo_tau"] = float(args.pseudo_tau)
    configs = [{**c, **extra} for c in PRESETS[args.sweep]]

    sweep_name = args.sweep
    if args.n_splits != 5:
        sweep_name += f"_{args.n_splits}fold"
    if args.fold_kind != "strat":
        sweep_name += f"_{args.fold_kind}"
    if args.pseudo_round is not None:
        sweep_name += f"_pseudo{args.pseudo_round}"
    run_sweep(
        name=sweep_name,
        configs=configs,
        stage_fn=run_full_evaluation,
        output_root=OUTPUT_ROOT / "sweep",
    )


if __name__ == "__main__":
    main()
