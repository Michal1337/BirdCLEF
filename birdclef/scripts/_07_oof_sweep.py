"""Run a preset sweep of Perch+SSM configs and write lean CSV + per-config JSON."""
from __future__ import annotations

import argparse

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.config.ssm_configs import SWEEP_BASELINE, SWEEP_CHEAP_WINS
from birdclef.sweep.runner import run_sweep
from birdclef.train.train_ssm_head import run_full_evaluation


PRESETS = {
    "baseline": SWEEP_BASELINE,
    "cheap_wins": SWEEP_CHEAP_WINS,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="cheap_wins", choices=list(PRESETS))
    args = ap.parse_args()
    configs = PRESETS[args.sweep]
    run_sweep(
        name=args.sweep,
        configs=configs,
        stage_fn=run_full_evaluation,
        output_root=OUTPUT_ROOT / "sweep",
    )


if __name__ == "__main__":
    main()
