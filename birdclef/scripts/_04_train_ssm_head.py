"""Run a single-config Perch+SSM head evaluation (stitched 5-fold OOF).

Writes stage metrics JSON + appends a row to the sweep summary CSV under
OUTPUT_ROOT/sweep/<name>. V-anchor was abandoned (see plan file); ranking
metric is now stitched OOF macro AUC minus site_auc_std penalty.
"""
from __future__ import annotations

import argparse
import json

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.config.ssm_configs import BASELINE
from birdclef.sweep.runner import run_sweep
from birdclef.train.train_ssm_head import run_full_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-name", default="ssm_sanity")
    ap.add_argument(
        "--n-splits", type=int, default=5, choices=[5, 10],
        help="Which static fold parquet to use. Default 5. "
             "Build them via `python -m birdclef.scripts._02_build_splits`.",
    )
    ap.add_argument("--config-overrides", type=str, default="{}",
                    help='JSON dict of baseline overrides to apply.')
    args = ap.parse_args()
    over = json.loads(args.config_overrides or "{}")
    cfg = {**BASELINE, "name": args.sweep_name, "n_splits": int(args.n_splits), **over}
    run_sweep(
        name=args.sweep_name,
        configs=[cfg],
        stage_fn=run_full_evaluation,
        output_root=OUTPUT_ROOT / "sweep",
    )


if __name__ == "__main__":
    main()
