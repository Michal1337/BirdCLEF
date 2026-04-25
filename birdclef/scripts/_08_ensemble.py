"""Build ensemble from a set of probability-member files and persist a recipe.

Each member is a .npz with keys {probs: (N_rows, C)} aligned to the
stitched-OOF rows (all labeled soundscape windows in canonical order).
Quick sanity: expects a y_true.npy and a meta.parquet siblings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.ensemble.blend import member_correlation, save_recipe, weight_search_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True,
                    help="List of .npz files with a 'probs' array")
    ap.add_argument("--y-true", required=True,
                    help=".npy of stitched-OOF labels (all labeled soundscape rows)")
    ap.add_argument("--meta", required=True, help=".parquet with site/hour_utc columns")
    ap.add_argument("--blend", default="sigmoid", choices=["sigmoid", "rank"])
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--out", default=str(OUTPUT_ROOT / "sweep" / "ensemble_final" / "best.json"))
    args = ap.parse_args()

    probs_list = [np.load(p)["probs"] for p in args.members]
    y = np.load(args.y_true)
    meta = pd.read_parquet(args.meta)
    corr = member_correlation(probs_list)
    print("[ensemble] member spearman (subsampled):")
    print(np.round(corr, 3))
    res = weight_search_grid(probs_list, y, meta, step=args.step, blend=args.blend)
    print(f"[ensemble] best weights={res['weights']}  "
          f"macro_auc={res['metrics']['macro_auc']:.4f}")
    save_recipe(Path(args.out), args.members, res["weights"], blend=args.blend,
                extra={"metrics": res["metrics"]})
    print(f"[ensemble] recipe saved to {args.out}")


if __name__ == "__main__":
    main()
