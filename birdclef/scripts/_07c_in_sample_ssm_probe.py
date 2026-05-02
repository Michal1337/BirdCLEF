"""In-sample SSM probe — train on ALL labeled rows, predict on those same rows.

This is the SSM equivalent of what Tucker Arrants' distilled SED bundle
appears to be (per the `_03c_eval_sed_kaggle_onnx.py --shuffle-offset`
probe): a model fit to the entire labeled pool with no held-out fold,
scored on the same data it trained on.

Purpose: calibrate our reading of SED's 0.985 "OOF" against a
comparable in-sample-fit measurement on SSM. Three possible readings of
the result:

  - SSM in-sample ≈ 0.985 (matches SED) → both architectures hit the
    same memorization ceiling on 708 files. SED has no real edge over
    SSM; its honest OOF is probably ~0.93 too. The +0.016 LB lift is
    purely architecture diversity, not a stronger model.

  - SSM in-sample ≫ 0.985 (e.g. 0.998+) → SSM has more capacity to
    memorize 708 files than SED does. SED's 0.985 represents its
    capacity ceiling, NOT a memorized fit — meaning SED's honest OOF
    could be close to 0.985. SED is the architecturally stronger model.

  - SSM in-sample < 0.985 (e.g. 0.95) → SSM's 0.93 honest OOF is close
    to its capacity; SED reaches 0.985 on the same data with more
    capacity. SED's honest OOF is likely 0.94–0.96 — genuinely better.

Implementation: re-uses `run_pipeline_for_split(cache, train_idx,
val_idx, cfg, ...)` with train_idx == val_idx == all labeled rows.
Same proxy fill, lambda-prior vector, temperatures, and post-processing
as `run_full_evaluation`, just with no fold split.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._07c_in_sample_ssm_probe
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._07c_in_sample_ssm_probe \\
        --seed 7 --out-dir birdclef/outputs/in_sample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.config.ssm_configs import BASELINE
from birdclef.data.soundscapes import load_taxonomy
from birdclef.eval.metrics import compute_stage_metrics, split_rare_frequent
from birdclef.train.train_ssm_head import (
    PerchCache,
    _lambda_prior_vector,
    _temperature_vector,
    load_perch_cache,
    primary_labels,
    run_pipeline_for_split,
    seed_everything,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(BASELINE["seed"]),
                    help="Override BASELINE seed for the SSM training.")
    ap.add_argument("--out-dir", type=Path,
                    default=OUTPUT_ROOT / "in_sample",
                    help="Where to write the metrics JSON + probs NPZ.")
    ap.add_argument("--config-name", type=str, default="ssm_in_sample_baseline",
                    help="Name to record in the metrics JSON.")
    ap.add_argument("--no-proxy", action="store_true",
                    help="Skip the genus-proxy fill on raw Perch logits "
                         "(default: ON, matches run_full_evaluation).")
    args = ap.parse_args()

    cfg = dict(BASELINE)
    cfg["seed"] = int(args.seed)
    cfg["name"] = args.config_name

    base_seed = int(cfg["seed"])
    seed_everything(base_seed)

    cache = load_perch_cache()
    if not args.no_proxy:
        from birdclef.models.perch import apply_proxy_to_scores
        cache.scores = apply_proxy_to_scores(cache.scores, cache.scores_proxy)

    labeled_idx = np.where(cache.labeled_mask)[0]
    cache_meta = cache.meta.iloc[labeled_idx].reset_index(drop=True)
    cache_scores = cache.scores[labeled_idx]
    cache_emb = cache.emb[labeled_idx]
    cache_Y = cache.Y[labeled_idx]
    cache_sub = PerchCache(
        meta=cache_meta,
        scores=cache_scores,
        scores_proxy=np.zeros_like(cache_scores),
        emb=cache_emb,
        Y=cache_Y,
        labeled_mask=np.ones(len(cache_meta), dtype=bool),
    )
    n_rows = len(cache_meta)
    n_files = cache_meta["filename"].nunique()
    print(f"[in-sample] labeled rows: {n_rows:,}   files: {n_files:,}")

    tax = load_taxonomy()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    labels = primary_labels()
    temperatures = _temperature_vector(labels, class_map)
    lambda_prior_vec = _lambda_prior_vector(
        labels, class_map,
        lambda_birds=float(cfg["lambda_prior"]),
        lambda_texture=float(cfg.get("lambda_prior_texture", cfg["lambda_prior"])),
    )

    support = cache_Y.sum(axis=0)
    rare_idx, freq_idx = split_rare_frequent(support)

    # Train on ALL labeled rows; predict on those SAME rows.
    # This is the SSM equivalent of full-pool memorization fit.
    all_idx = np.arange(n_rows, dtype=np.int64)
    seed_everything(base_seed + 1)
    print(f"[in-sample] training SSM pipeline on all {n_rows:,} rows "
          f"(seed={base_seed})")
    out = run_pipeline_for_split(
        cache_sub, all_idx, all_idx, cfg, temperatures,
        lambda_prior_vec=lambda_prior_vec,
    )

    # Score in-sample (training data) — this is the leak-equivalent number.
    final_metrics = compute_stage_metrics(
        cache_Y, out["final"], cache_meta,
        rare_idx=rare_idx, frequent_idx=freq_idx,
    )
    fp_metrics = compute_stage_metrics(
        cache_Y, out["first_pass"], cache_meta,
        rare_idx=rare_idx, frequent_idx=freq_idx,
    )

    summary = {
        "config_name": args.config_name,
        "seed": base_seed,
        "n_rows": int(n_rows),
        "n_files": int(n_files),
        "in_sample_final_macro_auc": float(final_metrics.get("macro_auc", float("nan"))),
        "in_sample_first_pass_macro_auc": float(fp_metrics.get("macro_auc", float("nan"))),
        "in_sample_site_auc_std": float(final_metrics.get("site_auc_std", float("nan"))),
        "in_sample_rare_auc": final_metrics.get("rare_auc", float("nan")),
        "in_sample_frequent_auc": final_metrics.get("frequent_auc", float("nan")),
        "stage_metrics_final": final_metrics,
        "stage_metrics_first_pass": fp_metrics,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.config_name}_seed{base_seed}.json"
    out_npz = out_dir / f"{args.config_name}_seed{base_seed}_probs.npz"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(
        out_npz,
        final=out["final"].astype(np.float32),
        first_pass=out["first_pass"].astype(np.float32),
        y_true=cache_Y.astype(np.uint8),
    )

    print()
    print(f"[in-sample] === SSM in-sample fit (seed={base_seed}) ===")
    print(f"[in-sample]   final macro_auc       = "
          f"{summary['in_sample_final_macro_auc']:.4f}")
    print(f"[in-sample]   first_pass macro_auc  = "
          f"{summary['in_sample_first_pass_macro_auc']:.4f}")
    print(f"[in-sample]   site_auc_std          = "
          f"{summary['in_sample_site_auc_std']:.4f}")
    print(f"[in-sample]   rare_auc              = "
          f"{summary.get('in_sample_rare_auc', float('nan'))}")
    print(f"[in-sample]   frequent_auc          = "
          f"{summary.get('in_sample_frequent_auc', float('nan'))}")
    print(f"[in-sample] wrote {out_json}")
    print(f"[in-sample] wrote {out_npz}")
    print()
    print(f"[in-sample] Compare to:")
    print(f"[in-sample]   - SSM honest stitched-OOF (mean-of-folds): ~0.93")
    print(f"[in-sample]   - SED Tucker leaked 'OOF':                  0.985")


if __name__ == "__main__":
    main()
