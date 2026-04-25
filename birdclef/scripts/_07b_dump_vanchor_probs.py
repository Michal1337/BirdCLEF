"""Dump V-anchor probability arrays for blend-weight search via _08_ensemble.py.

Two members are emitted, both aligned row-for-row to the V-anchor labeled rows
in the Perch cache:

    {out_dir}/ssm_probs.npz   {"probs": (N, C) float32}
    {out_dir}/sed_probs.npz   {"probs": (N, C) float32}
    {out_dir}/y_true.npy      (N, C) uint8
    {out_dir}/meta.parquet    columns: row_id, filename, site, hour_utc

Then the existing _08_ensemble.py search runs:

    python -m birdclef.scripts._08_ensemble \\
        --members .../ssm_probs.npz .../sed_probs.npz \\
        --y-true  .../y_true.npy \\
        --meta    .../meta.parquet

CAVEAT — keep in mind from STRATEGY_V2.md §1: V-anchor over-promised for the
SED single-fold case (V-anchor 0.89 → LB 0.738). The blend weight that wins
on V-anchor is *probably* directionally right but the absolute AUC is biased.
Use the ranking, not the magnitude.

The SSM member here is computed from `birdclef/train/train_ssm_head.py` (the
cleaned port: lambda_prior=0, correction_weight=0, threshold=[0.5]). That's
NOT the same stack as LB_093.ipynb. The blend curve we get is for the
cleaned stack. If LB_093 reaches 0.93 with prior=0.4/corr=0.30, the
LB-equivalent blend weight could differ. Best mitigation: pick a couple of
weights from the V-anchor curve and try them on Kaggle.
"""
from __future__ import annotations

import argparse
import glob as _glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    OUTPUT_ROOT,
    SOUNDSCAPES,
    WINDOW_SAMPLES,
)
from birdclef.data.splits import load_v_anchor


def _dump_ssm(out_dir: Path, ssm_config_name: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Train cleaned SSM stack on non-anchor labeled, predict on anchor.

    Returns (probs[N, C], meta[N rows], y_true[N, C]).
    """
    from birdclef.config.ssm_configs import BASELINE
    from birdclef.data.soundscapes import load_taxonomy
    from birdclef.train.train_ssm_head import (
        _temperature_vector,
        load_perch_cache,
        run_pipeline_for_split,
    )
    from birdclef.utils.seed import seed_everything
    from birdclef.data.soundscapes import primary_labels

    cfg = dict(BASELINE)
    cfg["name"] = ssm_config_name
    cfg["seed"] = int(cfg.get("seed", 42))
    seed_everything(cfg["seed"])

    print(f"[ssm-vanchor] loading Perch cache + computing on cleaned config '{cfg['name']}'")
    cache = load_perch_cache()
    labeled_idx = np.where(cache.labeled_mask)[0]
    cache_meta = cache.meta.iloc[labeled_idx].reset_index(drop=True)
    cache_scores = cache.scores[labeled_idx]
    cache_emb = cache.emb[labeled_idx]
    cache_Y = cache.Y[labeled_idx]
    from birdclef.train.train_ssm_head import PerchCache
    cache_sub = PerchCache(meta=cache_meta, scores=cache_scores, emb=cache_emb,
                           Y=cache_Y, labeled_mask=np.ones(len(cache_meta), dtype=bool))

    tax = load_taxonomy()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    labels = primary_labels()
    temperatures = _temperature_vector(labels, class_map)

    anchor_files = set(load_v_anchor())
    v_mask = cache_meta["filename"].isin(anchor_files).to_numpy()
    if not v_mask.any():
        raise SystemExit("V-anchor file list does not intersect labeled rows.")
    tr = np.where(~v_mask)[0]
    va = np.where(v_mask)[0]
    print(f"[ssm-vanchor] train rows={len(tr)}  v-anchor rows={len(va)}")

    out = run_pipeline_for_split(cache_sub, tr, va, cfg, temperatures)
    probs = out["final"].astype(np.float32)
    va_meta = cache_meta.iloc[va].reset_index(drop=True)
    y_true = cache_Y[va].astype(np.uint8)

    keep_cols = ["row_id", "filename", "site", "hour_utc"]
    keep_cols = [c for c in keep_cols if c in va_meta.columns]
    meta_out = va_meta[keep_cols].copy()
    return probs, meta_out, y_true


def _np_dtype_from_ort(type_str: str):
    if "float16" in type_str:
        return np.float16
    if "double" in type_str:
        return np.float64
    return np.float32


def _dump_sed(out_dir: Path, sed_paths: List[str], target_filenames: List[str],
              n_classes: int) -> np.ndarray:
    """Run the 5-fold SED ensemble on V-anchor OGGs in `target_filenames` order.

    `target_filenames` MUST match the row order of the SSM dump (one filename
    per 12 rows). We iterate that exact list rather than globbing so the
    per-row alignment is provable.
    """
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sed_sessions = [
        ort.InferenceSession(str(p), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        for p in sed_paths
    ]
    sed_input_names = [s.get_inputs()[0].name for s in sed_sessions]
    sed_input_dtypes = [_np_dtype_from_ort(s.get_inputs()[0].type) for s in sed_sessions]
    print(f"[sed-vanchor] loaded {len(sed_sessions)} SED ONNX sessions; "
          f"input dtypes={[str(np.dtype(d)) for d in sed_input_dtypes]}")

    rows = []
    for fname in tqdm(target_filenames, desc="SED V-anchor"):
        path = SOUNDSCAPES / fname
        if not path.exists():
            raise SystemExit(f"V-anchor file missing on disk: {path}")
        y, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        wins = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        ensemble = np.zeros((N_WINDOWS, n_classes), dtype=np.float32)
        for s, inp, dt in zip(sed_sessions, sed_input_names, sed_input_dtypes):
            x = wins if wins.dtype == dt else wins.astype(dt)
            logits = s.run(None, {inp: x})[0]
            ensemble += 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -30, 30)))
        ensemble /= len(sed_sessions)
        rows.append(ensemble)
    return np.concatenate(rows, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    sed_group = ap.add_mutually_exclusive_group(required=True)
    sed_group.add_argument("--sed-onnx", nargs="+", default=None,
                    help="Local paths to SED ONNX files (one per fold).")
    sed_group.add_argument("--sed-folds-glob", default=None,
                    help="Glob pattern, e.g. 'birdclef/models_ckpt/sed/sed_v2s/fold*/best.onnx'")
    ap.add_argument("--ssm-config-name", default="vanchor_dump",
                    help="Name stamped on the SSM run (for cache/log filenames). "
                         "The actual hyperparameters come from BASELINE in ssm_configs.py.")
    ap.add_argument("--out-dir",
                    default=str(OUTPUT_ROOT / "blend_search" / "vanchor"),
                    help="Where to write ssm_probs.npz, sed_probs.npz, y_true.npy, meta.parquet")
    args = ap.parse_args()

    if args.sed_onnx:
        sed_paths = list(args.sed_onnx)
    else:
        sed_paths = sorted(_glob.glob(args.sed_folds_glob))
        if not sed_paths:
            raise SystemExit(f"--sed-folds-glob '{args.sed_folds_glob}' matched no files")
    print(f"[dump] {len(sed_paths)} SED ONNX paths:")
    for p in sed_paths:
        print(f"  - {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) SSM pass — emits both the probs grid AND the canonical row order
    ssm_probs, meta_out, y_true = _dump_ssm(out_dir, args.ssm_config_name)

    # 2) SED pass — must visit files in the same order the SSM dump used
    file_order = meta_out.drop_duplicates("filename")["filename"].tolist()
    n_classes = ssm_probs.shape[1]
    sed_probs = _dump_sed(out_dir, sed_paths, file_order, n_classes)

    # Sanity: row counts should match
    if sed_probs.shape != ssm_probs.shape:
        raise SystemExit(
            f"shape mismatch: ssm={ssm_probs.shape}  sed={sed_probs.shape}. "
            "Check that V-anchor file count is consistent between Perch cache and disk."
        )

    np.savez_compressed(out_dir / "ssm_probs.npz", probs=ssm_probs)
    np.savez_compressed(out_dir / "sed_probs.npz", probs=sed_probs)
    np.save(out_dir / "y_true.npy", y_true)
    meta_out.to_parquet(out_dir / "meta.parquet", index=False)

    print()
    print(f"[dump] wrote 4 files to {out_dir}")
    print(f"[dump]   ssm_probs.npz  shape={ssm_probs.shape}  "
          f"mean={ssm_probs.mean():.4f}")
    print(f"[dump]   sed_probs.npz  shape={sed_probs.shape}  "
          f"mean={sed_probs.mean():.4f}")
    print(f"[dump]   y_true.npy     shape={y_true.shape}  positives={int(y_true.sum())}")
    print(f"[dump]   meta.parquet   rows={len(meta_out)}  files={len(file_order)}")
    print()
    print("Next step — blend weight search:")
    print(f"  python -m birdclef.scripts._08_ensemble \\")
    print(f"      --members {out_dir / 'ssm_probs.npz'} {out_dir / 'sed_probs.npz'} \\")
    print(f"      --y-true  {out_dir / 'y_true.npy'} \\")
    print(f"      --meta    {out_dir / 'meta.parquet'} \\")
    print(f"      --step 0.05")


if __name__ == "__main__":
    main()
