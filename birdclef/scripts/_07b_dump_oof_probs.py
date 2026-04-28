"""Dump stitched-OOF probability arrays for blend-weight search via _09_blend_search.

Two members are emitted, both aligned row-for-row to the labeled soundscape
rows in the Perch cache (all 59 fully-labeled files × 12 windows = 708 rows):

    {out_dir}/ssm_probs.npz   {"probs": (N, C) float32}
    {out_dir}/sed_probs.npz   {"probs": (N, C) float32}    (optional)
    {out_dir}/y_true.npy      (N, C) uint8
    {out_dir}/meta.parquet    cols: row_id, filename, site, hour_utc, fold,
                                    fold_kind

Default `out_dir` is `birdclef/outputs/blend_search/oof_<fold-kind>/`, so
you can re-run with different `--fold-kind` values without clobbering prior
dumps. Pass `--out-dir` to override.

Row order is sorted by (filename, end_sec) and is independent of fold-kind,
so AST/CNN prob dumps produced against any one of the per-kind dirs stay
row-aligned to the others — copy a single ast_probs.npz between dirs if you
want to compare blend search across CV schemes without re-running AST
inference.

Run examples:

    # filename-stratified (LB-correlated default)
    python -m birdclef.scripts._07b_dump_oof_probs --fold-kind strat

    # site-aware (stricter; sitedate balances better than pure site)
    python -m birdclef.scripts._07b_dump_oof_probs --fold-kind sitedate

How the OOF members are computed:
- SSM: for each fold i, train the SSM pipeline on n-1 folds and predict on
  fold i. Concatenate predictions across all folds -> per-row OOF prediction.
- SED: load each fold's `best.pt` ONNX, predict on that fold's val files,
  concatenate. Each row predicted by the model that didn't see that file.

The blend search then picks weights that maximize either stitched macro AUC
or mean-of-folds macro AUC (the LB-correlated default) over those 708 rows.
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
from birdclef.data.splits import load_folds


def _dump_ssm(out_dir: Path, ssm_config_name: str, n_splits: int,
              fold_kind: str = "strat") -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Stitched OOF over all labeled rows: train per fold on n-1 folds, predict on held-out fold.

    Returns (probs[N, C], meta[N rows], y_true[N, C]) where N = total labeled
    soundscape windows across all folds.
    """
    from birdclef.config.ssm_configs import BASELINE
    from birdclef.data.soundscapes import load_taxonomy, primary_labels
    from birdclef.train.train_ssm_head import (
        PerchCache,
        _lambda_prior_vector,
        _temperature_vector,
        load_perch_cache,
        run_pipeline_for_split,
    )
    from birdclef.utils.seed import seed_everything

    cfg = dict(BASELINE)
    cfg["name"] = ssm_config_name
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["n_splits"] = int(n_splits)
    cfg["fold_kind"] = str(fold_kind)
    seed_everything(cfg["seed"])

    print(f"[ssm-oof] loading Perch cache; cleaned config '{cfg['name']}', "
          f"n_splits={n_splits}, fold_kind={fold_kind}")
    cache = load_perch_cache()
    # Match the LB-notebook / sweep-runner pattern: explicit proxy merge
    # right after the load, before any downstream consumer touches scores.
    # Default on; flip to False in cfg to dump pure direct-mapped OOF.
    if bool(cfg.get("use_perch_proxy", True)):
        from birdclef.models.perch import apply_proxy_to_scores
        cache.scores = apply_proxy_to_scores(cache.scores, cache.scores_proxy)
    labeled_idx = np.where(cache.labeled_mask)[0]
    cache_meta = cache.meta.iloc[labeled_idx].reset_index(drop=True)
    cache_scores = cache.scores[labeled_idx]
    cache_emb = cache.emb[labeled_idx]
    cache_Y = cache.Y[labeled_idx]
    # Subset cache: proxy is already merged into cache_scores upstream (or
    # was never wanted), so the subset's `scores_proxy` is all-zero — any
    # downstream code that touches `cache_sub.scores_proxy` will see "no
    # extra fill to apply" and won't re-add the proxy.
    cache_sub = PerchCache(
        meta=cache_meta,
        scores=cache_scores,
        scores_proxy=np.zeros_like(cache_scores),
        emb=cache_emb,
        Y=cache_Y,
        labeled_mask=np.ones(len(cache_meta), dtype=bool),
    )

    tax = load_taxonomy()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    labels = primary_labels()
    temperatures = _temperature_vector(labels, class_map)
    lambda_prior_vec = _lambda_prior_vector(
        labels, class_map,
        lambda_birds=float(cfg["lambda_prior"]),
        lambda_texture=float(cfg.get("lambda_prior_texture", cfg["lambda_prior"])),
    )

    folds_df = load_folds(n_splits=int(n_splits), kind=fold_kind)
    fold_of = dict(zip(folds_df["filename"], folds_df["fold"].astype(int)))
    # fold=-1 means PINNED → in every fold's train, never in val (matches the
    # _07b_ convention that pinned/missing rows are not part of OOF eval).
    row_fold = cache_meta["filename"].map(fold_of).fillna(-1).astype(int).to_numpy()

    n_classes = cache_Y.shape[1]
    oof_final = np.zeros((len(cache_meta), n_classes), dtype=np.float32)
    keep = row_fold >= 0  # only rows assigned to a real fold contribute to OOF probs
    for f in range(int(n_splits)):
        # Train includes pinned (-1) and other folds; val is exact-match only.
        tr = np.where(row_fold != f)[0]
        va = np.where(row_fold == f)[0]
        if len(va) == 0:
            print(f"[ssm-oof]   fold {f}: empty val set, skipping")
            continue
        seed_everything(cfg["seed"] + int(f) + 1)
        out = run_pipeline_for_split(cache_sub, tr, va, cfg, temperatures,
                                     lambda_prior_vec=lambda_prior_vec)
        oof_final[va] = out["final"]
        print(f"[ssm-oof]   fold {f}: train={len(tr)} val={len(va)} done")

    if not keep.any():
        raise SystemExit("No labeled rows had a valid fold assignment.")
    keep_meta = cache_meta.iloc[keep].reset_index(drop=True)
    probs = oof_final[keep].astype(np.float32)
    y_true = cache_Y[keep].astype(np.uint8)
    keep_cols = [c for c in ["row_id", "filename", "site", "hour_utc"] if c in keep_meta.columns]
    meta_out = keep_meta[keep_cols].copy()
    # Stamp the per-row fold index + the kind that produced it so downstream
    # consumers (e.g. _09_blend_search) can compute mean-of-folds metrics
    # without re-loading the fold parquet.
    meta_out["fold"] = row_fold[keep].astype(np.int8)
    meta_out["fold_kind"] = str(fold_kind)
    return probs, meta_out, y_true


def _np_dtype_from_ort(type_str: str):
    if "float16" in type_str:
        return np.float16
    if "double" in type_str:
        return np.float64
    return np.float32


def _dump_sed(sed_paths: List[str], target_filenames: List[str],
              row_fold_per_file: dict, n_classes: int) -> np.ndarray:
    """Per-file: predict with the ONNX session whose fold did NOT include this file.

    `row_fold_per_file` maps filename -> fold index. Each fold's best.pt was
    trained excluding that fold's val files; predicting on that fold's files
    with that fold's checkpoint gives the honest OOF prediction.

    `sed_paths` is sorted by fold (path naming `fold0/`, `fold1/`, ...).
    """
    import onnxruntime as ort
    import re

    # Map fold-index -> session by parsing the path. Fall back to positional
    # order if the regex misses (typical when paths follow `fold{i}/best.onnx`).
    fold_re = re.compile(r"fold(\d+)")
    sess_by_fold: dict[int, "ort.InferenceSession"] = {}
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    for i, p in enumerate(sed_paths):
        m = fold_re.search(p)
        fold_idx = int(m.group(1)) if m else i
        sess = ort.InferenceSession(str(p), sess_options=sess_opts,
                                    providers=["CPUExecutionProvider"])
        sess_by_fold[fold_idx] = sess
        print(f"[sed-oof]   loaded fold {fold_idx} ONNX from {p}")

    # Per-session input name + dtype
    name_by_fold = {f: s.get_inputs()[0].name for f, s in sess_by_fold.items()}
    dtype_by_fold = {f: _np_dtype_from_ort(s.get_inputs()[0].type) for f, s in sess_by_fold.items()}

    rows = []
    for fname in tqdm(target_filenames, desc="SED stitched OOF"):
        path = SOUNDSCAPES / fname
        if not path.exists():
            raise SystemExit(f"Labeled file missing on disk: {path}")
        fold = row_fold_per_file.get(fname)
        if fold is None or fold not in sess_by_fold:
            raise SystemExit(
                f"No SED checkpoint for fold {fold} (file {fname}). "
                f"Available folds: {sorted(sess_by_fold)}."
            )
        sess = sess_by_fold[fold]
        inp = name_by_fold[fold]
        dt = dtype_by_fold[fold]
        y, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        wins = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        x = wins if wins.dtype == dt else wins.astype(dt)
        logits = sess.run(None, {inp: x})[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -30, 30)))
        rows.append(probs)
    return np.concatenate(rows, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    # SED is optional — without it we still emit the SSM probs (useful for the
    # per-class audit / standalone SSM analysis). Provide --sed-onnx or
    # --sed-folds-glob to also emit sed_probs.npz for the blend search.
    ap.add_argument("--sed-onnx", nargs="+", default=None,
                    help="Local paths to SED ONNX files (one per fold). Order matters: "
                         "must align with fold index (fold0, fold1, ...) — paths are "
                         "auto-mapped via the `foldN` substring.")
    ap.add_argument("--sed-folds-glob", default=None,
                    help="Glob pattern, e.g. 'birdclef/models_ckpt/sed/sed_v2s/fold*/best.onnx'")
    ap.add_argument("--n-splits", type=int, default=5, choices=[5, 10],
                    help="Static fold parquet to use. Default 5.")
    from birdclef.config.paths import FOLD_KINDS
    ap.add_argument("--fold-kind", default="strat", choices=list(FOLD_KINDS),
                    help="CV grouping. strat=filename stratified (default). "
                         "site=GroupKFold on site (strict, unbalanced). "
                         "sitedate=GroupKFold on (site, date) (balanced site-aware).")
    ap.add_argument("--ssm-config-name", default="oof_dump",
                    help="Name stamped on the SSM run (for log filenames). "
                         "Hyperparameters come from BASELINE in ssm_configs.py.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write ssm_probs.npz [, sed_probs.npz], "
                         "y_true.npy, meta.parquet. Default: "
                         "birdclef/outputs/blend_search/oof_<fold-kind>/ "
                         "(auto-suffixed by --fold-kind so different CV "
                         "schemes don't clobber each other).")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = str(OUTPUT_ROOT / "blend_search" / f"oof_{args.fold_kind}")

    sed_paths: List[str] = []
    if args.sed_onnx:
        sed_paths = list(args.sed_onnx)
    elif args.sed_folds_glob:
        sed_paths = sorted(_glob.glob(args.sed_folds_glob))
        if not sed_paths:
            raise SystemExit(f"--sed-folds-glob '{args.sed_folds_glob}' matched no files")
    if sed_paths:
        print(f"[dump] {len(sed_paths)} SED ONNX paths:")
        for p in sed_paths:
            print(f"  - {p}")
    else:
        print("[dump] No SED paths provided — SSM-only mode (sed_probs.npz will not be written).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) SSM stitched OOF — emits canonical row order
    ssm_probs, meta_out, y_true = _dump_ssm(
        out_dir, args.ssm_config_name, args.n_splits, fold_kind=args.fold_kind,
    )
    np.savez_compressed(out_dir / "ssm_probs.npz", probs=ssm_probs)
    np.save(out_dir / "y_true.npy", y_true)
    meta_out.to_parquet(out_dir / "meta.parquet", index=False)

    sed_probs = None
    if sed_paths:
        # 2) SED stitched OOF — predict each row by the fold's checkpoint that
        # didn't see it
        folds_df = load_folds(n_splits=args.n_splits, kind=args.fold_kind)
        fold_of = dict(zip(folds_df["filename"].astype(str), folds_df["fold"].astype(int)))
        file_order = meta_out.drop_duplicates("filename")["filename"].astype(str).tolist()
        n_classes = ssm_probs.shape[1]
        sed_probs = _dump_sed(sed_paths, file_order, fold_of, n_classes)
        if sed_probs.shape != ssm_probs.shape:
            raise SystemExit(
                f"shape mismatch: ssm={ssm_probs.shape}  sed={sed_probs.shape}. "
                "Verify all labeled files have a fold assignment and exist on disk."
            )
        np.savez_compressed(out_dir / "sed_probs.npz", probs=sed_probs)

    print()
    print(f"[dump] wrote files to {out_dir}")
    print(f"[dump]   ssm_probs.npz  shape={ssm_probs.shape}  mean={ssm_probs.mean():.4f}")
    if sed_probs is not None:
        print(f"[dump]   sed_probs.npz  shape={sed_probs.shape}  mean={sed_probs.mean():.4f}")
    print(f"[dump]   y_true.npy     shape={y_true.shape}  positives={int(y_true.sum())}")
    print(f"[dump]   meta.parquet   rows={len(meta_out)}  files={meta_out['filename'].nunique()}")
    print()
    if sed_probs is not None:
        print("Next step — blend weight search:")
        print(f"  python -m birdclef.scripts._08_ensemble \\")
        print(f"      --members {out_dir / 'ssm_probs.npz'} {out_dir / 'sed_probs.npz'} \\")
        print(f"      --y-true  {out_dir / 'y_true.npy'} \\")
        print(f"      --meta    {out_dir / 'meta.parquet'} \\")
        print(f"      --step 0.05")
    else:
        print("Next step — per-class audit:")
        print(f"  python -m birdclef.scripts._08b_per_class_audit --top-k 30")
        print(f"  (add --include-perch to compare against raw Perch baseline)")


if __name__ == "__main__":
    main()
