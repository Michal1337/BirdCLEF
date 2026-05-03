"""Evaluate a SED ONNX ensemble on labeled soundscapes.

Designed for full-data multi-seed bundles (no fold structure): every model
predicts on every labeled file, predictions are averaged across the
ensemble, then scored. The macro AUC reflects the **deployed** ensemble's
in-sample behavior — comparable to Tucker's leaked 0.985 number and to our
fold-aware stitched OOF (0.972), but produced by the actual artifact you'd
ship to LB.

Use cases:
  1. **Full-data 5-seed sanity check.** After training 5 models with
     `--fold -1` on different seeds, this verifies all 5 trained
     successfully (collapse → ensemble AUC ≈ 0.5) and gives the
     in-sample-fit ceiling the LB will draw from.
  2. **Per-model variance analysis.** Reports each individual model's
     AUC alongside the ensemble; if one seed is an outlier (e.g.
     0.85 vs others at 0.99), it's pulling down the ensemble.
  3. **Cross-bundle comparison.** Pointed at Tucker's bundle, gives the
     ensemble-on-labeled number for apples-to-apples vs ours.

Note: when the bundle was trained fold-aware (different held-out fold per
model), this script's "ensemble in-sample" number is partially leaked —
fold-0 files were SEEN by 4 of 5 models. Useful as a deployment proxy but
not honest OOF. For honest OOF on fold-aware bundles, use
`_03b_stitched_oof_sed.py`.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._03g_eval_sed_ensemble \\
        --onnx-dir models/sed_inhouse_fulldata
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    REPO,
    SOUNDSCAPES,
    SR,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta
from birdclef.eval.metrics import compute_stage_metrics


# Tucker / our defaults — both bundles use the same mel parameters
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000
TOP_DB = 80
SED_SMOOTH_SIGMA = 0.65


def _audio_to_mel(chunks: np.ndarray) -> np.ndarray:
    """(N_WINDOWS, WINDOW_SAMPLES) → (N_WINDOWS, 1, n_mels, n_frames)."""
    import librosa
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def _file_to_chunks(path: Path) -> np.ndarray:
    y, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    else:
        y = y[:FILE_SAMPLES]
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def _make_session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _predict_one_session(sess, wins: np.ndarray) -> np.ndarray:
    """Run one ONNX session on the 12 windows of a single 60s file. Auto-detects
    waveform-input (rank 2) vs mel-input (rank 4) and applies dual-head agg.
    Returns (N_WINDOWS, n_classes) probabilities, smoothed across windows.
    """
    from scipy.ndimage import gaussian_filter1d
    inp = sess.get_inputs()[0]
    inp_rank = len(inp.shape)
    if inp_rank == 2:
        x = wins.astype(np.float32)
    elif inp_rank == 4:
        x = _audio_to_mel(wins)
    else:
        raise SystemExit(f"Unsupported ONNX input rank {inp_rank}")
    outs = sess.run(None, {inp.name: x})
    if len(outs) >= 2:
        clip_logits = outs[0]
        fw_max = outs[1].max(axis=1)
        p = 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(fw_max)
    else:
        p = _sigmoid(outs[0])
    if p.shape[0] > 1:
        p = gaussian_filter1d(p, sigma=SED_SMOOTH_SIGMA, axis=0,
                              mode="nearest").astype(np.float32)
    return p.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path,
                    default=REPO / "models" / "sed_inhouse_fulldata",
                    help="Dir containing sed_fold{0..K-1}.onnx (any K).")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Default: <onnx-dir>/ensemble_eval.json")
    ap.add_argument("--out-npz", type=Path, default=None,
                    help="Default: <onnx-dir>/ensemble_probs.npz")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        raise SystemExit(f"No ONNX dir at {onnx_dir}")
    fold_paths = sorted(
        onnx_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    if not fold_paths:
        raise SystemExit(f"No sed_fold*.onnx files in {onnx_dir}")
    print(f"[ens-eval] onnx_dir={onnx_dir}")
    print(f"[ens-eval] models: {[p.name for p in fold_paths]}")

    sessions = [_make_session(p) for p in fold_paths]

    # All labeled soundscape rows (in canonical (filename, end_sec) order)
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    sc = sc.sort_values(["filename", "end_sec"]).reset_index(drop=True)
    idx_map = label_to_idx()
    n_classes = len(idx_map)
    files = sc.drop_duplicates("filename")["filename"].astype(str).tolist()
    print(f"[ens-eval] labeled files: {len(files)}  rows: {len(sc):,}  classes: {n_classes}")

    # Build GT matrix aligned to sc row order
    Y = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lb in lbls:
            j = idx_map.get(lb)
            if j is not None:
                Y[i, j] = 1

    # Prediction matrix per model (M x N x C) → averaged → ensemble probs
    n_models = len(sessions)
    P_per_model = np.zeros((n_models, len(sc), n_classes), dtype=np.float32)

    t0 = time.time()
    for fi, fn in enumerate(files):
        path = SOUNDSCAPES / fn
        try:
            wins = _file_to_chunks(path)
        except Exception as e:
            print(f"[ens-eval] WARN: failed to read {fn}: {e}; leaving zeros")
            continue
        # Find the row indices in sc for this file (12 consecutive rows)
        row_idx = sc.index[sc["filename"] == fn].to_numpy()
        if len(row_idx) == 0:
            continue
        n = min(len(row_idx), N_WINDOWS)
        for mi, sess in enumerate(sessions):
            probs = _predict_one_session(sess, wins)
            P_per_model[mi, row_idx[:n]] = probs[:n]
        if fi == 0 or (fi + 1) % 10 == 0 or (fi + 1) == len(files):
            elapsed = (time.time() - t0) / 60.0
            print(f"[ens-eval] {fi + 1}/{len(files)} files  elapsed={elapsed:.1f}m")

    # Ensemble = mean across models
    P_ensemble = P_per_model.mean(axis=0)

    # Per-model + ensemble macro AUC
    eval_meta = sc[["site", "hour_utc"]].reset_index(drop=True)
    per_model_metrics = []
    for mi in range(n_models):
        m = compute_stage_metrics(Y, P_per_model[mi], eval_meta)
        per_model_metrics.append({
            "model": fold_paths[mi].name,
            "macro_auc": float(m.get("macro_auc", float("nan"))),
            "site_auc_std": float(m.get("site_auc_std", float("nan"))),
            "rare_auc": m.get("rare_auc", float("nan")),
            "frequent_auc": m.get("frequent_auc", float("nan")),
        })

    ens_metrics = compute_stage_metrics(Y, P_ensemble, eval_meta)

    # Cross-model agreement: mean Spearman across pairs (proxy for diversity)
    from scipy.stats import spearmanr
    if n_models >= 2:
        # Subsample to 5k rows for speed
        rng = np.random.default_rng(42)
        n_sample = min(5000, P_per_model.shape[1])
        idx = rng.choice(P_per_model.shape[1], n_sample, replace=False)
        flat = P_per_model[:, idx, :].reshape(n_models, -1)
        pair_corrs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                rho, _ = spearmanr(flat[i], flat[j])
                pair_corrs.append(float(rho))
        spearman_mean = float(np.mean(pair_corrs))
        spearman_min = float(np.min(pair_corrs))
        spearman_max = float(np.max(pair_corrs))
    else:
        spearman_mean = spearman_min = spearman_max = float("nan")

    summary = {
        "onnx_dir": str(onnx_dir),
        "models": [str(p.name) for p in fold_paths],
        "n_models": n_models,
        "n_files": int(len(files)),
        "n_rows": int(len(sc)),
        "n_classes": int(n_classes),
        "ensemble_macro_auc": float(ens_metrics.get("macro_auc", float("nan"))),
        "ensemble_site_auc_std": float(ens_metrics.get("site_auc_std", float("nan"))),
        "ensemble_rare_auc": ens_metrics.get("rare_auc", float("nan")),
        "ensemble_frequent_auc": ens_metrics.get("frequent_auc", float("nan")),
        "per_model": per_model_metrics,
        "per_model_macro_auc_mean": float(np.mean(
            [m["macro_auc"] for m in per_model_metrics if not np.isnan(m["macro_auc"])]
        )) if per_model_metrics else float("nan"),
        "per_model_macro_auc_std": float(np.std(
            [m["macro_auc"] for m in per_model_metrics if not np.isnan(m["macro_auc"])]
        )) if per_model_metrics else float("nan"),
        "ensemble_lift_over_mean": float(
            ens_metrics.get("macro_auc", float("nan")) -
            np.mean([m["macro_auc"] for m in per_model_metrics if not np.isnan(m["macro_auc"])])
        ) if per_model_metrics else float("nan"),
        "spearman_mean_across_pairs": spearman_mean,
        "spearman_range": [spearman_min, spearman_max],
    }

    print()
    print("=" * 72)
    print("Per-model AUC (in-sample on labeled):")
    print(f"{'model':<30}{'macro_auc':>12}{'site_std':>12}")
    print("-" * 54)
    for m in per_model_metrics:
        print(f"{m['model']:<30}{m['macro_auc']:>12.4f}{m['site_auc_std']:>12.4f}")
    print()
    print(f"{'mean per-model macro_auc:':<30}{summary['per_model_macro_auc_mean']:>12.4f}  "
          f"(std={summary['per_model_macro_auc_std']:.4f})")
    print()
    print("=" * 72)
    print(f"ENSEMBLE macro_auc       = {summary['ensemble_macro_auc']:.4f}")
    print(f"ENSEMBLE site_auc_std    = {summary['ensemble_site_auc_std']:.4f}")
    print(f"ensemble lift over mean  = {summary['ensemble_lift_over_mean']:+.4f}  "
          f"(positive = ensemble exploits diversity)")
    print(f"Spearman across models   = {spearman_mean:.4f}  "
          f"(range: {spearman_min:.4f} – {spearman_max:.4f})  "
          f"(lower = more diverse predictions)")
    print()

    # Diagnostic verdicts
    aucs = [m["macro_auc"] for m in per_model_metrics if not np.isnan(m["macro_auc"])]
    if any(a < 0.7 for a in aucs):
        print("[verdict] AT LEAST ONE MODEL FAILED — macro_auc < 0.7. "
              "Likely mode collapse on that seed; investigate train_history.jsonl.")
    elif summary["per_model_macro_auc_std"] > 0.02:
        print("[verdict] Per-model AUC std > 0.02 — high seed variance. "
              "Worth re-running outliers with different seed before LB submission.")
    else:
        print("[verdict] All models look healthy; ensemble ready for deployment.")

    out_json = Path(args.out_json) if args.out_json else (onnx_dir / "ensemble_eval.json")
    out_npz = Path(args.out_npz) if args.out_npz else (onnx_dir / "ensemble_probs.npz")
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(out_npz, probs=P_ensemble, y_true=Y,
                        per_model_probs=P_per_model)
    print(f"[ens-eval] wrote {out_json}")
    print(f"[ens-eval] wrote {out_npz}")


if __name__ == "__main__":
    main()
