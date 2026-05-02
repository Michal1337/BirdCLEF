"""Compute stitched-OOF macro AUC for the 5-fold distilled SED ONNX bundle.

ONNX equivalent of `_03b_stitched_oof_sed.py`. The bundle in
`models/sed_kaggle/sed_fold{0..4}.onnx` (Tucker Arrants distilled SED, the
member behind the public LB 0.942 notebook) doesn't have torch checkpoints —
we evaluate it as it deploys, via ONNXRuntime, with the dual-head
aggregation `0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)` and the
sigma=0.65 Gaussian window-smoothing used in cell 26 of LB_0942_seed.ipynb.

Honest OOF: for each labeled file, predict using ONLY the fold whose val set
contains that file (exactly the SED contribution that fold's deployed model
would have made on real test data). Stitched-OOF macro AUC across all
labeled rows is the LB-proxy primary metric.

Mel parameters (must match what the SED bundle was trained with — copied
from the public notebook):
    n_mels=256, n_fft=2048, hop=512, fmin=20, fmax=16000, top_db=80
    per-window standardization: (s - s.mean()) / (s.std() + 1e-6)

Usage:
    python -m birdclef.scripts._03c_eval_sed_kaggle_onnx --n-splits 5
    python -m birdclef.scripts._03c_eval_sed_kaggle_onnx \\
        --onnx-dir models/sed_kaggle --n-splits 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    REPO,
    SOUNDSCAPES,
    SR,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta
from birdclef.data.splits import load_folds
from birdclef.eval.metrics import compute_stage_metrics


# Mel parameters baked into the distilled SED ONNX bundle.
N_MELS_SED = 256
N_FFT_SED = 2048
HOP_SED = 512
FMIN_SED = 20
FMAX_SED = 16000
TOP_DB_SED = 80
SED_SMOOTH_SIGMA = 0.65   # Gaussian window-smoothing across the 12 windows


def _audio_to_mel(chunks: np.ndarray) -> np.ndarray:
    """(N_WINDOWS, WINDOW_SAMPLES) float32 → (N_WINDOWS, 1, n_mels, n_frames) float32."""
    import librosa
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def _file_to_chunks(path: Path) -> np.ndarray:
    """60s file → (N_WINDOWS, WINDOW_SAMPLES) float32, padded/cropped at FILE_SAMPLES."""
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
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


def _predict_file(sess, path: Path) -> np.ndarray:
    """Return (N_WINDOWS, n_classes) probabilities under one fold's ONNX session.

    Aggregation matches the public notebook (cell 26):
        0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)
    Followed by sigma=0.65 Gaussian smoothing across the 12 windows.
    """
    from scipy.ndimage import gaussian_filter1d
    chunks = _file_to_chunks(path)
    mel = _audio_to_mel(chunks)
    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: mel})
    clip_logits = outs[0]              # (N_WINDOWS, n_classes)
    frame_max = outs[1].max(axis=1)    # (N_WINDOWS, n_classes)
    p = 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(frame_max)
    if len(p) > 1:
        p = gaussian_filter1d(p, sigma=SED_SMOOTH_SIGMA, axis=0,
                              mode="nearest").astype(np.float32)
    return p


def _resolve_fold_paths(onnx_dir: Path) -> dict[int, Path]:
    """Map fold idx → onnx path, by parsing `sed_fold{N}.onnx` filenames."""
    out: dict[int, Path] = {}
    for p in sorted(onnx_dir.glob("sed_fold*.onnx")):
        m = re.search(r"sed_fold(\d+)\.onnx$", p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    if not out:
        raise SystemExit(f"No sed_fold*.onnx files found under {onnx_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=str(REPO / "models" / "sed_kaggle"),
                    help="Directory containing sed_fold{0..K-1}.onnx.")
    ap.add_argument("--n-splits", type=int, default=5, choices=[5, 10],
                    help="Static fold parquet to use for fold→files lookup.")
    ap.add_argument("--shuffle-offset", type=int, default=0,
                    help="Leak probe: rotate the fold→ONNX assignment by this "
                         "offset. With K folds, our fold k uses sed_fold{(k+offset) %% K}. "
                         "offset=0 (default) is the honest assignment. offset>0 "
                         "guarantees no ONNX evaluates the val set its filename "
                         "implies. If AUC stays high under offset>0, the per-fold "
                         "assignment was meaningless and the model has seen our val "
                         "files in training (full leak). If AUC drops big, the fold "
                         "assignment was real and the original number is honest OOF.")
    ap.add_argument("--out-json", default=None,
                    help="Override output JSON path (default: <onnx-dir>/stitched_oof_metrics.json, "
                         "or stitched_oof_metrics_shuffle{offset}.json when shuffling)")
    ap.add_argument("--out-npz", default=None,
                    help="Override output NPZ path (default: <onnx-dir>/stitched_oof_probs.npz, "
                         "or stitched_oof_probs_shuffle{offset}.npz when shuffling)")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        raise SystemExit(f"No ONNX dir at {onnx_dir}")
    fold_paths = _resolve_fold_paths(onnx_dir)
    print(f"[oof-sed-onnx] onnx_dir={onnx_dir}  folds={sorted(fold_paths.keys())}")

    K = len(fold_paths)
    offset = int(args.shuffle_offset) % K
    if offset != 0:
        print(f"[oof-sed-onnx] LEAK PROBE: shuffle-offset={offset} (mod {K}) — "
              f"each fold will be evaluated by a DIFFERENT ONNX")
        for k in sorted(fold_paths.keys()):
            mapped = (k + offset) % K
            print(f"[oof-sed-onnx]   our fold {k}  →  {fold_paths[mapped].name}")
    fold_to_onnx_idx = {k: (k + offset) % K for k in sorted(fold_paths.keys())}

    folds_df = load_folds(n_splits=args.n_splits)
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    idx_map = label_to_idx()
    n_classes = len(idx_map)

    sc = sc.sort_values(["filename", "end_sec"]).reset_index(drop=True)

    fold_of = dict(zip(folds_df["filename"].astype(str), folds_df["fold"].astype(int)))
    sc["fold"] = sc["filename"].astype(str).map(fold_of)
    sc = sc.dropna(subset=["fold"]).reset_index(drop=True)
    sc["fold"] = sc["fold"].astype(int)
    n_pinned_rows = int((sc["fold"] == -1).sum())
    if n_pinned_rows:
        n_pinned_files = sc.loc[sc["fold"] == -1, "filename"].nunique()
        print(f"[oof-sed-onnx] excluding {n_pinned_rows} rows from "
              f"{n_pinned_files} pinned (always-train) files")
        sc = sc[sc["fold"] >= 0].reset_index(drop=True)
    if sc.empty:
        raise SystemExit("No labeled rows have fold assignment — split parquet missing or stale.")

    Y = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lb in lbls:
            j = idx_map.get(lb)
            if j is not None:
                Y[i, j] = 1

    P = np.zeros_like(Y, dtype=np.float32)
    fold_metrics: dict[int, dict] = {}
    n_folds_seen = 0

    for fold in sorted(sc["fold"].unique()):
        if fold not in fold_paths:
            print(f"[oof-sed-onnx] WARN: no ONNX for fold {fold}; "
                  f"those val rows will stay 0")
            continue
        onnx_idx = fold_to_onnx_idx.get(int(fold), int(fold))
        if onnx_idx not in fold_paths:
            print(f"[oof-sed-onnx] WARN: shuffle mapped fold {fold} → ONNX idx "
                  f"{onnx_idx} which doesn't exist; skipping")
            continue
        sess = _make_session(fold_paths[onnx_idx])
        n_folds_seen += 1

        sub = sc[sc["fold"] == fold]
        files = sub.drop_duplicates("filename")["filename"].astype(str).tolist()
        print(f"[oof-sed-onnx] fold {fold}: {len(files)} files, "
              f"onnx={fold_paths[onnx_idx].name}")
        fold_row_idxs: list[int] = []
        for fn in files:
            row_idx = sub.index[sub["filename"] == fn].to_numpy()
            if len(row_idx) == 0:
                continue
            probs = _predict_file(sess, SOUNDSCAPES / fn)
            n = min(len(row_idx), N_WINDOWS)
            P[row_idx[:n]] = probs[:n]
            fold_row_idxs.extend(row_idx[:n].tolist())

        if fold_row_idxs:
            fr = np.asarray(fold_row_idxs, dtype=int)
            fold_meta = sc.iloc[fr][["site", "hour_utc"]].reset_index(drop=True)
            fold_metrics[int(fold)] = compute_stage_metrics(Y[fr], P[fr], fold_meta)
            print(f"[oof-sed-onnx]   fold {fold} macro_auc = "
                  f"{fold_metrics[int(fold)].get('macro_auc', float('nan')):.4f}")

        del sess

    if n_folds_seen == 0:
        raise SystemExit(f"No fold ONNX files matched fold parquet under {onnx_dir}")

    meta_for_metrics = sc[["site", "hour_utc"]].reset_index(drop=True)
    stitched = compute_stage_metrics(Y, P, meta_for_metrics)

    fold_aucs = [m.get("macro_auc", float("nan")) for m in fold_metrics.values()]
    fold_aucs = [a for a in fold_aucs if not np.isnan(a)]
    mean_oof_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    macro_auc = float(stitched.get("macro_auc", float("nan")))
    drift = (mean_oof_auc - macro_auc) if not (np.isnan(mean_oof_auc) or np.isnan(macro_auc)) else float("nan")

    summary = {
        # Primary = mean of per-fold AUCs (LB-proxy: each fold = one
        # deployed-style model on one held-out test set).
        "primary": mean_oof_auc,
        "mean_oof_auc": mean_oof_auc,
        # Diagnostic = stitched AUC. If `mean_oof_auc - macro_auc` is large
        # (>0.02), per-fold calibration drift; deployed model wouldn't have it.
        "macro_auc": macro_auc,
        "calibration_drift": drift,
        "site_auc_std": float(stitched.get("site_auc_std", float("nan"))),
        "rare_auc": stitched.get("rare_auc", float("nan")),
        "frequent_auc": stitched.get("frequent_auc", float("nan")),
        "per_fold": {str(k): v for k, v in fold_metrics.items()},
        "stitched_full": stitched,
        "n_folds_seen": n_folds_seen,
        "n_splits": int(args.n_splits),
        "onnx_dir": str(onnx_dir),
        "fold_files": {str(k): str(v.name) for k, v in sorted(fold_paths.items())},
        "shuffle_offset": int(offset),
        "fold_to_onnx_assignment": {
            str(k): fold_paths[v].name for k, v in sorted(fold_to_onnx_idx.items())
        },
        "mel_params": dict(
            n_mels=N_MELS_SED, n_fft=N_FFT_SED, hop=HOP_SED,
            fmin=FMIN_SED, fmax=FMAX_SED, top_db=TOP_DB_SED,
            smooth_sigma=SED_SMOOTH_SIGMA,
        ),
    }

    suffix = f"_shuffle{offset}" if offset != 0 else ""
    out_json = Path(args.out_json) if args.out_json else (
        onnx_dir / f"stitched_oof_metrics{suffix}.json"
    )
    out_npz = Path(args.out_npz) if args.out_npz else (
        onnx_dir / f"stitched_oof_probs{suffix}.npz"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(out_npz, probs=P, y_true=Y)

    print()
    if offset != 0:
        print(f"[oof-sed-onnx] *** LEAK PROBE shuffle-offset={offset} ***")
    print(f"[oof-sed-onnx] onnx_dir={onnx_dir}  n_splits={args.n_splits}  "
          f"folds_seen={n_folds_seen}")
    print(f"[oof-sed-onnx] primary (mean_oof_auc) = {mean_oof_auc:.4f}")
    print(f"[oof-sed-onnx] macro_auc (stitched)   = {macro_auc:.4f}")
    print(f"[oof-sed-onnx] calibration_drift      = {drift:+.4f}  "
          f"({'OK' if abs(drift) < 0.02 else 'investigate post-proc'})")
    print(f"[oof-sed-onnx] site_auc_std           = {summary['site_auc_std']:.4f}")
    print(f"[oof-sed-onnx] wrote {out_json}")
    print(f"[oof-sed-onnx] wrote {out_npz}")


if __name__ == "__main__":
    main()
