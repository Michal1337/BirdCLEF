"""Entropy / confidence probe — does Tucker's SED memorize unlabeled soundscapes too?

Builds on the earlier full-leak finding (`_03c --shuffle-offset`) which
showed Tucker's bundle has trained on every labeled soundscape file. The
question this script answers: did Tucker ALSO train on the unlabeled
soundscape files (e.g. via pseudo-labels), or does the bundle generalize
to those?

Method: run the 5-fold ONNX bundle on every soundscape file in the Perch
cache (~10,658 files), aggregate per-window probabilities via the
canonical dual-head + fold-mean recipe, then compute per-file confidence
stats and compare the distributions of labeled vs unlabeled files.

Memorization fingerprint (a model that has seen a file in training):
  - very high max_prob per window (~1.0)
  - very low mean_prob across 234 classes (~0.01) — sparse
  - very low binary-entropy summed across classes (bimodal at 0/1)

Generalization fingerprint (model has NOT seen file):
  - moderate max_prob (~0.7–0.9 even for present classes)
  - higher mean_prob (uncertain across classes)
  - higher entropy

Reading the result:
  - **labeled << unlabeled** (labeled more confident): Tucker only saw labeled.
    Unlabeled files give a generalization signal — round-2 pseudo on those is
    a clean teacher.
  - **labeled ≈ unlabeled** (similar confidence): Tucker also memorized
    unlabeled (probably via pseudo-train). Pseudo round 2 from this teacher
    leaks Tucker's training signal back into our students.
  - **labeled > unlabeled** (unlabeled MORE confident than labeled): unlikely;
    would suggest a dataset / index bug.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._03f_kaggle_sed_entropy_probe
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._03f_kaggle_sed_entropy_probe \\
        --providers CUDAExecutionProvider CPUExecutionProvider --max-files 200

Outputs `<onnx-dir>/entropy_probe_metrics.json` with the summary table
plus `<onnx-dir>/entropy_probe_perfile.npz` with the full per-file
confidence stats for downstream histogram plots.
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
    PERCH_META,
    REPO,
    SOUNDSCAPES,
    SR,
    WINDOW_SAMPLES,
)


# Tucker's mel parameters — must match `_03c_eval_sed_kaggle_onnx.py`.
_N_MELS = 256
_N_FFT = 2048
_HOP = 512
_FMIN = 20
_FMAX = 16000
_TOP_DB = 80
_SMOOTH_SIGMA = 0.65


def _audio_to_mel(chunks: np.ndarray) -> np.ndarray:
    import librosa
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=_N_FFT, hop_length=_HOP,
            n_mels=_N_MELS, fmin=_FMIN, fmax=_FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=_TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Per-element binary entropy, summed over the class axis.

    Input p shape: (..., n_classes) probabilities in [0, 1].
    Returns: (...) float64 — sum_c [-p_c log p_c - (1-p_c) log(1-p_c)].
    """
    eps = 1e-9
    pp = np.clip(p, eps, 1.0 - eps)
    h = -(pp * np.log(pp) + (1.0 - pp) * np.log(1.0 - pp))
    return h.sum(axis=-1)


def _make_session(path: Path, providers):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=list(providers))


def _predict_one_file(audio: np.ndarray, sessions) -> np.ndarray:
    """Run all SED ONNX folds on a 60s waveform, dual-head + fold-mean.
    Returns (N_WINDOWS, n_classes) probabilities, smoothed across windows.
    """
    from scipy.ndimage import gaussian_filter1d
    if audio.shape[0] < FILE_SAMPLES:
        audio = np.pad(audio, (0, FILE_SAMPLES - audio.shape[0]))
    else:
        audio = audio[:FILE_SAMPLES]
    chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
    mel = _audio_to_mel(chunks)
    p_sum = None
    for sess in sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]              # (N_WINDOWS, n_classes)
        frame_max = outs[1].max(axis=1)    # (N_WINDOWS, n_classes)
        p_fold = 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(frame_max)
        p_sum = p_fold if p_sum is None else (p_sum + p_fold)
    p_mean = (p_sum / max(1, len(sessions))).astype(np.float32)
    if len(p_mean) > 1:
        p_mean = gaussian_filter1d(p_mean, sigma=_SMOOTH_SIGMA, axis=0,
                                   mode="nearest").astype(np.float32)
    return p_mean


def _summarize(group: np.ndarray, name: str) -> dict:
    """Per-metric mean/std/percentiles for one group of files."""
    if len(group) == 0:
        return {"n": 0}
    pct = np.percentile(group, [5, 25, 50, 75, 95])
    return {
        "n": int(len(group)),
        "mean": float(np.mean(group)),
        "std": float(np.std(group)),
        "p05": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
    }


def _ks_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Kolmogorov-Smirnov 2-sample test. Returns statistic + p-value."""
    try:
        from scipy.stats import ks_2samp
        s, p = ks_2samp(a, b)
        return {"ks_statistic": float(s), "ks_pvalue": float(p)}
    except Exception:
        return {"ks_statistic": float("nan"), "ks_pvalue": float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path,
                    default=REPO / "models" / "sed_kaggle",
                    help="Directory containing sed_fold{0..K-1}.onnx.")
    ap.add_argument("--providers", nargs="+",
                    default=["CPUExecutionProvider"],
                    help="ONNXRuntime providers, e.g. "
                         "'CUDAExecutionProvider CPUExecutionProvider' for hopper.")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Cap files per group for a quick smoke test. 0 = all files.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Default: <onnx-dir>/entropy_probe_metrics.json")
    ap.add_argument("--out-npz", type=Path, default=None,
                    help="Default: <onnx-dir>/entropy_probe_perfile.npz")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        raise SystemExit(f"No ONNX dir at {onnx_dir}")

    fold_paths = sorted(
        onnx_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    if not fold_paths:
        raise SystemExit(f"No sed_fold*.onnx files found in {onnx_dir}")
    print(f"[entropy] onnx_dir={onnx_dir}  folds={[p.name for p in fold_paths]}")
    print(f"[entropy] providers={args.providers}")

    sessions = [_make_session(p, args.providers) for p in fold_paths]

    print(f"[entropy] loading Perch cache meta from {PERCH_META}")
    meta = pd.read_parquet(PERCH_META)
    if "is_labeled" not in meta.columns:
        raise SystemExit("Perch meta has no 'is_labeled' column")
    if "filename" not in meta.columns:
        raise SystemExit("Perch meta has no 'filename' column")

    # Cache rows are 12-per-file consecutive — group by filename, preserving
    # cache order. Each file is either fully labeled or fully unlabeled.
    file_rows: dict[str, np.ndarray] = {}
    file_labeled: dict[str, bool] = {}
    for i, row in meta.iterrows():
        fn = str(row["filename"])
        if fn not in file_rows:
            file_rows[fn] = []
            file_labeled[fn] = bool(row["is_labeled"])
        file_rows[fn].append(int(i))
    files = list(file_rows.keys())
    n_files = len(files)
    print(f"[entropy] total soundscape files: {n_files:,}  "
          f"labeled: {sum(file_labeled.values()):,}  "
          f"unlabeled: {n_files - sum(file_labeled.values()):,}")

    if args.max_files and args.max_files > 0:
        # Take first N labeled + first N unlabeled (preserves ordering)
        labs = [f for f in files if file_labeled[f]][: int(args.max_files)]
        unls = [f for f in files if not file_labeled[f]][: int(args.max_files)]
        files = labs + unls
        print(f"[entropy] --max-files={args.max_files} → using "
              f"{len(labs)} labeled + {len(unls)} unlabeled = {len(files)} files")

    # Per-file confidence stats — averaged across the file's 12 windows.
    max_prob = np.zeros(len(files), dtype=np.float32)
    top3_mean = np.zeros(len(files), dtype=np.float32)
    mean_prob = np.zeros(len(files), dtype=np.float32)
    entropy = np.zeros(len(files), dtype=np.float32)
    is_labeled = np.zeros(len(files), dtype=bool)

    t0 = time.time()
    for fi, fn in enumerate(files):
        is_labeled[fi] = file_labeled[fn]
        path = SOUNDSCAPES / fn
        try:
            y, _sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[entropy] WARN: failed to read {fn}: {e}; skipping")
            continue
        if y.ndim == 2:
            y = y.mean(axis=1)
        probs = _predict_one_file(y, sessions)   # (12, 234)
        # Per-window stats, then average over the 12 windows.
        per_win_max = probs.max(axis=1)                          # (12,)
        per_win_top3 = np.sort(probs, axis=1)[:, -3:].mean(axis=1)
        per_win_mean = probs.mean(axis=1)                        # (12,)
        per_win_ent = _binary_entropy(probs)                     # (12,)
        max_prob[fi] = float(per_win_max.mean())
        top3_mean[fi] = float(per_win_top3.mean())
        mean_prob[fi] = float(per_win_mean.mean())
        entropy[fi] = float(per_win_ent.mean())

        if fi == 0 or (fi + 1) % 100 == 0 or (fi + 1) == len(files):
            elapsed = (time.time() - t0) / 60.0
            rate = (fi + 1) / max(1e-6, time.time() - t0)
            eta = (len(files) - fi - 1) / max(1e-6, rate) / 60.0
            print(f"[entropy] {fi + 1}/{len(files)}  "
                  f"elapsed={elapsed:.1f}m  rate={rate:.1f} files/s  eta={eta:.1f}m")

    lab_mask = is_labeled
    unl_mask = ~is_labeled

    summary: dict = {
        "n_files_total": int(len(files)),
        "n_labeled": int(lab_mask.sum()),
        "n_unlabeled": int(unl_mask.sum()),
        "metrics": {},
    }
    print()
    print(f"{'metric':<14} {'group':<10} {'n':>6} {'mean':>8} {'std':>8} "
          f"{'p05':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    print("-" * 88)
    for name, arr in [
        ("max_prob", max_prob),
        ("top3_mean", top3_mean),
        ("mean_prob", mean_prob),
        ("entropy", entropy),
    ]:
        lab_summary = _summarize(arr[lab_mask], "labeled")
        unl_summary = _summarize(arr[unl_mask], "unlabeled")
        ks = _ks_test(arr[lab_mask], arr[unl_mask])
        summary["metrics"][name] = {
            "labeled": lab_summary,
            "unlabeled": unl_summary,
            "labeled_minus_unlabeled_mean": (
                float(lab_summary.get("mean", 0.0) - unl_summary.get("mean", 0.0))
                if lab_summary.get("n", 0) and unl_summary.get("n", 0)
                else float("nan")
            ),
            **ks,
        }
        for grp_name, s in [("labeled", lab_summary), ("unlabeled", unl_summary)]:
            if s.get("n", 0) == 0:
                continue
            print(f"{name:<14} {grp_name:<10} {s['n']:>6} "
                  f"{s['mean']:>8.4f} {s['std']:>8.4f} "
                  f"{s['p05']:>8.4f} {s['p25']:>8.4f} "
                  f"{s['p50']:>8.4f} {s['p75']:>8.4f} {s['p95']:>8.4f}")
        print(f"{'':<14} {'KS':<10} stat={ks['ks_statistic']:.4f}  "
              f"p={ks['ks_pvalue']:.2e}  "
              f"Δmean(lab−unl)={summary['metrics'][name]['labeled_minus_unlabeled_mean']:+.4f}")
        print()

    # Verdict heuristic on max_prob: labeled > unlabeled by > ~0.05 mean and
    # KS p < 1e-6 = strong memorization-only-on-labeled signal.
    mp = summary["metrics"]["max_prob"]
    delta = mp.get("labeled_minus_unlabeled_mean", float("nan"))
    ks_p = mp.get("ks_pvalue", 1.0)
    if not np.isnan(delta) and not np.isnan(ks_p):
        print("=" * 88)
        if delta > 0.05 and ks_p < 1e-6:
            print(f"[verdict] LIKELY: Tucker memorized labeled but not unlabeled "
                  f"(Δmax_prob lab−unl = {delta:+.3f}, KS p = {ks_p:.2e}). "
                  f"Pseudo round 2 on unlabeled = clean teacher signal.")
        elif abs(delta) < 0.02 and ks_p > 0.01:
            print(f"[verdict] LIKELY: Tucker memorized BOTH labeled and unlabeled "
                  f"(Δmax_prob lab−unl = {delta:+.3f}, KS p = {ks_p:.2e}). "
                  f"Pseudo round 2 leaks Tucker's training signal into students.")
        else:
            print(f"[verdict] AMBIGUOUS: Δmax_prob lab−unl = {delta:+.3f}, "
                  f"KS p = {ks_p:.2e}. Inspect histograms manually.")

    out_json = Path(args.out_json) if args.out_json else (
        onnx_dir / "entropy_probe_metrics.json"
    )
    out_npz = Path(args.out_npz) if args.out_npz else (
        onnx_dir / "entropy_probe_perfile.npz"
    )
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(
        out_npz,
        filename=np.array(files, dtype=object),
        is_labeled=is_labeled,
        max_prob=max_prob, top3_mean=top3_mean,
        mean_prob=mean_prob, entropy=entropy,
    )
    print()
    print(f"[entropy] wrote {out_json}")
    print(f"[entropy] wrote {out_npz}")


if __name__ == "__main__":
    main()
