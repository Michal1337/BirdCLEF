"""Pseudo-label orchestrator — runs a teacher over all soundscapes and writes
soft-target probabilities that the SED student consumes via
`SEDTrainDataset(pseudo_round=N)`.

Output layout — `cache/pseudo/round{N}/`:
    probs.npz     : {'final': (N_rows, C) float32,
                     'first_pass': (N_rows, C) float32 — optional,
                     'keep_mask': (N_rows, C) uint8}
    meta.parquet  : row_id, filename, window, is_labeled
                    (same row ordering as the Perch cache meta)
    info.json     : teacher name, seeds, confidence filter, coverage stats

Three teacher paths:
  - SSM pipeline ensemble (round-0 default): trains the full proto+MLP+residual
    stack on ALL labeled rows, once per seed, and runs Perch-cache
    inference on all 10 658 soundscape files. Averages across seeds → final
    pseudo-label. Doesn't require a trained SED yet.
  - SED checkpoint ensemble: loads one or more SED `best.pt` files (PyTorch)
    and averages their predictions over the same soundscapes. Use once you
    have a student stronger than the SSM teacher.
  - **SSM + SED ONNX blend** (round-2 default after the public LB 0.942
    notebook): runs the SSM ensemble path AND the 5-fold distilled SED
    ONNX bundle (`models/sed_kaggle/sed_fold{0..4}.onnx`), then blends
    probabilities at `blend_weight * SSM + (1 - blend_weight) * SED`.
    NOTE: this is a probability-weighted blend, NOT the rank-blend used
    for LB submission. Reason: pseudo-label downstream applies a
    confidence threshold τ on the saved `final` probs, which requires
    real probability semantics. Rank-blend percentiles would break the
    τ filter (50th percentile != "more likely than not"). Architectural
    diversity is preserved either way; the blend-weight is what differs.

All three paths share the same output schema so the downstream consumer
doesn't care which teacher produced round N.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    PERCH_META,
    PERCH_NPZ,
    PSEUDO_DIR,
    SOUNDSCAPES,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import primary_labels
from birdclef.utils.seed import seed_everything


def _round_dir(rnd: int) -> Path:
    d = PSEUDO_DIR / f"round{rnd}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _apply_confidence_filter(
    probs: np.ndarray,
    confidence_tau: float,
    topk_per_species: int,
) -> np.ndarray:
    """Build a {0,1} uint8 mask of positions the student should supervise on."""
    keep_mask = np.zeros_like(probs, dtype=np.uint8)
    if confidence_tau > 0:
        keep_mask = (probs >= confidence_tau).astype(np.uint8)
    if topk_per_species > 0:
        n_files = probs.shape[0] // N_WINDOWS
        n_classes = probs.shape[1]
        view = probs.reshape(n_files, N_WINDOWS, n_classes)
        k = min(topk_per_species, N_WINDOWS)
        topk_idx = np.argpartition(-view, kth=k - 1, axis=1)[:, :k]
        km = np.zeros_like(view, dtype=np.uint8)
        for f in range(n_files):
            for c in range(n_classes):
                km[f, topk_idx[f, :, c], c] = 1
        keep_mask = np.maximum(keep_mask, km.reshape(n_files * N_WINDOWS, n_classes))
    return keep_mask


@torch.no_grad()
def _predict_sed_on_file(model, wav_60s: np.ndarray, device) -> np.ndarray:
    """Returns 12xC probabilities for a 60s waveform."""
    if wav_60s.shape[0] < FILE_SAMPLES:
        wav_60s = np.pad(wav_60s, (0, FILE_SAMPLES - wav_60s.shape[0]))
    else:
        wav_60s = wav_60s[:FILE_SAMPLES]
    wins = wav_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
    x = torch.from_numpy(wins.astype(np.float32)).to(device)
    logits = model(x)
    return torch.sigmoid(logits).cpu().numpy()


def pseudo_label_with_sed(
    checkpoints: List[Path], output_round: int,
    confidence_tau: float = 0.5, topk_per_species: int = 0,
) -> None:
    from birdclef.models.sed import SED, SEDConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = primary_labels()
    n_classes = len(labels)
    out_dir = _round_dir(output_round)

    models = []
    for ckpt in checkpoints:
        state = torch.load(ckpt, map_location="cpu")
        cfg = state["cfg"]
        sed_cfg = SEDConfig(
            backbone=cfg["backbone"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
            sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], f_min=cfg["f_min"], f_max=cfg["f_max"],
        )
        m = SED(sed_cfg).to(device)
        # Load full state_dict first so BN running buffers come from training,
        # then overwrite trainable params with the EMA shadow if present.
        # (EMA only contains parameters; using EMA-only loses the BN buffers
        # and produces NaN activations at inference time.)
        m.load_state_dict(state["state_dict"], strict=False)
        ema_shadow = state.get("ema")
        if ema_shadow:
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n in ema_shadow:
                        p.data.copy_(ema_shadow[n].to(p.device))
        m.eval()
        models.append(m)

    paths = sorted(SOUNDSCAPES.glob("*.ogg"))
    meta_rows = []
    probs_rows = []
    for p in tqdm(paths, desc="pseudo[sed]"):
        try:
            y, _sr = sf.read(str(p), dtype="float32", always_2d=False)
        except Exception:
            continue
        if y.ndim == 2:
            y = y.mean(axis=1)
        ensemble = np.zeros((N_WINDOWS, n_classes), dtype=np.float32)
        for m in models:
            ensemble += _predict_sed_on_file(m, y, device)
        ensemble /= max(1, len(models))
        for w in range(N_WINDOWS):
            meta_rows.append({
                "row_id": f"{p.stem}_{(w + 1) * 5}",
                "filename": p.name,
                "window": w,
            })
        probs_rows.append(ensemble)
    meta = pd.DataFrame(meta_rows)
    probs = np.concatenate(probs_rows, axis=0).astype(np.float32)
    keep_mask = _apply_confidence_filter(probs, confidence_tau, topk_per_species)

    np.savez_compressed(out_dir / "probs.npz", probs=probs, keep_mask=keep_mask)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    (out_dir / "info.json").write_text(
        json.dumps({
            "teacher": "sed_ckpt",
            "n_checkpoints": len(checkpoints),
            "confidence_tau": confidence_tau,
            "topk_per_species": topk_per_species,
            "n_files": int(probs.shape[0] // N_WINDOWS),
            "keep_fraction": float(keep_mask.mean()),
        }, indent=2), encoding="utf-8",
    )
    print(f"[pseudo] round {output_round} written to {out_dir}")


def pseudo_label_with_ssm_pipeline(
    teacher_cfg: dict,
    seeds: Sequence[int],
    output_round: int,
    confidence_tau: float = 0.5,
    topk_per_species: int = 0,
) -> None:
    """Round-0-capable teacher: SSM pipeline as an ensemble over `seeds`.

    Each seed independently (a) trains proto+MLP+residual on ALL labeled
    rows, (b) predicts final-stage probs on every soundscape row in
    the Perch cache (labeled + unlabeled). Probs averaged across seeds.
    V-anchor was abandoned — every labeled file is now training-eligible.
    """
    from birdclef.train.train_ssm_head import (
        PerchCache,
        _lambda_prior_vector,
        _temperature_vector,
        load_perch_cache,
        run_pipeline_for_split,
    )
    from birdclef.data.soundscapes import load_taxonomy

    if len(seeds) == 0:
        raise ValueError("Need at least one seed for the SSM teacher")

    print(f"[pseudo:ssm] loading Perch cache...")
    cache = load_perch_cache()
    # Apply genus-proxy fill at the load stage (matches LB notebook + SSM
    # sweep). Defaults on, set teacher_cfg["use_perch_proxy"] = False to
    # ablate.
    if bool(teacher_cfg.get("use_perch_proxy", True)):
        from birdclef.models.perch import apply_proxy_to_scores
        cache.scores = apply_proxy_to_scores(cache.scores, cache.scores_proxy)
    n_rows, n_classes = cache.scores.shape

    is_labeled = cache.labeled_mask
    train_mask = is_labeled
    train_idx = np.where(train_mask)[0]
    all_idx = np.arange(n_rows)
    print(f"[pseudo:ssm] train rows={len(train_idx)} (all labeled), "
          f"predict rows={len(all_idx)} (every soundscape window)")

    labels = primary_labels()
    tax = load_taxonomy()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    temperatures = _temperature_vector(labels, class_map)
    lambda_prior_vec = _lambda_prior_vector(
        labels, class_map,
        lambda_birds=float(teacher_cfg["lambda_prior"]),
        lambda_texture=float(teacher_cfg.get("lambda_prior_texture", teacher_cfg["lambda_prior"])),
    )

    t0 = time.time()
    agg_final = np.zeros((n_rows, n_classes), dtype=np.float64)
    agg_first_pass = np.zeros((n_rows, n_classes), dtype=np.float64)
    for si, seed in enumerate(seeds):
        print(f"[pseudo:ssm] seed {seed} ({si + 1}/{len(seeds)})...")
        seed_everything(int(seed))
        cfg_seed = {**teacher_cfg, "seed": int(seed),
                    "name": f"{teacher_cfg.get('name', 'teacher')}_s{seed}"}
        out = run_pipeline_for_split(cache, train_idx, all_idx, cfg_seed, temperatures,
                                     lambda_prior_vec=lambda_prior_vec)
        agg_final += out["final"].astype(np.float64)
        agg_first_pass += out["first_pass"].astype(np.float64)
    agg_final = (agg_final / len(seeds)).astype(np.float32)
    agg_first_pass = (agg_first_pass / len(seeds)).astype(np.float32)
    elapsed = (time.time() - t0) / 60.0

    keep_mask = _apply_confidence_filter(agg_final, confidence_tau, topk_per_species)

    meta = cache.meta[["row_id", "filename"]].copy()
    meta["window"] = np.tile(np.arange(N_WINDOWS), n_rows // N_WINDOWS)
    meta["is_labeled"] = cache.meta["is_labeled"].astype(int)

    out_dir = _round_dir(output_round)
    np.savez_compressed(
        out_dir / "probs.npz",
        final=agg_final, first_pass=agg_first_pass, keep_mask=keep_mask,
    )
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    info = {
        "teacher": "ssm_pipeline",
        "teacher_name": teacher_cfg.get("name", "unnamed"),
        "seeds": list(int(s) for s in seeds),
        "confidence_tau": float(confidence_tau),
        "topk_per_species": int(topk_per_species),
        "n_files": int(n_rows // N_WINDOWS),
        "n_rows": int(n_rows),
        "n_train_rows": int(len(train_idx)),
        "keep_fraction": float(keep_mask.mean()),
        "runtime_min": round(elapsed, 2),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[pseudo:ssm] round {output_round} written to {out_dir}  "
          f"keep_fraction={info['keep_fraction']:.4f}  "
          f"runtime={elapsed:.1f}m")


# ─── SSM + SED ONNX blend teacher ──────────────────────────────────────


# Mel + smoothing parameters baked into the distilled SED ONNX bundle.
# Must match `_03c_eval_sed_kaggle_onnx.py` and cell 26 of LB_0942_seed.ipynb.
_SED_N_MELS = 256
_SED_N_FFT = 2048
_SED_HOP = 512
_SED_FMIN = 20
_SED_FMAX = 16000
_SED_TOP_DB = 80
_SED_SMOOTH_SIGMA = 0.65


def _make_sed_session(path: Path, providers: Sequence[str] | None = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=so,
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )


def _sed_audio_to_mel(chunks: np.ndarray) -> np.ndarray:
    """(N_WINDOWS, WINDOW_SAMPLES) float32 → (N_WINDOWS, 1, n_mels, n_frames)."""
    import librosa
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=32_000, n_fft=_SED_N_FFT, hop_length=_SED_HOP,
            n_mels=_SED_N_MELS, fmin=_SED_FMIN, fmax=_SED_FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=_SED_TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def _sed_predict_one_file(audio: np.ndarray, sessions) -> np.ndarray:
    """Run the 5-ONNX SED bundle on a single 60s waveform.
    Returns (N_WINDOWS, n_classes) probabilities, smoothed across windows.
    """
    from scipy.ndimage import gaussian_filter1d
    if audio.shape[0] < FILE_SAMPLES:
        audio = np.pad(audio, (0, FILE_SAMPLES - audio.shape[0]))
    else:
        audio = audio[:FILE_SAMPLES]
    chunks = audio.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
    mel = _sed_audio_to_mel(chunks)

    p_sum = None
    for sess in sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]              # (N_WINDOWS, n_classes)
        frame_max = outs[1].max(axis=1)    # (N_WINDOWS, n_classes)
        p_fold = (0.5 * (1.0 / (1.0 + np.exp(-np.clip(clip_logits, -50, 50))))
                  + 0.5 * (1.0 / (1.0 + np.exp(-np.clip(frame_max,   -50, 50)))))
        p_sum = p_fold if p_sum is None else (p_sum + p_fold)
    p_mean = (p_sum / max(1, len(sessions))).astype(np.float32)
    if len(p_mean) > 1:
        p_mean = gaussian_filter1d(
            p_mean, sigma=_SED_SMOOTH_SIGMA, axis=0, mode="nearest",
        ).astype(np.float32)
    return p_mean


def _sed_predict_all_files(
    cache, sed_onnx_dir: Path, providers: Sequence[str] | None = None,
) -> np.ndarray:
    """Predict (n_rows, n_classes) SED probabilities aligned to cache row order.

    Cache rows are organized as 12 consecutive windows per file in
    `cache.meta["filename"]` order, so we walk file-by-file and write to
    a contiguous slice each time.
    """
    import re
    fold_paths = sorted(
        sed_onnx_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    if not fold_paths:
        raise SystemExit(f"No sed_fold*.onnx files found under {sed_onnx_dir}")
    sessions = [_make_sed_session(p, providers) for p in fold_paths]
    print(f"[pseudo:blend] SED bundle: {[p.name for p in fold_paths]}")

    n_rows, n_classes = cache.scores.shape
    if n_rows % N_WINDOWS != 0:
        raise SystemExit(f"Cache rows ({n_rows}) not divisible by N_WINDOWS={N_WINDOWS}")
    n_files = n_rows // N_WINDOWS

    sed_probs = np.zeros((n_rows, n_classes), dtype=np.float32)
    for f_idx in tqdm(range(n_files), desc="pseudo[sed-onnx]"):
        row_start = f_idx * N_WINDOWS
        fn = str(cache.meta.iloc[row_start]["filename"])
        path = SOUNDSCAPES / fn
        try:
            y, _sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[pseudo:blend] WARN: failed to read {fn}: {e}; leaving zeros")
            continue
        if y.ndim == 2:
            y = y.mean(axis=1)
        sed_probs[row_start:row_start + N_WINDOWS] = _sed_predict_one_file(y, sessions)

    return sed_probs


def _ssm_predict_all_rows(
    cache, teacher_cfg: dict, seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run the SSM teacher ensemble; return (final_probs, first_pass_probs)
    aligned to cache row order. Logic factored from
    `pseudo_label_with_ssm_pipeline` so the blend path can re-use it
    without writing intermediate round files.
    """
    from birdclef.train.train_ssm_head import (
        _lambda_prior_vector,
        _temperature_vector,
        run_pipeline_for_split,
    )
    from birdclef.data.soundscapes import load_taxonomy

    n_rows, n_classes = cache.scores.shape
    train_idx = np.where(cache.labeled_mask)[0]
    all_idx = np.arange(n_rows)
    print(f"[pseudo:blend] SSM teacher: train rows={len(train_idx)} "
          f"(all labeled), predict rows={len(all_idx)}")

    labels = primary_labels()
    tax = load_taxonomy()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    temperatures = _temperature_vector(labels, class_map)
    lambda_prior_vec = _lambda_prior_vector(
        labels, class_map,
        lambda_birds=float(teacher_cfg["lambda_prior"]),
        lambda_texture=float(teacher_cfg.get("lambda_prior_texture",
                                             teacher_cfg["lambda_prior"])),
    )

    agg_final = np.zeros((n_rows, n_classes), dtype=np.float64)
    agg_fp = np.zeros((n_rows, n_classes), dtype=np.float64)
    for si, seed in enumerate(seeds):
        print(f"[pseudo:blend] SSM seed {seed} ({si + 1}/{len(seeds)})...")
        seed_everything(int(seed))
        cfg_seed = {**teacher_cfg, "seed": int(seed),
                    "name": f"{teacher_cfg.get('name', 'teacher')}_s{seed}"}
        out = run_pipeline_for_split(cache, train_idx, all_idx, cfg_seed,
                                     temperatures, lambda_prior_vec=lambda_prior_vec)
        agg_final += out["final"].astype(np.float64)
        agg_fp += out["first_pass"].astype(np.float64)

    agg_final = (agg_final / len(seeds)).astype(np.float32)
    agg_fp = (agg_fp / len(seeds)).astype(np.float32)
    return agg_final, agg_fp


def pseudo_label_with_ssm_sed_blend(
    teacher_cfg: dict,
    seeds: Sequence[int],
    sed_onnx_dir: Path,
    output_round: int,
    blend_weight: float = 0.6,
    confidence_tau: float = 0.5,
    topk_per_species: int = 0,
    sed_providers: Sequence[str] | None = None,
) -> None:
    """Round-N teacher: SSM ensemble + 5-fold distilled SED ONNX bundle,
    probability-weighted blended at `blend_weight * SSM + (1-w) * SED`.
    """
    if len(seeds) == 0:
        raise ValueError("Need at least one seed for the SSM teacher")
    if not (0.0 <= blend_weight <= 1.0):
        raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}")

    from birdclef.train.train_ssm_head import load_perch_cache

    print(f"[pseudo:blend] loading Perch cache...")
    cache = load_perch_cache()
    if bool(teacher_cfg.get("use_perch_proxy", True)):
        from birdclef.models.perch import apply_proxy_to_scores
        cache.scores = apply_proxy_to_scores(cache.scores, cache.scores_proxy)
    n_rows, n_classes = cache.scores.shape

    t0 = time.time()
    print(f"[pseudo:blend] === phase 1: SSM ensemble ({len(seeds)} seeds) ===")
    ssm_final, ssm_fp = _ssm_predict_all_rows(cache, teacher_cfg, seeds)
    t_ssm = (time.time() - t0) / 60.0
    print(f"[pseudo:blend] SSM done in {t_ssm:.1f}m")

    t1 = time.time()
    print(f"[pseudo:blend] === phase 2: SED ONNX bundle ({sed_onnx_dir}) ===")
    sed_probs = _sed_predict_all_files(cache, sed_onnx_dir, providers=sed_providers)
    t_sed = (time.time() - t1) / 60.0
    print(f"[pseudo:blend] SED done in {t_sed:.1f}m")

    # Probability-weighted blend (NOT rank-blend — see module docstring).
    blend = (float(blend_weight) * ssm_final
             + (1.0 - float(blend_weight)) * sed_probs).astype(np.float32)
    blend = np.clip(blend, 0.0, 1.0)

    keep_mask = _apply_confidence_filter(blend, confidence_tau, topk_per_species)
    elapsed = (time.time() - t0) / 60.0

    meta = cache.meta[["row_id", "filename"]].copy()
    meta["window"] = np.tile(np.arange(N_WINDOWS), n_rows // N_WINDOWS)
    meta["is_labeled"] = cache.meta["is_labeled"].astype(int)

    out_dir = _round_dir(output_round)
    np.savez_compressed(
        out_dir / "probs.npz",
        final=blend,
        first_pass=ssm_fp,            # SSM-only first-pass (diagnostic)
        ssm_final=ssm_final,          # per-member outputs for re-blending later
        sed_final=sed_probs,
        keep_mask=keep_mask,
    )
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    info = {
        "teacher": "ssm_sed_blend",
        "teacher_name": teacher_cfg.get("name", "unnamed"),
        "seeds": list(int(s) for s in seeds),
        "blend_weight_ssm": float(blend_weight),
        "blend_weight_sed": float(1.0 - blend_weight),
        "blend_kind": "probability_weighted",
        "sed_onnx_dir": str(sed_onnx_dir),
        "sed_providers": list(sed_providers) if sed_providers else ["CPUExecutionProvider"],
        "confidence_tau": float(confidence_tau),
        "topk_per_species": int(topk_per_species),
        "n_files": int(n_rows // N_WINDOWS),
        "n_rows": int(n_rows),
        "keep_fraction": float(keep_mask.mean()),
        "runtime_min_total": round(elapsed, 2),
        "runtime_min_ssm": round(t_ssm, 2),
        "runtime_min_sed": round(t_sed, 2),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[pseudo:blend] round {output_round} written to {out_dir}")
    print(f"[pseudo:blend]   keep_fraction={info['keep_fraction']:.4f}  "
          f"blend_w_ssm={blend_weight:.2f}  total_runtime={elapsed:.1f}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["ssm", "sed", "blend"], default="ssm",
                    help="Which teacher path to use. 'ssm' = SSM ensemble (round-0). "
                         "'sed' = SED checkpoint(s). 'blend' = SSM + SED ONNX bundle "
                         "probability-blend (round-2 default).")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Confidence floor for keep_mask. 0 disables.")
    ap.add_argument("--topk-per-species", type=int, default=0,
                    help="Union with per-file top-k-window mask per species. 0 disables.")
    # SSM teacher options
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Seeds for the SSM teacher ensemble. Default from teacher config.")
    # SED checkpoint teacher options
    ap.add_argument("--ckpts", type=str, nargs="*", default=[],
                    help="For --teacher sed: paths to SED best.pt files (one or more).")
    # Blend teacher options
    ap.add_argument("--sed-onnx-dir", type=str, default=None,
                    help="For --teacher blend: dir containing sed_fold{0..K-1}.onnx. "
                         "Default = REPO/models/sed_kaggle.")
    ap.add_argument("--blend-weight", type=float, default=0.6,
                    help="For --teacher blend: SSM weight; SED weight = 1 - this. "
                         "Default 0.6 matches the public LB 0.942 notebook's rank-blend.")
    ap.add_argument("--sed-providers", type=str, nargs="+", default=None,
                    help="ONNXRuntime providers for SED inference, e.g. "
                         "'CUDAExecutionProvider CPUExecutionProvider'. "
                         "Default: CPUExecutionProvider.")
    args = ap.parse_args()

    if args.teacher == "ssm":
        from birdclef.config.teacher import TEACHER_LATEST

        seeds = args.seeds if args.seeds else TEACHER_LATEST["seeds"]
        pseudo_label_with_ssm_pipeline(
            teacher_cfg=TEACHER_LATEST["config"],
            seeds=seeds,
            output_round=args.round,
            confidence_tau=args.tau,
            topk_per_species=args.topk_per_species,
        )
    elif args.teacher == "sed":
        if not args.ckpts:
            raise SystemExit("--teacher sed requires --ckpts <path> [<path> ...]")
        pseudo_label_with_sed(
            [Path(p) for p in args.ckpts], args.round,
            confidence_tau=args.tau, topk_per_species=args.topk_per_species,
        )
    else:  # blend
        from birdclef.config.teacher import TEACHER_LATEST
        from birdclef.config.paths import REPO

        seeds = args.seeds if args.seeds else TEACHER_LATEST["seeds"]
        sed_dir = Path(args.sed_onnx_dir) if args.sed_onnx_dir else (
            REPO / "models" / "sed_kaggle"
        )
        if not sed_dir.exists():
            raise SystemExit(f"SED ONNX dir not found: {sed_dir}")
        pseudo_label_with_ssm_sed_blend(
            teacher_cfg=TEACHER_LATEST["config"],
            seeds=seeds,
            sed_onnx_dir=sed_dir,
            output_round=args.round,
            blend_weight=args.blend_weight,
            confidence_tau=args.tau,
            topk_per_species=args.topk_per_species,
            sed_providers=args.sed_providers,
        )


if __name__ == "__main__":
    main()
