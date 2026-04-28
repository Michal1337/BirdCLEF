"""Pseudo-label orchestrator — runs a teacher over all soundscapes and writes
soft-target probabilities that the SED student consumes via
`SEDTrainDataset(pseudo_round=N)`.

Output layout — `cache/pseudo/round{N}/`:
    probs.npz     : {'probs': (N_rows, C) float32,
                     'keep_mask': (N_rows, C) uint8}
    meta.parquet  : row_id, filename, window, is_labeled
                    (same row ordering as the Perch cache meta)
    info.json     : teacher name, seeds, confidence filter, coverage stats

Two teacher paths:
  - SSM pipeline ensemble (current default): trains the full proto+MLP+residual
    stack on non-anchor labeled rows, once per seed, and runs Perch-cache
    inference on all 10 658 soundscape files. Averages across seeds → final
    pseudo-label. This is the round-0 teacher since it doesn't require a
    trained SED yet.
  - SED checkpoint ensemble: loads one or more SED `best.pt` files and
    averages their predictions over the same soundscapes. Use this for
    round≥1 once you have a student that's stronger than the SSM teacher.

Both paths share the same output schema so the downstream consumer doesn't
care which teacher produced round N.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["ssm", "sed"], default="ssm",
                    help="Which teacher path to use. Default 'ssm' (round-0 capable).")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Confidence floor for keep_mask. 0 disables.")
    ap.add_argument("--topk-per-species", type=int, default=0,
                    help="Union with per-file top-k-window mask per species. 0 disables.")
    # SSM teacher options
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Seeds for the SSM teacher ensemble. Default from teacher config.")
    # SED teacher options
    ap.add_argument("--ckpts", type=str, nargs="*", default=[],
                    help="For --teacher sed: paths to SED best.pt files (one or more).")
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
    else:
        if not args.ckpts:
            raise SystemExit("--teacher sed requires --ckpts <path> [<path> ...]")
        pseudo_label_with_sed(
            [Path(p) for p in args.ckpts], args.round,
            confidence_tau=args.tau, topk_per_species=args.topk_per_species,
        )


if __name__ == "__main__":
    main()
