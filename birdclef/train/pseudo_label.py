"""Pseudo-label orchestrator — runs a teacher over unlabeled + labeled soundscapes.

Writes to cache/pseudo/round{N}/:
    probs.npz     : {'probs': (N_rows, C) float32}
    meta.parquet  : same columns as perch cache meta (row_id, filename, site, ...)

The next SED training run reads these and mixes them in at 50 % of the batch.

Supports two teacher types:
  - "ssm"     : runs train_ssm_head.run_pipeline_for_split on all files
                (trains on non-anchor rows, predicts on everything).
  - "sed_ckpt": loads one or more SED checkpoints from MODEL_ROOT/sed/**/best.pt
                and averages their predictions on the soundscapes.

The top-k-per-species / τ confidence filter is applied on the raw probs and
stored in meta as `keep_mask` for training-time sampling.
"""
from __future__ import annotations

import argparse
import json
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
    PERCH_META,
    PERCH_NPZ,
    PSEUDO_DIR,
    SOUNDSCAPES,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import primary_labels


def _round_dir(rnd: int) -> Path:
    d = PSEUDO_DIR / f"round{rnd}"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
):
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
        sd = state.get("ema") or state["state_dict"]
        # Handle EMA-shadow-only dicts (keys match named_parameters()).
        if all(k in dict(m.named_parameters()) for k in sd):
            for n, p in m.named_parameters():
                if n in sd:
                    p.data.copy_(sd[n].to(p.device))
        else:
            m.load_state_dict(state["state_dict"], strict=False)
        m.eval()
        models.append(m)

    paths = sorted(SOUNDSCAPES.glob("*.ogg"))
    meta_rows = []
    probs_rows = []
    for p in tqdm(paths, desc="pseudo"):
        try:
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
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
                "row_id": f"{p.stem}_{(w+1)*5}",
                "filename": p.name,
                "window": w,
            })
        probs_rows.append(ensemble)
    meta = pd.DataFrame(meta_rows)
    probs = np.concatenate(probs_rows, axis=0).astype(np.float32)

    # Confidence filter
    keep_mask = np.zeros_like(probs, dtype=np.uint8)
    if confidence_tau > 0:
        keep_mask = (probs >= confidence_tau).astype(np.uint8)
    if topk_per_species > 0:
        # per-file per-species top-k windows
        n_files = probs.shape[0] // N_WINDOWS
        view = probs.reshape(n_files, N_WINDOWS, n_classes)
        topk_idx = np.argpartition(-view, kth=topk_per_species - 1, axis=1)[:, :topk_per_species]
        km = np.zeros_like(view, dtype=np.uint8)
        for f in range(n_files):
            for c in range(n_classes):
                km[f, topk_idx[f, :, c], c] = 1
        keep_mask = np.maximum(keep_mask, km.reshape(n_files * N_WINDOWS, n_classes))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="Paths to SED best.pt checkpoints (one or more)")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--topk-per-species", type=int, default=0)
    args = ap.parse_args()
    pseudo_label_with_sed([Path(p) for p in args.ckpts], args.round,
                           confidence_tau=args.tau,
                           topk_per_species=args.topk_per_species)


if __name__ == "__main__":
    main()
