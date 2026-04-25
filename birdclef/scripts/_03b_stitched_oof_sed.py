"""Compute stitched-OOF macro AUC for a trained SED config (all folds done).

After running `_03_train_sed.py` (or `_06_train_sed_student.py`) for every
fold of the same config, this script:
  1) Loads each `<MODEL_ROOT>/sed/<config>/fold{i}/best.pt` checkpoint.
  2) Predicts on fold i's val files (via PyTorch — uses the same `SED` class
     the trainer saved). One model per fold = honest OOF.
  3) Concatenates per-row probs across all folds, runs `compute_stage_metrics`
     once, prints the global stitched-OOF macro AUC + per-site/per-hour
     subgroups.
  4) Writes `<MODEL_ROOT>/sed/<config>/stitched_oof_metrics.json` and
     `stitched_oof_probs.npz` for downstream blend search.

Why a separate script: stitched OOF requires all folds to be trained first,
and the training script (`train_sed_ddp.py`) only sees one fold at a time.
This is the SED equivalent of what `train_ssm_head.py:run_full_evaluation`
does in-process — but for SED we can't fit all 5 folds in memory at once,
so we run them sequentially here from saved checkpoints.

Usage:
    python -m birdclef.scripts._03b_stitched_oof_sed \
        --config sed_v2s --n-splits 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from birdclef.config.paths import (
    FILE_SAMPLES,
    MODEL_ROOT,
    N_WINDOWS,
    SOUNDSCAPES,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta
from birdclef.data.splits import load_folds
from birdclef.eval.metrics import compute_stage_metrics


def _load_sed_from_ckpt(ckpt_path: Path, device: torch.device):
    """Load SED from .pt checkpoint with proper BN-buffer + EMA handling.

    Same pattern used in pseudo_label.py and export_onnx.py — load full
    state_dict first to keep BN running statistics, then overwrite trainable
    params from the EMA shadow if present.
    """
    from birdclef.models.sed import SED, SEDConfig

    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state["cfg"]
    sed_cfg = SEDConfig(
        backbone=cfg["backbone"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], f_min=cfg["f_min"], f_max=cfg["f_max"],
    )
    m = SED(sed_cfg).to(device)
    m.load_state_dict(state["state_dict"], strict=False)
    ema_shadow = state.get("ema")
    if ema_shadow:
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in ema_shadow:
                    p.data.copy_(ema_shadow[n].to(p.device))
    m.eval()
    return m, cfg


@torch.no_grad()
def _predict_file(model, fpath: Path, device: torch.device) -> np.ndarray:
    """Return (N_WINDOWS, n_classes) sigmoid probabilities for one 60s OGG."""
    y, _ = sf.read(str(fpath), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    else:
        y = y[:FILE_SAMPLES]
    wins = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)).to(device)
    logits = model(wins)
    return torch.sigmoid(logits).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="SED config name (matches the dir under MODEL_ROOT/sed/).")
    ap.add_argument("--n-splits", type=int, default=5, choices=[5, 10],
                    help="Static fold parquet to use for fold->files lookup.")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | cuda:0 ... (default: auto)")
    ap.add_argument("--ckpt-dir", default=None,
                    help=f"Override the per-config dir under {MODEL_ROOT/'sed'}/<config>/.")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(args.ckpt_dir) if args.ckpt_dir else (Path(MODEL_ROOT) / "sed" / args.config)
    if not base_dir.exists():
        raise SystemExit(f"No SED config dir at {base_dir}")

    folds_df = load_folds(n_splits=args.n_splits)
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    idx_map = label_to_idx()
    n_classes = len(idx_map)

    # Ground truth and meta over all labeled rows
    sc = sc.sort_values(["filename", "end_sec"]).reset_index(drop=True)
    Y = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lb in lbls:
            j = idx_map.get(lb)
            if j is not None:
                Y[i, j] = 1

    fold_of = dict(zip(folds_df["filename"].astype(str), folds_df["fold"].astype(int)))
    sc["fold"] = sc["filename"].astype(str).map(fold_of)
    sc = sc.dropna(subset=["fold"]).reset_index(drop=True)
    sc["fold"] = sc["fold"].astype(int)
    if sc.empty:
        raise SystemExit("No labeled rows have fold assignment — split parquet missing or stale.")

    # Re-build Y aligned to the post-dropna sc (since we lost rows with no fold)
    Y = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lb in lbls:
            j = idx_map.get(lb)
            if j is not None:
                Y[i, j] = 1

    P = np.zeros_like(Y, dtype=np.float32)

    # Walk folds: for each fold, load its best.pt, predict on its val files
    n_folds_seen = 0
    for fold in sorted(sc["fold"].unique()):
        ckpt = base_dir / f"fold{fold}" / "best.pt"
        if not ckpt.exists():
            print(f"[stitched-oof] WARN: missing {ckpt}; fold {fold} val rows will stay 0")
            continue
        model, _cfg = _load_sed_from_ckpt(ckpt, device)
        n_folds_seen += 1

        sub = sc[sc["fold"] == fold]
        files = sub.drop_duplicates("filename")["filename"].astype(str).tolist()
        print(f"[stitched-oof] fold {fold}: {len(files)} files, ckpt={ckpt}")
        for fn in files:
            row_idx = sub.index[sub["filename"] == fn].to_numpy()
            if len(row_idx) == 0:
                continue
            probs = _predict_file(model, SOUNDSCAPES / fn, device)
            # `row_idx` is sorted by end_sec → 1-to-1 with the 12 ascending windows
            n = min(len(row_idx), N_WINDOWS)
            P[row_idx[:n]] = probs[:n]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_folds_seen == 0:
        raise SystemExit(f"No fold checkpoints found under {base_dir}/fold*/best.pt")

    meta_for_metrics = sc[["site", "hour_utc"]].reset_index(drop=True)
    metrics = compute_stage_metrics(Y, P, meta_for_metrics)

    out_json = base_dir / "stitched_oof_metrics.json"
    out_npz = base_dir / "stitched_oof_probs.npz"
    out_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(out_npz, probs=P, y_true=Y)

    print()
    print(f"[stitched-oof] config={args.config}  n_splits={args.n_splits}  folds_seen={n_folds_seen}")
    print(f"[stitched-oof] macro_auc       = {metrics.get('macro_auc', float('nan')):.4f}")
    print(f"[stitched-oof] site_auc_std    = {metrics.get('site_auc_std', float('nan')):.4f}")
    print(f"[stitched-oof] rare_auc        = {metrics.get('rare_auc', float('nan'))}")
    print(f"[stitched-oof] frequent_auc    = {metrics.get('frequent_auc', float('nan'))}")
    print(f"[stitched-oof] wrote {out_json}")
    print(f"[stitched-oof] wrote {out_npz}")


if __name__ == "__main__":
    main()
