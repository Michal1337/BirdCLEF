"""Dump focal-CNN predictions on the labeled-soundscape val pool, row-aligned
to the SSM OOF dump.

Mirrors `_08c_dump_ast_val.py` but for the focal CNN best checkpoint
(`birdclef_example/outputs/focal/<config>/best_model.pt`). Output goes
to the same blend_search/oof directory so `_09_blend_search.py` can mix
the SSM OOF, AST val, and CNN val probs in any combination.

Run on Hopper (or any machine with the trained checkpoint):
    python -m birdclef.scripts._08d_dump_cnn_val \\
        --cnn-ckpt birdclef_example/outputs/focal/sota14_nfnet_lr7e-04_1e-04/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import (
    FILE_SAMPLES, N_WINDOWS, OUTPUT_ROOT, SOUNDSCAPES, WINDOW_SAMPLES,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-ckpt", type=Path, required=True,
                    help="Path to focal CNN best_model.pt (from train_ddp_focal).")
    ap.add_argument("--ssm-meta", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "meta.parquet",
                    help="SSM OOF meta.parquet — defines the row order to align to.")
    ap.add_argument("--out-npz", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "cnn_probs.npz",
                    help="Output npz path. Match this in the blend search.")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | cuda:0. Default: auto.")
    args = ap.parse_args()

    if not args.cnn_ckpt.exists():
        raise SystemExit(f"CNN checkpoint missing: {args.cnn_ckpt}")
    if not args.ssm_meta.exists():
        raise SystemExit(
            f"SSM OOF meta missing: {args.ssm_meta}\n"
            f"Run `python -m birdclef.scripts._07b_dump_oof_probs --n-splits 5` first."
        )

    sys.path.insert(0, str(REPO_ROOT / "birdclef_example"))
    from birdclef_example.train_ddp_sota import TimmSpectrogramClassifier

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.cnn_ckpt, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    print(f"[cnn-val] CNN checkpoint: epoch={ckpt.get('epoch')} "
          f"best_val_auc_focal_seen={ckpt.get('best_val_auc_focal_seen')}")
    print(f"[cnn-val]   backbone={mc['backbone_name']} image_size={mc['image_size']} "
          f"n_mels={mc['n_mels']}")
    model = TimmSpectrogramClassifier(
        n_classes=int(mc["n_classes"]),
        backbone_name=str(mc["backbone_name"]),
        pretrained=False,                               # we load the fine-tuned weights below
        image_size=int(mc["image_size"]),
        dropout=float(mc.get("dropout", 0.35)),
        sample_rate=int(mc.get("sample_rate", 32000)),
        n_mels=int(mc["n_mels"]),
        n_fft=int(mc["n_fft"]),
        hop_length=int(mc["hop_length"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    meta = pd.read_parquet(args.ssm_meta)
    print(f"[cnn-val] aligning to {len(meta)} rows from {args.ssm_meta}")
    n_rows = len(meta)
    n_classes = int(mc["n_classes"])
    out = np.zeros((n_rows, n_classes), dtype=np.float32)

    file_groups: dict[str, list[int]] = {}
    for i, fn in enumerate(meta["filename"].astype(str).tolist()):
        file_groups.setdefault(fn, []).append(i)

    for fname, row_idx in tqdm(file_groups.items(), desc="CNN inference"):
        path = SOUNDSCAPES / fname
        if not path.exists():
            print(f"  WARN: {path} missing — leaving zeros for {len(row_idx)} rows")
            continue
        y, _sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        # CNN expects (B, 1, T) — 12 windows of 5s @ 32kHz.
        wins = (
            torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32))
            .unsqueeze(1)                                      # (12, 1, T)
            .to(device)
        )
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(wins)                           # (12, n_classes)
            probs_12 = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
        n = min(len(row_idx), N_WINDOWS)
        for j in range(n):
            out[row_idx[j]] = probs_12[j]

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, probs=out)
    print()
    print(f"[cnn-val] wrote {args.out_npz}")
    print(f"[cnn-val]   shape={out.shape}  mean={out.mean():.4f}  "
          f"non-zero rows={(out.sum(axis=1) > 0).sum()}/{n_rows}")


if __name__ == "__main__":
    main()
