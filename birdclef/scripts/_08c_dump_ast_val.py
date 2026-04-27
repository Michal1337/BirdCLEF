"""Dump AST predictions on the labeled-soundscape val pool, row-aligned to
the SSM OOF dump.

Produces `outputs/blend_search/oof/ast_probs.npz` (`{"probs": (N, C) float32}`)
where N matches the row count and order of `ssm_probs.npz` from
`_07b_dump_oof_probs.py`. Both files can then be fed into the blend search.

Note on honesty: the AST model was trained on train.csv (focal recordings),
NOT on labeled soundscapes — so its predictions on the soundscape val pool
are already held-out (no fold split needed). The SSM, by contrast, requires
5-fold OOF because it trained on the labeled soundscapes.

Run:
    python -m birdclef.scripts._08c_dump_ast_val \\
        --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt
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
    ap.add_argument("--ast-ckpt", type=Path, required=True,
                    help="Path to AST best_model.pt (from train_ddp_ast).")
    ap.add_argument("--ssm-meta", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "meta.parquet",
                    help="SSM OOF meta.parquet — defines the row order to align to.")
    ap.add_argument("--out-npz", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "ast_probs.npz",
                    help="Output npz path. Match this in the blend search.")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda | cuda:0. Default: auto.")
    args = ap.parse_args()

    if not args.ast_ckpt.exists():
        raise SystemExit(f"AST checkpoint missing: {args.ast_ckpt}")
    if not args.ssm_meta.exists():
        raise SystemExit(
            f"SSM OOF meta missing: {args.ssm_meta}\n"
            f"Run `python -m birdclef.scripts._07b_dump_oof_probs --n-splits 5` first."
        )

    # Load AST model with the inline-style helper from the trainer module.
    sys.path.insert(0, str(REPO_ROOT / "birdclef_example"))
    from birdclef_example.train_ddp_ast import ASTSpectrogramClassifier

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ast_ckpt, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    print(f"[ast-val] AST checkpoint: epoch={ckpt.get('epoch')} "
          f"best_val_auc_focal_seen={ckpt.get('best_val_auc_focal_seen')}")
    model = ASTSpectrogramClassifier(
        n_classes=int(mc["n_classes"]),
        hf_model_name=str(mc["hf_model_name"]),
        max_length=int(mc.get("max_length", 512)),
        num_mel_bins=int(mc.get("num_mel_bins", 128)),
        input_sample_rate=int(mc.get("input_sample_rate", 32000)),
        target_sample_rate=int(mc.get("target_sample_rate", 16000)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Pull row order from the SSM OOF meta. `meta.parquet` rows correspond
    # one-for-one with `ssm_probs.npz` rows. For each unique file run AST
    # once and place its 12 outputs at the right offsets.
    meta = pd.read_parquet(args.ssm_meta)
    print(f"[ast-val] aligning to {len(meta)} rows from {args.ssm_meta}")
    if "filename" not in meta.columns:
        raise SystemExit("SSM meta missing 'filename' column")
    n_rows = len(meta)
    n_classes = int(mc["n_classes"])
    out = np.zeros((n_rows, n_classes), dtype=np.float32)

    # Group meta rows by filename (preserves first-appearance order).
    file_groups: dict[str, list[int]] = {}
    for i, fn in enumerate(meta["filename"].astype(str).tolist()):
        file_groups.setdefault(fn, []).append(i)

    for fname, row_idx in tqdm(file_groups.items(), desc="AST inference"):
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
        wins = torch.from_numpy(
            y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32),
        ).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(wins)                              # (12, C)
            probs_12 = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
        # The SSM meta is sorted by (filename, end_sec) so row_idx is already
        # in the same window order as the model's natural 0..11 output.
        n = min(len(row_idx), N_WINDOWS)
        for j in range(n):
            out[row_idx[j]] = probs_12[j]

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, probs=out)
    print()
    print(f"[ast-val] wrote {args.out_npz}")
    print(f"[ast-val]   shape={out.shape}  mean={out.mean():.4f}  "
          f"non-zero rows={(out.sum(axis=1) > 0).sum()}/{n_rows}")


if __name__ == "__main__":
    main()
