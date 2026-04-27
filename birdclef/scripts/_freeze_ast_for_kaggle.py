"""Freeze the fine-tuned AST checkpoint into a self-contained HuggingFace
directory that loads OFFLINE on Kaggle (no internet needed).

Reads `best_model.pt` (which has the fine-tuned state_dict but no full
ASTConfig), reconstructs the model the same way `train_ddp_ast.py` does
(downloads the base AST config + interpolates position embeddings to the
trained max_length, then loads the fine-tuned weights), and writes the
result via `model.save_pretrained(out_dir)`.

The output dir contains `config.json` + `model.safetensors` (or
`pytorch_model.bin`). On Kaggle:

    from transformers import ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained(
        "/kaggle/input/birdclef-ast-best/ast_lr3e-05_e15_offline/",
        local_files_only=True,
    )

— no `from_pretrained(hf_name, ...)` call at inference time, no Hub fetch.

Run on Hopper (or any machine with internet):
    python -m birdclef.scripts._freeze_ast_for_kaggle \\
        --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
        --out-dir birdclef_example/outputs/ast/ast_lr3e-05_e15_offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ast-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory; will contain config.json + model weights.")
    args = ap.parse_args()

    if not args.ast_ckpt.exists():
        raise SystemExit(f"AST checkpoint missing: {args.ast_ckpt}")

    # Same reconstruction path the trainer uses: download base AST config,
    # build with target max_length, interpolate position embeddings,
    # load fine-tuned weights from best_model.pt.
    from birdclef_example.train_ddp_ast import ASTSpectrogramClassifier

    ckpt = torch.load(args.ast_ckpt, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    print(f"[freeze] AST checkpoint: epoch={ckpt.get('epoch')} "
          f"best_val_auc_focal_seen={ckpt.get('best_val_auc_focal_seen')}")
    print(f"[freeze] reconstructing AST model with max_length={mc.get('max_length', 512)}, "
          f"num_mel_bins={mc.get('num_mel_bins', 128)}, n_classes={mc['n_classes']}")
    print(f"[freeze] (this downloads the base AST config + weights once on Hopper to bootstrap)")

    wrapper = ASTSpectrogramClassifier(
        n_classes=int(mc["n_classes"]),
        hf_model_name=str(mc["hf_model_name"]),
        max_length=int(mc.get("max_length", 512)),
        num_mel_bins=int(mc.get("num_mel_bins", 128)),
        input_sample_rate=int(mc.get("input_sample_rate", 32000)),
        target_sample_rate=int(mc.get("target_sample_rate", 16000)),
    )
    wrapper.load_state_dict(ckpt["model_state"], strict=False)
    wrapper.eval()

    # Save just the inner HF model (ASTForAudioClassification). Its
    # save_pretrained writes config.json + model.safetensors so Kaggle's
    # `from_pretrained(local_path, local_files_only=True)` reads it back
    # with zero network calls.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save_pretrained(args.out_dir)
    print(f"[freeze] wrote HF-format model to {args.out_dir}")
    for p in sorted(args.out_dir.iterdir()):
        size_mb = p.stat().st_size / 1024**2
        print(f"  {p.name}  ({size_mb:.1f} MB)")

    # Save a small sidecar with our wrapper config (max_length, etc.) so
    # the LB notebook can reconstruct the kaldi fbank step the same way.
    sidecar = {
        "input_sample_rate":  int(mc.get("input_sample_rate", 32000)),
        "target_sample_rate": int(mc.get("target_sample_rate", 16000)),
        "max_length":         int(mc.get("max_length", 512)),
        "num_mel_bins":       int(mc.get("num_mel_bins", 128)),
        "n_classes":          int(mc["n_classes"]),
        "hf_model_name_at_train": str(mc["hf_model_name"]),
        "best_val_auc_focal_seen": ckpt.get("best_val_auc_focal_seen"),
        "best_val_auc_seen":       ckpt.get("best_val_auc_seen"),
        "trained_epoch":           ckpt.get("epoch"),
    }
    (args.out_dir / "wrapper_config.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8",
    )
    print(f"  wrapper_config.json (sidecar with wrapper-level params)")
    print()
    print("Upload this entire directory to Kaggle as a dataset.")
    print("Recommended dataset name: birdclef-ast-best")
    print("Mount path the LB notebook expects:")
    print(f"  /kaggle/input/birdclef-ast-best/{args.out_dir.name}/")


if __name__ == "__main__":
    main()
