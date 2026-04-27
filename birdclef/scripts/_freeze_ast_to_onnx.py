"""Export the fine-tuned AST model to ONNX for fast CPU inference on Kaggle.

PyTorch transformer eager-mode is slow on CPU (no graph fusion, no MKL
optimization). ONNX Runtime applies operator fusion, vectorized matmul,
and graph-level optimizations — typically 3-5× faster than torch.no_grad
on the same CPU. Critical for fitting AST inference into Kaggle's 3-hour
test budget.

Workflow on Hopper (one-time):
    python -m birdclef.scripts._freeze_ast_to_onnx \\
        --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
        --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx \\
        --validate

Outputs (upload the whole directory as a Kaggle dataset):
    model.onnx              ~330 MB — the exported AST model
    wrapper_config.json     wrapper-level params (max_length, sample rates)

The validate flag runs a torch vs onnxruntime sanity check on a random
batch and warns if max abs diff > 1e-3 (small numerical drift from fp32
ops is fine; large drift means the export went wrong).

On Kaggle:
    import onnxruntime as ort
    sess = ort.InferenceSession(
        "/kaggle/input/birdclef-ast-onnx/model.onnx",
        providers=["CPUExecutionProvider"],
    )
    # fbank in torch, hand off as numpy:
    logits = sess.run(["logits"], {"input_values": fb_np.astype(np.float32)})[0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ast-ckpt", type=Path, required=True,
                    help="Path to the fine-tuned best_model.pt from train_ddp_ast.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory; will contain model.onnx + wrapper_config.json.")
    ap.add_argument("--opset", type=int, default=14,
                    help="ONNX opset version. 14 supports the ops AST uses; "
                         "higher (17+) enables more optimizations but needs a "
                         "newer onnxruntime on Kaggle.")
    ap.add_argument("--validate", action="store_true",
                    help="Compare ONNX output to torch output on a random batch.")
    ap.add_argument("--validate-batch", type=int, default=12,
                    help="Batch size for the validation check.")
    args = ap.parse_args()

    if not args.ast_ckpt.exists():
        raise SystemExit(f"AST checkpoint missing: {args.ast_ckpt}")

    # Reconstruct the fine-tuned model the same way the trainer does:
    # download base AST config + weights, interpolate position embeddings
    # to the trained max_length, load fine-tuned state.
    from birdclef_example.train_ddp_ast import ASTSpectrogramClassifier

    ckpt = torch.load(args.ast_ckpt, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    print(f"[onnx] AST checkpoint: epoch={ckpt.get('epoch')} "
          f"best_val_auc_focal_seen={ckpt.get('best_val_auc_focal_seen')}")
    print(f"[onnx] reconstructing model with max_length={mc.get('max_length', 512)}, "
          f"num_mel_bins={mc.get('num_mel_bins', 128)}, n_classes={mc['n_classes']}")
    print(f"[onnx] (downloads base AST weights once on Hopper to bootstrap; "
          f"no internet needed at Kaggle inference time)")

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

    # We export ONLY the inner ASTForAudioClassification model. The kaldi
    # fbank preprocessing stays in PyTorch on Kaggle (it's fast — ~1ms per
    # window — and torchaudio.compliance.kaldi.fbank doesn't ONNX-export
    # cleanly anyway). The ONNX boundary is `(input_values: B,T,F) → logits`.
    inner = wrapper.model
    inner.eval()

    max_length = int(mc.get("max_length", 512))
    num_mel_bins = int(mc.get("num_mel_bins", 128))
    n_classes = int(mc["n_classes"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "model.onnx"

    # Dummy input: AST takes (batch, max_length, num_mel_bins) post-fbank.
    # Use a small batch — the dynamic axis lets the runtime accept any size.
    dummy = torch.randn(2, max_length, num_mel_bins, dtype=torch.float32)

    print(f"[onnx] exporting → {onnx_path}  (opset={args.opset})")

    # Wrap the inner model so the onnx export sees a clean signature.
    class _ASTONNXWrapper(torch.nn.Module):
        def __init__(self, inner_model):
            super().__init__()
            self.inner = inner_model

        def forward(self, input_values):
            return self.inner(input_values=input_values).logits

    export_model = _ASTONNXWrapper(inner).eval()

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (dummy,),
            str(onnx_path),
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch"},
                "logits":       {0: "batch"},
            },
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    size_mb = onnx_path.stat().st_size / 1024**2
    print(f"[onnx] wrote {onnx_path}  ({size_mb:.1f} MB)")

    # Sidecar config — wrapper-level (max_length, sample rates) so the LB
    # notebook can reconstruct kaldi fbank with matching params.
    sidecar = {
        "input_sample_rate":  int(mc.get("input_sample_rate", 32000)),
        "target_sample_rate": int(mc.get("target_sample_rate", 16000)),
        "max_length":         int(max_length),
        "num_mel_bins":       int(num_mel_bins),
        "n_classes":          int(n_classes),
        "hf_model_name_at_train": str(mc["hf_model_name"]),
        "best_val_auc_focal_seen": ckpt.get("best_val_auc_focal_seen"),
        "best_val_auc_seen":       ckpt.get("best_val_auc_seen"),
        "trained_epoch":           ckpt.get("epoch"),
        "onnx_opset":              int(args.opset),
    }
    (args.out_dir / "wrapper_config.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8",
    )
    print(f"[onnx] wrote {args.out_dir / 'wrapper_config.json'}")

    if args.validate:
        # Compare ONNX runtime output vs torch eager output on a random
        # batch. They should match to ~1e-4 fp32 numerical noise.
        try:
            import onnxruntime as ort
        except ImportError:
            print("[onnx] WARN: --validate requested but onnxruntime not "
                  "installed. `pip install onnxruntime`.")
            return
        print(f"[onnx] validating: torch eager vs onnxruntime on batch={args.validate_batch}")
        torch.manual_seed(0)
        x = torch.randn(int(args.validate_batch), max_length, num_mel_bins, dtype=torch.float32)
        with torch.no_grad():
            y_torch = export_model(x).numpy()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        y_onnx = sess.run(["logits"], {"input_values": x.numpy().astype(np.float32)})[0]
        diff = np.abs(y_torch - y_onnx)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        agree_status = "OK" if max_diff < 1e-3 else "WARN — large drift"
        print(f"[onnx]   max_abs_diff = {max_diff:.6e}  mean = {mean_diff:.6e}  ({agree_status})")
        if max_diff >= 1e-2:
            raise SystemExit(
                f"ONNX export validation failed: max_abs_diff = {max_diff:.4e} ≥ 1e-2. "
                "Check the export — possible dynamic-shape or op-version mismatch."
            )

    print()
    print("Upload the directory to Kaggle as a dataset (recommended name: birdclef-ast-onnx).")
    print("Mount path the LB notebook expects:")
    print(f"  /kaggle/input/birdclef-ast-onnx/{args.out_dir.name}/model.onnx")


if __name__ == "__main__":
    main()
