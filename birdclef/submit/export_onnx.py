"""Export a trained SED PyTorch checkpoint to ONNX (optionally FP16-cast).

Usage:
    python -m birdclef.submit.export_onnx \
        --ckpt birdclef/models_ckpt/sed/sed_v2s/fold0/best.pt \
        --out birdclef/models_ckpt/sed/sed_v2s/fold0/best.onnx
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from birdclef.config.paths import SR, WINDOW_SAMPLES


def export(ckpt: Path, out: Path, fp16: bool = True, opset: int = 17) -> None:
    # Defer training-dep imports to this function so the submit-side import
    # guard only trips on the guarded module chain.
    from birdclef.models.sed import SED, SEDConfig

    state = torch.load(ckpt, map_location="cpu")
    cfg = state["cfg"]
    sed_cfg = SEDConfig(
        backbone=cfg["backbone"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], f_min=cfg["f_min"], f_max=cfg["f_max"],
    )
    model = SED(sed_cfg)
    sd = state.get("ema") or state["state_dict"]
    if all(k in dict(model.named_parameters()) for k in sd):
        for n, p in model.named_parameters():
            if n in sd:
                p.data.copy_(sd[n])
    else:
        model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    dummy = torch.zeros(1, WINDOW_SAMPLES, dtype=torch.float32)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["wave"], output_names=["logits"],
        dynamic_axes={"wave": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[onnx] exported {out}")

    if fp16:
        try:
            from onnxconverter_common import float16
            import onnx

            m = onnx.load(str(out))
            m16 = float16.convert_float_to_float16(m)
            out16 = out.with_suffix(".fp16.onnx")
            onnx.save(m16, str(out16))
            print(f"[onnx] FP16 cast saved to {out16}")
        except Exception as exc:
            print(f"[onnx] FP16 cast failed ({exc}); keep FP32 model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    export(Path(args.ckpt), Path(args.out), fp16=not args.no_fp16, opset=args.opset)


if __name__ == "__main__":
    main()
