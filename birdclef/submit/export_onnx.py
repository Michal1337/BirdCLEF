"""Export a trained SED PyTorch checkpoint to ONNX (FP32 + optional FP16 cast).

Two important hardening steps vs. naive `torch.onnx.export(...)`:

1. `dynamo=False` — pin the legacy TorchScript exporter. PyTorch 2.5+ defaults
   to the dynamo-based exporter which is still maturing and trips on the mel
   front-end's Pad op when version-converting to opset 17.
2. The FP16 cast `op_block_list` keeps the audio front-end (STFT/FFT, mel
   matmul, log) in FP32. FP16 FFT routinely overflows on real audio and emits
   NaN/Inf — the exact symptom you'd see as `ValueError: Input contains NaN`
   downstream.

Usage:
    python -m birdclef.submit.export_onnx \
        --ckpt birdclef/models_ckpt/sed/sed_v2s/fold0/best.pt \
        --out birdclef/models_ckpt/sed/sed_v2s/fold0/best.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from birdclef.config.paths import WINDOW_SAMPLES


# Ops that must stay in FP32 — these overflow / underflow at fp16 precision
# and produce NaN, sometimes silently. Conservatively block every op that
# touches the audio frontend.
FP16_OP_BLOCK_LIST = [
    "Pad",        # zero-pad before STFT
    "STFT",       # complex twiddle factors
    "MatMul",     # too aggressive globally? — see below
    "Mul",
    "Sub",
    "Div",
    "Log",        # log-mel
    "ReduceMean", # variance / std normalization
    "ReduceMax",
    "ReduceMin",
    "Sqrt",
    "Pow",
    "Erf",
]


def export(ckpt: Path, out: Path, fp16: bool = True, opset: int = 17,
           fp16_block_list=None) -> None:
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
        dynamo=False,                  # pin legacy TorchScript exporter
    )
    print(f"[onnx] exported {out}  (opset={opset})")

    if fp16:
        try:
            from onnxconverter_common import float16
            import onnx

            block = list(fp16_block_list) if fp16_block_list is not None else list(FP16_OP_BLOCK_LIST)
            m = onnx.load(str(out))
            # keep_io_types=True : input/output tensors stay float32 (no caller-
            # side dtype dance). op_block_list : keep frontend ops in fp32 so
            # the FFT doesn't overflow into NaN.
            m16 = float16.convert_float_to_float16(
                m,
                keep_io_types=True,
                op_block_list=block,
            )
            out16 = out.with_suffix(".fp16.onnx")
            onnx.save(m16, str(out16))
            print(f"[onnx] FP16 cast saved to {out16}  "
                  f"(fp32 ops kept: {len(block)})")
        except Exception as exc:
            print(f"[onnx] FP16 cast failed ({exc}); keep FP32 model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16-block", nargs="*", default=None,
                    help=f"ONNX op types to keep in fp32 during the FP16 cast. "
                         f"Default: {FP16_OP_BLOCK_LIST}")
    args = ap.parse_args()
    export(Path(args.ckpt), Path(args.out),
           fp16=not args.no_fp16, opset=args.opset,
           fp16_block_list=args.fp16_block)


if __name__ == "__main__":
    main()
