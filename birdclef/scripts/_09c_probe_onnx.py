"""Low-level ONNX probe: feed synthetic + real audio into one ONNX, dump
output stats. Bypasses inference_template entirely so we can tell whether
the file itself is broken or only the inference wiring is.

Usage:
    python -m birdclef.scripts._09c_probe_onnx --onnx <path-to-onnx>

Use this when `_09b_test_submission_local` reports NaN — if THIS script
also shows NaN, the ONNX export is broken (re-export). If THIS script is
clean, the issue is in inference_template.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from birdclef.config.paths import SOUNDSCAPES, WINDOW_SAMPLES, N_WINDOWS, FILE_SAMPLES


def _stats(x: np.ndarray, label: str) -> None:
    n_nan = int(np.isnan(x).sum())
    n_inf = int(np.isinf(x).sum())
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        print(f"  {label}: shape={x.shape} dtype={x.dtype}  ALL NaN/Inf "
              f"(nan={n_nan}, inf={n_inf})")
        return
    print(f"  {label}: shape={x.shape} dtype={x.dtype}  "
          f"min={finite.min():.4f}  max={finite.max():.4f}  mean={finite.mean():.4f}  "
          f"nan={n_nan}  inf={n_inf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--soundscape", default=None,
                    help="Optional: a real soundscape OGG to test with. "
                         "Default: pick the first one in SOUNDSCAPES.")
    args = ap.parse_args()

    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"No such file: {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    print(f"[probe] {onnx_path}")
    print(f"  input : {inp.name}  type={inp.type}  shape={inp.shape}")
    print(f"  output: {out_meta.name}  type={out_meta.type}  shape={out_meta.shape}")

    # Determine input dtype
    if "float16" in inp.type:
        in_dtype = np.float16
    elif "double" in inp.type:
        in_dtype = np.float64
    else:
        in_dtype = np.float32

    # 1. Zero input — should produce a finite (likely class-bias) output
    print("\n[probe] zero input (12, win):")
    z = np.zeros((N_WINDOWS, WINDOW_SAMPLES), dtype=in_dtype)
    out_z = sess.run(None, {inp.name: z})[0]
    _stats(out_z, "logits")

    # 2. Random Gaussian input — should produce diverse logits
    print("\n[probe] random Gaussian input (12, win) std=0.05:")
    rng = np.random.default_rng(0)
    g = (rng.standard_normal((N_WINDOWS, WINDOW_SAMPLES)) * 0.05).astype(in_dtype)
    out_g = sess.run(None, {inp.name: g})[0]
    _stats(out_g, "logits")

    # 3. Real soundscape input
    sc = Path(args.soundscape) if args.soundscape else None
    if sc is None:
        cands = sorted(SOUNDSCAPES.glob("*.ogg"))
        if cands:
            sc = cands[0]
    if sc and sc.exists():
        print(f"\n[probe] real soundscape: {sc.name}")
        import soundfile as sf
        y, sr = sf.read(str(sc), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        print(f"  audio: shape={y.shape} dtype={y.dtype} sr={sr}  "
              f"min={y.min():.3f}  max={y.max():.3f}  std={y.std():.3f}")
        wins = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(in_dtype)
        out_r = sess.run(None, {inp.name: wins})[0]
        _stats(out_r, "logits")

    print()
    nan_overall = (
        np.isnan(out_z).any() or np.isnan(out_g).any() or
        (sc and np.isnan(out_r).any())
    )
    if nan_overall:
        print("[probe] VERDICT: NaN detected → ONNX export is broken.")
        print("  Most likely cause: torch.onnx.export went through the dynamo path")
        print("  and the graph version-conversion to opset 17 silently produced bad ops.")
        print("  Fix: ensure birdclef/submit/export_onnx.py contains 'dynamo=False'")
        print("  (and 'op_block_list' for the FP16 cast), then re-export.")
    else:
        print("[probe] VERDICT: ONNX produces finite outputs. The file itself is fine.")
        print("  If _09b still reports NaN, the issue is in the inference template")
        print("  or the input pipeline — share that script's exact output.")


if __name__ == "__main__":
    main()
