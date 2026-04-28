"""Apply INT8 dynamic quantization to the AST ONNX model.

Dynamic quantization keeps activations in FP32 (no calibration data needed)
but quantizes the heavy weight matrices to INT8. ONNX Runtime then uses
INT8 matmul kernels which are 2-4× faster on x86 CPUs than FP32 matmul.
Accuracy drift is typically tiny for transformer classifiers (~0.001-
0.005 AUC degradation).

Run on Hopper after _freeze_ast_to_onnx:
    python -m birdclef.scripts._quantize_ast_onnx \\
        --in-onnx  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/model.onnx \\
        --out-onnx birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/model_int8.onnx \\
        --validate

The quantized file replaces `model.onnx` in the Kaggle dataset (or upload
both side-by-side and edit AST_ONNX_PATH in the LB notebook).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-onnx", type=Path, required=True)
    ap.add_argument("--out-onnx", type=Path, required=True)
    ap.add_argument("--validate", action="store_true",
                    help="Time fp32 vs int8 on a fixed batch + check max abs diff.")
    ap.add_argument("--validate-batch", type=int, default=12,
                    help="Batch size for the timing/diff check.")
    args = ap.parse_args()

    if not args.in_onnx.exists():
        raise SystemExit(f"input ONNX missing: {args.in_onnx}")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import onnx
    except ImportError as e:
        raise SystemExit(
            f"missing dep ({e.name}). `pip install onnxruntime onnx`"
        ) from e

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: pre-process. The exported ONNX from torch.onnx.export often
    # has stale or incomplete shape annotations on tensors whose shape was
    # introduced by reshape/squeeze ops (e.g. the freshly-init 234-class
    # classifier head). quant_pre_process runs constant folding + light
    # graph optimizations + ONNX's regular shape inference to clean it up.
    #
    # We skip ONNX Runtime's *symbolic* shape inference because it fails
    # on HuggingFace transformer exports — its Concat/Reshape inference
    # is strict about merging int dims that the export doesn't always
    # agree on (raises an AssertionError on `_merge_symbols`). The
    # regular onnx.shape_inference is enough for the quantizer's needs.
    preprocessed = args.out_onnx.with_suffix(".preprocessed.onnx")
    print(f"[quant] step 1: quant_pre_process → {preprocessed}")
    try:
        quant_pre_process(
            input_model=str(args.in_onnx),
            output_model_path=str(preprocessed),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,    # known-bad on HF transformer exports
            auto_merge=True,
        )
    except Exception as e:
        # Last-resort fallback: no preprocessing. Quantizer's shape inference
        # may still fail, but worth trying — sometimes onnx_shape alone
        # is enough.
        print(f"[quant] WARN: quant_pre_process failed ({e}). "
              f"Falling back to in-place input ONNX (no preprocessing).")
        import shutil
        shutil.copy(args.in_onnx, preprocessed)

    # Step 2: strip stale value_info before quantize_dynamic.
    #
    # quant_pre_process injects value_info entries for every initializer
    # (~750 on the AST graph). At least one of those carries a stale shape
    # that conflicts with what onnx.shape_inference computes, and
    # quantize_dynamic's internal `save_and_reload_model_with_shape_infer`
    # call then crashes with `Inferred shape and existing shape differ in
    # dimension 0: (768) vs (234)` — the (234) is the freshly-init head's
    # n_classes, the (768) is AST's hidden size.
    #
    # Clearing value_info forces shape inference to recompute from scratch
    # without any conflicting hints. The fresh inference succeeds and the
    # quantizer is happy.
    stripped = args.out_onnx.with_suffix(".stripped.onnx")
    print(f"[quant] step 2: strip stale value_info → {stripped}")
    m = onnx.load(str(preprocessed), load_external_data=False)
    n_before = len(m.graph.value_info)
    del m.graph.value_info[:]
    m_inf = onnx.shape_inference.infer_shapes(m, check_type=False, strict_mode=True, data_prop=True)
    print(f"[quant]   value_info: {n_before} → {len(m_inf.graph.value_info)} (recomputed fresh)")
    onnx.save(m_inf, str(stripped))

    # Step 3: dynamic quantize from the cleaned model.
    print(f"[quant] step 3: dynamic quantization → {args.out_onnx}")
    print(f"[quant]   weight_type=QInt8 (activations stay FP32)")
    quantize_dynamic(
        model_input=str(stripped),
        model_output=str(args.out_onnx),
        weight_type=QuantType.QInt8,
        # per_channel=True for slightly better accuracy on the large
        # FFN/attention matmuls.
        per_channel=True,
        # reduce_range=True for x86 compatibility (no overflow on AVX2-only
        # CPUs). On AVX-512-VNNI you can set False for slightly more
        # accuracy headroom; Kaggle CPUs are typically AVX2 only.
        reduce_range=True,
    )

    # Clean up the intermediate files (each ~330 MB).
    for tmp in (preprocessed, stripped):
        try:
            tmp.unlink()
            for p in tmp.parent.glob(tmp.stem + ".data"):
                p.unlink()
        except Exception:
            pass
    # Size accounting: torch.onnx.export with external_data leaves weights
    # in a sidecar `model.onnx.data` file, so the bare .onnx is tiny. Sum
    # the sidecar to report a fair before/after.
    def _onnx_total_size(p: Path) -> float:
        sz = p.stat().st_size
        for sib in p.parent.iterdir():
            if sib.name.startswith(p.stem + ".") and sib.suffix == ".data":
                sz += sib.stat().st_size
        return sz / 1024**2
    in_mb = _onnx_total_size(args.in_onnx)
    out_mb = _onnx_total_size(args.out_onnx)
    print(f"[quant] sizes: fp32 {in_mb:.1f} MB → int8 {out_mb:.1f} MB  "
          f"({(1 - out_mb/in_mb)*100:.0f}% smaller)")

    if not args.validate:
        return

    import onnxruntime as ort
    import time as _time

    print(f"[quant] timing both models on batch={args.validate_batch} (3 iters each)...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp = ort.InferenceSession(str(args.in_onnx), sess_options=so,
                                   providers=["CPUExecutionProvider"])
    sess_q  = ort.InferenceSession(str(args.out_onnx), sess_options=so,
                                   providers=["CPUExecutionProvider"])

    in_name = sess_fp.get_inputs()[0].name
    in_shape = sess_fp.get_inputs()[0].shape
    n_mels = in_shape[2] if isinstance(in_shape[2], int) else 128
    max_length = in_shape[1] if isinstance(in_shape[1], int) else 512
    np.random.seed(0)
    x = np.random.randn(int(args.validate_batch), max_length, n_mels).astype(np.float32)

    def _time_run(sess, n=3):
        _ = sess.run(None, {in_name: x})       # warm-up
        t0 = _time.time()
        for _ in range(n):
            out = sess.run(None, {in_name: x})
        return (_time.time() - t0) / n, out[0]

    t_fp, y_fp = _time_run(sess_fp)
    t_q,  y_q  = _time_run(sess_q)
    diff = np.abs(y_fp - y_q)
    print()
    print(f"[quant]   fp32 forward: {t_fp*1000:.1f} ms")
    print(f"[quant]   int8 forward: {t_q*1000:.1f} ms  ({t_fp/t_q:.2f}× speedup)")
    print(f"[quant]   max abs diff: {float(diff.max()):.6e}")
    print(f"[quant]   mean abs diff: {float(diff.mean()):.6e}")
    print()
    if t_q >= t_fp:
        print("[quant] WARN: int8 isn't faster. CPU may lack VNNI/AVX-VNNI; "
              "try `reduce_range=False` or upgrade onnxruntime.")
    if float(diff.max()) > 0.5:
        print("[quant] WARN: large output drift — quantization may have "
              "broken some op. Consider falling back to fp32.")


if __name__ == "__main__":
    main()
