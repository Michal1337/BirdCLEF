"""Find the first ONNX node whose output is NaN/Inf.

Strategy:
  1. Hash + mtime the ONNX file (to confirm it's the freshly-exported one).
  2. Scan ALL initializers for embedded NaN/Inf — bad weights often slip in
     during constant-folding, especially with FP16 casts.
  3. Print ops-by-type histogram so we can see the graph shape at a glance.
  4. Add EVERY intermediate tensor as a model output, run inference, and
     report which node first produces NaN.

That last step is the smoking gun: it tells us exactly which op
(Reduce*, Pow, Sqrt, Div, BN, …) is breaking, so we can patch the source.

Usage:
    python -m birdclef.scripts._09d_onnx_first_nan \
        --onnx birdclef/models_ckpt/sed/sed_v2s/fold0/best.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from birdclef.config.paths import N_WINDOWS, WINDOW_SAMPLES


def _sha256_short(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _file_meta(p: Path) -> str:
    st = os.stat(p)
    mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return f"size={st.st_size:,}B  mtime={mtime}  sha256={_sha256_short(p)}"


def _scan_initializers(onnx_module, model) -> int:
    """Return number of initializers containing NaN/Inf."""
    bad = 0
    for init in model.graph.initializer:
        arr = onnx_module.numpy_helper.to_array(init)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            bad += 1
            print(f"  ✗ {init.name:60s} shape={list(arr.shape)} dtype={arr.dtype}  "
                  f"nan={n_nan}  inf={n_inf}")
    if bad == 0:
        print("  ok — no NaN/Inf in any of "
              f"{len(model.graph.initializer)} initializers")
    return bad


def _ops_histogram(model) -> None:
    counter = Counter(node.op_type for node in model.graph.node)
    print(f"  total nodes: {len(model.graph.node)}")
    for op, n in counter.most_common():
        print(f"    {op:25s} {n}")


def _expose_all_intermediate_outputs(onnx_module, model):
    """Return a copy of `model` whose graph emits every intermediate tensor
    as a graph output, in topological order.

    Uses onnx.shape_inference so we can declare intermediate tensors with
    their true dtypes — ORT rejects UNDEFINED.
    """
    import copy
    from onnx import shape_inference

    m = copy.deepcopy(model)
    # Run shape inference; populates m.graph.value_info with type+shape per
    # intermediate tensor. Best-effort — if it fails we fall back to FLOAT.
    try:
        m = shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as exc:
        print(f"  warning: shape inference failed ({exc}); declaring all "
              "intermediates as FLOAT and skipping non-float tensors.")

    type_by_name = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        type_by_name[vi.name] = vi.type

    existing_outs = {o.name for o in m.graph.output}
    intermediate_names = []
    for node in m.graph.node:
        for oname in node.output:
            if oname and oname not in existing_outs and oname not in intermediate_names:
                intermediate_names.append(oname)

    n_added = 0
    n_skipped_no_type = 0
    for name in intermediate_names:
        ttype = type_by_name.get(name)
        if ttype is None:
            # No inferred type — skip rather than risk an UNDEFINED tensor.
            n_skipped_no_type += 1
            continue
        vi = onnx_module.helper.ValueInfoProto()
        vi.name = name
        vi.type.CopyFrom(ttype)
        m.graph.output.append(vi)
        n_added += 1
    print(f"  exposed {n_added} intermediate tensors as outputs "
          f"(skipped {n_skipped_no_type} with no inferred type)")
    return m, intermediate_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input-mode", choices=["zero", "small_random"], default="zero",
                    help="Synthetic input. zero is the easiest case to break NaN-free.")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    p = Path(args.onnx)
    if not p.exists():
        raise SystemExit(f"No such file: {p}")
    print(f"== file ==")
    print(f"  {p}")
    print(f"  {_file_meta(p)}")

    print()
    print("== load + ops histogram ==")
    model = onnx.load(str(p))
    _ops_histogram(model)

    print()
    print("== initializer scan (NaN/Inf) ==")
    bad_init = _scan_initializers(onnx, model)
    if bad_init:
        print(f"  → {bad_init} initializer(s) corrupted at export time. "
              "Re-export with do_constant_folding=False.")

    print()
    print("== first-NaN trace ==")
    inst, names = _expose_all_intermediate_outputs(onnx, model)
    inst_path = p.with_suffix(".instrumented.onnx")
    onnx.save(inst, str(inst_path))
    print(f"  instrumented copy: {inst_path}  (+{len(names)} intermediate outputs)")

    # Build input
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(inst_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_dtype = np.float16 if "float16" in inp.type else np.float32
    if args.input_mode == "zero":
        x = np.zeros((N_WINDOWS, WINDOW_SAMPLES), dtype=in_dtype)
    else:
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((N_WINDOWS, WINDOW_SAMPLES)) * 0.05).astype(in_dtype)
    print(f"  input: {args.input_mode}  shape={x.shape}  dtype={x.dtype}")

    # ORT returns outputs in declared order. Map name → output index.
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {inp.name: x})
    by_name = dict(zip(out_names, outs))

    # Walk the graph in node order, find first node whose output has NaN.
    first_bad = None
    n_clean_before = 0
    for node in model.graph.node:
        for oname in node.output:
            if not oname or oname not in by_name:
                continue
            arr = by_name[oname]
            if not isinstance(arr, np.ndarray):
                continue
            n_nan = int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0
            n_inf = int(np.isinf(arr).sum()) if arr.dtype.kind == "f" else 0
            if n_nan or n_inf:
                first_bad = (node, oname, arr, n_nan, n_inf)
                break
        if first_bad:
            break
        n_clean_before += 1

    print(f"  walked {n_clean_before} clean nodes before any NaN/Inf")
    if first_bad is None:
        print("  no NaN/Inf in intermediate tensors — bug must be in the final output op.")
        # Print graph outputs
        for o in model.graph.output:
            arr = by_name.get(o.name)
            if isinstance(arr, np.ndarray):
                n_nan = int(np.isnan(arr).sum())
                n_inf = int(np.isinf(arr).sum())
                print(f"    graph output '{o.name}': shape={arr.shape} nan={n_nan} inf={n_inf}")
    else:
        node, oname, arr, n_nan, n_inf = first_bad
        print(f"  ← first node to emit NaN/Inf:")
        print(f"      op_type : {node.op_type}")
        print(f"      name    : {node.name}")
        print(f"      output  : {oname}")
        print(f"      shape   : {list(arr.shape)}")
        print(f"      nan/inf : {n_nan}/{n_inf}")
        # Show input tensor stats so we can see WHY this op produced NaN.
        for in_name in node.input:
            if in_name and in_name in by_name:
                a = by_name[in_name]
                if isinstance(a, np.ndarray) and a.dtype.kind == "f":
                    finite = a[np.isfinite(a)]
                    if finite.size:
                        print(f"      input '{in_name}': shape={list(a.shape)} dtype={a.dtype} "
                              f"min={finite.min():.4g} max={finite.max():.4g} "
                              f"mean={finite.mean():.4g} nan={int(np.isnan(a).sum())} "
                              f"inf={int(np.isinf(a).sum())}")
                    else:
                        print(f"      input '{in_name}': all NaN/Inf (shape={list(a.shape)})")
                elif isinstance(a, np.ndarray):
                    print(f"      input '{in_name}': shape={list(a.shape)} dtype={a.dtype}")
                else:
                    print(f"      input '{in_name}': non-tensor")
            else:
                # initializer
                for init in model.graph.initializer:
                    if init.name == in_name:
                        ia = onnx.numpy_helper.to_array(init)
                        finite = ia[np.isfinite(ia)] if ia.dtype.kind == "f" else ia
                        if hasattr(finite, "size") and finite.size:
                            print(f"      input '{in_name}' (init): shape={list(ia.shape)} "
                                  f"dtype={ia.dtype} "
                                  f"min={float(finite.min()):.4g} max={float(finite.max()):.4g}")
                        break


if __name__ == "__main__":
    main()
