"""Fork LB_093.ipynb and append SED-5-fold inference + SSM/SED blend cells.

Why fork rather than re-implement the SSM stack:
  LB_093.ipynb is the load-bearing artifact that scored 0.93 on public LB.
  Re-implementing its stack (Perch + ProtoSSM + MLP probes + ResidualSSM +
  per-class temperatures + 4-step post-processing + isotonic thresholds)
  inside the new birdclef/ package would be ~2k lines of code that we cannot
  validate against LB until we burn a quota. The cheaper path is to keep the
  notebook as-is and graft SED on the end.

The output notebook does this in order:
  1) Cells 0..25 of LB_093.ipynb run unchanged. After cell 25 the variables
     `probs`, `meta_te`, `test_paths`, `PRIMARY_LABELS`, `N_WINDOWS`,
     `WINDOW_SAMPLES`, `FILE_SAMPLES`, `tqdm`, `np`, `pd`, `sf` are all in
     scope. submission.csv has been written with pure-SSM probabilities.
  2) NEW CELL — load N SED ONNX checkpoints from a Kaggle dataset path,
     decode each test OGG once, run all folds, sigmoid + average → sed_probs
     aligned to meta_te["row_id"].
  3) NEW CELL — blended = w_ssm * probs + w_sed * sed_probs, clipped, written
     over submission.csv.

If the SED dataset is missing (e.g. you forgot to attach it), the appended
cells degrade gracefully: they print a warning, do NOT overwrite
submission.csv, and the notebook still produces the pure-SSM output. So a
broken attach is at worst a no-op, never a regression.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


# Default Kaggle dataset paths. These match the conventions in
# birdclef/submit/build_notebook.py so users only have to attach one
# additional dataset (their SED ONNX export) on top of LB_093.ipynb's
# existing attachments.
DEFAULT_SED_GLOB = "/kaggle/input/birdclef-sed-onnx/fold*/best.onnx"


def _make_sed_inference_cell(sed_paths: List[str], sed_glob: str | None) -> str:
    """Build the SED inference cell.

    If `sed_paths` is provided, the list is hard-coded (most reliable on
    Kaggle since dataset slugs are deterministic per upload). Otherwise the
    cell uses a glob expansion so the user can re-export checkpoints without
    rebuilding the notebook.
    """
    if sed_paths:
        paths_literal = "SED_ONNX_PATHS = [\n" + "".join(
            f"    {json.dumps(p)},\n" for p in sed_paths
        ) + "]"
    else:
        paths_literal = (
            "import glob as _glob\n"
            f"SED_ONNX_PATHS = sorted(_glob.glob({json.dumps(sed_glob or DEFAULT_SED_GLOB)}))\n"
            "print(f'[blend] glob expanded to {len(SED_ONNX_PATHS)} SED ONNX file(s)')"
        )

    return f'''# === SED 5-fold ensemble inference (appended by build_blend_notebook.py) ===
# Runs after the SSM pipeline finishes. The variable `probs` from cell 25 holds
# the pure-SSM submission-shaped probabilities; we compute SED-ensemble probs in
# the same row order and store them in `sed_probs` for the next cell to blend.
import onnxruntime as ort

{paths_literal}

def _np_dtype_from_ort(type_str):
    if "float16" in type_str:
        return np.float16
    if "double" in type_str:
        return np.float64
    return np.float32

# Detect missing checkpoints early — at worst we want to fall back to pure SSM
# (which we already wrote to submission.csv), never crash.
_sed_paths_existing = [p for p in SED_ONNX_PATHS if Path(p).exists()]
if len(_sed_paths_existing) == 0:
    print("[blend] WARNING: no SED ONNX paths exist on disk")
    print(f"[blend] tried: {{SED_ONNX_PATHS}}")
    print("[blend] keeping pure-SSM submission.csv from cell 25")
    sed_probs = None
else:
    if len(_sed_paths_existing) < len(SED_ONNX_PATHS):
        print(f"[blend] WARNING: only {{len(_sed_paths_existing)}}/{{len(SED_ONNX_PATHS)}} "
              "SED ONNX files exist; ensembling the available subset")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sed_sessions = [
        ort.InferenceSession(p, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        for p in _sed_paths_existing
    ]
    sed_input_names = [s.get_inputs()[0].name for s in sed_sessions]
    sed_input_dtypes = [_np_dtype_from_ort(s.get_inputs()[0].type) for s in sed_sessions]
    print(f"[blend] loaded {{len(sed_sessions)}} SED ONNX session(s); "
          f"input dtypes={{[str(np.dtype(d)) for d in sed_input_dtypes]}}")

    n_classes_blend = len(PRIMARY_LABELS)
    sed_probs_rows = []
    for p in tqdm(test_paths, desc="SED 5-fold"):
        y, _ = sf.read(str(p), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        wins = y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        ensemble = np.zeros((N_WINDOWS, n_classes_blend), dtype=np.float32)
        for s, inp, dt in zip(sed_sessions, sed_input_names, sed_input_dtypes):
            x = wins if wins.dtype == dt else wins.astype(dt)
            logits = s.run(None, {{inp: x}})[0]
            ensemble += 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -30, 30)))
        ensemble /= len(sed_sessions)
        sed_probs_rows.append(ensemble)
    sed_probs = np.concatenate(sed_probs_rows, axis=0).astype(np.float32)
    print(f"[blend] sed_probs shape={{sed_probs.shape}}  "
          f"mean={{float(sed_probs.mean()):.4f}}  "
          f"max={{float(sed_probs.max()):.4f}}")
    if sed_probs.shape != probs.shape:
        print(f"[blend] WARNING: sed_probs {{sed_probs.shape}} vs probs {{probs.shape}} mismatch; "
              "discarding SED branch")
        sed_probs = None
'''


def _make_blend_cell(w_ssm: float, w_sed: float) -> str:
    """Cell that blends sed_probs + probs and overwrites submission.csv."""
    return f'''# === SSM/SED blend (appended by build_blend_notebook.py) ===
# w_ssm = {w_ssm} (pure-SSM weight, leave most of the mass here since SSM owns
#                 the LB)
# w_sed = {w_sed} (small SED contribution for diversity)
# If sed_probs is None (SED dataset missing or shape-mismatched), skip the blend
# entirely — the pure-SSM submission.csv from cell 25 stays in place.
W_SSM = {w_ssm}
W_SED = {w_sed}

if sed_probs is None:
    print("[blend] no SED probs; submission.csv stays pure-SSM (LB-0.93 baseline)")
else:
    # Both inputs are already in [0, 1] (sigmoid + post-proc on the SSM side,
    # sigmoid on the SED side). Blend on the probability scale, not the logit
    # scale — matches what _08_ensemble.py does and avoids amplifying near-0/1
    # tails when one member is overconfident.
    blended = W_SSM * probs + W_SED * sed_probs
    blended = np.clip(blended, 0.0, 1.0)

    sub = pd.DataFrame(blended.astype(np.float32), columns=PRIMARY_LABELS)
    sub.insert(0, "row_id", meta_te["row_id"].values)
    assert list(sub.columns) == ["row_id"] + PRIMARY_LABELS
    assert len(sub) == len(test_paths) * N_WINDOWS
    assert not sub.isna().any().any()
    sub.to_csv("submission.csv", index=False)
    print(f"[blend] OVERWROTE submission.csv with blend "
          f"W_SSM={{W_SSM}}, W_SED={{W_SED}} — shape {{sub.shape}}")
    print(f"[blend] blended range [{{float(blended.min()):.4f}}, "
          f"{{float(blended.max()):.4f}}]  mean={{float(blended.mean()):.4f}}")
'''


def build(
    src_ipynb: Path,
    out_ipynb: Path,
    sed_paths: List[str],
    sed_glob: str | None,
    w_ssm: float,
    w_sed: float,
) -> None:
    if not src_ipynb.exists():
        raise SystemExit(f"Source notebook not found: {src_ipynb}")
    nb = json.loads(src_ipynb.read_text(encoding="utf-8"))

    # Strip empty trailing cells (cells 26, 27 in LB_093.ipynb are blank).
    # Keeps the appended cells in the obvious place at the end.
    while nb["cells"] and not "".join(nb["cells"][-1].get("source", [])).strip():
        nb["cells"].pop()

    sed_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": _make_sed_inference_cell(sed_paths, sed_glob).splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }
    blend_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": _make_blend_cell(w_ssm, w_sed).splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"## Blend appendix — SSM (LB-0.93 baseline) + 5-fold SED ensemble\n\n",
            f"`W_SSM = {w_ssm}`, `W_SED = {w_sed}`. SED checkpoints are loaded from "
            f"the Kaggle attachment(s); if missing, the notebook falls back silently "
            f"to the pure-SSM submission.csv from cell 25.\n",
        ],
    }
    nb["cells"].extend([header_cell, sed_cell, blend_cell])

    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    out_ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_sed = len(sed_paths) if sed_paths else f"glob({sed_glob or DEFAULT_SED_GLOB})"
    print(f"[blend-nb] wrote {out_ipynb}")
    print(f"[blend-nb]   source: {src_ipynb}")
    print(f"[blend-nb]   appended: header + SED ensemble cell + blend cell")
    print(f"[blend-nb]   SED members: {n_sed}")
    print(f"[blend-nb]   weights: W_SSM={w_ssm}  W_SED={w_sed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="LB_093.ipynb",
                    help="Source SSM notebook (the one that already scores 0.93). Default: LB_093.ipynb")
    ap.add_argument("--out", required=True,
                    help="Output notebook path (e.g. outputs/submit/blend_w90.ipynb)")
    sed_group = ap.add_mutually_exclusive_group()
    sed_group.add_argument("--sed-onnx", nargs="+", default=None,
                    help="Explicit SED ONNX paths on Kaggle "
                         "(e.g. /kaggle/input/birdclef-sed-onnx/fold0/best.onnx ...)")
    sed_group.add_argument("--sed-glob", default=None,
                    help=f"Glob pattern (resolved at notebook runtime). "
                         f"Default if neither flag set: {DEFAULT_SED_GLOB}")
    ap.add_argument("--w-ssm", type=float, default=0.90,
                    help="Weight on the SSM (LB-0.93) member. Default 0.90.")
    ap.add_argument("--w-sed", type=float, default=0.10,
                    help="Weight on the SED ensemble member. Default 0.10.")
    args = ap.parse_args()

    if abs((args.w_ssm + args.w_sed) - 1.0) > 1e-6:
        print(f"[blend-nb] warning: weights sum to {args.w_ssm + args.w_sed:.4f}, "
              "not 1.0 — they'll be applied as-is, not renormalized")

    build(
        src_ipynb=Path(args.src),
        out_ipynb=Path(args.out),
        sed_paths=args.sed_onnx,
        sed_glob=args.sed_glob,
        w_ssm=args.w_ssm,
        w_sed=args.w_sed,
    )


if __name__ == "__main__":
    main()
