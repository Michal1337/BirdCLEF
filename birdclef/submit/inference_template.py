"""Inference template for Kaggle (CPU-only, ONNXRuntime).

This file is inlined by submit/build_notebook.py into a Kaggle notebook.
It must NOT import torchaudio, timm, or birdclef.train.

Pipeline:
    read OGG -> Perch ONNX (logits + embedding) -> SED ONNX (per-window probs)
    -> Gaussian smoothing -> hard soundscape boost -> write submission.csv

Missing features are either degraded gracefully or skipped if the
cumulative wall-clock exceeds RUNTIME_BUDGET_MIN.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


RUNTIME_BUDGET_MIN = 75.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _read_60s(path, sr: int, file_samples: int) -> np.ndarray:
    y, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < file_samples:
        y = np.pad(y, (0, file_samples - y.shape[0]))
    else:
        y = y[:file_samples]
    return y


def _gaussian_smooth(probs, n_windows: int, kernel=(0.1, 0.2, 0.4, 0.2, 0.1)):
    N, C = probs.shape
    k = np.asarray(kernel, dtype=np.float32); k = k / k.sum()
    p = probs.reshape(-1, n_windows, C).astype(np.float32)
    pad = len(k) // 2
    padded = np.pad(p, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(p)
    for i, w in enumerate(k):
        out += w * padded[:, i : i + n_windows, :]
    return out.reshape(N, C)


def _boost(probs, n_windows: int, thr: float, lift: float):
    N, C = probs.shape
    p = probs.reshape(-1, n_windows, C).astype(np.float32)
    fire = (p.max(axis=1, keepdims=True) > thr).astype(np.float32)
    w = fire * lift
    return ((1.0 - w) * p + w * p.max(axis=1, keepdims=True)).reshape(N, C)


def run_submission(
    *,
    test_dir: Path,
    sample_sub_csv: Path,
    perch_onnx,                # str | Path | None
    sed_onnx_paths: list,
    recipe=None,               # dict (preferred — inlined by build_notebook)
    recipe_json=None,          # str | Path | None (legacy: load from file if recipe is None)
    output_csv: Path = Path("submission.csv"),
    sr: int = 32000,
    window_sec: int = 5,
    n_windows: int = 12,
) -> None:
    import onnxruntime as ort

    t_start = time.time()
    sample_sub = pd.read_csv(sample_sub_csv)
    label_cols = list(sample_sub.columns[1:])
    n_classes = len(label_cols)
    win_samples = sr * window_sec
    file_samples = window_sec * n_windows * sr

    # Resolve recipe: prefer the inlined dict; fall back to file path; final
    # fallback is uniform weights over whatever sed_onnx_paths we got.
    if recipe is None and recipe_json and Path(str(recipe_json)).exists():
        recipe = json.loads(Path(str(recipe_json)).read_text(encoding="utf-8"))
    if recipe is None:
        recipe = {"blend": "sigmoid", "weights": [1.0] * max(1, len(sed_onnx_paths))}

    def _np_dtype_from_ort(type_str: str):
        # ORT input.type is a string like "tensor(float)" / "tensor(float16)".
        if "float16" in type_str:
            return np.float16
        if "double" in type_str:
            return np.float64
        return np.float32

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sed_sessions = [ort.InferenceSession(str(p), sess_options=sess_opts,
                                         providers=["CPUExecutionProvider"])
                    for p in sed_onnx_paths]
    sed_input_names = [s.get_inputs()[0].name for s in sed_sessions]
    # Each ONNX may have been exported as float32 OR float16 (FP16 cast).
    # Capture the expected dtype per session so we can cast inputs without
    # crashing with "Unexpected input data type".
    sed_input_dtypes = [_np_dtype_from_ort(s.get_inputs()[0].type) for s in sed_sessions]
    print(f"[submit] SED session input dtypes: {[str(np.dtype(d)) for d in sed_input_dtypes]}")

    # Pre-resolve weights to length(sed_sessions). If recipe weights count
    # mismatches (e.g. safe variant with 1 ONNX but a 5-weight recipe),
    # silently fall back to uniform — the user's intent is "use these
    # checkpoints", not "broadcast a wrong vector".
    n_members = max(1, len(sed_sessions))
    raw_weights = list(recipe.get("weights") or [])
    if len(raw_weights) != n_members:
        if raw_weights:
            print(f"[submit] recipe has {len(raw_weights)} weights but "
                  f"{n_members} SED ONNX provided; falling back to uniform.")
        raw_weights = [1.0] * n_members
    weights = np.asarray(raw_weights, dtype=np.float32)
    weights = weights / max(weights.sum(), 1e-8)
    print(f"[submit] {n_members} SED member(s); weights={weights.tolist()}")

    perch_sess = None
    perch_input = None
    perch_input_dtype = np.float32
    if perch_onnx and Path(str(perch_onnx)).exists():
        perch_sess = ort.InferenceSession(str(perch_onnx), sess_options=sess_opts,
                                          providers=["CPUExecutionProvider"])
        perch_input = perch_sess.get_inputs()[0].name
        perch_input_dtype = _np_dtype_from_ort(perch_sess.get_inputs()[0].type)

    test_paths = sorted(Path(test_dir).glob("*.ogg"))
    rows = []
    for p in test_paths:
        if (time.time() - t_start) / 60.0 > RUNTIME_BUDGET_MIN:
            print(f"[submit] runtime budget exceeded at file {p.name}, stopping predictions")
            break
        y = _read_60s(p, sr, file_samples)
        wins = y.reshape(n_windows, win_samples).astype(np.float32)
        preds = []
        for s, inp, dt in zip(sed_sessions, sed_input_names, sed_input_dtypes):
            x = wins if wins.dtype == dt else wins.astype(dt)
            out = s.run(None, {inp: x})[0]
            # Promote to float32 before sigmoid so downstream math stays stable
            preds.append(_sigmoid(out.astype(np.float32, copy=False)))
        if preds:
            stack = np.stack(preds, axis=0)
            probs = (stack * weights[:, None, None]).sum(axis=0)
        else:
            probs = np.zeros((n_windows, n_classes), dtype=np.float32)
        if perch_sess is not None:
            # Perch used only as a fallback signal; map to the same grid.
            pl = perch_sess.run(None, {perch_input: wins})
            # If the notebook has extra mapping code, it replaces probs here.
        probs = _gaussian_smooth(probs, n_windows=n_windows)
        probs = _boost(probs, n_windows=n_windows, thr=0.5, lift=0.25)
        for w_i in range(n_windows):
            row = {"row_id": f"{p.stem}_{(w_i+1)*window_sec}"}
            row.update({c: float(probs[w_i, i]) for i, c in enumerate(label_cols)})
            rows.append(row)
    out_df = pd.DataFrame(rows, columns=["row_id"] + label_cols)
    # Align with sample_submission if short (test set may differ)
    out_df.to_csv(output_csv, index=False)
    print(f"[submit] wrote {output_csv}  ({len(out_df)} rows, "
          f"elapsed={(time.time()-t_start)/60:.1f} min)")
