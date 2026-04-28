"""Compare fp32 vs int8 AST ONNX on the labeled-soundscape val pool.

Runs both ONNX sessions on the same files (the 708-row OOF pool that
backs `ssm_probs.npz` / `y_true.npy`) and reports macro / per-taxon AUC
plus probability drift, so we can decide whether INT8 quantization is
safe to ship before uploading to Kaggle.

Aligns to `outputs/blend_search/oof/meta.parquet` for honest
shape-matching with the SSM OOF.

Run:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._compare_ast_int8 \\
        --fp32 birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/model.onnx \\
        --int8 birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/model_int8.onnx \\
        --wrapper-cfg birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/wrapper_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import (
    FILE_SAMPLES, N_WINDOWS, OUTPUT_ROOT, SOUNDSCAPES, WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import load_taxonomy, primary_labels


_FBANK_MEAN = -4.2677
_FBANK_STD = 4.5689


def _fbank_batch(
    waveform_BxT: torch.Tensor,
    target_sr: int,
    max_length: int,
    num_mel_bins: int,
) -> torch.Tensor:
    """AudioSet-recipe kaldi fbank for a (B, T) waveform batch at target_sr.

    Matches `ASTSpectrogramClassifier._fbank_one` exactly so the ONNX inputs
    line up with what the model was trained on.
    """
    fbanks = []
    for b in range(waveform_BxT.size(0)):
        x = waveform_BxT[b : b + 1]
        fb = torchaudio.compliance.kaldi.fbank(
            x,
            htk_compat=True,
            sample_frequency=target_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=int(num_mel_bins),
            dither=0.0,
            frame_shift=10,
            frame_length=25,
        )
        fb = (fb - _FBANK_MEAN) / (_FBANK_STD * 2.0)
        n = fb.size(0)
        if n < max_length:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, max_length - n))
        elif n > max_length:
            fb = fb[:max_length, :]
        fbanks.append(fb)
    return torch.stack(fbanks, dim=0)  # (B, max_length, num_mel_bins)


def _per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    n_cls = y_true.shape[1]
    out = np.full(n_cls, np.nan, dtype=np.float64)
    pos = y_true.sum(axis=0)
    valid = (pos > 0) & (pos < y_true.shape[0])
    for c in np.where(valid)[0]:
        try:
            out[c] = roc_auc_score(y_true[:, c], y_score[:, c])
        except ValueError:
            pass
    return out


def _macro_per_taxon(
    aucs: np.ndarray, labels: list[str], tax: pd.DataFrame,
) -> dict[str, tuple[int, float]]:
    cls_map = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
    by_taxon: dict[str, list[float]] = {}
    for lb, a in zip(labels, aucs):
        if np.isnan(a):
            continue
        by_taxon.setdefault(cls_map.get(lb, "?"), []).append(float(a))
    return {k: (len(v), float(np.mean(v))) for k, v in sorted(by_taxon.items())}


def _make_session(path: Path, threads: int = 4) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", type=Path, required=True, help="fp32 model.onnx")
    ap.add_argument("--int8", type=Path, required=True, help="int8 model.onnx")
    ap.add_argument("--wrapper-cfg", type=Path, required=True,
                    help="wrapper_config.json sidecar from _freeze_ast_to_onnx")
    ap.add_argument("--ssm-meta", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "meta.parquet")
    ap.add_argument("--y-true", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "y_true.npy")
    ap.add_argument("--out-dir", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof",
                    help="Where to drop ast_fp32_probs.npz / ast_int8_probs.npz")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    for p in (args.fp32, args.int8, args.wrapper_cfg, args.ssm_meta, args.y_true):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    cfg = json.loads(args.wrapper_cfg.read_text(encoding="utf-8"))
    input_sr = int(cfg.get("input_sample_rate", 32000))
    target_sr = int(cfg.get("target_sample_rate", 16000))
    max_length = int(cfg.get("max_length", 512))
    n_mels = int(cfg.get("num_mel_bins", 128))
    n_classes = int(cfg.get("n_classes", 234))
    print(f"[cmp] wrapper: input_sr={input_sr} target_sr={target_sr} "
          f"max_length={max_length} n_mels={n_mels} n_classes={n_classes}")

    resample = (torchaudio.transforms.Resample(input_sr, target_sr)
                if input_sr != target_sr else torch.nn.Identity())

    print(f"[cmp] loading sessions...")
    sess_fp = _make_session(args.fp32, args.threads)
    sess_q = _make_session(args.int8, args.threads)
    in_name_fp = sess_fp.get_inputs()[0].name
    in_name_q = sess_q.get_inputs()[0].name

    meta = pd.read_parquet(args.ssm_meta)
    y_true = np.load(args.y_true)
    n_rows = len(meta)
    print(f"[cmp] aligning to {n_rows} rows; y_true shape={y_true.shape}")
    if y_true.shape[0] != n_rows or y_true.shape[1] != n_classes:
        raise SystemExit(f"y_true shape {y_true.shape} mismatches meta/n_classes")

    out_fp = np.zeros((n_rows, n_classes), dtype=np.float32)
    out_q = np.zeros((n_rows, n_classes), dtype=np.float32)

    file_groups: dict[str, list[int]] = {}
    for i, fn in enumerate(meta["filename"].astype(str).tolist()):
        file_groups.setdefault(fn, []).append(i)

    t_fp_total = 0.0
    t_q_total = 0.0
    skipped = 0

    for fname, row_idx in tqdm(file_groups.items(), desc="AST cmp"):
        path = SOUNDSCAPES / fname
        if not path.exists():
            print(f"  WARN: {path} missing — leaving zeros for {len(row_idx)} rows")
            skipped += 1
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
        )
        with torch.no_grad():
            wins_16k = resample(wins)
            fb = _fbank_batch(wins_16k, target_sr, max_length, n_mels)
        fb_np = fb.numpy().astype(np.float32)

        t0 = time.time()
        logits_fp = sess_fp.run(None, {in_name_fp: fb_np})[0]
        t_fp_total += time.time() - t0

        t0 = time.time()
        logits_q = sess_q.run(None, {in_name_q: fb_np})[0]
        t_q_total += time.time() - t0

        probs_fp = 1.0 / (1.0 + np.exp(-logits_fp))
        probs_q = 1.0 / (1.0 + np.exp(-logits_q))

        n = min(len(row_idx), N_WINDOWS)
        for j in range(n):
            out_fp[row_idx[j]] = probs_fp[j]
            out_q[row_idx[j]] = probs_q[j]

    if skipped:
        print(f"[cmp] WARN: skipped {skipped} missing files")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "ast_fp32_probs.npz", probs=out_fp)
    np.savez_compressed(args.out_dir / "ast_int8_probs.npz", probs=out_q)
    print(f"[cmp] wrote ast_fp32_probs.npz / ast_int8_probs.npz to {args.out_dir}")

    # ----- metrics -----
    labels = primary_labels()
    tax = load_taxonomy()

    aucs_fp = _per_class_auc(y_true, out_fp)
    aucs_q = _per_class_auc(y_true, out_q)
    valid = ~np.isnan(aucs_fp) & ~np.isnan(aucs_q)
    macro_fp = float(np.nanmean(aucs_fp[valid]))
    macro_q = float(np.nanmean(aucs_q[valid]))

    per_tax_fp = _macro_per_taxon(aucs_fp, labels, tax)
    per_tax_q = _macro_per_taxon(aucs_q, labels, tax)

    diff = out_q.astype(np.float64) - out_fp.astype(np.float64)
    abs_diff = np.abs(diff)
    print()
    print("=" * 78)
    print("AST  fp32  vs  int8  on labeled soundscapes")
    print("=" * 78)
    print(f"  rows: {n_rows}    classes evaluated: {int(valid.sum())}/{n_classes}")
    print(f"  fp32 forward total: {t_fp_total:.1f}s    "
          f"int8 forward total: {t_q_total:.1f}s    "
          f"speedup: {t_fp_total / max(t_q_total, 1e-9):.2f}x")
    print()
    print(f"  prob drift  (int8 - fp32):  "
          f"max|diff|={abs_diff.max():.4f}  mean|diff|={abs_diff.mean():.5f}")
    print()
    print(f"  macro AUC   fp32 = {macro_fp:.4f}")
    print(f"  macro AUC   int8 = {macro_q:.4f}    "
          f"delta = {macro_q - macro_fp:+.4f}")
    print()
    print("  per-taxon AUC")
    print(f"    {'taxon':<10}  {'n':>4}  {'fp32':>7}  {'int8':>7}  {'delta':>7}")
    for tx in sorted(set(per_tax_fp) | set(per_tax_q)):
        n_fp, a_fp = per_tax_fp.get(tx, (0, float("nan")))
        n_q, a_q = per_tax_q.get(tx, (0, float("nan")))
        print(f"    {tx:<10}  {n_fp:>4}  {a_fp:>7.4f}  {a_q:>7.4f}  "
              f"{a_q - a_fp:+.4f}")
    print()
    n_drop_05 = int(((aucs_fp - aucs_q) > 0.05).sum())
    n_drop_02 = int(((aucs_fp - aucs_q) > 0.02).sum())
    print(f"  per-class drops:  >0.02 AUC: {n_drop_02}    >0.05 AUC: {n_drop_05}")
    if n_drop_05 > 0:
        bad = np.argsort(aucs_q - aucs_fp)[:5]
        print(f"  worst-5 (label, fp32→int8):")
        for c in bad:
            if not np.isnan(aucs_fp[c]) and not np.isnan(aucs_q[c]):
                print(f"    {labels[c]:<10}  {aucs_fp[c]:.4f} -> {aucs_q[c]:.4f}  "
                      f"({aucs_q[c] - aucs_fp[c]:+.4f})")


if __name__ == "__main__":
    main()
