"""Compare two Perch caches on the labeled-soundscape pool to localize WHERE
their macro-AUC gap comes from.

Outputs:
  - Per-cache: shape, mean/std, embedding shape, row coverage, macro AUC
  - Side-by-side per-class AUC table sorted by |delta|
  - Score-distribution and embedding-distribution diff stats
  - Class-coverage diff (classes that have signal in one cache but not the other)
  - Top-K classes contributing most to the macro-AUC delta

Run:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._compare_perch_caches \\
        --cache-a-meta .../A_meta.parquet --cache-a-npz .../A_arrays.npz \\
        --cache-b-meta .../B_meta.parquet --cache-b-npz .../B_arrays.npz
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.scripts._check_perch_cache import (
    _build_full_file_label_frame,
    _resolve_score_array,
)


def _resolve_emb_array(npz: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray] | tuple[None, None]:
    for key in ("emb_full", "embs", "perch_embs", "embeddings"):
        if key in npz.files:
            return key, npz[key].astype(np.float32)
    return None, None


def _per_class_auc(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    n_cls = Y.shape[1]
    out = np.full(n_cls, np.nan, dtype=np.float64)
    pos = Y.sum(axis=0)
    valid = (pos > 0) & (pos < Y.shape[0])
    for c in np.where(valid)[0]:
        try:
            out[c] = roc_auc_score(Y[:, c], P[:, c])
        except ValueError:
            pass
    return out


def _macro_oof_auc(P: np.ndarray, Y: np.ndarray, meta: pd.DataFrame, n_splits: int) -> float:
    groups = meta["filename"].to_numpy()
    gkf = GroupKFold(n_splits=int(n_splits))
    oof = np.zeros_like(P, dtype=np.float32)
    for _, va_idx in gkf.split(P, groups=groups):
        oof[va_idx] = P[va_idx]
    keep = Y.sum(axis=0) > 0
    if not keep.any():
        return float("nan")
    return float(roc_auc_score(Y[:, keep], oof[:, keep], average="macro"))


def _filter_cache_to_labeled(
    meta: pd.DataFrame, scores: np.ndarray, embs: np.ndarray | None,
    full_rows: pd.DataFrame, Y_full: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray]:
    full_idx_by_row = dict(zip(full_rows["row_id"].astype(str), full_rows.index.tolist()))
    cache_row_ids = meta["row_id"].astype(str).to_numpy()
    in_full = np.array([rid in full_idx_by_row for rid in cache_row_ids])
    sub_meta = meta.loc[in_full].reset_index(drop=True)
    sub_scores = scores[in_full]
    sub_embs = embs[in_full] if embs is not None else None
    cache_to_full = np.array(
        [full_idx_by_row[rid] for rid in sub_meta["row_id"].astype(str)],
        dtype=np.int64,
    )
    Y_aligned = Y_full[cache_to_full]
    return sub_meta, sub_scores, sub_embs, Y_aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-a-meta", type=Path, required=True)
    ap.add_argument("--cache-a-npz",  type=Path, required=True)
    ap.add_argument("--cache-b-meta", type=Path, required=True)
    ap.add_argument("--cache-b-npz",  type=Path, required=True)
    ap.add_argument("--soundscape-labels", type=Path,
                    default=REPO_ROOT / "data" / "train_soundscapes_labels.csv")
    ap.add_argument("--sample-sub", type=Path,
                    default=REPO_ROOT / "data" / "sample_submission.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=15,
                    help="Print top-K classes by |AUC delta|.")
    args = ap.parse_args()

    full_rows, Y_full, primary_labels = _build_full_file_label_frame(
        args.soundscape_labels, args.sample_sub,
    )
    n_classes = len(primary_labels)

    def load_cache(label, meta_path, npz_path):
        print(f"[{label}] meta: {meta_path}")
        print(f"[{label}] npz : {npz_path}")
        meta = pd.read_parquet(meta_path)
        npz = np.load(npz_path)
        s_key, scores = _resolve_score_array(npz)
        e_key, embs = _resolve_emb_array(npz)
        if scores.shape[1] != n_classes:
            raise SystemExit(
                f"{label}: score width {scores.shape[1]} != {n_classes} classes."
            )
        n_finite_pct = float(np.isfinite(scores).mean()) * 100.0
        print(f"[{label}] scores '{s_key}'  shape={scores.shape}  "
              f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
              f"finite={n_finite_pct:.2f}%  "
              f"min={scores.min():.4f}  max={scores.max():.4f}  "
              f"dtype={scores.dtype}")
        if embs is not None:
            print(f"[{label}] embs   '{e_key}'  shape={embs.shape}  "
                  f"mean={embs.mean():.4f}  std={embs.std():.4f}")
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        meta_f, scores_f, embs_f, Y_f = _filter_cache_to_labeled(
            meta, scores, embs, full_rows, Y_full,
        )
        print(f"[{label}] kept {len(meta_f)}/{len(meta)} rows after filtering "
              f"to fully-labeled pool")
        return meta_f, scores_f, embs_f, Y_f

    print("=" * 78)
    print("Loading cache A")
    print("=" * 78)
    meta_a, sc_a, emb_a, Y_a = load_cache("A", args.cache_a_meta, args.cache_a_npz)
    print()
    print("=" * 78)
    print("Loading cache B")
    print("=" * 78)
    meta_b, sc_b, emb_b, Y_b = load_cache("B", args.cache_b_meta, args.cache_b_npz)
    print()

    # Row-alignment check (must hold for an apples-to-apples diff)
    common_rows = sorted(set(meta_a["row_id"]) & set(meta_b["row_id"]))
    print(f"[align] cache A rows={len(meta_a)}, B rows={len(meta_b)}, "
          f"common={len(common_rows)}")
    if not common_rows:
        raise SystemExit("No row_ids in common — caches don't overlap on labeled rows.")
    a_idx = {r: i for i, r in enumerate(meta_a["row_id"].astype(str))}
    b_idx = {r: i for i, r in enumerate(meta_b["row_id"].astype(str))}
    a_pick = np.array([a_idx[r] for r in common_rows], dtype=np.int64)
    b_pick = np.array([b_idx[r] for r in common_rows], dtype=np.int64)
    sc_a, sc_b = sc_a[a_pick], sc_b[b_pick]
    Y_a, Y_b = Y_a[a_pick], Y_b[b_pick]
    if not np.array_equal(Y_a, Y_b):
        raise SystemExit(
            "Y matrices disagree on common rows — implies sample_sub or label "
            "parsing differs between cache contexts. Bug in this script."
        )
    Y = Y_a
    common_meta = meta_a.iloc[a_pick].reset_index(drop=True)
    print()

    # Score distribution diff
    abs_diff = np.abs(sc_a - sc_b)
    print(f"[diff] score |a-b|: max={abs_diff.max():.4f}  "
          f"mean={abs_diff.mean():.5f}  median={np.median(abs_diff):.5f}")
    # Rank-correlation (Spearman ~= cheap proxy here)
    try:
        from scipy.stats import spearmanr
        rho_per_class = []
        for c in range(sc_a.shape[1]):
            r, _ = spearmanr(sc_a[:, c], sc_b[:, c])
            if np.isfinite(r):
                rho_per_class.append(r)
        if rho_per_class:
            arr = np.array(rho_per_class)
            print(f"[diff] per-class Spearman ρ(a, b): "
                  f"median={np.median(arr):.4f}  min={arr.min():.4f}  "
                  f"5%={np.percentile(arr, 5):.4f}")
    except ImportError:
        pass
    print()

    # OOF macro AUCs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_a = _macro_oof_auc(sc_a, Y, common_meta, n_splits=args.n_splits)
        auc_b = _macro_oof_auc(sc_b, Y, common_meta, n_splits=args.n_splits)
    print(f"[auc] cache A macro-OOF = {auc_a:.6f}")
    print(f"[auc] cache B macro-OOF = {auc_b:.6f}")
    print(f"[auc] delta (B - A)     = {auc_b - auc_a:+.6f}")
    print()

    # Per-class AUC + delta
    pa = _per_class_auc(Y, sc_a)
    pb = _per_class_auc(Y, sc_b)
    delta = pb - pa
    pos_per_cls = Y.sum(axis=0)
    valid = ~np.isnan(pa) & ~np.isnan(pb)
    print(f"[per-class] {int(valid.sum())} / {n_classes} classes evaluable in both caches")

    # Class-coverage diff: classes evaluable in one but not the other
    a_only = np.where(~np.isnan(pa) & np.isnan(pb))[0]
    b_only = np.where(np.isnan(pa) & ~np.isnan(pb))[0]
    if len(a_only) or len(b_only):
        print(f"[per-class] WARN: {len(a_only)} classes evaluable only in A, "
              f"{len(b_only)} only in B (different row coverage)")

    # Top-K classes by |delta|
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(-np.abs(delta[valid_idx]))]
    print()
    print(f"[per-class] Top-{args.top_k} classes by |B - A|:")
    print(f"  {'class':<12s}  {'pos':>4s}  {'auc_A':>7s}  {'auc_B':>7s}  {'delta':>7s}")
    for c in order[:args.top_k]:
        print(f"  {primary_labels[c]:<12s}  {int(pos_per_cls[c]):>4d}  "
              f"{pa[c]:>7.4f}  {pb[c]:>7.4f}  {delta[c]:>+7.4f}")

    # Cumulative macro-delta contribution: how much of the 0.01 gap comes from
    # the top-K classes vs the long tail?
    print()
    sorted_abs_delta = np.sort(np.abs(delta[valid]))[::-1]
    n_total = len(sorted_abs_delta)
    macro_delta_total = float(delta[valid].mean())  # this equals macro_b - macro_a IF computed over the valid set
    for k in (5, 10, 20, 50):
        if k > n_total:
            continue
        topk_contrib = float(np.abs(delta[valid])[np.argsort(-np.abs(delta[valid]))][:k].sum() / n_total)
        print(f"[per-class] top-{k} |delta| account for "
              f"{topk_contrib / max(abs(macro_delta_total), 1e-9) * 100.0:.1f}% "
              f"of the macro-delta magnitude")

    # Embedding sanity
    if emb_a is not None and emb_b is not None and emb_a.shape == emb_b.shape:
        print()
        emb_a_s = emb_a[a_pick]
        emb_b_s = emb_b[b_pick]
        d = np.abs(emb_a_s - emb_b_s)
        print(f"[emb] embedding shapes match {emb_a.shape}; "
              f"|a-b| max={d.max():.4f}  mean={d.mean():.5f}")
        if d.max() < 1e-4:
            print("[emb] embeddings are bit-identical → the gap is in the SCORES path "
                  "(different head / different proxy logits / different post-proc), "
                  "not in the encoder.")
        else:
            print("[emb] embeddings DIFFER → the gap is upstream of the head "
                  "(different Perch model / different preprocessing). Check sample "
                  "rate, mono-mix, and Perch checkpoint version.")
    elif emb_a is None or emb_b is None:
        print()
        print("[emb] one or both caches lack embeddings — can't isolate "
              "encoder vs head. Build with embs to localize.")
    else:
        print()
        print(f"[emb] embedding shapes DIFFER: A={emb_a.shape} vs B={emb_b.shape}. "
              "Different Perch backbone or different aggregation.")


if __name__ == "__main__":
    main()
