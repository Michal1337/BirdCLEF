"""Quick standalone check on a Perch cache: honest OOF macro-AUC of raw scores.

Mirrors the `honest_oof_auc` helper from the LB notebook (GroupKFold by
filename, never split a 60-s file across folds). Runs in seconds — no
training. If macro-AUC ≈ 0.81 on the labeled-soundscape pool, the cache is
healthy. A big drop (e.g. 0.5x or NaNs) means score arrays are scrambled,
classes are misordered, or the cache was built against a different
sample_submission.csv.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._check_perch_cache \\
        --meta /path/to/perch_meta.parquet \\
        --npz  /path/to/perch_arrays.npz

Auto-detects array key names — handles both:
    {scores_full_raw, emb_full}            (majkel1337 cache)
    {scores, embs}                         (vyankteshdwivedi cache)
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


def _resolve_score_array(npz: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray]:
    """Pick the raw-Perch-scores array out of the npz, handling both naming
    conventions used by different cache builders.
    """
    for key in ("scores_full_raw", "scores", "perch_scores", "logits"):
        if key in npz.files:
            return key, npz[key].astype(np.float32)
    raise SystemExit(
        f"npz has no recognised score array. Found keys: {list(npz.files)}.\n"
        "Expected one of: scores_full_raw, scores, perch_scores, logits."
    )


def _build_full_file_label_frame(
    soundscape_labels_csv: Path, sample_sub_csv: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Replicates the cand.ipynb / LB notebook label-parsing pipeline.

    Returns:
        full_rows: DataFrame of (row_id, filename, end_sec, original index)
                   for every window of every fully-labeled (12-window) file.
        Y_FULL:    multi-hot label matrix (N_rows, N_classes), aligned to
                   full_rows.
        primary_labels: canonical class order from sample_submission columns.
    """
    sample_sub = pd.read_csv(sample_sub_csv)
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    N_WINDOWS = 12

    sc_labels = pd.read_csv(soundscape_labels_csv)

    def union(series):
        out: set[str] = set()
        for x in series:
            if pd.notna(x):
                for t in str(x).split(";"):
                    t = t.strip()
                    if t:
                        out.add(t)
        return sorted(out)

    sc = (
        sc_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union)
        .reset_index(name="label_list")
    )
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = (
        sc["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + sc["end_sec"].astype(str)
    )

    Y_SC = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    windows_per_file = sc.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc["fully_labeled"] = sc["filename"].isin(full_files)

    full_rows = (
        sc[sc["fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    Y_full = Y_SC[full_rows["index"].to_numpy()]
    return full_rows, Y_full, primary_labels


def _macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    if not keep.any():
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def _honest_oof_macro_auc(
    scores: np.ndarray, Y: np.ndarray, meta: pd.DataFrame, n_splits: int,
) -> float:
    """GroupKFold-by-filename macro AUC. Identical to the cand.ipynb helper.

    The 'OOF' loop here is a no-op for raw Perch logits (no model fits per
    fold) — we still run the iteration so the function shape matches the
    notebook helper and a future swap to a per-fold model is trivial.
    """
    if "filename" not in meta.columns:
        raise SystemExit("cache meta is missing 'filename' column required for GroupKFold.")
    groups = meta["filename"].to_numpy()
    gkf = GroupKFold(n_splits=int(n_splits))
    oof = np.zeros_like(scores, dtype=np.float32)
    for _, va_idx in gkf.split(scores, groups=groups):
        oof[va_idx] = scores[va_idx]
    return _macro_auc(Y, oof)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True,
                    help="Path to the cache meta parquet (must have 'row_id' + 'filename').")
    ap.add_argument("--npz", type=Path, required=True,
                    help="Path to the cache arrays npz (must have a scores array).")
    ap.add_argument("--soundscape-labels", type=Path,
                    default=REPO_ROOT / "data" / "train_soundscapes_labels.csv")
    ap.add_argument("--sample-sub", type=Path,
                    default=REPO_ROOT / "data" / "sample_submission.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    for p in (args.meta, args.npz, args.soundscape_labels, args.sample_sub):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    print(f"[check] cache meta: {args.meta}")
    print(f"[check] cache npz : {args.npz}")
    meta = pd.read_parquet(args.meta)
    if "row_id" not in meta.columns or "filename" not in meta.columns:
        raise SystemExit(
            f"cache meta lacks required cols. Found: {list(meta.columns)}; "
            "expected at least 'row_id' and 'filename'."
        )
    npz = np.load(args.npz)
    score_key, sc_tr = _resolve_score_array(npz)
    print(f"[check] score array: '{score_key}'  shape={sc_tr.shape}  "
          f"mean={sc_tr.mean():.4f}  std={sc_tr.std():.4f}")
    if not np.isfinite(sc_tr).all():
        n_bad = int((~np.isfinite(sc_tr)).sum())
        print(f"[check] WARN: {n_bad} non-finite cells in scores — clipping for AUC")
        sc_tr = np.nan_to_num(sc_tr, nan=0.0, posinf=0.0, neginf=0.0)

    full_rows, Y_full, primary_labels = _build_full_file_label_frame(
        args.soundscape_labels, args.sample_sub,
    )
    n_classes = len(primary_labels)
    if sc_tr.shape[1] != n_classes:
        raise SystemExit(
            f"score array width {sc_tr.shape[1]} != {n_classes} primary labels.\n"
            "Cache was built against a different sample_submission.csv. "
            "Either re-build the cache or point --sample-sub at the matching version."
        )

    # The cache typically covers ALL soundscapes (labeled + unlabeled, ~10k
    # files × 12 windows = ~125k rows). Only ~59 files are fully-labeled.
    # Filter cache rows down to the fully-labeled subset before computing
    # honest OOF — unlabeled rows have no GT and would only add noise.
    full_idx_by_row = dict(zip(full_rows["row_id"].astype(str), full_rows.index.tolist()))
    cache_row_ids = meta["row_id"].astype(str).to_numpy()
    in_full = np.array([rid in full_idx_by_row for rid in cache_row_ids])
    n_total = len(cache_row_ids)
    n_kept = int(in_full.sum())
    n_dropped = n_total - n_kept
    print(f"[check] cache rows = {n_total}; fully-labeled rows kept = {n_kept}  "
          f"(dropped {n_dropped} unlabeled-soundscape rows)")
    if n_kept == 0:
        raise SystemExit(
            "No cache rows match the fully-labeled pool. Either the cache is "
            "from a totally different sample_sub, or row_id formatting drifted."
        )
    if n_kept != len(full_rows):
        # Some labeled files aren't represented in the cache — flag but don't fail.
        print(f"[check] WARN: {len(full_rows) - n_kept} fully-labeled rows "
              f"have no cache entry; AUC will skip those rows.")
    sc_tr = sc_tr[in_full]
    meta = meta.loc[in_full].reset_index(drop=True)
    cache_to_full = np.array(
        [full_idx_by_row[rid] for rid in meta["row_id"].astype(str)],
        dtype=np.int64,
    )
    Y_aligned = Y_full[cache_to_full]
    n_seen = int((Y_aligned.sum(axis=0) > 0).sum())
    print(f"[check] active classes in evaluation subset (≥1 positive) = {n_seen}/{n_classes}")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = _honest_oof_macro_auc(sc_tr, Y_aligned, meta, n_splits=args.n_splits)
    print(f"[check] honest OOF macro-AUC ({args.n_splits}-fold by filename) = {auc:.6f}")
    print()
    # Calibration thresholds based on observed BirdCLEF26 raw-Perch behaviour
    # on this 708-row labeled-soundscape pool: known healthy caches land at
    # ~0.73-0.74 macro AUC. The "expected" number depends on the Perch
    # checkpoint, proxy-logit map for unmapped species, and aggregation —
    # so treat anything in the 0.70-0.76 range as plausible and use
    # _compare_perch_caches.py to localize a small (~0.01) delta.
    if not np.isfinite(auc):
        print("[check] STATUS: BROKEN — AUC is non-finite. Cache likely scrambled.")
    elif auc < 0.55:
        print("[check] STATUS: BROKEN — AUC < 0.55, barely better than random.")
    elif auc < 0.65:
        print("[check] STATUS: SUSPICIOUS — well below the ~0.73 expected for raw Perch. "
              "Likely class-index drift or row-order mismatch; spot-check a known class.")
    elif auc < 0.70:
        print("[check] STATUS: LOW — close to but below the typical raw-Perch range.")
    elif auc <= 0.78:
        print("[check] STATUS: OK — within the typical raw-Perch range "
              "(~0.73, ±~0.03 depending on proxy logits / Perch version).")
    else:
        print("[check] STATUS: BETTER-THAN-EXPECTED — likely the cache stores something "
              "stronger than raw Perch (e.g. fine-tuned head, ensemble logits).")


if __name__ == "__main__":
    main()
