"""Per-class weakness audit for the Perch+SSM stack.

Produces a CSV + console table sorted by `ssm_auc` ascending — the classes the
Perch+SSM pipeline does worst on. Use this output to decide which classes
need (a) more focal recordings (Xeno-Canto, train.csv mining), (b) targeted
soundscape labelling, or (c) better post-processing.

Inputs expected (produced by `birdclef.scripts._07b_dump_oof_probs`):
    outputs/blend_search/oof/ssm_probs.npz   # full SSM stitched OOF probs
    outputs/blend_search/oof/y_true.npy
    outputs/blend_search/oof/meta.parquet

Optional:
    --include-perch    Also run raw Perch ONNX on the same files for a
                       per-class comparison (perch_auc, gap = perch - ssm).
                       Negative gap = SSM made the class worse.

Output:
    outputs/eda/per_class_audit.csv
    outputs/eda/per_class_audit.json   (summary + top-K weakest)

Run:
    # Step 1 — produce SSM OOF probs (if not already done)
    python -m birdclef.scripts._07b_dump_oof_probs --n-splits 5

    # Step 2 — audit
    python -m birdclef.scripts._08b_per_class_audit --top-k 30
    # or, with per-class Perch comparison:
    python -m birdclef.scripts._08b_per_class_audit --include-perch --top-k 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.data.soundscapes import load_taxonomy, primary_labels


def _per_class_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Per-class ROC-AUC. Returns NaN for classes with 0 or all positives."""
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


def _train_csv_class_counts(train_csv: Path, labels: list[str]) -> dict[str, int]:
    """Per-label primary-recording count in train.csv (secondaries excluded
    here so the column is comparable across classes — secondaries are
    asymmetric)."""
    df = pd.read_csv(train_csv)
    if "primary_label" not in df.columns:
        return {lb: 0 for lb in labels}
    counts = df["primary_label"].astype(str).value_counts().to_dict()
    return {lb: int(counts.get(lb, 0)) for lb in labels}


def _maybe_run_perch(
    meta: pd.DataFrame, repo_root: Path, batch_files: int = 8,
) -> np.ndarray | None:
    """Optional raw-Perch baseline aligned to the same N rows as `meta`.

    Returns (N, n_classes) score matrix, or None if Perch artifacts missing.
    """
    try:
        from birdclef_example.perch_eval_full_soundscapes import (
            DEFAULT_PERCH_LABELS, DEFAULT_PERCH_LABELS_FALLBACK,
        )
        from birdclef_example.perch_proxy_inference_auc import (
            build_mapping_context, build_scores_with_proxy,
            load_perch_labels, run_perch_onnx,
        )
    except ImportError as e:
        print(f"[audit] --include-perch requested but imports failed: {e}")
        return None

    onnx_path = repo_root / "models" / "perch_onnx" / "perch_v2.onnx"
    perch_labels_csv = DEFAULT_PERCH_LABELS
    if not perch_labels_csv.exists() and DEFAULT_PERCH_LABELS_FALLBACK.exists():
        perch_labels_csv = DEFAULT_PERCH_LABELS_FALLBACK
    soundscape_dir = repo_root / "data" / "train_soundscapes"
    taxonomy = pd.read_csv(repo_root / "data" / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    taxonomy["scientific_name"] = taxonomy["scientific_name"].astype(str)

    if not onnx_path.exists():
        print(f"[audit] Perch ONNX missing at {onnx_path} — skipping Perch comparison")
        return None
    if not perch_labels_csv.exists():
        print(f"[audit] Perch labels CSV missing — skipping Perch comparison")
        return None

    labels = primary_labels()
    perch_labels = load_perch_labels(perch_labels_csv)
    mapped_pos, mapped_perch_idx, mapping_ctx = build_mapping_context(
        taxonomy_df=taxonomy, perch_labels_df=perch_labels, primary_labels=labels,
    )

    files = meta["filename"].astype(str).drop_duplicates().tolist()
    paths = [soundscape_dir / f for f in files]
    print(f"[audit] running raw Perch on {len(paths)} soundscape files...")
    pred_meta, perch_logits = run_perch_onnx(
        soundscape_paths=paths, onnx_path=onnx_path,
        n_perch_classes=len(perch_labels),
        batch_files=batch_files, verbose=True,
    )

    full_score = build_scores_with_proxy(
        perch_logits=perch_logits, n_classes=len(labels),
        mapped_positions=mapped_pos, mapped_perch_indices=mapped_perch_idx,
        proxy_map=mapping_ctx["baseline_proxy_map"],
        proxy_reduce="max",
    )

    # Align Perch's (12 windows × n_files) rows to `meta` row order via row_id
    pred_meta = pred_meta.copy()
    pred_meta["row_id"] = pred_meta["row_id"].astype(str)
    if "row_id" not in meta.columns:
        # _07b emits row_id; if missing, build it from filename + end_sec
        ms = meta.copy()
        if "end_sec" not in ms.columns:
            ms["end_sec"] = pd.to_timedelta(ms["end"]).dt.total_seconds().astype(int)
        ms["row_id"] = ms["filename"].str.replace(".ogg", "", regex=False) + "_" + ms["end_sec"].astype(str)
        meta_row_ids = ms["row_id"].tolist()
    else:
        meta_row_ids = meta["row_id"].astype(str).tolist()

    by_row_id = dict(zip(pred_meta["row_id"].tolist(), range(len(pred_meta))))
    missing = [rid for rid in meta_row_ids if rid not in by_row_id]
    if missing:
        print(f"[audit] WARN: {len(missing)} meta row_ids not in Perch output (sample: {missing[:3]})")
    src_idx = np.array([by_row_id[rid] for rid in meta_row_ids if rid in by_row_id], dtype=np.int64)
    if len(src_idx) != len(meta):
        print(f"[audit] WARN: returning Perch matrix with {len(src_idx)} rows vs meta {len(meta)} — caller must align")
    return full_score[src_idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm-probs", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "ssm_probs.npz",
                    help="SSM OOF probs (from _07b_dump_oof_probs).")
    ap.add_argument("--y-true", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "y_true.npy")
    ap.add_argument("--meta", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "meta.parquet")
    ap.add_argument("--include-perch", action="store_true",
                    help="Also run raw Perch ONNX for per-class comparison "
                         "(adds perch_auc + gap columns).")
    ap.add_argument("--train-csv", type=Path,
                    default=Path("data/train.csv"))
    ap.add_argument("--top-k", type=int, default=30,
                    help="Console: print this many bottom-AUC classes.")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("outputs/eda/per_class_audit.csv"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("outputs/eda/per_class_audit.json"))
    args = ap.parse_args()

    for p in (args.ssm_probs, args.y_true, args.meta):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}\n"
                f"Run `python -m birdclef.scripts._07b_dump_oof_probs --n-splits 5` first."
            )

    ssm_probs = np.load(args.ssm_probs)
    P_ssm = ssm_probs["probs"] if "probs" in ssm_probs.files else ssm_probs[ssm_probs.files[0]]
    Y = np.load(args.y_true).astype(np.uint8)
    meta = pd.read_parquet(args.meta)
    if P_ssm.shape != Y.shape:
        raise SystemExit(f"shape mismatch: SSM probs {P_ssm.shape} vs y_true {Y.shape}")
    print(f"[audit] loaded SSM probs {P_ssm.shape}, y_true {Y.shape}, meta {len(meta)} rows")

    labels = primary_labels()
    n_classes = len(labels)
    if Y.shape[1] != n_classes:
        raise SystemExit(f"y_true has {Y.shape[1]} cols vs {n_classes} sample-sub classes")

    auc_ssm = _per_class_auc(Y, P_ssm)
    auc_perch = None
    if args.include_perch:
        P_perch = _maybe_run_perch(meta, REPO_ROOT)
        if P_perch is not None and P_perch.shape == Y.shape:
            auc_perch = _per_class_auc(Y, P_perch)
        else:
            print("[audit] Perch comparison unavailable; proceeding SSM-only.")

    # Per-class metadata
    tax = load_taxonomy().set_index("primary_label")
    train_counts = _train_csv_class_counts(args.train_csv, labels)
    pos_files = [
        meta.loc[Y[:, c] > 0, "filename"].nunique() if Y[:, c].sum() > 0 else 0
        for c in range(n_classes)
    ]

    rows = []
    for c, lb in enumerate(labels):
        support = int(Y[:, c].sum())
        rows.append({
            "primary_label": lb,
            "class_name":     str(tax.loc[lb, "class_name"]) if lb in tax.index else "?",
            "scientific_name": str(tax.loc[lb, "scientific_name"]) if lb in tax.index else "?",
            "support":        support,
            "pos_files":      int(pos_files[c]),
            "ssm_auc":        float(auc_ssm[c]) if not np.isnan(auc_ssm[c]) else None,
            "perch_auc":      None if auc_perch is None else
                              (float(auc_perch[c]) if not np.isnan(auc_perch[c]) else None),
            "gap_perch_minus_ssm": (None if auc_perch is None or np.isnan(auc_perch[c]) or np.isnan(auc_ssm[c])
                                    else float(auc_perch[c] - auc_ssm[c])),
            "n_train_csv":    int(train_counts.get(lb, 0)),
        })

    df = pd.DataFrame(rows)
    df_seen = df[df["support"] > 0].copy()
    df_seen = df_seen.sort_values(["ssm_auc", "support"], ascending=[True, False],
                                  na_position="first").reset_index(drop=True)
    df_seen.insert(0, "rank", df_seen.index + 1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_seen.to_csv(args.out_csv, index=False)

    # Headline summary
    macro_ssm_seen = float(np.nanmean([r["ssm_auc"] for r in rows
                                       if r["ssm_auc"] is not None and r["support"] > 0]))
    summary = {
        "n_classes_total":     int(n_classes),
        "n_classes_with_support": int((df["support"] > 0).sum()),
        "macro_auc_ssm_seen":  macro_ssm_seen,
        "macro_auc_ssm_seen_focal": float(np.nanmean([
            r["ssm_auc"] for r in rows
            if r["ssm_auc"] is not None and r["support"] > 0 and r["n_train_csv"] > 0
        ])),
        "macro_auc_ssm_seen_no_focal": float(np.nanmean([
            r["ssm_auc"] for r in rows
            if r["ssm_auc"] is not None and r["support"] > 0 and r["n_train_csv"] == 0
        ])),
        "include_perch":       bool(auc_perch is not None),
        "out_csv":             str(args.out_csv),
    }
    if auc_perch is not None:
        gaps = [r["gap_perch_minus_ssm"] for r in rows if r.get("gap_perch_minus_ssm") is not None]
        summary["n_classes_ssm_helps"] = int(sum(1 for g in gaps if g < -0.001))
        summary["n_classes_ssm_neutral"] = int(sum(1 for g in gaps if abs(g) <= 0.001))
        summary["n_classes_ssm_hurts"] = int(sum(1 for g in gaps if g > 0.001))
        summary["mean_gap_perch_minus_ssm"] = float(np.mean(gaps)) if gaps else float("nan")

    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console output
    print()
    print("=" * 80)
    print(f"Per-class SSM (+Perch) audit — {summary['n_classes_with_support']} val-seen classes")
    print(f"  macro AUC (val-seen)             : {summary['macro_auc_ssm_seen']:.4f}")
    print(f"  macro AUC (val-seen ∩ focal)     : {summary['macro_auc_ssm_seen_focal']:.4f}")
    print(f"  macro AUC (val-seen, NO focal)   : {summary['macro_auc_ssm_seen_no_focal']:.4f}")
    if auc_perch is not None:
        print(f"  classes where SSM helps          : {summary['n_classes_ssm_helps']}")
        print(f"  classes where SSM hurts          : {summary['n_classes_ssm_hurts']}")
        print(f"  mean gap (perch - ssm)           : {summary['mean_gap_perch_minus_ssm']:+.4f}")
    print()
    print(f"Bottom {args.top_k} classes by SSM AUC (the targets):")
    cols = ["rank", "primary_label", "class_name", "scientific_name",
            "support", "pos_files", "ssm_auc", "n_train_csv"]
    if auc_perch is not None:
        cols += ["perch_auc", "gap_perch_minus_ssm"]
    head = df_seen.head(args.top_k)[cols]
    print(head.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"[audit] full table → {args.out_csv}")
    print(f"[audit] summary    → {args.out_json}")


if __name__ == "__main__":
    main()
