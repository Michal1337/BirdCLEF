#!/usr/bin/env python3
"""Evaluate Perch (with proxy mapping) on ALL labeled soundscapes.

Produces the same two metrics as `train_ddp_focal.py` so Perch is directly
comparable to a focal-trained CNN on the same val pool:

    val_auc_seen        macro AUC over the ~75 val classes with positives.
                        Mirrors the official LB metric. Primary number.
    val_auc_focal_seen  macro AUC over the ~47 classes that are BOTH in
                        train.csv AND have val positives. Diagnostic — what
                        a focal-only model could in principle reach.

Differences vs `perch_proxy_inference_auc.py`:
  - That script restricts to the 59 fully-labeled files (708 rows). This one
    uses all 66 labeled files (full + partial = 739 rows after dedup), exactly
    matching the val pool the focal CNN sees.
  - That script sweeps proxy strategies; this one fixes the strategy to the
    LB_093 baseline (genus-proxy on Aves/Amphibia/Insecta, max reduce) — i.e.
    what the SSM pipeline already uses — so the comparison stays apples-to-apples.

Run:
    python -m birdclef_example.perch_eval_full_soundscapes

Output JSON: outputs/eda/perch_eval_full_soundscapes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import prepare_soundscape_metadata, parse_primary_labels  # noqa: E402
from birdclef_example.perch_proxy_inference_auc import (  # noqa: E402
    DEFAULT_PERCH_LABELS,
    DEFAULT_PERCH_LABELS_FALLBACK,
    build_mapping_context,
    build_scores_with_proxy,
    load_perch_labels,
    run_perch_onnx,
)
from birdclef_example.utils import birdclef_roc_auc  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "perch_onnx" / "perch_v2.onnx"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "eda" / "perch_eval_full_soundscapes.json"


def _build_truth_matrix(val_meta: pd.DataFrame, primary_labels: List[str]) -> np.ndarray:
    """One row per (filename, start, end) in val_meta; multi-hot over 234 classes."""
    label_to_idx = {lb: i for i, lb in enumerate(primary_labels)}
    Y = np.zeros((len(val_meta), len(primary_labels)), dtype=np.float32)
    for i, raw in enumerate(val_meta["primary_label"].astype(str)):
        for lb in parse_primary_labels(raw):
            j = label_to_idx.get(lb)
            if j is not None:
                Y[i, j] = 1.0
    return Y


def _focal_class_indices(train_csv_path: Path, primary_labels: List[str]) -> np.ndarray:
    """Indices (in primary_labels order) of classes that appear in train.csv."""
    train_df = pd.read_csv(train_csv_path)
    label_to_idx = {lb: i for i, lb in enumerate(primary_labels)}
    seen: set[int] = set()
    for raw in train_df["primary_label"].astype(str).tolist():
        for lb in parse_primary_labels(raw):
            if lb in label_to_idx:
                seen.add(label_to_idx[lb])
    return np.array(sorted(seen), dtype=np.int64)


def _macro_auc_over_classes(
    y_true: np.ndarray, y_score: np.ndarray, class_subset: np.ndarray | None = None,
) -> float:
    """birdclef_roc_auc restricted to a column subset (or full if None)."""
    Y = y_true if class_subset is None else y_true[:, class_subset]
    P = y_score if class_subset is None else y_score[:, class_subset]
    try:
        return float(birdclef_roc_auc(Y, P))
    except ValueError:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Perch macro-AUC eval on the full labeled-soundscape pool.")
    ap.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    ap.add_argument("--soundscape-dir", type=Path, default=DEFAULT_DATA_DIR / "train_soundscapes")
    ap.add_argument("--sound-labels-csv", type=Path, default=DEFAULT_DATA_DIR / "train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_DATA_DIR / "taxonomy.csv")
    ap.add_argument("--sample-submission", type=Path, default=DEFAULT_DATA_DIR / "sample_submission.csv")
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_DATA_DIR / "train.csv")
    ap.add_argument("--perch-labels-csv", type=Path, default=DEFAULT_PERCH_LABELS)
    ap.add_argument("--batch-files", type=int, default=8)
    ap.add_argument("--proxy-reduce", choices=["max", "mean"], default="max",
                    help="How to combine multiple Perch indices for a genus-proxied class. "
                         "Default 'max' = LB_093 baseline.")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="Optional debug limit on the number of soundscape files.")
    ap.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    perch_labels_csv = args.perch_labels_csv
    if not perch_labels_csv.exists() and DEFAULT_PERCH_LABELS_FALLBACK.exists():
        perch_labels_csv = DEFAULT_PERCH_LABELS_FALLBACK
    for path in (args.onnx_path, args.soundscape_dir, args.sound_labels_csv,
                 args.taxonomy_csv, args.sample_submission, args.train_csv,
                 perch_labels_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    sample_sub = pd.read_csv(args.sample_submission)
    primary_labels: List[str] = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    taxonomy = pd.read_csv(args.taxonomy_csv)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    taxonomy["scientific_name"] = taxonomy["scientific_name"].astype(str)

    perch_labels = load_perch_labels(perch_labels_csv)
    mapped_positions, mapped_perch_indices, mapping_ctx = build_mapping_context(
        taxonomy_df=taxonomy,
        perch_labels_df=perch_labels,
        primary_labels=primary_labels,
    )

    # Val meta: ALL labeled soundscapes (fully + partially), dedup'd (the raw
    # CSV ships every row twice — prepare_soundscape_metadata handles it).
    sscape_df = pd.read_csv(args.sound_labels_csv)
    val_meta = prepare_soundscape_metadata(sscape_df, args.soundscape_dir)
    # end_sec mirrors the row_id format the Perch runner emits.
    val_meta = val_meta.copy()
    val_meta["end_sec"] = pd.to_timedelta(val_meta["end"]).dt.total_seconds().astype(int)
    val_meta["row_id"] = (
        val_meta["filename"].str.replace(".ogg", "", regex=False)
        + "_" + val_meta["end_sec"].astype(str)
    )

    val_files = sorted(val_meta["filename"].astype(str).unique().tolist())
    if args.limit_files > 0:
        val_files = val_files[: args.limit_files]
        val_meta = val_meta[val_meta["filename"].isin(val_files)].reset_index(drop=True)
    val_files = [f for f in val_files if (args.soundscape_dir / f).exists()]
    val_meta = val_meta[val_meta["filename"].isin(val_files)].reset_index(drop=True)
    if not val_files:
        raise SystemExit("No labeled soundscape files found on disk.")

    soundscape_paths = [args.soundscape_dir / f for f in val_files]
    print(f"Running Perch on {len(soundscape_paths)} soundscape files "
          f"(all labeled = full + partial)...", flush=True)
    pred_meta, perch_logits = run_perch_onnx(
        soundscape_paths=soundscape_paths,
        onnx_path=args.onnx_path,
        n_perch_classes=len(perch_labels),
        batch_files=int(args.batch_files),
        verbose=not args.quiet,
    )

    # 234-class score matrix using the LB_093 baseline proxy strategy.
    baseline_proxy_map = mapping_ctx["baseline_proxy_map"]
    full_y_score = build_scores_with_proxy(
        perch_logits=perch_logits,
        n_classes=n_classes,
        mapped_positions=mapped_positions,
        mapped_perch_indices=mapped_perch_indices,
        proxy_map=baseline_proxy_map,
        proxy_reduce=str(args.proxy_reduce),
    )

    # Align predictions (12 rows per file) to val_meta rows by row_id.
    # Partial-file rows are picked up here too — anything missing is dropped.
    pred_score_by_row_id = dict(zip(pred_meta["row_id"].tolist(),
                                    range(len(pred_meta))))
    keep_mask = val_meta["row_id"].isin(pred_score_by_row_id)
    val_meta = val_meta[keep_mask].reset_index(drop=True)
    pred_indices = np.array([pred_score_by_row_id[rid]
                             for rid in val_meta["row_id"].tolist()], dtype=np.int64)
    y_score = full_y_score[pred_indices]
    y_true = _build_truth_matrix(val_meta, primary_labels)

    # Class subsets matching train_ddp_focal.py's two metrics
    val_seen_classes = np.where(y_true.sum(axis=0) > 0)[0]
    focal_classes = _focal_class_indices(args.train_csv, primary_labels)
    focal_seen_classes = np.array(sorted(set(val_seen_classes.tolist()) &
                                         set(focal_classes.tolist())), dtype=np.int64)

    val_auc_seen = _macro_auc_over_classes(y_true, y_score, val_seen_classes)
    val_auc_focal_seen = _macro_auc_over_classes(y_true, y_score, focal_seen_classes)

    # Bonus: per-class_name breakdown so sonotype perf doesn't hide in macro
    class_name_map: Dict[str, str] = mapping_ctx["class_name_map"]
    by_class_name: Dict[str, Dict[str, float | int]] = {}
    for cn in sorted({class_name_map.get(primary_labels[i], "Unknown")
                      for i in val_seen_classes.tolist()}):
        cn_mask = np.array([class_name_map.get(primary_labels[i]) == cn
                            for i in val_seen_classes.tolist()], dtype=bool)
        cn_idx = val_seen_classes[cn_mask]
        by_class_name[cn] = {
            "n_classes": int(len(cn_idx)),
            "macro_auc": _macro_auc_over_classes(y_true, y_score, cn_idx),
        }

    summary = {
        "onnx_path": str(args.onnx_path),
        "perch_labels_csv": str(perch_labels_csv),
        "proxy_reduce": str(args.proxy_reduce),
        "n_files": int(len(val_files)),
        "n_val_rows": int(len(val_meta)),
        "n_total_classes": int(n_classes),
        "n_val_seen_classes": int(len(val_seen_classes)),
        "n_focal_classes": int(len(focal_classes)),
        "n_focal_seen_classes": int(len(focal_seen_classes)),
        # The two headline numbers — comparable to train_ddp_focal output.
        "val_auc_seen": val_auc_seen,
        "val_auc_focal_seen": val_auc_focal_seen,
        # Diagnostic: how Perch does on the 28 val-only classes (sonotypes etc.)
        # — should be near 0.5 since Perch has no training signal for them.
        "val_auc_sonotype_etc": _macro_auc_over_classes(
            y_true, y_score,
            np.array(sorted(set(val_seen_classes.tolist()) -
                            set(focal_classes.tolist())), dtype=np.int64),
        ),
        "by_class_name": by_class_name,
        "mapping": {
            "n_perch_direct": int(mapped_positions.size),
            "n_perch_proxy": int(len(baseline_proxy_map)),
            "n_no_perch_signal": int(mapping_ctx["n_unmapped"] - len(baseline_proxy_map)),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Perch eval — full labeled soundscape pool")
    print(f"  files                  : {summary['n_files']}")
    print(f"  val rows               : {summary['n_val_rows']}")
    print(f"  val_auc_seen           : {val_auc_seen:.4f}   "
          f"(over {summary['n_val_seen_classes']} classes)")
    print(f"  val_auc_focal_seen     : {val_auc_focal_seen:.4f}   "
          f"(over {summary['n_focal_seen_classes']} classes ∩ train.csv)")
    print(f"  val_auc_sonotype_etc   : {summary['val_auc_sonotype_etc']:.4f}   "
          f"(over {summary['n_val_seen_classes'] - summary['n_focal_seen_classes']} "
          f"val-only classes — Perch can't see them)")
    print()
    print("By taxonomic class_name:")
    for cn, m in by_class_name.items():
        print(f"  {cn:<10s} n={m['n_classes']:3d}  macro_auc={m['macro_auc']:.4f}")
    print()
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
