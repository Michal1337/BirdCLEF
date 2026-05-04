"""Data-driven calibration of pseudo-label τ and topk.

For a built pseudo round, computes keep_mask at various (τ, topk) values
and measures keep_mask quality on the labeled rows where we have ground
truth. Reports precision / recall / F1 plus rare-class coverage so you
can pick (τ, topk) by an explicit criterion instead of guessing.

The keep_mask logic mirrors `_apply_confidence_filter` in `pseudo_label.py`:

    keep_mask[r, c] = 1   IFF   probs[r, c] ≥ τ                    (abs floor)
                         OR    (r is in top-k windows of class c
                                within this file)                  (rel floor)

Two parameters working together:
- **τ** filters positions by ABSOLUTE confidence. High τ → cleaner pseudo
  labels but rare-class coverage drops (some classes never clear τ).
- **topk** is the rare-class safety net. For each (file, class), keep
  the top-k windows regardless of τ. Without it (topk=0), classes the
  teacher is never confident about have ZERO supervision.

**Caveat — leaked OOF on labeled.** The round-2 blend teacher has been
trained on labeled (Tucker memorization, SSM trained on all labeled).
So precision/recall numbers ON LABELED rows are leak-optimistic. The
RELATIVE ranking across (τ, topk) is still informative; absolute numbers
should be discounted.

What to use this for:

- For **SED student** (soft target + masked loss): more positions ≈ more
  training signal. Pick (τ, topk) that maximizes recall while keeping
  precision above 0.7 (above 0.5 = noise floor for multi-label).
- For **SSM student** (hard 0/1 labels): false positives are damaging.
  Pick (τ, topk) that keeps F1 high, with high precision (≥0.85). Lower
  topk is generally safer for SSM.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._05b_pseudo_calibration \\
        --round 2

    # Then update the cache's keep_mask with chosen values (no full rebuild):
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._05b_pseudo_calibration \\
        --round 2 --write --tau 0.5 --topk 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import (
    N_WINDOWS,
    PERCH_LABELS,
    PERCH_META,
    PSEUDO_DIR,
)


def _build_keep_mask(probs: np.ndarray, tau: float, topk: int) -> np.ndarray:
    """Mirrors `pseudo_label._apply_confidence_filter` exactly."""
    keep_mask = np.zeros_like(probs, dtype=np.uint8)
    if tau > 0:
        keep_mask = (probs >= tau).astype(np.uint8)
    if topk > 0:
        n_rows, n_classes = probs.shape
        n_files = n_rows // N_WINDOWS
        view = probs.reshape(n_files, N_WINDOWS, n_classes)
        k = min(int(topk), N_WINDOWS)
        topk_idx = np.argpartition(-view, kth=k - 1, axis=1)[:, :k]
        km = np.zeros_like(view, dtype=np.uint8)
        for f in range(n_files):
            for c in range(n_classes):
                km[f, topk_idx[f, :, c], c] = 1
        keep_mask = np.maximum(
            keep_mask, km.reshape(n_files * N_WINDOWS, n_classes)
        )
    return keep_mask


def _eval_on_labeled(
    keep_mask: np.ndarray, probs: np.ndarray, Y: np.ndarray, is_labeled: np.ndarray,
) -> dict:
    """Precision / recall / F1 of keep_mask treated as a per-position
    "is positive" predictor, vs ground truth Y. Restricted to labeled rows.

    Note: keep_mask=1 doesn't mean "this position is a positive" — it means
    "supervise this position." For SED, the actual target is `probs[r,c]`
    (soft), so the student isn't told "this is positive" at low-prob kept
    positions. For SSM, the target is `(probs[r,c] >= runtime_tau) &
    keep_mask`. We're measuring the keep_mask coverage of TRUE positives,
    which is the relevant signal for both consumers.
    """
    km = keep_mask[is_labeled]
    y = Y[is_labeled]
    p = probs[is_labeled]

    # Position-level keep_mask vs y
    kept_positive = (km > 0) & (y > 0)
    kept_negative = (km > 0) & (y == 0)
    missed_positive = (km == 0) & (y > 0)

    tp = int(kept_positive.sum())
    fp = int(kept_negative.sum())
    fn = int(missed_positive.sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # For SSM-equivalent at runtime_tau=keep-time-tau: what fraction of
    # kept positions become hard pseudo-positives that match GT?
    hard_positive = (km > 0) & (p >= 0.5)   # placeholder: 0.5 reference
    hard_correct = hard_positive & (y > 0)
    ssm_pseudo_count = int(hard_positive.sum())
    ssm_pseudo_precision = (
        int(hard_correct.sum()) / max(1, ssm_pseudo_count)
    )

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_kept": int(km.sum()),
        "n_total": int(km.size),
        "keep_fraction": float(km.mean()),
        "ssm_pseudo_count": ssm_pseudo_count,
        "ssm_pseudo_precision": float(ssm_pseudo_precision),
    }


def _coverage_breakdown(
    keep_mask: np.ndarray, probs: np.ndarray, Y: np.ndarray, is_labeled: np.ndarray,
) -> dict:
    """How many classes get at least one supervised position per file?
    This is the rare-class metric topk specifically targets.
    """
    km = keep_mask
    n_rows, n_classes = km.shape
    n_files = n_rows // N_WINDOWS

    # File-level: for each file, how many classes have ≥1 keep_mask position
    km_file = km.reshape(n_files, N_WINDOWS, n_classes)
    classes_per_file = (km_file.sum(axis=1) > 0).sum(axis=1)  # (n_files,)

    # Class-level: how many files have ≥1 keep_mask position for this class
    files_per_class = (km_file.sum(axis=1) > 0).sum(axis=0)  # (n_classes,)

    # Rarity-weighted coverage on labeled: for each class, what fraction
    # of its true-positive labeled rows are caught by keep_mask
    Y_lab = Y[is_labeled]
    km_lab = km[is_labeled]
    per_class_recall = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_pos = int(Y_lab[:, c].sum())
        if n_pos == 0:
            per_class_recall[c] = float("nan")
        else:
            n_caught = int(((km_lab[:, c] > 0) & (Y_lab[:, c] > 0)).sum())
            per_class_recall[c] = n_caught / n_pos

    n_classes_with_labeled_pos = int((Y_lab.sum(axis=0) > 0).sum())
    valid = ~np.isnan(per_class_recall)
    per_class_recall_clean = per_class_recall[valid]
    n_classes_caught = int((per_class_recall_clean > 0).sum())
    n_classes_complete = int((per_class_recall_clean >= 0.99).sum())
    return {
        "classes_per_file_mean": float(classes_per_file.mean()),
        "files_per_class_mean": float(files_per_class.mean()),
        "labeled_classes_with_positives": n_classes_with_labeled_pos,
        "classes_caught_at_least_one": n_classes_caught,
        "classes_caught_completely": n_classes_complete,
        "per_class_recall_mean": float(per_class_recall_clean.mean()),
        "per_class_recall_median": float(np.median(per_class_recall_clean)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True,
                    help="Pseudo round to calibrate (e.g. 2).")
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    help="τ values to evaluate. Default 0.3–0.8 in 0.1 steps.")
    ap.add_argument("--topk-grid", type=int, nargs="+",
                    default=[0, 1, 2, 3],
                    help="topk values. 0 = no rare-class floor, 1 = single "
                         "best window per (file, class), 2/3 = more aggressive.")
    ap.add_argument("--write", action="store_true",
                    help="Overwrite the round's keep_mask in probs.npz with one "
                         "computed at --tau / --topk. Required if you want the "
                         "downstream student trainers to use the recalibrated mask.")
    ap.add_argument("--tau", type=float, default=None,
                    help="τ to use when --write is set.")
    ap.add_argument("--topk", type=int, default=None,
                    help="topk to use when --write is set.")
    args = ap.parse_args()

    rd = PSEUDO_DIR / f"round{int(args.round)}"
    if not rd.exists():
        raise SystemExit(f"No pseudo round at {rd}")

    print(f"[calib] loading pseudo cache from {rd}")
    arr = np.load(rd / "probs.npz")
    probs = arr["final"].astype(np.float32) if "final" in arr.files else arr["probs"].astype(np.float32)
    n_rows, n_classes = probs.shape
    print(f"[calib] probs shape: {probs.shape}")

    perch_meta = pd.read_parquet(PERCH_META)
    if len(perch_meta) != n_rows:
        raise SystemExit(
            f"Perch meta rows ({len(perch_meta)}) != pseudo probs rows ({n_rows}). "
            "Stale cache or stale pseudo — rebuild one."
        )
    is_labeled = perch_meta["is_labeled"].astype(bool).to_numpy()
    Y = np.load(PERCH_LABELS).astype(np.uint8)
    if Y.shape != probs.shape:
        raise SystemExit(f"Y shape {Y.shape} != probs shape {probs.shape}")
    Y_lab = Y[is_labeled]
    print(f"[calib] labeled rows: {is_labeled.sum():,}  "
          f"true positives in labeled: {int(Y_lab.sum()):,}")
    print(f"[calib] **caveat: precision/recall on labeled is LEAK-OPTIMISTIC** "
          f"(blend teacher trained on these rows). Trust *relative* ranking.")
    print()

    # ── Grid search ────────────────────────────────────────────────────
    results = []
    for tau in args.tau_grid:
        for topk in args.topk_grid:
            km = _build_keep_mask(probs, float(tau), int(topk))
            ev = _eval_on_labeled(km, probs, Y, is_labeled)
            cov = _coverage_breakdown(km, probs, Y, is_labeled)
            row = {"tau": float(tau), "topk": int(topk), **ev, **cov}
            results.append(row)

    # ── Print table sorted by F1 desc ──────────────────────────────────
    print(f"{'τ':>5} {'topk':>5} | {'prec':>6} {'recall':>7} {'F1':>6} | "
          f"{'keep_%':>7} {'cls_caught':>11} {'recall_mean':>11}")
    print("-" * 92)
    for r in sorted(results, key=lambda x: (-x["f1"], -x["recall"])):
        print(f"{r['tau']:>5.2f} {r['topk']:>5} | "
              f"{r['precision']:>6.3f} {r['recall']:>7.3f} {r['f1']:>6.3f} | "
              f"{100 * r['keep_fraction']:>6.2f}% "
              f"{r['classes_caught_at_least_one']:>5}/{r['labeled_classes_with_positives']:>3} "
              f"{r['per_class_recall_mean']:>11.3f}")
    print()

    # ── Recommendations ────────────────────────────────────────────────
    by_f1 = max(results, key=lambda x: x["f1"])
    by_recall = max(results, key=lambda x: x["recall"])
    by_precision = max(
        [r for r in results if r["recall"] > 0.5], key=lambda x: x["precision"],
        default=results[0],
    )
    by_class_coverage = max(
        results, key=lambda x: (x["classes_caught_at_least_one"], x["per_class_recall_mean"]),
    )

    print("=" * 92)
    print("Recommendations by criterion:")
    print(f"  best F1                : τ={by_f1['tau']:.2f} topk={by_f1['topk']}  "
          f"(F1={by_f1['f1']:.3f}, prec={by_f1['precision']:.3f}, recall={by_f1['recall']:.3f})")
    print(f"  best recall            : τ={by_recall['tau']:.2f} topk={by_recall['topk']}  "
          f"(recall={by_recall['recall']:.3f}, prec={by_recall['precision']:.3f}, F1={by_recall['f1']:.3f})")
    print(f"  best precision (rec>.5): τ={by_precision['tau']:.2f} topk={by_precision['topk']}  "
          f"(prec={by_precision['precision']:.3f}, recall={by_precision['recall']:.3f})")
    print(f"  best class coverage    : τ={by_class_coverage['tau']:.2f} topk={by_class_coverage['topk']}  "
          f"(classes_caught={by_class_coverage['classes_caught_at_least_one']}, "
          f"per-class-recall={by_class_coverage['per_class_recall_mean']:.3f})")
    print()
    print("For SED student (soft target + masked loss): prefer high recall + "
          "decent precision (≥0.7). Often best F1 or best recall.")
    print("For SSM student (hard 0/1 labels):           prefer high precision (≥0.85) "
          "to avoid teaching false positives. Often higher τ + lower topk.")
    print()

    # ── Save full report ───────────────────────────────────────────────
    out_json = rd / "calibration_report.json"
    out_json.write_text(
        json.dumps({
            "n_rows": int(n_rows), "n_classes": int(n_classes),
            "n_labeled": int(is_labeled.sum()),
            "n_labeled_positives": int(Y_lab.sum()),
            "results": results,
            "recommendations": {
                "best_f1": {"tau": by_f1["tau"], "topk": by_f1["topk"]},
                "best_recall": {"tau": by_recall["tau"], "topk": by_recall["topk"]},
                "best_precision_min_recall_05": {
                    "tau": by_precision["tau"], "topk": by_precision["topk"],
                },
                "best_class_coverage": {
                    "tau": by_class_coverage["tau"], "topk": by_class_coverage["topk"],
                },
            },
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[calib] wrote full report → {out_json}")

    # ── Optional: rebuild keep_mask in cache ───────────────────────────
    if args.write:
        if args.tau is None or args.topk is None:
            raise SystemExit("--write requires --tau and --topk")
        new_mask = _build_keep_mask(probs, float(args.tau), int(args.topk))
        # Reload all arrays + replace keep_mask
        new_arrs = {k: arr[k] for k in arr.files}
        new_arrs["keep_mask"] = new_mask
        np.savez_compressed(rd / "probs.npz", **new_arrs)
        print(f"[calib] WRITE: keep_mask updated in {rd / 'probs.npz'}  "
              f"(τ={args.tau:.2f} topk={args.topk}, "
              f"keep_fraction={float(new_mask.mean()):.4f})")
        # Patch info.json
        info_path = rd / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["confidence_tau"] = float(args.tau)
            info["topk_per_species"] = int(args.topk)
            info["keep_fraction"] = float(new_mask.mean())
            info["calibration_history"] = info.get("calibration_history", []) + [{
                "tau": float(args.tau), "topk": int(args.topk),
                "keep_fraction": float(new_mask.mean()),
            }]
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"[calib] info.json updated.")


if __name__ == "__main__":
    main()
