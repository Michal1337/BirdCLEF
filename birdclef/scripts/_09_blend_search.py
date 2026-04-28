"""Per-taxon convex-combination search over multiple blend members.

Given N ≥ 2 row-aligned probability arrays (one per blend member) and the
ground-truth label matrix, finds the per-taxon weights that maximize the
chosen AUC metric. Outputs a JSON of weights you can paste into the LB
notebook (replacing the hand-picked _TAXON_WEIGHTS dict).

Metric (--metric):
    mean_of_folds (default, requires meta.parquet with a `fold` column) —
        average of per-fold macro AUCs. LB-correlated; mirrors how a deployed
        model fits its calibration once on all data. Use --folds to drop
        problem folds (e.g. site-CV fold 0 trains on too little data).
    stitched — single macro AUC over all rows concatenated. Biased by
        inter-fold calibration drift; legacy default.

Search space: for each taxon, sweep weight combinations on a coarse grid
(default step=0.1) over the simplex (weights sum to 1). For 2 members it's a
1-D sweep; for 3 members a 2-D triangle; for N members a Dirichlet grid.
Coarse step is intentional — finer grids overfit val with so few rows.

Inputs (default Locations match _07b_dump_oof_probs + _08c_dump_ast_val):
    outputs/blend_search/oof/ssm_probs.npz  → SSM stitched OOF predictions
    outputs/blend_search/oof/ast_probs.npz  → AST val predictions
    outputs/blend_search/oof/y_true.npy     → ground truth (uint8)
    outputs/blend_search/oof/meta.parquet   → row metadata; needs `fold` +
                                              `fold_kind` columns for
                                              mean-of-folds metric

Run:
    # rebuild OOF inputs with fold info under the desired CV scheme
    python -m birdclef.scripts._07b_dump_oof_probs --n-splits 5 --fold-kind strat

    # mean-of-folds search across all folds
    python -m birdclef.scripts._09_blend_search \\
        --members ssm:outputs/blend_search/oof/ssm_probs.npz \\
                  ast:outputs/blend_search/oof/ast_probs.npz \\
        --step 0.2

    # restrict to specific folds (e.g. drop site-CV fold 0 = all-S22)
    python -m birdclef.scripts._09_blend_search \\
        --members ssm:... ast:... \\
        --folds 1 2 3 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.data.soundscapes import load_taxonomy, primary_labels


def _macro_auc(Y: np.ndarray, P: np.ndarray, class_idx: np.ndarray) -> Tuple[float, int]:
    """Macro AUC over `class_idx` columns, skipping classes with no positives."""
    if len(class_idx) == 0:
        return float("nan"), 0
    pos = Y[:, class_idx].sum(axis=0)
    keep = (pos > 0) & (pos < Y.shape[0])
    if int(keep.sum()) == 0:
        return float("nan"), 0
    sub = class_idx[keep]
    try:
        return float(roc_auc_score(Y[:, sub], P[:, sub], average="macro")), int(keep.sum())
    except ValueError:
        return float("nan"), int(keep.sum())


def _mean_of_folds_auc(
    Y: np.ndarray, P: np.ndarray, class_idx: np.ndarray, fold_of_row: np.ndarray,
) -> Tuple[float, int]:
    """Mean of per-fold macro AUCs (within `class_idx` columns).

    For each fold f, computes macro AUC over rows where `fold_of_row == f`,
    averaging only over classes with at least one positive in that fold's
    val rows. Returns (mean over folds where AUC is finite, n_folds_used).

    This is the LB-correlated metric: each fold scores its own self-consistent
    predictions, mirroring how a deployed model fits its calibration once on
    all data. Stitched-OOF (concat all folds → one global AUC) mixes score
    distributions across differently-trained fold models and is misleading.
    """
    folds = np.unique(fold_of_row[fold_of_row >= 0])
    if len(folds) == 0:
        return float("nan"), 0
    per_fold = []
    for f in folds:
        rows = np.where(fold_of_row == f)[0]
        if len(rows) == 0:
            continue
        auc_f, _ = _macro_auc(Y[rows], P[rows], class_idx)
        if np.isfinite(auc_f):
            per_fold.append(auc_f)
    if not per_fold:
        return float("nan"), 0
    return float(np.mean(per_fold)), len(per_fold)


def _simplex_grid(n_members: int, step: float) -> List[Tuple[float, ...]]:
    """All weight tuples on the n-simplex with `step` granularity."""
    n_steps = int(round(1.0 / step))
    pts = []
    for combo in itertools.product(range(n_steps + 1), repeat=n_members - 1):
        if sum(combo) > n_steps:
            continue
        last = n_steps - sum(combo)
        weights = tuple((c / n_steps) for c in combo) + (last / n_steps,)
        pts.append(weights)
    return pts


def _parse_member_arg(s: str) -> Tuple[str, Path]:
    if ":" not in s:
        raise SystemExit(f"--members must be name:path (got {s!r})")
    name, p = s.split(":", 1)
    return name.strip(), Path(p.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True,
                    help="member specs as name:path.npz (e.g. ssm:outputs/.../ssm_probs.npz "
                         "ast:outputs/.../ast_probs.npz). Each .npz must have 'probs' key.")
    ap.add_argument("--y-true", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "y_true.npy")
    ap.add_argument("--meta", type=Path,
                    default=OUTPUT_ROOT / "blend_search" / "oof" / "meta.parquet")
    ap.add_argument("--step", type=float, default=0.1,
                    help="Grid step on the simplex. Default 0.1 = coarse "
                         "(11 points per axis for 2 members). Lower = finer "
                         "search but higher overfit-to-val risk on few rows.")
    ap.add_argument("--metric", default="mean_of_folds",
                    choices=["mean_of_folds", "stitched"],
                    help="Optimization target. mean_of_folds = average of "
                         "per-fold macro AUCs (LB-correlated; default when "
                         "meta has a 'fold' column). stitched = single macro "
                         "AUC over all rows concatenated (legacy; biased by "
                         "inter-fold calibration drift).")
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="Restrict mean-of-folds to a subset of fold ids "
                         "(e.g. --folds 1 2 3 4 to drop fold 0). Default: "
                         "all folds present in meta. Ignored if --metric "
                         "stitched.")
    ap.add_argument("--out-json", type=Path,
                    default=Path("outputs/blend/blend_weights.json"),
                    help="Output: per-taxon weights JSON.")
    args = ap.parse_args()

    members = [_parse_member_arg(m) for m in args.members]
    if len(members) < 2:
        raise SystemExit("Need ≥2 members for a blend search")
    for name, path in members:
        if not path.exists():
            raise SystemExit(f"Missing prob file for member {name!r}: {path}")
    if not args.y_true.exists() or not args.meta.exists():
        raise SystemExit(
            f"Missing y_true or meta. Run _07b_dump_oof_probs first.\n"
            f"  y_true: {args.y_true}\n  meta:   {args.meta}"
        )

    member_names = [n for n, _ in members]
    P_list: List[np.ndarray] = []
    for name, path in members:
        arr = np.load(path)
        key = "probs" if "probs" in arr.files else arr.files[0]
        P_list.append(arr[key].astype(np.float32))
    Y = np.load(args.y_true).astype(np.uint8)
    meta = pd.read_parquet(args.meta)

    n_rows, n_cls = Y.shape
    print(f"[blend] members: {member_names}")
    print(f"[blend] rows={n_rows}  classes={n_cls}")
    for n, P in zip(member_names, P_list):
        if P.shape != Y.shape:
            raise SystemExit(f"shape mismatch: {n}={P.shape} vs y_true={Y.shape}")
        print(f"[blend]   {n:>6s}  shape={P.shape}  mean={P.mean():.4f}")

    # Resolve the AUC metric. mean_of_folds requires a `fold` column in meta.
    metric = args.metric
    fold_of_row: np.ndarray | None = None
    if metric == "mean_of_folds":
        if "fold" not in meta.columns:
            print(f"[blend] WARN: meta has no 'fold' column — falling back "
                  f"to stitched. Re-dump OOF probs with the updated "
                  f"_07b_dump_oof_probs to get fold-aware ranking.")
            metric = "stitched"
        else:
            fold_of_row = meta["fold"].astype(int).to_numpy()
            available = sorted(int(f) for f in np.unique(fold_of_row) if f >= 0)
            if args.folds is not None:
                requested = set(int(f) for f in args.folds)
                missing = requested - set(available)
                if missing:
                    raise SystemExit(
                        f"--folds requested {sorted(missing)} but meta only has {available}"
                    )
                # Mask non-selected rows by setting their fold to -1, which the
                # mean-of-folds helper already filters out.
                mask = np.isin(fold_of_row, sorted(requested))
                fold_of_row = np.where(mask, fold_of_row, -1).astype(np.int8)
                print(f"[blend] mean-of-folds restricted to folds {sorted(requested)} "
                      f"(of available {available}, "
                      f"{int(mask.sum())}/{len(mask)} rows)")
            else:
                print(f"[blend] mean-of-folds over folds {available} "
                      f"(fold_kind={meta.get('fold_kind', pd.Series(['?'])).iloc[0]})")

    def _score(Yp: np.ndarray, Pp: np.ndarray, class_idx: np.ndarray) -> Tuple[float, int]:
        if metric == "mean_of_folds" and fold_of_row is not None:
            return _mean_of_folds_auc(Yp, Pp, class_idx, fold_of_row)
        return _macro_auc(Yp, Pp, class_idx)

    # Build per-taxon class-index lists from taxonomy.
    labels = primary_labels()
    tax = load_taxonomy()
    label_to_taxon = dict(zip(tax["primary_label"].astype(str),
                              tax["class_name"].astype(str)))
    by_taxon: Dict[str, List[int]] = {}
    for ci, lb in enumerate(labels):
        t = label_to_taxon.get(str(lb), "Unknown")
        by_taxon.setdefault(t, []).append(ci)

    grid = _simplex_grid(len(members), args.step)
    print(f"[blend] simplex grid: {len(grid)} points at step={args.step}")
    print()

    # Search: per-taxon, find the weight combination maximizing macro AUC.
    # Also track each taxon's single-member baselines for the diagnostic
    # table — showing how much the blend wins over each member alone.
    results: Dict[str, Dict] = {}
    print(f"{'taxon':<10s}  {'n_cls':>5s}  " +
          "  ".join(f"{n:>6s}" for n in member_names) +
          f"  {'best':>6s}  {'best_w':<25s}")
    print("-" * (28 + 8 * len(members) + 35))

    for taxon, cols in sorted(by_taxon.items()):
        idx = np.array(cols, dtype=np.int64)
        # single-member baselines
        single = []
        for P in P_list:
            auc, n_eval = _score(Y, P, idx)
            single.append(auc)
        if not all(np.isfinite(s) for s in single):
            # If any member has no positives in this taxon, skip the search
            # (degenerate). Default to equal weights as fallback.
            best_w = tuple(1.0 / len(members) for _ in members)
            best_score = float("nan")
        else:
            best_score = -np.inf
            best_w = grid[0]
            for w in grid:
                blended = sum(wi * Pi for wi, Pi in zip(w, P_list))
                auc, _ = _score(Y, blended, idx)
                if not np.isfinite(auc):
                    continue
                if auc > best_score:
                    best_score = float(auc)
                    best_w = w
        n_eval_str = f"{int((Y[:, idx].sum(axis=0) > 0).sum())}/{len(idx)}"
        single_str = "  ".join(f"{s:.4f}" if np.isfinite(s) else "  nan " for s in single)
        bw_str = "  ".join(f"{n[:3]}={w:.2f}" for n, w in zip(member_names, best_w))
        print(f"{taxon:<10s}  {n_eval_str:>5s}  {single_str}  "
              f"{best_score:.4f}  {bw_str}")
        results[taxon] = {
            "n_classes_total": int(len(idx)),
            "n_classes_evaluable": int((Y[:, idx].sum(axis=0) > 0).sum()),
            "single_aucs": {n: (None if not np.isfinite(s) else round(s, 4))
                            for n, s in zip(member_names, single)},
            "best_weights": {n: round(float(w), 4) for n, w in zip(member_names, best_w)},
            "best_blended_auc": (None if not np.isfinite(best_score) else round(best_score, 4)),
        }

    # Overall (all val-seen classes) for sanity.
    all_seen = np.where(Y.sum(axis=0) > 0)[0]
    print()
    print(f"=== Overall (all val-seen classes, metric={metric}) ===")
    overall_singles = [_score(Y, P, all_seen)[0] for P in P_list]
    for n, s in zip(member_names, overall_singles):
        print(f"  {n:>6s} alone: {s:.4f}")
    # Overall best with the per-taxon weight assignment we just chose
    blended_per_taxon = np.zeros_like(Y, dtype=np.float32)
    for taxon, cols in by_taxon.items():
        w = tuple(results[taxon]["best_weights"][n] for n in member_names)
        b = sum(wi * Pi for wi, Pi in zip(w, P_list))
        blended_per_taxon[:, cols] = b[:, cols]
    per_taxon_overall = _score(Y, blended_per_taxon, all_seen)[0]
    print(f"  per-taxon blend: {per_taxon_overall:.4f}")
    # Also a single-α baseline for comparison
    if len(members) == 2:
        best_global_alpha, best_global_auc = 0.5, -np.inf
        for alpha in np.arange(0.0, 1.0 + args.step, args.step):
            b = alpha * P_list[0] + (1 - alpha) * P_list[1]
            a, _ = _score(Y, b, all_seen)
            if a > best_global_auc:
                best_global_auc, best_global_alpha = a, float(alpha)
        print(f"  best global α (single weight): "
              f"{member_names[0]}={best_global_alpha:.2f} → {best_global_auc:.4f}")
        results["_overall_global_blend"] = {
            f"{member_names[0]}_weight": round(best_global_alpha, 4),
            f"{member_names[1]}_weight": round(1 - best_global_alpha, 4),
            "auc": round(float(best_global_auc), 4),
        }

    results["_overall"] = {
        "single_aucs": {n: round(float(s), 4) for n, s in zip(member_names, overall_singles)},
        "per_taxon_blend_auc": round(float(per_taxon_overall), 4),
    }
    results["_meta"] = {
        "members": member_names,
        "step": args.step,
        "n_grid_points": len(grid),
        "n_rows": int(n_rows),
        "n_classes": int(n_cls),
        "metric": metric,
        "fold_kind": (str(meta["fold_kind"].iloc[0])
                      if "fold_kind" in meta.columns else None),
        "folds_used": (sorted(set(int(f) for f in fold_of_row if f >= 0))
                       if fold_of_row is not None else None),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"[blend] wrote {args.out_json}")
    print()
    print("=== Paste this into LB_0931_seed.ipynb cell 27's _TAXON_WEIGHTS ===")
    print("_TAXON_WEIGHTS = {")
    for taxon in ("Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"):
        if taxon not in results:
            continue
        ws = results[taxon]["best_weights"]
        print(f"    \"{taxon}\":     ({ws.get(member_names[0], 0.5):.2f}, "
              f"{ws.get(member_names[1], 0.5):.2f}),")
    print("}")


if __name__ == "__main__":
    main()
