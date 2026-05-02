"""Local blend-design diagnostic on stitched-OOF probabilities.

Consumes the artifacts dumped by `_07b_dump_oof_probs.py` (ssm_probs.npz,
sed_probs.npz, y_true.npy, meta.parquet) and runs three analyses:

  1. **Global blend-weight grid** at multiple aggregators
     (rank-mean, prob-mean, geometric-mean). Reports both stitched and
     mean-of-folds macro AUC, and the SSM-weight optimum for each.

  2. **Per-taxon AUC** of each member (SSM, SED) — independent per-taxon
     class-AUC averages. Tells you which taxon prefers which member.

  3. **Per-taxon optimal blend weights** — independent grid search for
     each of {Aves, Amphibia, Insecta, Mammalia}. The output suggests
     taxon-conditional weights to use in cell 27 of the LB notebook.

**Tucker-leak caveat.** SED on labeled is fully memorized (per the
`--shuffle-offset` probe), so absolute SED OOF AUC is inflated. The
*relative* signal is still useful in:
  - aggregator comparisons (leak symmetric across rank/prob/geom)
  - per-taxon ranking (which member is stronger on a taxon)
  - shape of weight curve (where the maximum sits)
But absolute optimal α tends to be biased toward heavier-SED than the
true LB optimum because SED appears artificially strong.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._08c_blend_local_diagnostic \\
        --oof-dir birdclef/outputs/blend_search/oof_strat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from birdclef.config.paths import OUTPUT_ROOT
from birdclef.data.soundscapes import load_taxonomy, primary_labels


# ───────────────────────────── aggregators ──────────────────────────────


def _rank_pct(p: np.ndarray) -> np.ndarray:
    """Per-class percentile rank in [0, 1]. Same convention as cell 27."""
    return pd.DataFrame(p).rank(axis=0, pct=True).to_numpy(np.float32)


def aggregate(p_ssm: np.ndarray, p_sed: np.ndarray, w_ssm: float,
              kind: str) -> np.ndarray:
    """Combine two prob arrays at SSM-weight `w_ssm`."""
    w_sed = 1.0 - float(w_ssm)
    eps = 1e-5
    pa = np.clip(p_ssm, eps, 1 - eps).astype(np.float32)
    pb = np.clip(p_sed, eps, 1 - eps).astype(np.float32)
    if kind == "rank":
        return w_ssm * _rank_pct(pa) + w_sed * _rank_pct(pb)
    if kind == "prob":
        return w_ssm * pa + w_sed * pb
    if kind == "geom":
        # weighted geometric mean of probs (rank invariant under monotone
        # warps but not under additive shifts)
        return np.exp(w_ssm * np.log(pa) + w_sed * np.log(pb)).astype(np.float32)
    raise ValueError(f"unknown aggregator {kind!r}")


# ───────────────────────────── metrics ──────────────────────────────────


def per_class_auc(y: np.ndarray, p: np.ndarray, class_idxs: list[int] | None = None
                  ) -> dict[int, float]:
    """Per-class ROC-AUC for the requested class indices (or all)."""
    if class_idxs is None:
        class_idxs = list(range(y.shape[1]))
    out: dict[int, float] = {}
    for c in class_idxs:
        if y[:, c].sum() == 0 or y[:, c].sum() == y.shape[0]:
            continue
        try:
            out[c] = float(roc_auc_score(y[:, c], p[:, c]))
        except ValueError:
            continue
    return out


def macro_auc(y: np.ndarray, p: np.ndarray, class_idxs: list[int] | None = None) -> float:
    aucs = per_class_auc(y, p, class_idxs)
    return float(np.mean(list(aucs.values()))) if aucs else float("nan")


def mean_of_folds_auc(y: np.ndarray, p: np.ndarray, fold: np.ndarray,
                      class_idxs: list[int] | None = None) -> float:
    """Average per-fold macro AUC. The LB-correlated metric."""
    aucs = []
    for f in sorted(np.unique(fold)):
        if f < 0:
            continue
        idx = np.where(fold == f)[0]
        if len(idx) == 0:
            continue
        a = macro_auc(y[idx], p[idx], class_idxs)
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


# ───────────────────────────── analysis ─────────────────────────────────


def _grid_search_alpha(p_ssm, p_sed, y, fold, class_idxs, kind: str,
                       grid=np.linspace(0.0, 1.0, 21)):
    """Return ((best_alpha, best_mof_auc), [(alpha, mof_auc), ...])."""
    curve = []
    for a in grid:
        b = aggregate(p_ssm, p_sed, float(a), kind)
        m = mean_of_folds_auc(y, b, fold, class_idxs)
        curve.append((float(a), float(m)))
    best = max(curve, key=lambda x: x[1])
    return best, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=str(OUTPUT_ROOT / "blend_search" / "oof_strat"),
                    help="Directory containing ssm_probs.npz, sed_probs.npz, "
                         "y_true.npy, meta.parquet (produced by _07b_dump_oof_probs).")
    ap.add_argument("--grid-step", type=float, default=0.05)
    args = ap.parse_args()

    oof = Path(args.oof_dir)
    ssm = np.load(oof / "ssm_probs.npz")["probs"].astype(np.float32)
    sed = np.load(oof / "sed_probs.npz")["probs"].astype(np.float32)
    y = np.load(oof / "y_true.npy").astype(np.uint8)
    meta = pd.read_parquet(oof / "meta.parquet")
    fold = meta["fold"].to_numpy().astype(np.int8)

    if ssm.shape != sed.shape or ssm.shape != y.shape:
        raise SystemExit(
            f"shape mismatch: ssm={ssm.shape} sed={sed.shape} y={y.shape}"
        )
    print(f"[diag] rows={ssm.shape[0]}  classes={ssm.shape[1]}  "
          f"folds={sorted(np.unique(fold).tolist())}")

    # ── 1. global weight grid × aggregator ──────────────────────────────
    print()
    print("=" * 88)
    print("1. Global blend weight grid (mean-of-folds AUC)")
    print("=" * 88)
    grid = np.arange(0.0, 1.0 + args.grid_step / 2, args.grid_step)
    print(f"{'α (SSM weight)':<16}", end="")
    for kind in ("rank", "prob", "geom"):
        print(f"{kind:>12}", end="")
    print()
    print("-" * 64)
    rows = []
    for a in grid:
        row = {"alpha_ssm": float(round(a, 3))}
        for kind in ("rank", "prob", "geom"):
            b = aggregate(ssm, sed, float(a), kind)
            row[kind] = mean_of_folds_auc(y, b, fold)
        rows.append(row)
        print(f"{a:<16.2f}", end="")
        for kind in ("rank", "prob", "geom"):
            print(f"{row[kind]:>12.4f}", end="")
        print()
    df = pd.DataFrame(rows)
    print()
    for kind in ("rank", "prob", "geom"):
        best = df.iloc[df[kind].idxmax()]
        print(f"  {kind:>4} optimum: α*={best['alpha_ssm']:.2f}  "
              f"AUC={best[kind]:.4f}")
    print()
    rank_at_06 = df.loc[(df["alpha_ssm"] - 0.60).abs().idxmin(), "rank"]
    print(f"  baseline (rank, α=0.60): {rank_at_06:.4f}  ← current LB 0.943 anchor")
    print()

    # ── 2. per-taxon analysis ───────────────────────────────────────────
    print("=" * 88)
    print("2. Per-taxon member AUC + optimal blend weight")
    print("=" * 88)
    tax = load_taxonomy()
    class_to_taxon = tax.set_index("primary_label")["class_name"].to_dict()
    labels = primary_labels()
    taxon_classes: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        cn = class_to_taxon.get(lbl, "Aves")
        taxon_classes.setdefault(cn, []).append(i)

    print(f"\n{'taxon':<12}{'n_classes':>10}{'SSM_alone':>12}{'SED_alone':>12}"
          f"{'rank_α=.6':>12}{'rank_α*':>12}{'best_AUC':>12}")
    print("-" * 80)
    per_taxon_optimum: dict[str, float] = {}
    for taxon, idxs in sorted(taxon_classes.items()):
        ssm_taxon = mean_of_folds_auc(y, ssm, fold, idxs)
        sed_taxon = mean_of_folds_auc(y, sed, fold, idxs)
        rank_06 = mean_of_folds_auc(y, aggregate(ssm, sed, 0.6, "rank"), fold, idxs)
        # Per-taxon grid search
        (best_a, best_auc), _ = _grid_search_alpha(ssm, sed, y, fold, idxs, "rank",
                                                   grid=grid)
        per_taxon_optimum[taxon] = best_a
        print(f"{taxon:<12}{len(idxs):>10d}{ssm_taxon:>12.4f}{sed_taxon:>12.4f}"
              f"{rank_06:>12.4f}{best_a:>12.2f}{best_auc:>12.4f}")

    # Synthetic per-taxon blend (uses each taxon's α*)
    print()
    print("Composite per-taxon blend (each taxon's optimum α* applied):")
    blend = np.zeros_like(ssm)
    pa = _rank_pct(np.clip(ssm, 1e-5, 1 - 1e-5))
    pb = _rank_pct(np.clip(sed, 1e-5, 1 - 1e-5))
    for taxon, idxs in taxon_classes.items():
        a = per_taxon_optimum[taxon]
        for c in idxs:
            blend[:, c] = a * pa[:, c] + (1.0 - a) * pb[:, c]
    pt_global = mean_of_folds_auc(y, blend, fold)
    print(f"  per-taxon-optimal mean-of-folds AUC: {pt_global:.4f}")
    print(f"  (baseline 0.6 rank-blend was {rank_at_06:.4f}; "
          f"Δ={pt_global - rank_at_06:+.4f})")
    print()

    # ── 3. recommendations ──────────────────────────────────────────────
    print("=" * 88)
    print("3. Recommended LB submissions (top-3 by local OOF lift, leak-aware)")
    print("=" * 88)
    print()
    rank_best_a = float(df.iloc[df["rank"].idxmax()]["alpha_ssm"])
    geom_best_a = float(df.iloc[df["geom"].idxmax()]["alpha_ssm"])
    geom_best_auc = float(df["geom"].max())
    rank_best_auc = float(df["rank"].max())

    candidates = []
    # Candidate A: global rank-blend at empirically-optimal α (caveat: SED leak biases this toward SED)
    candidates.append((
        "rank-blend at OOF optimum",
        f"PROTO_W={rank_best_a:.2f}  rank-blend",
        rank_best_auc,
        f"caveat: SED leaked → α* may be biased toward SED-heavy. "
        f"Submit only if α* ≥ 0.60 (SSM-tilted)."
    ))
    # Candidate B: per-taxon optimal weights
    pt_str = "  ".join(f"{t}={per_taxon_optimum[t]:.2f}" for t in sorted(per_taxon_optimum))
    candidates.append((
        "per-taxon rank blend",
        pt_str,
        pt_global,
        f"Lift over global α=0.6: {pt_global - rank_at_06:+.4f}. "
        f"Robust to leak if Δ is positive across ≥2 taxa."
    ))
    # Candidate C: best aggregator at α=0.6 (leak-symmetric)
    rank_06_auc = float(df.loc[(df["alpha_ssm"] - 0.60).abs().idxmin(), "rank"])
    prob_06_auc = float(df.loc[(df["alpha_ssm"] - 0.60).abs().idxmin(), "prob"])
    geom_06_auc = float(df.loc[(df["alpha_ssm"] - 0.60).abs().idxmin(), "geom"])
    best_agg_at_06 = max(("rank", rank_06_auc), ("prob", prob_06_auc), ("geom", geom_06_auc),
                         key=lambda x: x[1])
    candidates.append((
        f"{best_agg_at_06[0]}-blend at α=0.6 (best aggregator)",
        f"aggregator={best_agg_at_06[0]}",
        best_agg_at_06[1],
        f"Leak-symmetric: rank/prob/geom react proportionally to leak. "
        f"Pick wins are honest."
    ))

    for i, (name, recipe, auc, note) in enumerate(candidates):
        print(f"  [{i+1}] {name}")
        print(f"      recipe : {recipe}")
        print(f"      OOF mean-of-folds AUC: {auc:.4f}")
        print(f"      note   : {note}")
        print()

    # Save full curves for plotting / further inspection
    out_json = oof / "blend_diagnostic.json"
    summary = {
        "global_grid": rows,
        "rank_optimum_alpha": rank_best_a,
        "rank_at_alpha_0.6": rank_at_06,
        "per_taxon_aucs": {
            t: {
                "n_classes": len(idxs),
                "ssm_alone": mean_of_folds_auc(y, ssm, fold, idxs),
                "sed_alone": mean_of_folds_auc(y, sed, fold, idxs),
                "alpha_star": per_taxon_optimum[t],
            } for t, idxs in taxon_classes.items()
        },
        "per_taxon_blend_oof": pt_global,
        "candidates": [
            {"name": n, "recipe": r, "oof_auc": a, "note": note}
            for n, r, a, note in candidates
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[diag] wrote {out_json}")


if __name__ == "__main__":
    main()
