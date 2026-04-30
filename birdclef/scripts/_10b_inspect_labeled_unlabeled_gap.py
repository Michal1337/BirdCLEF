"""Diagnose WHY labeled vs unlabeled rows are 0.99 adversarial-AUC distinct.

Three checks, each ~10s:
  1. Per-site composition: how labeled is each site?
  2. XGBoost feature importance: how concentrated is the discrimination?
  3. Within-site adversarial AUC + accuracy: site fingerprint vs deeper
     distributional gap.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10b_inspect_labeled_unlabeled_gap
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import PERCH_META, PERCH_NPZ


def main() -> None:
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("xgboost not installed.")
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    print(f"[inspect] loading Perch cache from {PERCH_META.parent}")
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)
    emb = arr["emb_full"].astype(np.float32)
    is_labeled = meta["is_labeled"].astype(bool).to_numpy()

    # ── 1. Per-site composition ────────────────────────────────────
    print()
    print("=== 1. Per-site composition (how labeled is each site?) ===")
    if "site" not in meta.columns:
        print("[inspect] meta has no 'site' column; skipping.")
    else:
        df = pd.DataFrame({"site": meta["site"].astype(str), "is_labeled": is_labeled})
        per_site = df.groupby("site").agg(
            n_total=("is_labeled", "size"),
            n_labeled=("is_labeled", "sum"),
        )
        per_site["pct_labeled"] = (per_site["n_labeled"] / per_site["n_total"] * 100).round(1)
        per_site = per_site.sort_values("n_total", ascending=False)
        print(f"sites total: {len(per_site)}")
        print(per_site.head(20).to_string())
        n_pure_unlab = int((per_site["n_labeled"] == 0).sum())
        n_pure_lab   = int((per_site["n_total"] == per_site["n_labeled"]).sum())
        n_mixed      = int(((per_site["n_labeled"] > 0) &
                            (per_site["n_labeled"] < per_site["n_total"])).sum())
        print()
        print(f"  fully unlabeled sites (0% labeled): {n_pure_unlab}")
        print(f"  fully labeled sites (100% labeled): {n_pure_lab}")
        print(f"  mixed sites (some labeled, some unlabeled): {n_mixed}")

    # ── 2. XGBoost feature importance ──────────────────────────────
    print()
    print("=== 2. XGBoost feature importance (which Perch dims discriminate?) ===")
    rng = np.random.default_rng(42)
    n_lab = int(is_labeled.sum())
    # Match labeled count for speed (concentration of importance is the
    # question, not the AUC level)
    unlab_idx = np.where(~is_labeled)[0]
    lab_idx = np.where(is_labeled)[0]
    sub_unlab = rng.choice(unlab_idx, size=min(n_lab * 5, len(unlab_idx)), replace=False)
    keep = np.concatenate([lab_idx, sub_unlab])
    X = emb[keep]
    y = np.zeros(len(keep), dtype=np.int64)
    y[len(lab_idx):] = 1

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.7,
        tree_method="hist", random_state=42, eval_metric="auc",
    )
    clf.fit(X, y, verbose=False)
    importance = clf.feature_importances_
    top_k = 15
    top_idx = np.argsort(importance)[::-1][:top_k]
    cumsum = np.cumsum(np.sort(importance)[::-1])
    n_dims_for_50pct = int(np.searchsorted(cumsum, 0.5 * cumsum[-1]) + 1)
    n_dims_for_90pct = int(np.searchsorted(cumsum, 0.9 * cumsum[-1]) + 1)

    print(f"top {top_k} Perch dims by importance:")
    for r, d in enumerate(top_idx, 1):
        print(f"  rank {r:2d}: dim {int(d):4d}  importance={importance[d]:.4f}")
    print()
    print(f"  dims accounting for 50% of total importance: {n_dims_for_50pct} / 1536")
    print(f"  dims accounting for 90% of total importance: {n_dims_for_90pct} / 1536")
    if n_dims_for_50pct < 50:
        print("  → discrimination is LOW-RANK (concentrated in <50 dims). Likely site/mic")
        print("    fingerprint rather than deep species-level distributional difference.")
    elif n_dims_for_50pct < 200:
        print("  → discrimination is moderately concentrated. Mixed site + acoustic signal.")
    else:
        print("  → discrimination is broadly distributed. Deep distributional gap.")

    # ── 3. Within-site adversarial AUC + accuracy (S22 only) ───────
    print()
    print("=== 3. Within-site adversarial AUC + accuracy (S22 only) ===")
    if "site" not in meta.columns:
        print("[inspect] no 'site' column; skipping.")
        return
    s22_mask = meta["site"].astype(str).eq("S22").to_numpy()
    s22_lab = s22_mask & is_labeled
    s22_unlab = s22_mask & ~is_labeled
    n_s22_lab = int(s22_lab.sum())
    n_s22_unlab = int(s22_unlab.sum())
    print(f"S22 rows: {n_s22_lab + n_s22_unlab:,} "
          f"(labeled={n_s22_lab:,}, unlabeled={n_s22_unlab:,})")
    if n_s22_lab < 50 or n_s22_unlab < 50:
        print("  insufficient S22 unlabeled rows for adversarial val "
              "(need ≥50 of each). Skipping.")
        return

    Xs = emb[s22_mask]
    ys = is_labeled[s22_mask].astype(np.int64) ^ 1   # 0=labeled, 1=unlabeled
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = []
    accs = []
    for tr, va in skf.split(Xs, ys):
        cf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.7,
            tree_method="hist", random_state=42, eval_metric="auc",
        )
        cf.fit(Xs[tr], ys[tr], verbose=False)
        p = cf.predict_proba(Xs[va])[:, 1]
        y_pred = (p >= 0.5).astype(np.int64)
        aucs.append(roc_auc_score(ys[va], p))
        accs.append(float((y_pred == ys[va]).mean()))
    within_auc = float(np.mean(aucs))
    within_acc = float(np.mean(accs))
    print(f"  within-S22 adversarial AUC: {within_auc:.4f} ± {np.std(aucs):.4f}")
    print(f"  within-S22 adversarial ACC: {within_acc:.4f} ± {np.std(accs):.4f}  "
          f"(chance = 0.5; majority-class baseline = "
          f"{max((ys==0).mean(), (ys==1).mean()):.4f})")
    print()
    if within_auc < 0.65:
        print("  → within-site adv-AUC is LOW. The 0.99 cross-pool gap is mostly")
        print("    SITE FINGERPRINT, not deep distributional difference. Once you")
        print("    control for site, labeled and unlabeled audio look very similar.")
        print("    Implication: site×date CV is fine; pseudo training is hurt by site")
        print("    bias rather than fundamental domain mismatch.")
    elif within_auc < 0.85:
        print("  → moderate within-site gap. Both site fingerprint AND deeper")
        print("    differences contribute. Filtering pseudo by labeled-likeness might")
        print("    help by selecting unlabeled rows from sites/conditions similar to")
        print("    labeled.")
    else:
        print("  → within-site gap is also high. Even within S22, labeled and unlabeled")
        print("    are distinct — likely a TIME or RECORDING-EQUIPMENT difference.")
        print("    The labeled subset may be from a specific recording session that")
        print("    differs from the rest of S22's data.")


if __name__ == "__main__":
    main()
