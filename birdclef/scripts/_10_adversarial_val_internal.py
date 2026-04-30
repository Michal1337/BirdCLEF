"""Adversarial validation between LABELED and UNLABELED rows of the
existing Perch cache.

Tests internal heterogeneity within train_soundscapes/. If labeled vs
unlabeled rows are easily distinguishable (high adv-AUC), the labeled
val pool isn't representative of the unlabeled training pool — i.e.,
filtering pseudo-labels to "labeled-like" rows could give a cleaner
signal. If they're indistinguishable (adv-AUC ≈ 0.5), the unlabeled
pool is uniformly in-distribution with labeled and pseudo-filtering
along this axis won't help.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10_adversarial_val_internal

Optional:
    --balanced       Subsample unlabeled to match labeled count (faster, simpler interpretation)
    --out-npz PATH   Where to dump per-row labeled-likeness scores
                     (default: outputs/adversarial/labeled_likeness.npz)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import OUTPUT_ROOT, PERCH_META, PERCH_NPZ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--balanced", action="store_true",
                    help="Subsample unlabeled rows to match labeled count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-npz", type=Path,
                    default=OUTPUT_ROOT / "adversarial" / "labeled_likeness.npz")
    args = ap.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("xgboost not installed. `pip install xgboost`.")

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    # ── Load Perch cache ────────────────────────────────────────────
    if not PERCH_META.exists() or not PERCH_NPZ.exists():
        raise SystemExit(
            f"Perch cache not found at {PERCH_META.parent}. "
            "Build via `python -m birdclef.cache.build_perch_cache`."
        )
    print(f"[adv-int] loading Perch cache from {PERCH_META.parent}")
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)
    emb = arr["emb_full"].astype(np.float32)

    if "is_labeled" not in meta.columns:
        raise SystemExit(
            "meta.parquet has no 'is_labeled' column. The cache builder "
            "needs to write that column — rebuild the cache."
        )
    is_labeled = meta["is_labeled"].astype(bool).to_numpy()
    n_lab = int(is_labeled.sum())
    n_unlab = int((~is_labeled).sum())
    print(f"[adv-int] labeled={n_lab:,}  unlabeled={n_unlab:,}  emb_dim={emb.shape[1]}")

    # ── Build adversarial dataset ───────────────────────────────────
    rng = np.random.default_rng(int(args.seed))
    lab_idx = np.where(is_labeled)[0]
    unlab_idx = np.where(~is_labeled)[0]

    if args.balanced:
        n_take = min(n_lab, n_unlab)
        unlab_idx = rng.choice(unlab_idx, size=n_take, replace=False)
        lab_idx = rng.choice(lab_idx, size=n_take, replace=False) if n_take < n_lab else lab_idx
        print(f"[adv-int] balanced subsample: {len(lab_idx):,} labeled vs "
              f"{len(unlab_idx):,} unlabeled")

    keep_idx = np.concatenate([lab_idx, unlab_idx])
    X = emb[keep_idx]
    # Convention: labeled=0 (the "training-pool" class), unlabeled=1 (the "test"
    # class for the adversarial probe). Per-row score = P(row is unlabeled-like).
    y = np.zeros(len(keep_idx), dtype=np.int64)
    y[len(lab_idx):] = 1
    print(f"[adv-int] X={X.shape}  positives (unlabeled): {int(y.sum()):,}")

    # ── 5-fold XGBoost ──────────────────────────────────────────────
    scale_pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    xgb_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=float(scale_pos_weight),
        tree_method="hist", random_state=int(args.seed), eval_metric="auc",
    )

    aucs = []
    oof_scores = np.zeros(len(X), dtype=np.float32)
    skf = StratifiedKFold(int(args.n_folds), shuffle=True, random_state=int(args.seed))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        p = clf.predict_proba(X[va])[:, 1]
        oof_scores[va] = p
        auc = roc_auc_score(y[va], p)
        print(f"[adv-int] fold {fold}  AUC = {auc:.4f}  "
              f"(va pos={(y[va]==1).sum():,}, neg={(y[va]==0).sum():,})")
        aucs.append(auc)

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    print()
    print(f"[adv-int] === Internal adversarial AUC (labeled vs unlabeled) ===")
    print(f"[adv-int]   mean ± std = {auc_mean:.4f} ± {auc_std:.4f}")
    print()
    if auc_mean < 0.55:
        print("[adv-int] AUC < 0.55 → labeled and unlabeled are essentially the same")
        print("[adv-int]   distribution. The labeled val pool is representative; pseudo")
        print("[adv-int]   filtering by labeled-likeness wouldn't reshape the pool.")
        print("[adv-int]   If LB regression on pseudo CNN was real, the gap is BEYOND")
        print("[adv-int]   train_soundscapes — run variant 1 (vs test_soundscapes) to confirm.")
    elif auc_mean < 0.75:
        print("[adv-int] AUC ∈ [0.55, 0.75] → moderate internal heterogeneity.")
        print("[adv-int]   Some unlabeled rows look 'labeled-like', others don't. The")
        print("[adv-int]   per-row scores in the npz can be used as a soft filter for")
        print("[adv-int]   round-2 pseudo, though the LB-relevance is unclear without")
        print("[adv-int]   variant 1 to anchor against the actual test distribution.")
    else:
        print("[adv-int] AUC > 0.75 → labeled and unlabeled are clearly different.")
        print("[adv-int]   Surprising — these are from the same recording campaign.")
        print("[adv-int]   Possible reasons: temporal drift (labeling done in batches),")
        print("[adv-int]   or systematic per-site labeling. Worth inspecting.")

    # ── Use OOF (held-out) scores from CV as the canonical filter ───
    #
    # Why OOF rather than "train on all data, score everything":
    # every unlabeled row was in the training set as class 1 in the all-data
    # path, so the classifier would have memorized it and assigned it a
    # biased-down score. The OOF scores from the 5-fold CV above are clean
    # — each row is scored by a fold's classifier that DIDN'T see that row
    # in train, so the score reflects true generalization, not memorization.
    #
    # Note `oof_scores` was indexed only over `keep_idx` (the labeled +
    # subsampled-or-full unlabeled pool). For rows outside that pool (only
    # possible if --balanced is set), we fall back to the all-data classifier
    # output. Without --balanced, every cache row is in keep_idx and we have
    # honest OOF scores everywhere.
    print()
    print("[adv-int] composing canonical scores: OOF where available, all-data "
          "fallback elsewhere")

    # Build per-row score over the FULL cache (n_rows = len(emb)).
    full_scores = np.zeros(len(emb), dtype=np.float32)
    full_scores[keep_idx] = oof_scores

    # If --balanced subsampled the unlabeled, the un-sampled unlabeled rows
    # got no OOF score. Fill them via an all-data classifier (still trained
    # on the same balanced subset, which keeps the comparison apples-to-apples).
    missing = np.ones(len(emb), dtype=bool)
    missing[keep_idx] = False
    n_missing = int(missing.sum())
    if n_missing > 0:
        print(f"[adv-int]   {n_missing:,} rows outside CV pool (--balanced enabled);"
              f" filling via all-data classifier")
        clf_full = xgb.XGBClassifier(**xgb_params)
        clf_full.fit(X, y, verbose=False)
        full_scores[missing] = clf_full.predict_proba(emb[missing])[:, 1]

    # Print quantiles to sanity-check (these are P(unlabeled), labeled rows
    # should score low, unlabeled should score high)
    q = np.percentile(full_scores[is_labeled], [10, 50, 90])
    print(f"[adv-int] labeled rows OOF P(unlabeled) percentiles "
          f"10%={q[0]:.3f}  50%={q[1]:.3f}  90%={q[2]:.3f}")
    q = np.percentile(full_scores[~is_labeled], [10, 50, 90])
    print(f"[adv-int] unlabeled rows OOF P(unlabeled) percentiles "
          f"10%={q[0]:.3f}  50%={q[1]:.3f}  90%={q[2]:.3f}")

    # Inversion: labeled_likeness = 1 - P(unlabeled). Higher = more labeled-like.
    labeled_likeness = (1.0 - full_scores).astype(np.float32)

    # Sanity: how spread out are unlabeled scores under OOF? With memorization
    # bias removed, the top end of the unlabeled distribution should reach up
    # to meaningful values (e.g., > 0.1) instead of being clipped near 0.
    unl_scores = labeled_likeness[~is_labeled]
    pcs = np.percentile(unl_scores, [50, 75, 90, 95, 99, 99.5, 99.9])
    print(f"[adv-int] unlabeled rows labeled-likeness percentiles:")
    print(f"[adv-int]   50%={pcs[0]:.4f}  75%={pcs[1]:.4f}  90%={pcs[2]:.4f}  "
          f"95%={pcs[3]:.4f}  99%={pcs[4]:.4f}  99.5%={pcs[5]:.4f}  "
          f"99.9%={pcs[6]:.4f}")
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        n = int((unl_scores >= thr).sum())
        print(f"[adv-int]   unlabeled with labeled_likeness ≥ {thr}: {n:,} "
              f"({n/len(unl_scores)*100:.2f}%)")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        labeled_likeness=labeled_likeness,
        is_labeled=is_labeled.astype(np.uint8),
        filenames=meta["filename"].astype(str).to_numpy(),
        adv_auc_mean=np.float32(auc_mean),
        adv_auc_std=np.float32(auc_std),
        balanced=np.uint8(int(args.balanced)),
        oof_scoring=np.uint8(1),    # marker: this npz uses honest OOF scores
    )
    print()
    print(f"[adv-int] wrote OOF-based labeled-likeness scores to {args.out_npz}")
    print(f"[adv-int]   shape={labeled_likeness.shape}  "
          f"(no train-set memorization bias)")


if __name__ == "__main__":
    main()
