"""SANITY TEST: random 50/50 split of the labeled-soundscape pool.

Take the 708 labeled rows, randomly split into two halves of 354 each,
assign one half as class 0 and the other as class 1, then run the same
XGBoost adversarial-AUC framework. Both halves come from the SAME
population, so there's no signal — AUC and ACC should both land at ≈ 0.5.

This is the cleanest possible framework sanity check:
  - Same Perch embedding distribution on both sides (literally same pool)
  - No twin pairs (each row appears exactly once)
  - Standard 5-fold StratifiedKFold (no leakage paths)
  - Random label assignment (no real signal)

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10f_sanity_random_labeled_split
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

from birdclef.config.paths import PERCH_META, PERCH_NPZ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the 50/50 split.")
    ap.add_argument("--n-trials", type=int, default=3,
                    help="Repeat the experiment N times with different splits "
                         "to confirm AUC doesn't deviate from 0.5 by chance.")
    args = ap.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("xgboost not installed.")
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    print(f"[sanity] loading Perch cache from {PERCH_META.parent}")
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)
    emb = arr["emb_full"].astype(np.float32)
    is_labeled = meta["is_labeled"].astype(bool).to_numpy()
    lab_emb = emb[is_labeled]
    n_lab = len(lab_emb)
    print(f"[sanity] labeled rows: {n_lab:,}  emb_dim={emb.shape[1]}")

    if n_lab < 100:
        raise SystemExit(f"too few labeled rows ({n_lab}) for a meaningful 50/50 split")

    print(f"[sanity] running {args.n_trials} trials with different random splits "
          f"(base seed = {args.seed})")
    print(f"[sanity] expected: AUC ≈ 0.5 ± few%, ACC ≈ 0.5 ± few%")
    print()

    scale_pos_weight = 1.0   # balanced 50/50 split, no rebalancing needed
    xgb_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist", random_state=int(args.seed), eval_metric="auc",
    )

    trial_aucs = []
    trial_accs = []
    for trial in range(int(args.n_trials)):
        trial_seed = int(args.seed) + trial
        rng = np.random.default_rng(trial_seed)

        # Random 50/50 split: shuffle indices, first half → class 0, rest → class 1
        idx = np.arange(n_lab)
        rng.shuffle(idx)
        half = n_lab // 2
        idx0, idx1 = idx[:half], idx[half:half * 2]   # ensure even sizes
        keep = np.concatenate([idx0, idx1])

        X = lab_emb[keep]
        y = np.zeros(len(keep), dtype=np.int64)
        y[len(idx0):] = 1
        print(f"[trial {trial}]  seed={trial_seed}  "
              f"n_class0={int((y==0).sum())}, n_class1={int((y==1).sum())}")

        skf = StratifiedKFold(5, shuffle=True, random_state=trial_seed)
        fold_aucs = []
        fold_accs = []
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            clf = xgb.XGBClassifier(**{**xgb_params, "random_state": trial_seed})
            clf.fit(X[tr], y[tr], verbose=False)
            p = clf.predict_proba(X[va])[:, 1]
            y_pred = (p >= 0.5).astype(np.int64)
            auc = roc_auc_score(y[va], p)
            acc = float((y_pred == y[va]).mean())
            fold_aucs.append(auc)
            fold_accs.append(acc)
            print(f"  fold {fold}: AUC = {auc:.4f}  ACC = {acc:.4f}  "
                  f"(va: pos={int((y[va]==1).sum())}, neg={int((y[va]==0).sum())})")

        auc_mean = float(np.mean(fold_aucs))
        acc_mean = float(np.mean(fold_accs))
        trial_aucs.append(auc_mean)
        trial_accs.append(acc_mean)
        print(f"  trial {trial} summary: AUC = {auc_mean:.4f}  ACC = {acc_mean:.4f}")
        print()

    overall_auc = float(np.mean(trial_aucs))
    overall_acc = float(np.mean(trial_accs))
    auc_std = float(np.std(trial_aucs))
    acc_std = float(np.std(trial_accs))
    print("=" * 72)
    print(f"[summary] across {args.n_trials} trials:")
    print(f"[summary]   AUC = {overall_auc:.4f}  (std across trials = {auc_std:.4f})")
    print(f"[summary]   ACC = {overall_acc:.4f}  (std across trials = {acc_std:.4f})")
    print()
    if 0.45 <= overall_auc <= 0.55 and 0.45 <= overall_acc <= 0.55:
        print(f"[verdict] OK: both metrics ≈ 0.5 → framework is sound on a clean")
        print(f"          random-split control. The 0.99 labeled-vs-unlabeled signal")
        print(f"          measured in _10_adversarial_val_internal.py is genuine")
        print(f"          distributional structure, not framework bias.")
    elif overall_auc > 0.55 or overall_acc > 0.55:
        print(f"[verdict] WARN: random 50/50 split shows non-trivial discrimination.")
        print(f"          This means the labeled pool has INTERNAL structure that")
        print(f"          XGBoost can pick up (e.g., site / time / equipment clusters")
        print(f"          within labeled). The 0.99 vs unlabeled is partly inflated by")
        print(f"          this same internal structure. Investigate which axis drives")
        print(f"          this baseline.")
    else:
        print(f"[verdict] AUC < 0.45 — unexpected. Check label assignment.")


if __name__ == "__main__":
    main()
