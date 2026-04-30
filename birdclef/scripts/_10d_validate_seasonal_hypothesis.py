"""Stress-test the 'labeled-vs-unlabeled gap is mostly seasonal' hypothesis.

Six checks, each prints a conclusion line:

  1. Multi-classifier sanity check — confirms the 0.99 adv-AUC isn't
     an XGBoost-specific quirk. Re-runs with logistic regression,
     random forest, and a small MLP.
  2. Centroid + cosine similarity — compares labeled vs unlabeled
     mean Perch embedding directions. If centroids are far apart,
     the distributions are geometrically separable.
  3. Permutation test — shuffle labels, retrain. AUC must collapse
     to ~0.5; confirms the 0.99 is real signal, not artifact.
  4. **Match-on-month adversarial AUC** — the direct hypothesis test.
     Restrict to (labeled, unlabeled) pairs from the SAME calendar
     month, retrain XGBoost. If the seasonal hypothesis is correct,
     AUC should drop significantly from 0.99.
  5. Match-on-month-AND-site (S22) — tightest control. Same season,
     same site. If still 0.99, year/equipment is a separate axis.
  6. Year-stratified — labeled-year only adv-AUC. Decomposes year
     effect from season effect.

Run:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10d_validate_seasonal_hypothesis
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import PERCH_META, PERCH_NPZ

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_date(name: str) -> pd.Timestamp | None:
    m = FNAME_RE.match(name)
    if not m:
        return None
    return pd.to_datetime(m.group(3), format="%Y%m%d", errors="coerce")


def adv_auc(X, y, n_folds=5, model="xgb", random_state=42, balanced=True):
    """Generic adversarial AUC. Returns mean AUC over k-fold CV."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    if balanced:
        # Balance class counts via undersampling the majority
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n_take = min(n_pos, n_neg)
        if n_pos > n_take:
            keep_pos = rng.choice(np.where(y == 1)[0], n_take, replace=False)
        else:
            keep_pos = np.where(y == 1)[0]
        if n_neg > n_take:
            keep_neg = rng.choice(np.where(y == 0)[0], n_take, replace=False)
        else:
            keep_neg = np.where(y == 0)[0]
        keep = np.concatenate([keep_pos, keep_neg])
        X, y = X[keep], y[keep]

    if len(X) < n_folds * 4:
        return float("nan"), 0

    skf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
    aucs = []
    for tr, va in skf.split(X, y):
        if model == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.8, colsample_bytree=0.7,
                tree_method="hist", random_state=random_state, eval_metric="auc",
            )
            m.fit(X[tr], y[tr], verbose=False)
        elif model == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)
            m.fit(sc.transform(X[tr]), y[tr])
            X_va_scaled = sc.transform(X[va])
            p = m.predict_proba(X_va_scaled)[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            continue
        elif model == "rf":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=4,
                n_jobs=-1, random_state=random_state,
            )
            m.fit(X[tr], y[tr])
        elif model == "mlp":
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X[tr])
            m = MLPClassifier(
                hidden_layer_sizes=(256, 64), activation="relu",
                max_iter=200, random_state=random_state, early_stopping=True,
            )
            m.fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[va]))[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            continue
        else:
            raise ValueError(f"unknown model {model}")
        p = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs)), int(len(X))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=10_000,
                    help="Cap rows fed into the multi-classifier sanity test "
                         "(MLP/RF are slow at full 127k). Default 10k.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[validate] loading {PERCH_META} + {PERCH_NPZ}")
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)
    emb = arr["emb_full"].astype(np.float32)
    is_labeled = meta["is_labeled"].astype(bool).to_numpy()
    meta = meta.copy()
    meta["date"] = meta["filename"].astype(str).map(parse_date)
    meta["month"] = meta["date"].dt.month
    meta["year"]  = meta["date"].dt.year

    rng = np.random.default_rng(int(args.seed))
    n_lab = int(is_labeled.sum())
    n_unl = int((~is_labeled).sum())
    print(f"[validate] labeled={n_lab:,}  unlabeled={n_unl:,}")

    # ── 1. Multi-classifier sanity check (balanced subsample) ──────
    print()
    print("=" * 78)
    print("[1] MULTI-CLASSIFIER SANITY CHECK (balanced subsample)")
    print("=" * 78)
    print(f"    Confirms the 0.99 isn't XGBoost-specific.")
    n_take = min(n_lab, args.max_rows // 2)
    lab_idx = rng.choice(np.where(is_labeled)[0], n_take, replace=False) \
              if n_lab > n_take else np.where(is_labeled)[0]
    unl_idx = rng.choice(np.where(~is_labeled)[0], n_take, replace=False)
    keep = np.concatenate([lab_idx, unl_idx])
    X = emb[keep]
    y = np.zeros(len(keep), dtype=np.int64)
    y[len(lab_idx):] = 1
    print(f"    pool: {len(X):,} rows ({n_take} labeled vs {n_take} unlabeled)")

    for model in ("xgb", "logreg", "rf", "mlp"):
        auc, n = adv_auc(X, y, model=model, random_state=int(args.seed),
                         balanced=False)
        print(f"    {model:>8s}: adv-AUC = {auc:.4f}  (n={n})")

    # ── 2. Centroid + cosine similarity ────────────────────────────
    print()
    print("=" * 78)
    print("[2] CENTROID & COSINE SIMILARITY (full pool, no subsample)")
    print("=" * 78)
    print(f"    If centroids are well-separated → distributions are geometrically")
    print(f"    distinct (independent of any classifier).")

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, 1e-9)

    lab_emb = emb[is_labeled]
    unl_emb = emb[~is_labeled]
    lab_centroid = lab_emb.mean(axis=0)
    unl_centroid = unl_emb.mean(axis=0)
    centroid_cos = float(
        np.dot(normalize(lab_centroid), normalize(unl_centroid))
    )
    print(f"    labeled centroid · unlabeled centroid (cosine): {centroid_cos:.4f}")
    print(f"      (1.0 = same direction, 0 = orthogonal, -1 = opposite)")

    # Per-row cos similarity to each centroid
    lab_c_norm = normalize(lab_centroid)
    unl_c_norm = normalize(unl_centroid)
    emb_norm = normalize(emb)
    cos_to_lab = emb_norm @ lab_c_norm
    cos_to_unl = emb_norm @ unl_c_norm
    delta = cos_to_lab - cos_to_unl

    print(f"    cos(row, lab_centroid) - cos(row, unl_centroid):")
    print(f"      labeled rows  : median={np.median(delta[is_labeled]):+.4f}  "
          f"IQR=[{np.percentile(delta[is_labeled], 25):+.4f}, "
          f"{np.percentile(delta[is_labeled], 75):+.4f}]")
    print(f"      unlabeled rows: median={np.median(delta[~is_labeled]):+.4f}  "
          f"IQR=[{np.percentile(delta[~is_labeled], 25):+.4f}, "
          f"{np.percentile(delta[~is_labeled], 75):+.4f}]")

    overlap = float(((delta[is_labeled].min() <= delta[~is_labeled])
                     & (delta[~is_labeled] <= delta[is_labeled].max())).mean())
    print(f"    fraction of unlabeled rows in labeled-row delta range: {overlap:.3f}")
    if centroid_cos < 0.95:
        print(f"    → centroids are NOT aligned. Distributions distinct in feature space.")
    else:
        print(f"    → centroids are well-aligned (cos > 0.95). Within-pool variance is")
        print(f"      what the classifier picks up, not a global mean shift.")

    # ── 3. Permutation test ─────────────────────────────────────────
    print()
    print("=" * 78)
    print("[3] PERMUTATION TEST")
    print("=" * 78)
    print(f"    Shuffle labels, retrain. Should drop to ~0.5 → confirms 0.99 is real signal.")
    y_perm = y.copy()
    rng.shuffle(y_perm)
    auc_perm, n = adv_auc(X, y_perm, model="xgb", random_state=int(args.seed),
                          balanced=False)
    print(f"    permuted-label adv-AUC = {auc_perm:.4f}  (n={n})")
    if auc_perm > 0.6:
        print(f"    → WARN: permuted AUC > 0.6 — something fishy with the setup.")
    else:
        print(f"    → OK: permuted AUC ≈ 0.5 confirms the 0.99 is real signal.")

    # ── 4. Match-on-month adversarial AUC (THE hypothesis test) ────
    print()
    print("=" * 78)
    print("[4] MATCH-ON-MONTH ADVERSARIAL AUC  (the seasonal hypothesis test)")
    print("=" * 78)
    print(f"    Restrict to (labeled, unlabeled) pairs from the SAME month-of-year.")
    print(f"    If season was the axis: AUC drops from 0.99 → 0.6-0.8.")
    print(f"    If something else: AUC stays ≈ 0.99.")
    parsed_idx = ~meta["month"].isna().to_numpy()
    lab_months = set(meta.loc[is_labeled & parsed_idx, "month"].astype(int).tolist())
    print(f"    labeled months present: {sorted(lab_months)}")
    same_month_mask = meta["month"].isin(lab_months).to_numpy() & parsed_idx
    keep = same_month_mask
    X_m = emb[keep]
    y_m = is_labeled[keep].astype(np.int64) ^ 1   # 0=labeled, 1=unlabeled
    print(f"    pool: {len(X_m):,} rows total "
          f"(labeled={int((y_m==0).sum()):,}, unlabeled={int((y_m==1).sum()):,})")
    auc_match_month, n = adv_auc(X_m, y_m, model="xgb", random_state=int(args.seed),
                                 balanced=True)
    print(f"    match-on-month adv-AUC (balanced): {auc_match_month:.4f}  (n_eval={n:,})")
    drop_from_baseline = 0.9932 - auc_match_month
    print(f"    drop from baseline 0.9932: {drop_from_baseline:+.4f}")
    if drop_from_baseline > 0.10:
        print(f"    → STRONG SEASONAL EFFECT. The 0.99 baseline was driven by")
        print(f"      labeled rows being concentrated in specific months.")
    elif drop_from_baseline > 0.03:
        print(f"    → MODERATE SEASONAL EFFECT. Season explains some of the gap;")
        print(f"      remaining ~{(1-auc_match_month):.2f} is non-seasonal (year/equipment/curation).")
    else:
        print(f"    → SEASONAL HYPOTHESIS LARGELY DISPROVED. AUC stays high after")
        print(f"      restricting to matched months. Look elsewhere (year, equipment).")

    # ── 5. Match-on-month-AND-site (tightest control) ──────────────
    print()
    print("=" * 78)
    print("[5] MATCH-ON-MONTH-AND-SITE  (S22, the most concentrated site)")
    print("=" * 78)
    print(f"    Same site (S22), same months (Oct-Feb). Removes site + season axes.")
    print(f"    If still 0.99 → year/equipment/curation is a separate axis.")
    s22_mask = (meta["site"].astype(str) == "S22").to_numpy()
    lab_s22_months = set(
        meta.loc[is_labeled & s22_mask & parsed_idx, "month"].astype(int).tolist()
    )
    print(f"    S22 labeled months: {sorted(lab_s22_months)}")
    keep = s22_mask & parsed_idx & meta["month"].isin(lab_s22_months).to_numpy()
    X_ms = emb[keep]
    y_ms = is_labeled[keep].astype(np.int64) ^ 1
    n_ms_lab = int((y_ms == 0).sum()); n_ms_unl = int((y_ms == 1).sum())
    print(f"    pool: {len(X_ms):,} S22 rows in matched months "
          f"(labeled={n_ms_lab:,}, unlabeled={n_ms_unl:,})")
    if n_ms_lab < 100 or n_ms_unl < 100:
        print(f"    pool too small for meaningful CV; skipping.")
        auc_match_ms = float("nan")
    else:
        auc_match_ms, n = adv_auc(X_ms, y_ms, model="xgb",
                                  random_state=int(args.seed), balanced=True)
        print(f"    match-on-month+site adv-AUC: {auc_match_ms:.4f}  (n_eval={n:,})")

    # ── 6. Year-stratified AUC (labeled-year only) ─────────────────
    print()
    print("=" * 78)
    print("[6] YEAR-STRATIFIED ADVERSARIAL AUC  (within labeled year window)")
    print("=" * 78)
    print(f"    Restrict to years where labeled rows exist. Removes 'before-labeled-")
    print(f"    campaign-started' unlabeled. If AUC drops, year was the axis.")
    lab_years = set(meta.loc[is_labeled & parsed_idx, "year"].astype(int).tolist())
    print(f"    labeled years: {sorted(lab_years)}")
    keep = parsed_idx & meta["year"].isin(lab_years).to_numpy()
    X_y = emb[keep]
    y_y = is_labeled[keep].astype(np.int64) ^ 1
    print(f"    pool: {len(X_y):,} rows "
          f"(labeled={int((y_y==0).sum()):,}, unlabeled={int((y_y==1).sum()):,})")
    auc_year, n = adv_auc(X_y, y_y, model="xgb", random_state=int(args.seed),
                         balanced=True)
    print(f"    year-stratified adv-AUC: {auc_year:.4f}  (n_eval={n:,})")

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("[summary] Verdict on seasonal hypothesis")
    print("=" * 78)
    print(f"    baseline (full pool, XGBoost):             0.9932")
    print(f"    [4] match-on-month:                        {auc_match_month:.4f}  "
          f"(Δ {auc_match_month - 0.9932:+.4f})")
    if not np.isnan(auc_match_ms):
        print(f"    [5] match-on-month-AND-site (S22):         {auc_match_ms:.4f}  "
              f"(Δ {auc_match_ms - 0.9932:+.4f})")
    print(f"    [6] year-stratified:                       {auc_year:.4f}  "
          f"(Δ {auc_year - 0.9932:+.4f})")
    print()
    if (0.9932 - auc_match_month) > 0.10:
        print("    Seasonal hypothesis: SUPPORTED. Filtering pseudo by temporal_likeness")
        print("    is the right round-2 fix.")
    elif not np.isnan(auc_match_ms) and auc_match_ms > 0.90:
        print("    Seasonal hypothesis: PARTIAL. Even within-month-within-site rows are")
        print("    distinguishable. There's a real non-temporal axis (year/equipment).")
        print("    Temporal filter helps but won't fully close the gap.")
    else:
        print("    Seasonal hypothesis: WEAK. Most of the discriminability isn't seasonal.")
        print("    Investigate equipment / curation differences.")


if __name__ == "__main__":
    main()
