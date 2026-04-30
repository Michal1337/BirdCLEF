"""Test the curation-bias hypothesis: labelers picked files with more
prominent bird/insect calls, leaving sparse/quiet files in the unlabeled
pool. Curation bias would manifest as a systematic shift in Perch's
"how confidently do I see any species" signal.

Three checks:
  1. Max raw Perch logit distribution: labeled vs unlabeled.
  2. Top-K-mean Perch logit (smoother proxy for call density).
  3. Spectral activity proxy: variance of embeddings across the 12
     windows of each file. Files with more dynamic acoustic content
     have higher within-file variance.

If labeled rows are systematically higher on (1) and (2), the curation
bias hypothesis is confirmed.

Run:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10e_test_curation_hypothesis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import N_WINDOWS, PERCH_META, PERCH_NPZ


def quantile_summary(name: str, lab: np.ndarray, unl: np.ndarray) -> None:
    qs = [5, 25, 50, 75, 95]
    lq = np.percentile(lab, qs)
    uq = np.percentile(unl, qs)
    print(f"  {name}")
    print(f"    {'percentile':>10s}  {'5%':>8s}  {'25%':>8s}  {'50%':>8s}  {'75%':>8s}  {'95%':>8s}")
    print(f"    {'labeled':>10s}  " + "  ".join(f"{v:>8.3f}" for v in lq))
    print(f"    {'unlabeled':>10s}  " + "  ".join(f"{v:>8.3f}" for v in uq))
    delta = np.median(lab) - np.median(unl)
    rel = delta / max(abs(np.median(unl)), 1e-9) * 100.0
    print(f"    median Δ (lab - unl) = {delta:+.4f}  ({rel:+.1f}% relative)")


def main() -> None:
    print(f"[curation] loading {PERCH_META}, {PERCH_NPZ}")
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)

    # The honest direct-mapped Perch logits for the BirdCLEF column space.
    # If you have the new-format cache (post the proxy split), `scores_full_raw`
    # is the right field; on legacy caches it's the same name (proxy was
    # baked in but doesn't change the max-logit semantics meaningfully).
    scores = arr["scores_full_raw"].astype(np.float32)
    emb = arr["emb_full"].astype(np.float32)
    is_labeled = meta["is_labeled"].astype(bool).to_numpy()
    n_lab = int(is_labeled.sum()); n_unl = int((~is_labeled).sum())
    print(f"[curation] labeled={n_lab:,}  unlabeled={n_unl:,}")

    # ── 1. Max logit per row ──────────────────────────────────────
    print()
    print("=" * 78)
    print("[1] Max raw Perch logit per row")
    print("=" * 78)
    print("    'How confidently does Perch see ANY mapped species in this 5s window?'")
    print("    If labelers picked call-rich files, labeled max-logit > unlabeled.")
    max_logit = scores.max(axis=1)
    quantile_summary("max_logit", max_logit[is_labeled], max_logit[~is_labeled])

    # ── 2. Top-K mean logit per row (smoother proxy) ───────────────
    print()
    print("=" * 78)
    print("[2] Top-5-mean Perch logit per row")
    print("=" * 78)
    print("    Smoother than max — captures multi-species call density.")
    top5_mean = np.partition(scores, -5, axis=1)[:, -5:].mean(axis=1)
    quantile_summary("top5_mean_logit", top5_mean[is_labeled], top5_mean[~is_labeled])

    # ── 3. Within-file embedding variance ──────────────────────────
    print()
    print("=" * 78)
    print("[3] Within-file embedding variance across the 12 windows")
    print("=" * 78)
    print("    'How acoustically dynamic is this file?' Files with bursts of bird")
    print("    activity have high variance; quiet/uniform files have low variance.")
    # Rebuild file-level groups by filename
    fnames = meta["filename"].astype(str).to_numpy()
    file_to_rows = {}
    for i, fn in enumerate(fnames):
        file_to_rows.setdefault(fn, []).append(i)
    # Compute std of embeddings across rows of the same file, average over emb dims
    file_variance = np.zeros(len(meta), dtype=np.float32)
    for fn, rows in file_to_rows.items():
        if len(rows) < 2:
            continue
        sub = emb[rows]
        # std across windows for each emb dim, then mean across dims
        v = sub.std(axis=0).mean()
        for r in rows:
            file_variance[r] = v
    # Per-file means (one value per file)
    is_lab_file = pd.Series(is_labeled, index=fnames).groupby(level=0).first()
    var_per_file = pd.Series(file_variance, index=fnames).groupby(level=0).first()
    var_lab = var_per_file[is_lab_file].to_numpy()
    var_unl = var_per_file[~is_lab_file].to_numpy()
    print(f"    (per-file: labeled={len(var_lab)}, unlabeled={len(var_unl)})")
    quantile_summary("file_emb_variance", var_lab, var_unl)

    # ── 4. Adversarial AUC using just (max_logit, top5_mean, file_var) ─
    print()
    print("=" * 78)
    print("[4] Adversarial AUC using ONLY 3 hand-crafted signals")
    print("=" * 78)
    print("    If curation bias is the axis, just these 3 features should give")
    print("    a meaningful adv-AUC (>0.7) with no Perch embedding access.")
    try:
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise SystemExit("xgboost/sklearn missing")

    rng = np.random.default_rng(42)
    n_take = min(n_lab, 5000)
    lab_idx = rng.choice(np.where(is_labeled)[0], n_take, replace=False) \
              if n_lab > n_take else np.where(is_labeled)[0]
    unl_idx = rng.choice(np.where(~is_labeled)[0], n_take, replace=False)
    keep = np.concatenate([lab_idx, unl_idx])
    X3 = np.stack([
        max_logit[keep],
        top5_mean[keep],
        file_variance[keep],
    ], axis=1)
    y3 = np.concatenate([np.zeros(len(lab_idx)), np.ones(len(unl_idx))]).astype(np.int64)
    aucs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X3, y3):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.06,
                              tree_method="hist", random_state=42, eval_metric="auc")
        m.fit(X3[tr], y3[tr], verbose=False)
        aucs.append(roc_auc_score(y3[va], m.predict_proba(X3[va])[:, 1]))
    auc3 = float(np.mean(aucs))
    print(f"    3-feature adv-AUC: {auc3:.4f}")
    print(f"    baseline (1536 emb dims): 0.9932")
    print(f"    fraction of separability captured by 3 features: "
          f"{(auc3 - 0.5) / (0.9932 - 0.5):.1%}")

    # ── Verdict ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("[verdict] Curation-bias hypothesis check")
    print("=" * 78)
    lab_max = float(np.median(max_logit[is_labeled]))
    unl_max = float(np.median(max_logit[~is_labeled]))
    delta = lab_max - unl_max

    if delta > 1.0 and auc3 > 0.80:
        print(f"  STRONG: median max-logit Δ = {delta:+.3f} (lab {lab_max:.3f} vs unl {unl_max:.3f})")
        print(f"  3-feature adv-AUC {auc3:.3f} → curation-bias hypothesis CONFIRMED.")
        print(f"  Labelers selected acoustically richer files. The 0.99 gap is mostly")
        print(f"  a 'how much call content?' signal that's invariant to season/site/year.")
        print()
        print("  Implication for round-2 pseudo: filter unlabeled by call-richness")
        print("  (e.g., max_logit > median). Use those as the 'labeled-distribution-")
        print("  similar' rows, regardless of date or site.")
    elif delta > 0.3 or auc3 > 0.70:
        print(f"  PARTIAL: median max-logit Δ = {delta:+.3f}, 3-feature adv-AUC {auc3:.3f}")
        print(f"  Curation bias is present but not the whole story. There's an additional")
        print(f"  axis Perch embeddings see that hand-features don't.")
    else:
        print(f"  WEAK: median max-logit Δ = {delta:+.3f}, 3-feature adv-AUC {auc3:.3f}")
        print(f"  Curation bias unlikely to explain the 0.99. The gap is in")
        print(f"  higher-order embedding structure that's harder to interpret.")


if __name__ == "__main__":
    main()
