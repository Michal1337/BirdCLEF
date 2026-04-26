"""Isotonic calibration + per-class soft-threshold rescaling."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate_and_optimize_thresholds(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    threshold_grid=None,
    n_windows: int = 12,
) -> np.ndarray:
    """Per-class optimal threshold found on file-max predictions."""
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files = n_samples // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y = y_true.reshape(n_files, n_windows, n_cls).max(axis=1)
    for c in range(n_cls):
        yc = file_y[:, c]
        pc = file_oof[:, c]
        if yc.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(pc, yc)
            pc_cal = ir.transform(pc)
        except Exception:
            pc_cal = pc
        best = (0.0, 0.5)
        for t in threshold_grid:
            pred = (pc_cal >= t).astype(int)
            tp = ((pred == 1) & (yc == 1)).sum()
            fp = ((pred == 1) & (yc == 0)).sum()
            fn = ((pred == 0) & (yc == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best[0]:
                best = (f1, t)
        thresholds[c] = best[1]
    return thresholds


def apply_per_class_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    C = scores.shape[1]
    assert C == len(thresholds)
    out = np.copy(scores)
    for c in range(C):
        t = float(thresholds[c])
        above = scores[:, c] > t
        out[above, c] = 0.5 + 0.5 * (scores[above, c] - t) / (1.0 - t + 1e-8)
        out[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(out, 0.0, 1.0)


def logit_prior_shift(
    scores: np.ndarray,
    meta,  # DataFrame with site, hour_utc
    prior_tables: dict,
    lambda_prior=0.4,
) -> np.ndarray:
    """Additive prior shift in logit space (site + hour empirical rates).

    `lambda_prior` may be a scalar (uniform across classes) or a 1D array of
    length n_classes (per-class strength — e.g. stronger for Amphibia/Insecta
    where Perch is weak, weaker for birds where Perch is reliable).
    """
    eps = 1e-4
    n = len(scores)
    out = scores.copy()
    global_p = prior_tables["global_p"]
    p = np.tile(global_p, (n, 1))
    sites = meta["site"].to_numpy()
    hours = meta["hour_utc"].to_numpy()
    for i, h in enumerate(hours):
        h = int(h)
        if h in prior_tables["hour_to_i"]:
            j = prior_tables["hour_to_i"][h]
            nh = prior_tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * prior_tables["hour_p"][j] + (1 - w) * global_p
    for i, s in enumerate(sites):
        s = str(s)
        if s in prior_tables["site_to_i"]:
            j = prior_tables["site_to_i"][s]
            ns = prior_tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * prior_tables["site_p"][j] + (1 - w) * p[i]
    p = np.clip(p, eps, 1 - eps)
    logit_prior = np.log(p) - np.log1p(-p)
    lam = np.asarray(lambda_prior, dtype=np.float32)
    if lam.ndim == 0:
        return out + float(lam) * logit_prior
    return out + lam[None, :] * logit_prior


def build_prior_tables(meta, y_labels: np.ndarray) -> dict:
    sites = sorted(meta["site"].dropna().astype(str).unique())
    site_to_i = {s: i for i, s in enumerate(sites)}
    site_p = np.zeros((len(sites), y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(sites), dtype=np.float32)
    for s in sites:
        mask = meta["site"].astype(str).values == s
        site_n[site_to_i[s]] = mask.sum()
        site_p[site_to_i[s]] = y_labels[mask].mean(axis=0) if mask.sum() else 0
    hours = sorted(meta["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hours)}
    hour_p = np.zeros((len(hours), y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hours), dtype=np.float32)
    for h in hours:
        mask = meta["hour_utc"].astype(int).values == h
        hour_n[hour_to_i[h]] = mask.sum()
        hour_p[hour_to_i[h]] = y_labels[mask].mean(axis=0) if mask.sum() else 0
    return {
        "global_p": y_labels.mean(axis=0).astype(np.float32),
        "site_to_i": site_to_i,
        "site_p": site_p,
        "site_n": site_n,
        "hour_to_i": hour_to_i,
        "hour_p": hour_p,
        "hour_n": hour_n,
    }
