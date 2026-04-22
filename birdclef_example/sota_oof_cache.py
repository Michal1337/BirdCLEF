"""Converted from sota.ipynb.
Order: imports, definitions, configs/init variables, execution.
"""

# 1) Imports
import copy
import json
import subprocess, sys, os
from pathlib import Path
import os, re, gc, time, warnings
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
import re as _re
import concurrent.futures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
import torch.nn.functional as F
import onnxruntime as ort
try:
    from birdclef_example.oof_sweep_configs import OOF_SWEEP_CONFIGS
except ModuleNotFoundError:
    from oof_sweep_configs import OOF_SWEEP_CONFIGS

# 2) Definitions


def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    raise FileNotFoundError(pattern)


def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y


def run_perch(paths, batch_files=16, verbose=True):
    paths = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.zeros(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs = np.zeros((n_rows, 1536), dtype=np.float32)

    wr = 0
    itr = (
        tqdm(range(0, len(paths), batch_files), desc="Perch")
        if verbose
        else range(0, len(paths), batch_files)
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        # Prefetch first batch
        next_paths = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            # Prefetch next batch immediately
            next_start = start + batch_files
            if next_start < len(paths):
                next_paths = paths[next_start : next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr

            for bi, path in enumerate(batch_paths):
                y = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS : (bi + 1) * N_WINDOWS] = y.reshape(
                    N_WINDOWS, WINDOW_SAMPLES
                )
                row_ids[wr : wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr : wr + N_WINDOWS] = path.name
                sites[wr : wr + N_WINDOWS] = meta["site"]
                hours[wr : wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            # ── ONNX or TF inference ───────────────────────────────────
            outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
            logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
            emb = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)

            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs[br:wr] = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame(
        {"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours}
    )
    return meta_df, scores, embs


def macro_auc(y_true, y_score):
    """
    Exact replica of the competition metric:
    macro-averaged ROC-AUC, skipping classes with no positive labels.
    This is the ONLY number you should track locally.
    """
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


def honest_oof_auc(scores, Y, meta_df, n_splits=5, label="scores"):
    """
    GroupKFold by filename — files never split across folds.
    This is the only correct way to estimate LB performance locally.
    Leaking a file across train/val inflates AUC by ~0.01–0.03.
    """
    groups = meta_df["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(scores, dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores, groups=groups), 1):
        oof[va_idx] = scores[va_idx]

    auc = macro_auc(Y, oof)
    print(f"[{label}] honest OOF macro-AUC: {auc:.6f}")
    return auc, oof


def smooth_predictions(probs, n_windows=12, alpha=0.3):
    """
    For each file's 12 windows, blend each window with its neighbors.

    new[t] = (1 - alpha) * old[t] + 0.5*alpha * (old[t-1] + old[t+1])

    alpha=0: no smoothing (your current baseline)
    alpha=0.3: moderate smoothing (good starting point)

    Shape: (n_files * 12, n_classes) → same shape output
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    # Reshape to (n_files, 12, 234) so we can work file-by-file
    view = probs.reshape(-1, n_windows, C).copy()

    # Shift left and right (with edge padding = repeat boundary)
    prev_w = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)  # t-1
    next_w = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)  # t+1

    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)

    return smoothed.reshape(N, C)


def build_prior_tables(sc_df, Y_labels):
    """
    Build site-level and hour-level species frequency tables.

    These answer: "How often is species X observed at site S at hour H?"

    We use these as a soft prior: add them to raw Perch logits.
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)  # overall frequency

    # ── Site-level frequencies ──────────────────────────────────────────
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)

    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    # ── Hour-level frequencies ──────────────────────────────────────────
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)

    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_p": site_p,
        "site_n": site_n,
        "hour_to_i": hour_to_i,
        "hour_p": hour_p,
        "hour_n": hour_n,
    }


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    """
    Add a scaled prior logit to the raw Perch scores.

    lambda_prior=0: no effect (your baseline)
    lambda_prior=0.4: moderate influence from location/time

    The prior is converted to a logit (log-odds) before adding.
    This is mathematically correct — you add logits, not probabilities.
    """
    eps = 1e-4
    n = len(scores)
    out = scores.copy()

    # Start from global average
    p = np.tile(tables["global_p"], (n, 1))  # (n, 234)

    # Override with hour-level estimate (if enough data)
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)  # shrink toward global if little data
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]

    # Override with site-level estimate (if enough data)
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)  # same shrinkage logic
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]

    # Convert prior probability to logit and add
    p = np.clip(p, eps, 1 - eps)
    logit_prior = np.log(p) - np.log1p(-p)
    out += lambda_prior * logit_prior

    return out.astype(np.float32)


def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    """
    Scale each window's predictions by how confident the file is overall.

    Steps:
    1. For each file, find the top-k highest scores across all 12 windows
    2. Compute their mean → "file confidence"
    3. Multiply every window's scores by (file_confidence ** power)

    power=0: no effect (baseline)
    power=0.4: moderate suppression of uncertain files

    Why top-k and not max?
    Max is noisy (one lucky spike). Top-2 mean is more robust.
    """
    N, C = probs.shape
    assert N % n_windows == 0

    view = probs.reshape(-1, n_windows, C)  # (n_files, 12, 234)
    sorted_v = np.sort(view, axis=1)  # sort across time
    top_k_mean = sorted_v[:, -top_k:, :].mean(
        axis=1, keepdims=True
    )  # (n_files, 1, 234)

    scale = np.power(top_k_mean, power)  # (n_files, 1, 234)
    scaled = view * scale  # broadcast across 12 windows

    return scaled.reshape(N, C)


def build_class_freq_weights(Y, cap=10.0):
    total = Y.shape[0]
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq = pos_count / total
    weights = 1.0 / (freq**0.5)
    weights = np.clip(weights, 1.0, cap)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def build_sequential_features(scores_col, n_windows=12):
    N = len(scores_col)
    assert N % n_windows == 0
    x = scores_col.reshape(-1, n_windows)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean = np.repeat(x.mean(axis=1), n_windows)
    max_ = np.repeat(x.max(axis=1), n_windows)
    std = np.repeat(x.std(axis=1), n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    """
    CHANGE 1: Upgraded MLP probe.
    - pca_dim: 32 → 64  (more embedding information)
    - hidden:  (32,) → (128, 64)  (more capacity)
    - max_iter: 100 → 300  (longer training)
    - min_pos: 8 → 5  (catches more rare species)
    """
    # Step 1: Compress embeddings
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)
    if pca_dim > 0:
        pca = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
        Z = pca.fit_transform(emb_s).astype(np.float32)
        print(
        f"Embedding: {emb.shape} → PCA: {Z.shape}  "
        f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})"
    )
    else:
        pca = None
        Z = emb_s.astype(np.float32)

    class_weights = build_class_freq_weights(Y, cap=10.0)

    probe_models = {}
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(
        f"Training MLP probes for {len(active)} species (>= {min_pos} pos windows)..."
    )

    MAX_ROWS = 3000  # slightly higher budget for (128,64) layers

    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack(
            [
                Z,
                scores_raw[:, ci : ci + 1],
                prev[:, None],
                next_[:, None],
                mean[:, None],
                max_[:, None],
                std[:, None],
            ]
        )

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_idx = np.where(y == 1)[0]

        w = float(class_weights[ci])
        repeat = max(1, int(round(w * n_neg / max(n_pos, 1))))
        repeat = min(repeat, 8)
        if n_pos * repeat + len(y) > MAX_ROWS:
            repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))

        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),  # CHANGE 1: was (32,)
            activation="relu",
            max_iter=300,  # CHANGE 1: was 100
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,  # CHANGE 1: was 10
            random_state=42,
            learning_rate_init=5e-4,  # CHANGE 1: was 1e-3 (lower lr for deeper net)
            alpha=0.005,  # CHANGE 1: was 0.01
        )
        clf.fit(X_bal, y_bal)
        probe_models[ci] = clf

    print(f"Trained {len(probe_models)} MLP probes")
    return probe_models, scaler, pca, alpha_blend


def apply_mlp_probes(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    emb_s = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32) if pca is not None else emb_s.astype(np.float32)
    result = scores_test.copy()
    for ci, clf in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        X_test = np.hstack(
            [
                Z_test,
                scores_test[:, ci : ci + 1],
                prev[:, None],
                next_[:, None],
                mean[:, None],
                max_[:, None],
                std[:, None],
            ]
        )
        prob = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        result[:, ci] = (1 - alpha_blend) * scores_test[:, ci] + alpha_blend * logit
    return result


class VectorizedMLPProbes(nn.Module):
    """Stacks all per-class MLP weights into a single batched PyTorch model.
    Replaces the slow Python for-loop over probe_models at inference time."""

    def __init__(self, probe_models):
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)
        if V == 0:
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            self.n_layers = 0
            return

        sample = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for layer_idx in range(self.n_layers):
            W = np.stack(
                [probe_models[c].coefs_[layer_idx] for c in self.valid_classes], axis=0
            )  # (V, in, out)
            b = np.stack(
                [probe_models[c].intercepts_[layer_idx] for c in self.valid_classes],
                axis=0,
            )  # (V, out)
            self.weights.append(
                nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)
            )
            self.biases.append(
                nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False)
            )

    def forward(self, x):
        # x: (V, N, in_dim)
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)  # (V, N)


def apply_mlp_probes_vectorized(
    emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4
):
    """
    Drop-in replacement for apply_mlp_probes().
    Uses batched PyTorch matrix multiply instead of a Python for-loop —
    ~10-50x faster at inference time.
    """
    if len(probe_models) == 0:
        return scores_test.copy()

    emb_s = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32) if pca is not None else emb_s.astype(np.float32)

    valid_classes = sorted(probe_models.keys())
    V = len(valid_classes)
    N = len(scores_test)

    # Build sequential features for all classes at once
    raw = scores_test[:, valid_classes].T  # (V, N)
    n_files = N // N_WINDOWS
    raw_view = raw.reshape(V, n_files, N_WINDOWS)
    prev = np.concatenate([raw_view[:, :, :1], raw_view[:, :, :-1]], axis=2).reshape(
        V, N
    )
    nxt = np.concatenate([raw_view[:, :, 1:], raw_view[:, :, -1:]], axis=2).reshape(
        V, N
    )
    mean = np.repeat(raw_view.mean(axis=2), N_WINDOWS, axis=1)
    mx = np.repeat(raw_view.max(axis=2), N_WINDOWS, axis=1)
    std = np.repeat(raw_view.std(axis=2), N_WINDOWS, axis=1)

    # scalar_feats: (V, N, 6)
    scalar_feats = np.stack([raw, prev, nxt, mean, mx, std], axis=-1).astype(np.float32)

    # Z_test: (N, D) → broadcast to (V, N, D)
    Z_expanded = np.broadcast_to(Z_test, (V, N, Z_test.shape[1]))

    # X_all: (V, N, D+6)
    X_all = np.concatenate([Z_expanded.astype(np.float32), scalar_feats], axis=-1)

    vec_probe = VectorizedMLPProbes(probe_models)
    vec_probe.eval()
    with torch.no_grad():
        preds = vec_probe(torch.tensor(X_all)).numpy()  # (V, N)

    result = scores_test.copy()
    base_valid = scores_test[:, valid_classes]  # (N, V)
    result[:, valid_classes] = (1.0 - alpha_blend) * base_valid + alpha_blend * preds.T
    return result


def calibrate_and_optimize_thresholds(
    oof_probs, Y_FULL, threshold_grid=None, n_windows=12
):
    """
    CHANGE 2: For each species:
    1. Fit isotonic regression on OOF scores (calibrates overconfident/underconfident classes)
    2. Grid-search F1-optimal threshold over calibrated probs
    Returns: thresholds array of shape (n_classes,)
    """
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files = n_samples // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)

    n_calibrated = 0
    for c in range(n_cls):
        y_true = file_y[:, c]
        y_prob = file_oof[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except Exception:
            y_cal = y_prob

        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        n_calibrated += 1

    print(f"Calibrated {n_calibrated} classes")
    print(f"Mean threshold: {thresholds.mean():.3f}")
    print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds


def apply_per_class_thresholds(scores, thresholds):
    """
    Sharpens probabilities around the per-class threshold:
    - above threshold → push toward 1
    - below threshold → push toward 0
    """
    C = scores.shape[1]
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        above = scores[:, c] > t
        scaled[above, c] = 0.5 + 0.5 * (scores[above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def rank_aware_scaling(probs, n_windows=12, power=0.4):
    """
    CHANGE 6: Scale each window by the file's single peak confidence.

    How it works:
      1. For each file, find the MAX score across all 12 windows (per species)
      2. Raise it to power → scale factor
      3. Multiply every window's score by that scale factor

    Example for one species across 12 windows:
      Confident file:  max=0.90 → scale=0.90^0.4=0.96 → mild boost
      Uncertain file:  max=0.10 → scale=0.10^0.4=0.40 → strong suppression

    How this differs from Change 3 (file_confidence_scale):
      Change 3 uses top-2 MEAN → smoother, less aggressive
      Change 6 uses single MAX  → asks "does ANY window have strong evidence?"

    power=0.0 → no effect (baseline)
    power=0.4 → moderate suppression of uncertain files (recommended start)
    power=1.0 → multiply directly by file max (very aggressive)
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    view = probs.reshape(-1, n_windows, C)  # (n_files, 12, 234)
    file_max = view.max(axis=1, keepdims=True)  # (n_files, 1, 234)

    scale = np.power(file_max, power)  # (n_files, 1, 234)
    scaled = view * scale  # broadcast to all 12 windows

    return scaled.reshape(N, C)


def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """
    CHANGE 7: Smooth uncertain windows toward their neighbors,
    while leaving confident windows almost untouched.

    How it works:
      For each window t:
        conf  = max probability across all 234 species at window t
        alpha = base_alpha * (1 - conf)   ← KEY: adapts to confidence
        new[t] = (1 - alpha) * old[t] + alpha * avg(old[t-1], old[t+1])

    Why alpha adapts to confidence:
      Confident window (max=0.90):
        alpha = 0.20 * (1 - 0.90) = 0.02  → barely smoothed, peak preserved
      Uncertain window (max=0.10):
        alpha = 0.20 * (1 - 0.10) = 0.18  → smoothed more, noise reduced

    This is exactly why your Change 1 hurt (-0.005) but this one should help:
      Change 1 used fixed alpha=0.3 → diluted confident peaks equally
      Change 7 uses adaptive alpha  → protects confident peaks, smooths noise

    base_alpha=0.0  → no smoothing (baseline)
    base_alpha=0.20 → recommended starting point
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    result = probs.copy()
    view = probs.reshape(-1, n_windows, C)  # (n_files, 12, 234) original
    out = result.reshape(-1, n_windows, C)  # (n_files, 12, 234) to modify

    for t in range(n_windows):

        # Confidence at this window = max prob across all species
        # Shape: (n_files, 1) — one confidence value per file per window
        conf = view[:, t, :].max(axis=-1, keepdims=True)  # (n_files, 1)

        # Adaptive alpha — low confidence = more smoothing
        alpha = base_alpha * (1.0 - conf)  # (n_files, 1)

        # Neighbor average with edge padding
        if t == 0:
            # First window: left neighbor = itself
            neighbor_avg = (view[:, t, :] + view[:, t + 1, :]) / 2.0
        elif t == n_windows - 1:
            # Last window: right neighbor = itself
            neighbor_avg = (view[:, t - 1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t - 1, :] + view[:, t + 1, :]) / 2.0

        # Blend: confident windows barely change, uncertain ones smooth more
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg

    return result


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_model, -1)
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """
    CHANGE 4: LightProtoSSM with cross-attention between SSM layers.
    d_model=128, 2 SSM layers + 2-head cross-attention (CPU friendly).
    Trains in ~35–45s on 59 files.
    """

    def __init__(
        self,
        d_input=1536,
        d_model=128,
        d_state=16,
        n_classes=234,
        n_windows=12,
        dropout=0.15,
        n_sites=20,
        meta_dim=16,
        n_ssm_layers=2,
        use_cross_attn=True,
        cross_attn_heads=2,
    ):  # CHANGE 4
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.n_ssm_layers = int(n_ssm_layers)
        if self.n_ssm_layers < 1:
            raise ValueError("n_ssm_layers must be >= 1")

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(self.n_ssm_layers)]
        )
        self.ssm_bwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(self.n_ssm_layers)]
        )
        self.ssm_merge = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(self.n_ssm_layers)]
        )
        self.ssm_norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_ssm_layers)]
        )
        self.drop = nn.Dropout(dropout)

        # CHANGE 4: Cross-attention after each SSM layer
        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model,
                        num_heads=cross_attn_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(self.n_ssm_layers)
                ]
            )
            self.cross_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(self.n_ssm_layers)]
            )

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            )
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(
            zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)
        ):
            res = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h = norm(h + res)

            # CHANGE 4: cross-attention between SSM layers
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = (
            torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp)
            + self.class_bias[None, None, :]
        )
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_light_proto_ssm(
    emb_full,
    scores_full,
    Y_full,
    meta_full,
    emb_val=None,
    scores_val=None,
    y_val=None,
    meta_val=None,
    n_epochs=40,
    patience=8,
    lr=1e-3,
    n_sites=20,
    d_model=128,
    d_state=16,
    n_ssm_layers=2,
    dropout=0.15,
    meta_dim=16,
    distill_weight=0.15,
    pos_weight_cap=25.0,
    use_swa=True,
    swa_start_frac=0.65,
    swa_lr=4e-4,
    use_cross_attn=True,
    cross_attn_heads=2,
    label_smoothing=0.0,
    mixup_alpha=0.0,
    focal_gamma=0.0,
    use_cosine_restart=False,
    restart_period=20,
    verbose=False,
):
    n_files = len(emb_full) // N_WINDOWS
    emb_f = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fnames = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].dropna().astype(str).unique())
    site2i = {s: i + 1 for i, s in enumerate(sites_u)}
    site_ids = np.array(
        [
            min(
                site2i.get(
                    str(meta_full.loc[meta_full["filename"] == fn, "site"].iloc[0]), 0
                ),
                n_sites - 1,
            )
            for fn in fnames
        ],
        dtype=np.int64,
    )
    hour_ids = np.array(
        [
            int(meta_full.loc[meta_full["filename"] == fn, "hour_utc"].iloc[0]) % 24
            for fn in fnames
        ],
        dtype=np.int64,
    )

    model = LightProtoSSM(
        d_input=emb_full.shape[1],
        d_model=d_model,
        d_state=d_state,
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        dropout=dropout,
        n_sites=n_sites,
        meta_dim=meta_dim,
        n_ssm_layers=n_ssm_layers,
        use_cross_attn=use_cross_attn,
        cross_attn_heads=cross_attn_heads,
    )
    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32),
        torch.tensor(Y_full, dtype=torch.float32),
    )
    print(f"LightProtoSSM params: {model.count_parameters():,}")

    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    log_t = torch.tensor(log_f, dtype=torch.float32)
    lab_t = torch.tensor(lab_f, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    has_val = (
        emb_val is not None
        and scores_val is not None
        and y_val is not None
        and meta_val is not None
    )
    if has_val:
        n_val_files = len(emb_val) // N_WINDOWS
        emb_val_t = torch.tensor(
            emb_val.reshape(n_val_files, N_WINDOWS, -1), dtype=torch.float32
        )
        log_val_t = torch.tensor(
            scores_val.reshape(n_val_files, N_WINDOWS, -1), dtype=torch.float32
        )
        fnames_val = meta_val["filename"].unique()
        site_ids_val = np.array(
            [
                min(
                    site2i.get(
                        str(meta_val.loc[meta_val["filename"] == fn, "site"].iloc[0]), 0
                    ),
                    n_sites - 1,
                )
                for fn in fnames_val
            ],
            dtype=np.int64,
        )
        hour_ids_val = np.array(
            [
                int(meta_val.loc[meta_val["filename"] == fn, "hour_utc"].iloc[0]) % 24
                for fn in fnames_val
            ],
            dtype=np.int64,
        )
        site_val_t = torch.tensor(site_ids_val, dtype=torch.long)
        hour_val_t = torch.tensor(hour_ids_val, dtype=torch.long)
        y_val_flat = y_val.reshape(-1, N_CLASSES)

    pos_cnt = lab_t.sum(dim=(0, 1))
    total = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1.0)).clamp(max=float(pos_weight_cap))

    if label_smoothing > 0:
        lab_t_s = lab_t * (1 - label_smoothing) + label_smoothing * 0.5
    else:
        lab_t_s = lab_t

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    if use_cosine_restart:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=restart_period, eta_min=lr * 0.01
        )
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            epochs=n_epochs,
            steps_per_epoch=1,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    best_metric, best_state, wait = float("-inf"), None, 0
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start = int(n_epochs * swa_start_frac)
        swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=swa_lr)
    else:
        swa_model = None
        swa_start = n_epochs + 1
        swa_sched = None

    def _loss_fn(logits, targets, pw):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        if focal_gamma > 0:
            pt = torch.exp(-bce)
            return ((1 - pt) ** focal_gamma * bce).mean()
        return bce.mean()

    for ep in range(n_epochs):
        model.train()
        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(emb_t.shape[0])
            e_in = lam * emb_t + (1 - lam) * emb_t[idx]
            l_in = lam * log_t + (1 - lam) * log_t[idx]
            lab_in = lam * lab_t_s + (1 - lam) * lab_t_s[idx]
        else:
            e_in, l_in, lab_in = emb_t, log_t, lab_t_s

        out = model(e_in, l_in, site_ids=site_t, hours=hour_t)
        loss = _loss_fn(out, lab_in, pos_weight[None, None, :]) + distill_weight * F.mse_loss(out, l_in)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if use_swa and ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                out_val = model(emb_val_t, log_val_t, site_ids=site_val_t, hours=hour_val_t)
            val_probs = 1.0 / (
                1.0 + np.exp(-np.clip(out_val.numpy().reshape(-1, N_CLASSES), -30, 30))
            )
            monitor = float(macro_auc(y_val_flat, val_probs))
        else:
            monitor = -float(loss.item())

        if monitor > best_metric:
            best_metric = monitor
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose:
                print(f"  Early stop ep {ep + 1}")
            break

    if use_swa and ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
    metric_label = f"val_auc={best_metric:.4f}" if has_val else f"best_loss={-best_metric:.4f}"
    print(f"LightProtoSSM trained — {metric_label}")
    return model, site2i


def run_tta_proto(
    proto_model, emb_files, sc_files, site_t, hour_t, shifts=[0, 1, -1, 2, -2]
):
    """
    CHANGE 3: TTA by circular-shifting 12-window sequences.

    For each shift s:
      1. Roll embeddings and perch logits by s windows
      2. Run ProtoSSM → get predictions
      3. Roll predictions back by -s (undo shift)

    Finally average all predictions across shifts.

    Why this works:
      - ProtoSSM sees temporal context across all 12 windows
      - Different starting points expose different context patterns
      - Averaging over 5 views reduces temporal boundary artifacts
    """
    proto_model.eval()
    all_preds = []

    emb_t = torch.tensor(emb_files, dtype=torch.float32)
    sc_t = torch.tensor(sc_files, dtype=torch.float32)

    for shift in shifts:
        if shift == 0:
            e_shifted = emb_t
            s_shifted = sc_t
        else:
            e_shifted = torch.roll(emb_t, shift, dims=1)
            s_shifted = torch.roll(sc_t, shift, dims=1)

        with torch.no_grad():
            out = proto_model(
                e_shifted, s_shifted, site_ids=site_t, hours=hour_t
            ).numpy()  # (n_files, 12, 234)

        if shift != 0:
            out = np.roll(out, -shift, axis=1)  # undo shift

        all_preds.append(out)

    return np.mean(all_preds, axis=0)  # (n_files, 12, 234)


class ResidualSSM(nn.Module):
    """
    Lightweight second-pass model that learns to correct
    systematic errors from the first-pass ensemble.

    Input:  embeddings + first-pass scores (concatenated)
    Output: additive correction to first-pass scores

    Key design: output head initialized to zero
    so corrections start small and only grow if helpful.
    ~25s training on 59 files.
    """

    def __init__(
        self,
        d_input=1536,
        d_scores=234,
        d_model=64,
        d_state=8,
        n_classes=234,
        n_windows=12,
        dropout=0.1,
        n_sites=20,
        meta_dim=8,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)

        self.output_head = nn.Linear(d_model, n_classes)
        # Zero init — corrections start at zero, only grow if helpful
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat(
                    [
                        self.site_emb(
                            site_ids.clamp(0, self.site_emb.num_embeddings - 1)
                        ),
                        self.hour_emb(hours.clamp(0, 23)),
                    ],
                    dim=-1,
                )
            )
            h = h + meta.unsqueeze(1)

        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([h_f, h_b], dim=-1)))
        h = self.ssm_norm(h + res)

        return self.output_head(h)  # (B, T, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_residual_ssm(
    emb_full,
    first_pass_flat,
    Y_full,
    site_ids,
    hour_ids,
    n_epochs=30,
    patience=8,
    lr=1e-3,
    correction_weight=0.30,
    verbose=False,
):
    """
    Train ResidualSSM to predict (Y - sigmoid(first_pass)).
    Returns corrected flat scores (n_rows, n_classes).
    ~20s on CPU.
    """
    n_files = len(emb_full) // N_WINDOWS
    emb_f = emb_full.reshape(n_files, N_WINDOWS, -1)
    fp_f = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    # Residual target = label - sigmoid(first_pass)
    fp_prob = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals = lab_f - fp_prob  # values in [-1, 1]

    print(
        f"Residuals: mean={residuals.mean():.4f}  "
        f"std={residuals.std():.4f}  "
        f"abs_mean={np.abs(residuals).mean():.4f}"
    )

    # Train / val split (file level, no shuffle leakage)
    n_val = max(1, int(n_files * 0.15))
    rng = torch.Generator()
    rng.manual_seed(42)
    perm = torch.randperm(n_files, generator=rng).numpy()
    val_i = perm[:n_val]
    train_i = perm[n_val:]

    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    fp_t = torch.tensor(fp_f, dtype=torch.float32)
    res_t = torch.tensor(residuals, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    model = ResidualSSM(n_classes=N_CLASSES)
    print(f"ResidualSSM params: {model.count_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(n_epochs):
        model.train()
        corr = model(
            emb_t[train_i],
            fp_t[train_i],
            site_ids=site_t[train_i],
            hours=hour_t[train_i],
        )
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_corr = model(
                emb_t[val_i], fp_t[val_i], site_ids=site_t[val_i], hours=hour_t[val_i]
            )
            val_loss = F.mse_loss(val_corr, res_t[val_i])

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose:
                print(f"  Early stop ep {ep+1}")
            break

    model.load_state_dict(best_state)
    print(f"ResidualSSM trained — best val MSE={best_loss:.6f}")

    # Apply correction to ALL training data (for verification)
    model.eval()
    with torch.no_grad():
        all_corr = model(emb_t, fp_t, site_ids=site_t, hours=hour_t).numpy()
    print(
        f"Correction magnitude: "
        f"mean_abs={np.abs(all_corr).mean():.4f}  "
        f"max={np.abs(all_corr).max():.4f}"
    )

    return model, correction_weight


def run_pipeline_oof(emb_full, sc_full, Y_full, meta_full, n_splits=5):
    """
    Proper full-pipeline OOF.
    Trains ProtoSSM + MLP on K-1 folds, predicts on held-out fold.
    ~3-4 min total on CPU. Use this instead of the raw-Perch OOF.
    """
    file_meta = meta_full.drop_duplicates("filename").reset_index(drop=True)
    proto_cfg = CFG.get("proto_ssm_train", {})
    proto_n_sites = int(proto_cfg.get("n_sites", 20))

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        emb_tr_f = emb_full[tr_mask]
        sc_tr_f = sc_full[tr_mask]
        Y_tr_f = Y_full[tr_mask]
        meta_tr_f = meta_full[tr_mask].reset_index(drop=True)

        emb_va_f = emb_full[va_mask]
        sc_va_f = sc_full[va_mask]
        Y_va_f = Y_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        # ── Train ProtoSSM on train fold ───────────────────────────────
        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            meta_tr_f,
            emb_val=emb_va_f,
            scores_val=sc_va_f,
            y_val=Y_va_f,
            meta_val=meta_va_f,
            n_epochs=int(proto_cfg.get("n_epochs", 40)),
            patience=int(proto_cfg.get("patience", 8)),
            lr=float(proto_cfg.get("lr", 1e-3)),
            n_sites=proto_n_sites,
            d_model=int(proto_cfg.get("d_model", 128)),
            d_state=int(proto_cfg.get("d_state", 16)),
            n_ssm_layers=int(proto_cfg.get("n_ssm_layers", 2)),
            dropout=float(proto_cfg.get("dropout", 0.15)),
            meta_dim=int(proto_cfg.get("meta_dim", 16)),
            distill_weight=float(proto_cfg.get("distill_weight", 0.15)),
            pos_weight_cap=float(proto_cfg.get("pos_weight_cap", 25.0)),
            use_swa=bool(proto_cfg.get("use_swa", True)),
            swa_start_frac=float(proto_cfg.get("swa_start_frac", 0.65)),
            swa_lr=float(proto_cfg.get("swa_lr", 4e-4)),
            use_cross_attn=bool(proto_cfg.get("use_cross_attn", True)),
            cross_attn_heads=int(proto_cfg.get("cross_attn_heads", 2)),
            label_smoothing=float(proto_cfg.get("label_smoothing", 0.0)),
            mixup_alpha=float(proto_cfg.get("mixup_alpha", 0.0)),
            focal_gamma=float(proto_cfg.get("focal_gamma", 0.0)),
            use_cosine_restart=bool(proto_cfg.get("use_cosine_restart", False)),
            restart_period=int(proto_cfg.get("restart_period", 20)),
            verbose=False,
        )

        # ── ProtoSSM predict on val fold ───────────────────────────────
        n_va = len(emb_va_f) // N_WINDOWS

        va_fn_list = meta_va_f.drop_duplicates("filename")["filename"].tolist()

        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        str(meta_va_f.loc[meta_va_f["filename"] == fn, "site"].iloc[0]),
                        0,
                    ),
                    proto_n_sites - 1,
                )
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        va_hour_ids = np.array(
            [
                int(meta_va_f.loc[meta_va_f["filename"] == fn, "hour_utc"].iloc[0]) % 24
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        proto_model.eval()
        with torch.no_grad():
            proto_va = (
                proto_model(
                    torch.tensor(
                        emb_va_f.reshape(n_va, N_WINDOWS, -1),
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        sc_va_f.reshape(n_va, N_WINDOWS, -1),
                        dtype=torch.float32,
                    ),
                    site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                    hours=torch.tensor(va_hour_ids, dtype=torch.long),
                )
                .numpy()
                .reshape(-1, N_CLASSES)
            )

        # ── Train MLP on train fold ────────────────────────────────────
        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            min_pos=5,
            pca_dim=64,
            alpha_blend=0.4,
        )

        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f,
            sc_va_f,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )

        # ── Ensemble + sigmoid ─────────────────────────────────────────
        first_pass = 0.5 * proto_va + 0.5 * sc_va_mlp
        probs_va = 1.0 / (1.0 + np.exp(-np.clip(first_pass, -30, 30)))
        oof_probs[va_mask] = probs_va

        fold_auc = macro_auc(Y_full[va_mask], probs_va)
        print(
            f"  Fold {fold}/{n_splits}  val files={len(va_fnames)}  AUC={fold_auc:.6f}"
        )

    overall = macro_auc(Y_full, oof_probs)
    print(f"\nFull pipeline OOF AUC: {overall:.6f}")
    return overall, oof_probs


def run_pipeline_oof_fullstack(
    emb_full,
    sc_full,
    Y_full,
    meta_full,
    temperatures,
):
    """
    Fold-safe OOF that mirrors the final submission stack:
    ProtoSSM + prior + MLP + residual correction + post-processing.
    """
    n_splits = 5
    ensemble_w = float(CFG.get("ensemble_w", 0.50))
    lambda_prior = float(CFG.get("lambda_prior", 0.40))
    proto_cfg = CFG.get("proto_ssm_train", {})
    proto_n_epochs = int(proto_cfg.get("n_epochs", 40))
    proto_patience = int(proto_cfg.get("patience", 8))
    proto_lr = float(proto_cfg.get("lr", 1e-3))
    proto_n_sites = int(proto_cfg.get("n_sites", 20))
    proto_d_model = int(proto_cfg.get("d_model", 128))
    proto_d_state = int(proto_cfg.get("d_state", 16))
    proto_n_ssm_layers = int(proto_cfg.get("n_ssm_layers", 2))
    proto_dropout = float(proto_cfg.get("dropout", 0.15))
    proto_meta_dim = int(proto_cfg.get("meta_dim", 16))
    proto_distill_weight = float(proto_cfg.get("distill_weight", 0.15))
    proto_pos_weight_cap = float(proto_cfg.get("pos_weight_cap", 25.0))
    proto_use_swa = bool(proto_cfg.get("use_swa", True))
    proto_swa_start_frac = float(proto_cfg.get("swa_start_frac", 0.65))
    proto_swa_lr = float(proto_cfg.get("swa_lr", 4e-4))
    proto_use_cross_attn = bool(proto_cfg.get("use_cross_attn", True))
    proto_cross_attn_heads = int(proto_cfg.get("cross_attn_heads", 2))
    proto_label_smoothing = float(proto_cfg.get("label_smoothing", 0.0))
    proto_mixup_alpha = float(proto_cfg.get("mixup_alpha", 0.0))
    proto_focal_gamma = float(proto_cfg.get("focal_gamma", 0.0))
    proto_use_cosine_restart = bool(proto_cfg.get("use_cosine_restart", False))
    proto_restart_period = int(proto_cfg.get("restart_period", 20))
    mlp_min_pos = int(CFG.get("mlp_min_pos", 5))
    mlp_pca_dim = int(CFG.get("mlp_pca_dim", 64))
    mlp_alpha_blend = float(CFG.get("mlp_alpha_blend", 0.40))
    resid_cfg = CFG.get("residual_ssm", {})
    residual_n_epochs = int(resid_cfg.get("n_epochs", 30))
    residual_patience = int(resid_cfg.get("patience", 8))
    residual_lr = float(resid_cfg.get("lr", 1e-3))
    correction_weight = float(resid_cfg.get("correction_weight", 0.30))
    rank_power = float(CFG.get("rank_power", 0.40))
    smooth_alpha = float(CFG.get("smooth_alpha", 0.20))
    tta_shifts = [int(x) for x in CFG.get("tta_shifts", [0, 1, -1, 2, -2])]
    threshold_grid = CFG.get(
        "threshold_grid",
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    )
    cfg_name = str(CFG.get("name", "unnamed"))
    safe_cfg_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg_name)
    metrics_out_path = (
        BASE / "outputs" / "sweep" / f"oof_fullstack_stage_metrics_{safe_cfg_name}.json"
    )
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = meta_full.drop_duplicates("filename").reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    stage_order = [
        "raw_perch",
        "proto_only",
        "prior_only",
        "prior_mlp",
        "first_pass",
        "residual_plus_temp",
        "post_rank",
        "post_smooth",
        "final_after_threshold",
    ]
    stage_oof = {
        stage_name: np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
        for stage_name in stage_order
    }

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        emb_tr_f = emb_full[tr_mask]
        sc_tr_f = sc_full[tr_mask]
        Y_tr_f = Y_full[tr_mask]
        meta_tr_f = meta_full[tr_mask].reset_index(drop=True)

        emb_va_f = emb_full[va_mask]
        sc_va_f = sc_full[va_mask]
        Y_va_f = Y_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        # Train fold models
        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            meta_tr_f,
            emb_val=emb_va_f,
            scores_val=sc_va_f,
            y_val=Y_va_f,
            meta_val=meta_va_f,
            n_epochs=proto_n_epochs,
            patience=proto_patience,
            lr=proto_lr,
            n_sites=proto_n_sites,
            d_model=proto_d_model,
            d_state=proto_d_state,
            n_ssm_layers=proto_n_ssm_layers,
            dropout=proto_dropout,
            meta_dim=proto_meta_dim,
            distill_weight=proto_distill_weight,
            pos_weight_cap=proto_pos_weight_cap,
            use_swa=proto_use_swa,
            swa_start_frac=proto_swa_start_frac,
            swa_lr=proto_swa_lr,
            use_cross_attn=proto_use_cross_attn,
            cross_attn_heads=proto_cross_attn_heads,
            label_smoothing=proto_label_smoothing,
            mixup_alpha=proto_mixup_alpha,
            focal_gamma=proto_focal_gamma,
            use_cosine_restart=proto_use_cosine_restart,
            restart_period=proto_restart_period,
            verbose=False,
        )
        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            min_pos=mlp_min_pos,
            pca_dim=mlp_pca_dim,
            alpha_blend=mlp_alpha_blend,
        )

        # Fold-local priors (no leakage)
        prior_tables_fold = build_prior_tables(meta_tr_f[["site", "hour_utc"]], Y_tr_f)

        # Validation metadata at file level
        n_va = len(emb_va_f) // N_WINDOWS
        va_fn_list = meta_va_f.drop_duplicates("filename")["filename"].tolist()
        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        str(meta_va_f.loc[meta_va_f["filename"] == fn, "site"].iloc[0]),
                        0,
                    ),
                    proto_n_sites - 1,
                )
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )
        va_hour_ids = np.array(
            [
                int(meta_va_f.loc[meta_va_f["filename"] == fn, "hour_utc"].iloc[0]) % 24
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        # First-pass on validation fold (mirrors final test branch)
        proto_model.eval()
        with torch.no_grad():
            proto_va = (
                proto_model(
                    torch.tensor(
                        emb_va_f.reshape(n_va, N_WINDOWS, -1),
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        sc_va_f.reshape(n_va, N_WINDOWS, -1),
                        dtype=torch.float32,
                    ),
                    site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                    hours=torch.tensor(va_hour_ids, dtype=torch.long),
                )
                .numpy()
                .reshape(-1, N_CLASSES)
            )

        sc_va_prior = apply_prior(
            sc_va_f,
            sites=meta_va_f["site"].to_numpy(),
            hours=meta_va_f["hour_utc"].to_numpy(),
            tables=prior_tables_fold,
            lambda_prior=lambda_prior,
        )
        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f,
            sc_va_prior,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )
        first_pass_va = ensemble_w * proto_va + (1.0 - ensemble_w) * sc_va_mlp

        raw_perch_probs_va = sigmoid(sc_va_f)
        proto_probs_va = sigmoid(proto_va)
        prior_only_probs_va = sigmoid(sc_va_prior)
        prior_mlp_probs_va = sigmoid(sc_va_mlp)
        first_pass_probs_va = sigmoid(first_pass_va)

        # Build training first-pass for calibration and residual training
        n_tr = len(emb_tr_f) // N_WINDOWS
        emb_tr_seq = emb_tr_f.reshape(n_tr, N_WINDOWS, -1)
        sc_tr_seq = sc_tr_f.reshape(n_tr, N_WINDOWS, -1)
        tr_fn_list = meta_tr_f.drop_duplicates("filename")["filename"].tolist()
        tr_site_ids = np.array(
            [
                min(
                    site2i.get(
                        str(meta_tr_f.loc[meta_tr_f["filename"] == fn, "site"].iloc[0]),
                        0,
                    ),
                    proto_n_sites - 1,
                )
                for fn in tr_fn_list
            ],
            dtype=np.int64,
        )
        tr_hour_ids = np.array(
            [
                int(meta_tr_f.loc[meta_tr_f["filename"] == fn, "hour_utc"].iloc[0]) % 24
                for fn in tr_fn_list
            ],
            dtype=np.int64,
        )

        proto_tr_out = run_tta_proto(
            proto_model,
            emb_tr_seq,
            sc_tr_seq,
            site_t=torch.tensor(tr_site_ids, dtype=torch.long),
            hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
            shifts=tta_shifts,
        )
        proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

        sc_tr_prior = apply_prior(
            sc_tr_f,
            sites=meta_tr_f["site"].to_numpy(),
            hours=meta_tr_f["hour_utc"].to_numpy(),
            tables=prior_tables_fold,
            lambda_prior=lambda_prior,
        )
        sc_tr_mlp = apply_mlp_probes_vectorized(
            emb_tr_f,
            sc_tr_prior,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )
        first_pass_tr = ensemble_w * proto_tr_flat + (1.0 - ensemble_w) * sc_tr_mlp

        thresholds_fold = calibrate_and_optimize_thresholds(
            oof_probs=sigmoid(first_pass_tr),
            Y_FULL=Y_tr_f,
            threshold_grid=threshold_grid,
            n_windows=N_WINDOWS,
        )

        res_model, fold_corr_w = train_residual_ssm(
            emb_full=emb_tr_f,
            first_pass_flat=first_pass_tr,
            Y_full=Y_tr_f,
            site_ids=tr_site_ids,
            hour_ids=tr_hour_ids,
            n_epochs=residual_n_epochs,
            patience=residual_patience,
            lr=residual_lr,
            correction_weight=correction_weight,
            verbose=False,
        )

        # Residual correction on validation fold
        first_pass_va_seq = first_pass_va.reshape(n_va, N_WINDOWS, -1)
        res_model.eval()
        with torch.no_grad():
            va_correction = res_model(
                torch.tensor(emb_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                torch.tensor(first_pass_va_seq, dtype=torch.float32),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                hours=torch.tensor(va_hour_ids, dtype=torch.long),
            ).numpy()
        va_corr_flat = va_correction.reshape(-1, N_CLASSES).astype(np.float32)

        final_scores_va = first_pass_va + fold_corr_w * va_corr_flat
        final_scores_va = final_scores_va / temperatures[None, :]
        residual_plus_temp_probs_va = sigmoid(final_scores_va)
        post_rank_probs_va = rank_aware_scaling(
            residual_plus_temp_probs_va,
            n_windows=N_WINDOWS,
            power=rank_power,
        )
        post_smooth_probs_va = adaptive_delta_smooth(
            post_rank_probs_va,
            n_windows=N_WINDOWS,
            base_alpha=smooth_alpha,
        )
        post_smooth_probs_va = np.clip(post_smooth_probs_va, 0.0, 1.0)
        final_probs_va = apply_per_class_thresholds(post_smooth_probs_va, thresholds_fold)

        stage_probs_va = {
            "raw_perch": raw_perch_probs_va,
            "proto_only": proto_probs_va,
            "prior_only": prior_only_probs_va,
            "prior_mlp": prior_mlp_probs_va,
            "first_pass": first_pass_probs_va,
            "residual_plus_temp": residual_plus_temp_probs_va,
            "post_rank": post_rank_probs_va,
            "post_smooth": post_smooth_probs_va,
            "final_after_threshold": final_probs_va,
        }
        for stage_name, stage_probs in stage_probs_va.items():
            stage_oof[stage_name][va_mask] = stage_probs
        fold_auc = float(macro_auc(Y_full[va_mask], final_probs_va))
        print(
            f"  Fold {fold}/{n_splits}  val files={len(va_fnames)}  full-stack AUC={fold_auc:.6f}"
        )

    global_stage_metrics_raw = {
        stage_name: float(macro_auc(Y_full, stage_oof[stage_name]))
        for stage_name in stage_order
    }
    global_stage_metrics = {
        stage_name: round(metric_val, 4)
        for stage_name, metric_val in global_stage_metrics_raw.items()
    }
    oof_probs = stage_oof["final_after_threshold"]
    overall = global_stage_metrics_raw["final_after_threshold"]

    payload = {
        "config_name": cfg_name,
        "total_folds": int(n_splits),
        "stage_order": stage_order,
        "global_metrics": global_stage_metrics,
    }
    with metrics_out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nFull-stack OOF AUC: {overall:.6f}")
    print(f"Saved stage metrics: {metrics_out_path}")
    return overall, oof_probs, global_stage_metrics, str(metrics_out_path)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# 3) Configs And Init Variables
MODE = "train"  # ← change to "train" for local CV
CFG = {
    # inference
    "batch_files": 16,

    # train-only flags
    "run_oof": MODE == "train",
    "verbose": MODE == "train",
    "dryrun_n_files": 20 if MODE == "train" else 0,

    # ensemble / post-processing
    "ensemble_w": 0.50,
    "lambda_prior": 0.40,
    "rank_power": 0.40,
    "smooth_alpha": 0.20,
    "tta_shifts": [0, 1, -1, 2, -2],
    "threshold_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],

    # MLP probes
    "mlp_min_pos": 3,
    "mlp_pca_dim": 0,
    "mlp_alpha_blend": 0.15,
    "mlp_params": {
        "hidden_layer_sizes": (384, 192, 96),
        "activation": "relu",
        "max_iter": 500 if MODE == "train" else 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20 if MODE == "train" else 10,
        "random_state": 42,
        "learning_rate_init": 6e-4,
        "alpha": 0.005,
    },

    # proto_ssm config: ultra_attn_d448_s40_noswa_fastlr
    "proto_ssm_train": {
        "n_epochs": 75,
        "patience": 13,
        "lr": 5e-4 * 1.35,
        "n_sites": 24,
        "d_model": 448,
        "d_state": 40,
        "n_ssm_layers": 4,
        "dropout": 0.28,
        "meta_dim": 56,
        "distill_weight": 0.22,
        "pos_weight_cap": 30.0,
        "use_swa": False,
        "swa_start_frac": 1.1,
        "swa_lr": 0.0,
        "use_cross_attn": True,
        "cross_attn_heads": 8,
    },

    # residual SSM
    "residual_ssm": {
        "d_model": 128,
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.1,
        "n_epochs": 30,
        "patience": 8,
        "lr": 1e-3,
        "correction_weight": 0.30,
    },
}
INPUT_ROOT = Path("/kaggle/input")
ONNX_WHL = Path(
    "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
BASE = Path(".")
MODEL_DIR = BASE / "models" / "perch_v2_cpu" / "1"
ONNX_PERCH_PATH = BASE / "models" / "perch_onnx" / "perch_v2.onnx"
CACHE_META = BASE / "data" / "perch_cache_finetuned" / "full_perch_meta.parquet"
CACHE_NPZ = BASE / "data" / "perch_cache_finetuned" / "full_perch_arrays.npz"

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12  # 12 × 5s = 60s per file
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
TEXTURE_TAXA = {"Amphibia", "Insecta"}
proxy_map = {}  # label_idx -> list of bc_indices; finetuned cache already fills the remainder
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
TEXTURE_TAXA = {"Amphibia", "Insecta"}  # continuous callers
baseline_auc = None
oof_raw = None
n_sites_cap = 20
# 4) Execution
taxonomy = pd.read_csv("data/taxonomy.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
soundscape_labels = pd.read_csv("data/train_soundscapes_labels.csv")
test_paths = sorted((Path("data/test_soundscapes").glob("*.ogg")))

assert MODE in {"train", "submit"}
print("MODE =", MODE)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
_WALL_START = time.time()
print("✅ V18 CFG loaded")
print(
    f"  n_epochs={CFG['proto_ssm_train']['n_epochs']}  "
    f"patience={CFG['proto_ssm_train']['patience']}  "
    f"oof_n_splits=5  "
    f"mlp_max_iter={CFG['mlp_params']['max_iter']}"
)
print("Config ready")
print(
    f"  run_oof={CFG['run_oof']}  verbose={CFG['verbose']}  dryrun={CFG['dryrun_n_files']}"
)

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
sc = (
    soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"] = (
    sc["filename"].str.replace(".ogg", "", regex=False)
    + "_"
    + sc["end_sec"].astype(str)
)
_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)
Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1
windows_per_file = sc.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc["fully_labeled"] = sc["filename"].isin(full_files)
full_rows = (
    sc[sc["fully_labeled"]].sort_values(["filename", "end_sec"]).reset_index(drop=False)
)
Y_FULL = Y_SC[full_rows["index"].to_numpy()]
print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)}")
print(
    f"Full-file windows: {len(full_rows)} | Active classes: {int((Y_FULL.sum(0) > 0).sum())}"
)

birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]
_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
ONNX_SESSION = ort.InferenceSession(
    str(ONNX_PERCH_PATH), sess_options=_so, providers=["CPUExecutionProvider"]
)
ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
ONNX_OUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
print("Using ONNX Perch (150x faster)")
bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
NO_LABEL = len(bc_labels)
mapping = taxonomy.merge(
    bc_labels.rename(columns={"scientific_name": "scientific_name"}),
    on="scientific_name",
    how="left",
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK = BC_INDICES != NO_LABEL
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
unmapped_df = taxonomy[
    taxonomy["primary_label"].isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])
].copy()
for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci = str(row["scientific_name"])
    genus = sci.split()[0]
    # Find all Perch labels from the same genus
    hits = bc_labels[
        bc_labels["scientific_name"]
        .astype(str)
        .str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()
proxy_map = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}
print(f"Unmapped species total:        {len(UNMAPPED_POS)}")
print(f"Species with genus proxy:      {len(proxy_map)}")
print(f"Species covered by finetuned cache: {len(UNMAPPED_POS) - len(proxy_map)}")
print("\nProxy targets:")

for idx, bc_idxs in list(proxy_map.items())[:8]:
    label = PRIMARY_LABELS[idx]
    cls = CLASS_NAME_MAP.get(label, "?")
    print(f"  {label:12s} ({cls:10s}) ← {len(bc_idxs)} Perch genus matches")
print("✅ Perch inference engine (ONNX + multithreaded I/O) defined")
print("Loading Perch cache from:", CACHE_META.parent)
meta_tr = pd.read_parquet(CACHE_META)
_arr = np.load(CACHE_NPZ)
sc_tr = _arr["scores_full_raw"].astype(np.float32)
emb_tr = _arr["emb_full"].astype(np.float32)
Y_FULL_aligned = Y_SC[
    full_rows.set_index("row_id").loc[meta_tr["row_id"], "index"].to_numpy()
]
assert len(meta_tr) == len(full_files) * N_WINDOWS
print(
    f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  Y_FULL_aligned: {Y_FULL_aligned.shape}"
)
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    if cls in TEXTURE_TAXA:
        temperatures[ci] = 0.95  # frogs/insects: slightly sharper
    else:
        temperatures[ci] = 1.10  # birds: slightly softer
n_texture = (temperatures < 1.0).sum()
n_event = (temperatures > 1.0).sum()
print(
    f"✅ Temperatures: {n_event} event species (T=1.10), {n_texture} texture species (T=0.95)"
)
if CFG["run_oof"]:
    print("Running honest OOF evaluation on training data…")
    baseline_auc, oof_raw = honest_oof_auc(
        sc_tr, Y_FULL_aligned, meta_tr, n_splits=5, label="raw Perch"
    )
    print(f"\nBaseline OOF AUC: {baseline_auc:.6f}  ← your starting point")
else:
    print("Submit mode: skipping OOF evaluation")
if CFG["run_oof"]:
    if not OOF_SWEEP_CONFIGS:
        raise ValueError("OOF_SWEEP_CONFIGS is empty. Add at least one config preset.")

    print(
        f"Running full-stack fold-safe OOF sweep over {len(OOF_SWEEP_CONFIGS)} configs…"
    )
    base_cfg = copy.deepcopy(CFG)
    sweep_results = []
    sweep_dir = BASE / "outputs" / "sweep5"
    sweep_out_path = sweep_dir / "oof_fullstack_sweep_results.json"
    sweep_csv_path = sweep_dir / "oof_fullstack_sweep_results.csv"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    def round_metrics_dict(metrics_dict):
        return {k: round(float(v), 4) for k, v in metrics_dict.items()}

    def prepare_sweep_results_for_export(ranked_results):
        export_rows = []
        for result in ranked_results:
            row = copy.deepcopy(result)
            row["metric"] = round(float(result.get("metric", 0.0)), 4)
            row["stage_metrics"] = round_metrics_dict(result.get("stage_metrics", {}))
            export_rows.append(row)
        return export_rows

    def write_sweep_results_csv(ranked_results, csv_path):
        rows = []
        for rank, result in enumerate(ranked_results, 1):
            row = {
                "rank": int(rank),
                "config_name": str(result.get("config_name", "")),
                "metric": round(float(result.get("metric", 0.0)), 4),
                "stage_metrics_path": str(result.get("stage_metrics_path", "")),
            }
            for stage_name, stage_val in result.get("stage_metrics", {}).items():
                row[f"metric_{stage_name}"] = round(float(stage_val), 4)
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")

    sweep_iter = tqdm(
        OOF_SWEEP_CONFIGS,
        total=len(OOF_SWEEP_CONFIGS),
        desc="OOF sweep",
    )
    for run_idx, sweep_cfg in enumerate(sweep_iter, 1):
        CFG = copy.deepcopy(base_cfg)
        CFG.update(sweep_cfg)

        cfg_name = str(CFG.get("name", f"cfg_{run_idx:02d}"))
        sweep_iter.set_postfix_str(cfg_name)
        print(f"\n[{run_idx}/{len(OOF_SWEEP_CONFIGS)}] {cfg_name}")

        pipeline_auc, _, stage_metrics, stage_metrics_path = run_pipeline_oof_fullstack(
            emb_tr,
            sc_tr,
            Y_FULL_aligned,
            meta_tr,
            temperatures=temperatures,
        )
        print(f"Completed {cfg_name}: AUC={pipeline_auc:.6f}")

        sweep_results.append(
            {
                "config_name": cfg_name,
                "metric": float(pipeline_auc),
                "config": copy.deepcopy(CFG),
                "stage_metrics": stage_metrics,
                "stage_metrics_path": stage_metrics_path,
            }
        )

        partial_results = sorted(
            sweep_results,
            key=lambda r: r["metric"],
            reverse=True,
        )
        partial_results_export = prepare_sweep_results_for_export(partial_results)
        with sweep_out_path.open("w", encoding="utf-8") as f:
            json.dump(partial_results_export, f, indent=2)
        write_sweep_results_csv(partial_results_export, sweep_csv_path)
        print(
            f"Saved interim sweep results ({len(partial_results)}/{len(OOF_SWEEP_CONFIGS)}): {sweep_out_path} and {sweep_csv_path}"
        )

    full_cfg_results = sorted(
        sweep_results,
        key=lambda r: r["metric"],
        reverse=True,
    )
    full_cfg_results_export = prepare_sweep_results_for_export(full_cfg_results)
    with sweep_out_path.open("w", encoding="utf-8") as f:
        json.dump(full_cfg_results_export, f, indent=2)
    write_sweep_results_csv(full_cfg_results_export, sweep_csv_path)

    CFG = copy.deepcopy(base_cfg)

    print("\nSweep summary (best first):")
    for row in full_cfg_results:
        print(f"{row['config_name']}: {row['metric']:.6f}")

    best_row = full_cfg_results[0]
    print(f"\nBest config: {best_row['config_name']}  AUC={best_row['metric']:.6f}")
    print(f"Saved sweep results: {sweep_out_path}")
    print(f"Saved sweep CSV: {sweep_csv_path}")
else:
    print("Submit mode: skipping OOF sweep")

print(f"Total wall time: {(time.time() - _WALL_START)/60:.1f} min")
