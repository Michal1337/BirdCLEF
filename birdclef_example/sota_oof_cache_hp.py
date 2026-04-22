"""Converted from sota.ipynb.
Order: imports, definitions, configs/init variables, execution.
"""

# 1) Imports
import copy
import json
import random
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


class TorchProbeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_torch_probes(emb, scores_raw, Y, cfg):
    min_pos = int(cfg.get("min_pos", 5))
    pca_dim = int(cfg.get("pca_dim", 64))
    max_rows = int(cfg.get("max_rows", 3000))
    hidden_dims = tuple(cfg.get("hidden_dims", (128, 64)))
    dropout = float(cfg.get("dropout", 0.1))
    epochs = int(cfg.get("epochs", 80))
    batch_size = int(cfg.get("batch_size", 512))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    patience = int(cfg.get("patience", 10))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    standardize_features = bool(cfg.get("standardize_features", True))
    seed = int(cfg.get("seed", SEED))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb).astype(np.float32)
    if pca_dim > 0:
        n_components = min(int(pca_dim), emb_s.shape[1] - 1)
        pca = PCA(n_components=n_components, random_state=seed)
        z = pca.fit_transform(emb_s).astype(np.float32)
        print(
            f"Embedding: {emb.shape} → PCA: {z.shape}  "
            f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})"
        )
    else:
        pca = None
        z = emb_s

    class_weights = build_class_freq_weights(Y, cap=10.0)
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"Training torch probes for {len(active)} species (>= {min_pos} pos windows)...")

    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    probe_models = {}

    for ci in tqdm(active, desc="Torch probes"):
        yc = Y[:, ci].astype(np.float32)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        x = np.hstack([
            z,
            scores_raw[:, ci : ci + 1],
            prev[:, None], next_[:, None], mean[:, None], max_[:, None], std[:, None],
        ]).astype(np.float32)

        n_pos = int(yc.sum())
        n_neg = len(yc) - n_pos
        pos_idx = np.where(yc == 1)[0]
        w = float(class_weights[ci])
        repeat = min(max(1, int(round(w * n_neg / max(n_pos, 1)))), 8)
        if n_pos * repeat + len(yc) > max_rows:
            repeat = max(1, (max_rows - len(yc)) // max(n_pos, 1))

        x_bal = np.vstack([x, np.tile(x[pos_idx], (repeat, 1))]).astype(np.float32)
        y_bal = np.concatenate([yc, np.ones(n_pos * repeat, dtype=np.float32)])

        n = len(y_bal)
        n_val = max(1, int(round(n * val_fraction)))
        perm = rng.permutation(n)
        va_idx, tr_idx = perm[:n_val], perm[n_val:]

        x_tr, y_tr = x_bal[tr_idx], y_bal[tr_idx]
        x_va, y_va = x_bal[va_idx], y_bal[va_idx]

        if standardize_features:
            mu = x_tr.mean(axis=0, keepdims=True)
            sd = x_tr.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            x_tr = (x_tr - mu) / sd
            x_va = (x_va - mu) / sd
        else:
            mu = np.zeros((1, x_bal.shape[1]), dtype=np.float32)
            sd = np.ones((1, x_bal.shape[1]), dtype=np.float32)

        model = TorchProbeMLP(in_dim=x_bal.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        x_va_t = torch.tensor(x_va, dtype=torch.float32, device=device)
        y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

        best_val, best_state, wait = float("inf"), None, 0
        for _ep in range(epochs):
            model.train()
            order = torch.randperm(x_tr_t.shape[0], device=device)
            for start in range(0, x_tr_t.shape[0], batch_size):
                idx = order[start : start + batch_size]
                loss = criterion(model(x_tr_t[idx]), y_tr_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_va_t), y_va_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        probe_models[int(ci)] = {"model": model.cpu(), "mu": mu, "sd": sd}

    print(f"Trained {len(probe_models)} torch probes")
    return probe_models, scaler, pca


def predict_torch_probe_logits(emb_test, scores_test, probe_models, scaler, pca):
    emb_s = scaler.transform(emb_test).astype(np.float32)
    z_test = pca.transform(emb_s).astype(np.float32) if pca is not None else emb_s
    probe_logits = scores_test.copy()
    for ci, payload in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        x_test = np.hstack([
            z_test,
            scores_test[:, ci : ci + 1],
            prev[:, None], next_[:, None], mean[:, None], max_[:, None], std[:, None],
        ]).astype(np.float32)
        x_test = (x_test - payload["mu"]) / payload["sd"]
        with torch.no_grad():
            logits = payload["model"](torch.tensor(x_test, dtype=torch.float32)).numpy().astype(np.float32)
        probe_logits[:, ci] = logits
    return probe_logits


def blend_probe_logits(base_scores, probe_logits, alpha_blend):
    return (1.0 - alpha_blend) * base_scores + alpha_blend * probe_logits


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

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

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
    rng.manual_seed(SEED)
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
        _mlp_cfg = CFG.get("mlp_params", {})
        probe_models, emb_scaler, emb_pca = train_torch_probes(
            emb_tr_f, sc_tr_f, Y_tr_f, _mlp_cfg,
        )
        probe_logits_va = predict_torch_probe_logits(
            emb_va_f, sc_va_f, probe_models, emb_scaler, emb_pca,
        )
        sc_va_mlp = blend_probe_logits(
            sc_va_f, probe_logits_va, float(_mlp_cfg.get("alpha_blend", 0.15)),
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
    mlp_cfg = CFG.get("mlp_params", {})
    mlp_alpha_blend = float(mlp_cfg.get("alpha_blend", 0.40))
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
        probe_models, emb_scaler, emb_pca = train_torch_probes(
            emb_tr_f, sc_tr_f, Y_tr_f, mlp_cfg,
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
        sc_va_mlp = blend_probe_logits(
            sc_va_prior,
            predict_torch_probe_logits(emb_va_f, sc_va_prior, probe_models, emb_scaler, emb_pca),
            mlp_alpha_blend,
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
        sc_tr_mlp = blend_probe_logits(
            sc_tr_prior,
            predict_torch_probe_logits(emb_tr_f, sc_tr_prior, probe_models, emb_scaler, emb_pca),
            mlp_alpha_blend,
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


def run_hp_search_oof(emb_full, sc_full, Y_full, meta_full):
    """
    Trains LightProtoSSM and torch probes once per fold, then sweeps all
    combinations in LAMBDA_PRIOR_GRID × MLP_ALPHA_BLEND_GRID × ENSEMBLE_W_GRID
    (~1000 combos) using cached per-fold outputs.  Results saved to
    outputs/hp_search/.
    """
    n_splits = 5
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
    mlp_cfg = CFG.get("mlp_params", {})

    file_meta = meta_full.drop_duplicates("filename").reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)

    # ── Phase 1: train SSM + probes once per fold ──────────────────────
    print("Phase 1: training SSM and probes (once per fold)...")
    fold_data = []
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

        print(f"\n[Fold {fold}/{n_splits}] Training SSM...")
        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f, sc_tr_f, Y_tr_f, meta_tr_f,
            emb_val=emb_va_f, scores_val=sc_va_f, y_val=Y_va_f, meta_val=meta_va_f,
            n_epochs=proto_n_epochs, patience=proto_patience, lr=proto_lr,
            n_sites=proto_n_sites, d_model=proto_d_model, d_state=proto_d_state,
            n_ssm_layers=proto_n_ssm_layers, dropout=proto_dropout,
            meta_dim=proto_meta_dim, distill_weight=proto_distill_weight,
            pos_weight_cap=proto_pos_weight_cap,
            use_swa=proto_use_swa, swa_start_frac=proto_swa_start_frac,
            swa_lr=proto_swa_lr, use_cross_attn=proto_use_cross_attn,
            cross_attn_heads=proto_cross_attn_heads,
            label_smoothing=proto_label_smoothing, mixup_alpha=proto_mixup_alpha,
            focal_gamma=proto_focal_gamma, use_cosine_restart=proto_use_cosine_restart,
            restart_period=proto_restart_period, verbose=False,
        )

        n_va = len(emb_va_f) // N_WINDOWS
        va_fn_list = meta_va_f.drop_duplicates("filename")["filename"].tolist()
        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        str(meta_va_f.loc[meta_va_f["filename"] == fn, "site"].iloc[0]), 0
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
                    torch.tensor(emb_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                    torch.tensor(sc_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                    site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                    hours=torch.tensor(va_hour_ids, dtype=torch.long),
                )
                .numpy()
                .reshape(-1, N_CLASSES)
            )

        print(f"[Fold {fold}/{n_splits}] Training probes...")
        probe_models, emb_scaler, emb_pca = train_torch_probes(
            emb_tr_f, sc_tr_f, Y_tr_f, mlp_cfg,
        )

        prior_tables_fold = build_prior_tables(meta_tr_f[["site", "hour_utc"]], Y_tr_f)

        fold_data.append({
            "va_mask": va_mask,
            "sc_va_f": sc_va_f,
            "emb_va_f": emb_va_f,
            "proto_va": proto_va,
            "probe_models": probe_models,
            "emb_scaler": emb_scaler,
            "emb_pca": emb_pca,
            "prior_tables_fold": prior_tables_fold,
            "meta_va_f": meta_va_f,
        })
        print(f"[Fold {fold}/{n_splits}] Done — val files={len(va_fnames)}")

    # ── Phase 2: precompute per-lambda_prior caches ────────────────────
    # probe logits depend on sc_va_prior (a function of lambda_prior),
    # so we precompute once per lambda value to avoid redundant MLP passes
    print(f"\nPhase 2: precomputing probe logits for {len(LAMBDA_PRIOR_GRID)} lambda values...")
    per_lp_prior = {}   # lp -> list[sc_va_prior] over folds
    per_lp_probe = {}   # lp -> list[probe_logits_va] over folds
    for lp in LAMBDA_PRIOR_GRID:
        prior_list, probe_list = [], []
        for fd in fold_data:
            sc_va_prior = apply_prior(
                fd["sc_va_f"],
                sites=fd["meta_va_f"]["site"].to_numpy(),
                hours=fd["meta_va_f"]["hour_utc"].to_numpy(),
                tables=fd["prior_tables_fold"],
                lambda_prior=lp,
            )
            probe_logits_va = predict_torch_probe_logits(
                fd["emb_va_f"], sc_va_prior,
                fd["probe_models"], fd["emb_scaler"], fd["emb_pca"],
            )
            prior_list.append(sc_va_prior)
            probe_list.append(probe_logits_va)
        per_lp_prior[lp] = prior_list
        per_lp_probe[lp] = probe_list

    # ── Precompute stage metrics that don't depend on all 3 params ────
    # proto_only: constant across all combos
    oof_proto = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
    for fd in fold_data:
        oof_proto[fd["va_mask"]] = sigmoid(fd["proto_va"])
    proto_only_auc = round(float(macro_auc(Y_full, oof_proto)), 6)
    print(f"\nProto-only OOF AUC: {proto_only_auc:.6f}")

    # prior_only: depends on lambda_prior only (10 values)
    prior_only_by_lp = {}
    for lp in LAMBDA_PRIOR_GRID:
        oof_prior = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
        for fi, fd in enumerate(fold_data):
            oof_prior[fd["va_mask"]] = sigmoid(per_lp_prior[lp][fi])
        prior_only_by_lp[lp] = round(float(macro_auc(Y_full, oof_prior)), 6)

    # prior_mlp: depends on lambda_prior × mlp_alpha_blend (100 values)
    prior_mlp_by_lp_ab = {}
    for lp in LAMBDA_PRIOR_GRID:
        for ab in MLP_ALPHA_BLEND_GRID:
            oof_mlp = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
            for fi, fd in enumerate(fold_data):
                sc_va_mlp = blend_probe_logits(
                    per_lp_prior[lp][fi], per_lp_probe[lp][fi], ab
                )
                oof_mlp[fd["va_mask"]] = sigmoid(sc_va_mlp)
            prior_mlp_by_lp_ab[(lp, ab)] = round(float(macro_auc(Y_full, oof_mlp)), 6)

    # ── Phase 3: sweep all 10×10×10 combinations ──────────────────────
    total_combos = len(LAMBDA_PRIOR_GRID) * len(MLP_ALPHA_BLEND_GRID) * len(ENSEMBLE_W_GRID)
    print(f"\nPhase 3: sweeping {total_combos} combinations...")

    all_combos = [
        (lp, ab, ew)
        for lp in LAMBDA_PRIOR_GRID
        for ab in MLP_ALPHA_BLEND_GRID
        for ew in ENSEMBLE_W_GRID
    ]
    sweep_results = []
    for lp, ab, ew in tqdm(all_combos, total=total_combos, desc="HP sweep"):
        oof_first_pass = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)
        for fi, fd in enumerate(fold_data):
            sc_va_mlp = blend_probe_logits(per_lp_prior[lp][fi], per_lp_probe[lp][fi], ab)
            first_pass_va = ew * fd["proto_va"] + (1.0 - ew) * sc_va_mlp
            oof_first_pass[fd["va_mask"]] = sigmoid(first_pass_va)
        auc = round(float(macro_auc(Y_full, oof_first_pass)), 6)
        sweep_results.append({
            "lambda_prior": float(lp),
            "mlp_alpha_blend": float(ab),
            "ensemble_w": float(ew),
            "metric_proto_only": proto_only_auc,
            "metric_prior_only": prior_only_by_lp[lp],
            "metric_prior_mlp": prior_mlp_by_lp_ab[(lp, ab)],
            "metric_first_pass": auc,
        })

    sweep_results.sort(key=lambda r: r["metric_first_pass"], reverse=True)

    sweep_dir = BASE / "outputs" / "hp_search2"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_json = sweep_dir / "hp_search_results.json"
    out_csv = sweep_dir / "hp_search_results.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    pd.DataFrame(sweep_results).to_csv(out_csv, index=False, float_format="%.6f")

    print(f"\nSaved {len(sweep_results)} results → {out_json}")
    print("\nTop 10 combinations (by first_pass AUC):")
    for r in sweep_results[:10]:
        print(
            f"  lp={r['lambda_prior']:.2f}  ab={r['mlp_alpha_blend']:.2f}"
            f"  ew={r['ensemble_w']:.2f}"
            f"  proto={r['metric_proto_only']:.6f}"
            f"  prior={r['metric_prior_only']:.6f}"
            f"  mlp={r['metric_prior_mlp']:.6f}"
            f"  first_pass={r['metric_first_pass']:.6f}"
        )
    return sweep_results


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
    "ensemble_w": 0.85,
    "lambda_prior": 0.0,
    "rank_power": 0.40,
    "smooth_alpha": 0.20,
    "tta_shifts": [0, 1, -1, 2, -2],
    "threshold_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],

    # MLP probes (torch)
    "mlp_params": {
        "min_pos": 3,
        "pca_dim": 0,
        "alpha_blend": 1,
        "hidden_dims": (128, 64),
        "dropout": 0.1,
        "epochs": 80,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 10,
        "val_fraction": 0.15,
        "standardize_features": True,
        "max_rows": 3000,
        "seed": 1337,
    },

    # proto_ssm config: ultra_attn_d448_s40_noswa_fastlr
    "proto_ssm_train": {
        "n_epochs": 90,
        "patience": 18,
        "lr": 3.5e-4,
        "n_sites": 28,
        "d_model": 640,
        "d_state": 64,
        "n_ssm_layers": 5,
        "dropout": 0.32,
        "meta_dim": 72,
        "distill_weight": 0.30,
        "pos_weight_cap": 35.0,
        "use_swa": True,
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

# Hyperparameter grids swept by run_hp_search_oof (~10×10×10 = 1000 combos)
LAMBDA_PRIOR_GRID    = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
MLP_ALPHA_BLEND_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
ENSEMBLE_W_GRID      = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# 4) Execution
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
tf.random.set_seed(SEED)

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
    f"mlp_epochs={CFG['mlp_params']['epochs']}"
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
    hp_results = run_hp_search_oof(emb_tr, sc_tr, Y_FULL_aligned, meta_tr)
else:
    print("Submit mode: skipping HP search")

print(f"Total wall time: {(time.time() - _WALL_START)/60:.1f} min")
