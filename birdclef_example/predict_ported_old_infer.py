#!/usr/bin/env python3

# 1. Imports
from pathlib import Path
import os
import subprocess, sys, os
import random
import tensorflow as tf
import torch
import numpy as np
import gc
import json
import re
import time
import warnings
from safetensors.torch import load_file
from collections import defaultdict
import pandas as pd
import soundfile as sf
import onnxruntime as ort
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.isotonic import IsotonicRegression
import concurrent.futures
from sklearn.model_selection import StratifiedGroupKFold

# 2. Function and Class Definitions
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)

def get_cosine_restart_scheduler(optimizer, restart_period=20):
    return CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_period, T_mult=1, eta_min=1e-5
    )


def restore_cfg_paths(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = restore_cfg_paths(v)
            elif isinstance(v, list):
                out[k] = [restore_cfg_paths(x) for x in v]
            elif isinstance(v, str) and (k.endswith("_dir") or k.endswith("_path")):
                out[k] = Path(v)
            else:
                out[k] = v
        return out
    return obj

def mixup_cutmix(emb, logits, labels, alpha=0.4, cutmix_prob=0.3):
    B, T, D = emb.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B)

    if np.random.rand() < cutmix_prob:

        cut_len = max(1, int(T * (1 - lam)))
        cut_start = np.random.randint(0, T - cut_len + 1)
        new_emb = emb.clone()
        new_emb[:, cut_start:cut_start+cut_len, :] = emb[idx, cut_start:cut_start+cut_len, :]
        new_logits = logits.clone()
        new_logits[:, cut_start:cut_start+cut_len, :] = logits[idx, cut_start:cut_start+cut_len, :]
        lam_actual = 1.0 - cut_len / T
        new_labels = lam_actual * labels + (1-lam_actual) * labels[idx]
    else:

        new_emb    = lam * emb    + (1-lam) * emb[idx]
        new_logits = lam * logits + (1-lam) * logits[idx]
        new_labels = lam * labels + (1-lam) * labels[idx]

    return new_emb, new_logits, new_labels

def build_class_freq_weights(Y_FULL, cap=10.0):
    pos_count = Y_FULL.sum(axis=0).astype(np.float32) + 1.0
    total     = Y_FULL.shape[0]
    freq      = pos_count / total
    weights   = 1.0 / (freq ** 0.5)
    weights   = np.clip(weights, 1.0, cap)
    weights   = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def species_focal_loss(logits, targets, class_weights,
                       gamma=2.5, label_smoothing=0.03):
    targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2.0
    bce    = F.binary_cross_entropy_with_logits(
                 logits, targets_smooth, reduction="none")
    pt     = torch.exp(-bce)
    focal  = ((1 - pt) ** gamma) * bce
    w      = class_weights.to(logits.device).unsqueeze(0)
    return (focal * w).mean()

def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL,
                                       threshold_grid, n_windows=12):
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files  = n_samples // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y   = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)

    for c in range(n_cls):
        y_true, y_prob = file_y[:, c], file_oof[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except:
            y_cal = y_prob

        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred==1)&(y_true==1)).sum()
            fp = ((pred==1)&(y_true==0)).sum()
            fn = ((pred==0)&(y_true==1)).sum()
            prec = tp/(tp+fp+1e-8)
            rec  = tp/(tp+fn+1e-8)
            f1   = 2*prec*rec/(prec+rec+1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t

    print(f"Mean threshold: {thresholds.mean():.3f}")
    print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds

def sweep_ensemble_weight(oof_proto, oof_mlp, Y_FULL,
                          n_windows=12,
                          candidates=np.arange(0.3, 0.8, 0.05)):
    n_files = oof_proto.shape[0] // n_windows
    file_y  = Y_FULL.reshape(n_files, n_windows, -1).max(axis=1)
    best_auc, best_w = 0.0, 0.6

    for w in candidates:
        blended   = w * oof_proto + (1-w) * oof_mlp
        file_pred = blended.reshape(n_files, n_windows, -1).max(axis=1)
        try:
            auc = macro_auc_skip_empty(file_y, file_pred)
        except:
            continue
        if auc > best_auc:
            best_auc, best_w = auc, w

    print(f"Best ensemble weight (proto): {best_w:.2f}")
    print(f"Best AUC: {best_auc:.5f}")
    return best_w

def get_genus_hits(scientific_name):
    genus = str(scientific_name).split()[0]
    labels = globals().get("bc_labels")
    if labels is None or "scientific_name" not in labels.columns:
        return genus, pd.DataFrame(columns=["scientific_name"])
    hits = labels[
        labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
    ].copy()
    return genus, hits

def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def smooth_cols_fixed12(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()

    s = scores.copy()
    assert len(s) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    view = s.reshape(-1, N_WINDOWS, s.shape[1])

    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)

    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s

def smooth_events_fixed12(scores, cols, alpha=0.15):
    """Soft max-pool context for event birds (Aves).
    Uses local_max instead of average neighbor, preserving transient call detection."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s

def seq_features_1d(v):
    """
    v: shape (n_rows,), ordered as full-file blocks of 12 windows
    Extended: tambah std_v untuk capture variance temporal dalam file
    """
    assert len(v) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    x = v.reshape(-1, N_WINDOWS)

    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
    std_v  = np.repeat(x.std(axis=1),  N_WINDOWS)

    return prev_v, next_v, mean_v, max_v, std_v

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, reduction="mean"):
    """Focal loss for multi-label classification.
    Reduces contribution of easy examples, focuses on hard ones."""
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    return loss

def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    """Rank 1/2 technique: scale each window's predictions by the file's top-K mean confidence."""
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)

def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1], max_batch_size=512):
    """
    TTA by circular-shifting the 12-window embedding sequence.
    Batched and optimized for faster single-pass inference.
    """
    n_files = emb_files.shape[0]
    n_shifts = len(shifts)

    if n_shifts == 0:
        return np.zeros((n_files, emb_files.shape[1], logits_files.shape[2]), dtype=np.float32)

    e_list, l_list = [], []
    for shift in shifts:
        if shift == 0:
            e_list.append(emb_files)
            l_list.append(logits_files)
        else:
            e_list.append(np.roll(emb_files, shift, axis=1))
            l_list.append(np.roll(logits_files, shift, axis=1))

    e_batch = np.concatenate(e_list, axis=0)
    l_batch = np.concatenate(l_list, axis=0)

    site_batch = np.tile(site_ids, n_shifts)
    hour_batch = np.tile(hours, n_shifts)

    model.eval()
    pred_batch_list = []

    with torch.no_grad():
        total_samples = e_batch.shape[0]
        for start_idx in range(0, total_samples, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_samples)

            out, _, _ = model(
                torch.tensor(e_batch[start_idx:end_idx], dtype=torch.float32),
                torch.tensor(l_batch[start_idx:end_idx], dtype=torch.float32),
                site_ids=torch.tensor(site_batch[start_idx:end_idx], dtype=torch.long),
                hours=torch.tensor(hour_batch[start_idx:end_idx], dtype=torch.long),
            )
            pred_batch_list.append(out.numpy())

    pred_batch = np.concatenate(pred_batch_list, axis=0)
    pred_batch = pred_batch.reshape(n_shifts, n_files, pred_batch.shape[1], pred_batch.shape[2])

    all_preds = []
    for i, shift in enumerate(shifts):
        pred_i = pred_batch[i]
        if shift != 0:
            pred_i = np.roll(pred_i, -shift, axis=1)
        all_preds.append(pred_i)
    return np.mean(all_preds, axis=0)

def rank_aware_scaling(scores, n_windows=12, power=0.5):
    """V17: 2025 Rank 3 technique. Scale each window by (file_max)^power.
    Suppresses predictions in uncertain files, boosts confident files."""
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows

    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)

    scale = np.power(file_max, power)

    scaled = view * scale
    return scaled.reshape(N, C)

def delta_shift_smooth(scores, n_windows=12, alpha=0.15):
    """V17: 2025 Rank 1 technique. Temporal smoothing across windows.
    new[t] = (1-alpha)*old[t] + 0.5*alpha*(old[t-1] + old[t+1])"""
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows

    view = scores.reshape(n_files, n_windows, C)

    prev_view = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_view = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)

    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_view + next_view)

    return smoothed.reshape(N, C)

def optimize_per_class_thresholds(oof_scores, y_true, n_windows=12, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """V17: Find optimal decision threshold per class from OOF predictions.
    Optimizes F1-like metric (precision-recall balance) for each species."""
    n_classes = oof_scores.shape[1]
    best_thresholds = np.zeros(n_classes)
    best_scores = np.zeros(n_classes)

    for c in range(n_classes):
        y_c = y_true[:, c]
        scores_c = oof_scores[:, c]

        if y_c.sum() == 0:
            best_thresholds[c] = 0.5
            continue

        best_f1 = 0
        best_t = 0.5

        for t in thresholds:
            pred_c = (scores_c > t).astype(int)
            tp = ((pred_c == 1) & (y_c == 1)).sum()
            fp = ((pred_c == 1) & (y_c == 0)).sum()
            fn = ((pred_c == 0) & (y_c == 1)).sum()

            if tp + fp == 0 or tp + fn == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[c] = best_t
        best_scores[c] = best_f1

    return best_thresholds, best_scores

def apply_per_class_thresholds(scores, thresholds, n_windows=12):
    """V17: Apply per-class thresholds to convert scores to binary predictions."""
    N, C = scores.shape
    assert C == len(thresholds)

    scaled = np.copy(scores)

    for c in range(C):
        t = thresholds[c]

        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)

    return np.clip(scaled, 0, 1)

def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y

def infer_perch_with_embeddings(paths, batch_files=16, verbose=True, proxy_reduce="max"):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:

        next_paths = paths[0:batch_files]
        future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

        for start in iterator:
            batch_paths = next_paths
            batch_n = len(batch_paths)

            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < n_files:
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            batch_row_start = write_row
            x_pos = 0

            for i, path in enumerate(batch_paths):
                y = batch_audio[i]
                x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

                meta = parse_soundscape_filename(path.name)
                stem = path.stem

                row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[write_row:write_row + N_WINDOWS] = path.name
                sites[write_row:write_row + N_WINDOWS] = meta["site"]
                hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

                x_pos += N_WINDOWS
                write_row += N_WINDOWS

            if USE_ONNX_PERCH:
                onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = onnx_outs[ONNX_OUTPUT_MAP["label"]].astype(np.float32, copy=False)
                emb = onnx_outs[ONNX_OUTPUT_MAP["embedding"]].astype(np.float32, copy=False)
            else:
                outputs = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = outputs["label"].numpy().astype(np.float32, copy=False)
                emb = outputs["embedding"].numpy().astype(np.float32, copy=False)

            scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
            embeddings[batch_row_start:write_row] = emb

            for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
                sub = logits[:, bc_idx_arr]
                if proxy_reduce == "max":
                    proxy_score = sub.max(axis=1)
                elif proxy_reduce == "mean":
                    proxy_score = sub.mean(axis=1)
                elif proxy_reduce == "median":
                    proxy_score = np.median(sub, axis=1)
                else:
                    raise ValueError("proxy_reduce must be 'max' or 'mean' or 'median'")
                scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

            if USE_ONNX_PERCH:
                del x, onnx_outs, logits, emb, batch_audio
            else:
                del x, outputs, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings

def resolve_full_cache_paths():
    candidates = []

    candidates.append((
        BUNDLE_DIR / "full_perch_meta.parquet",
        BUNDLE_DIR / "full_perch_arrays.npz"
    ))

    for meta_path, npz_path in candidates:
        if meta_path.exists() and npz_path.exists():
            return meta_path, npz_path

    return None, None

def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)

    global_p = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)

    for s in site_keys:
        i = site_to_i[s]
        mask = prior_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_prior[mask].mean(axis=0)

    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)

    for h in hour_keys:
        i = hour_to_i[h]
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_prior[mask].mean(axis=0)

    sh_to_i = {}
    sh_n_list = []
    sh_p_list = []

    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))

    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if len(sh_p_list) else np.zeros((0, Y_prior.shape[1]), dtype=np.float32)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_n": site_n,
        "site_p": site_p,
        "hour_to_i": hour_to_i,
        "hour_n": hour_n,
        "hour_p": hour_p,
        "sh_to_i": sh_to_i,
        "sh_n": sh_n,
        "sh_p": sh_p,
    }

def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter(
        (tables["site_to_i"].get(str(s), -1) for s in sites),
        dtype=np.int32,
        count=n
    )
    hour_idx = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        dtype=np.int32,
        count=n
    )
    sh_idx = np.fromiter(
        (tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)),
        dtype=np.int32,
        count=n
    )

    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        wh = nh / (nh + 8.0)
        p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]

    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        ws = ns / (ns + 8.0)
        p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]

    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        wsh = nsh / (nsh + 4.0)
        p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)

def fuse_scores_with_tables(base_scores, sites, hours, tables,
                            lambda_event=None,
                            lambda_texture=None,
                            lambda_proxy_texture=None,
                            smooth_texture=None,
                            smooth_event=None):
    if lambda_event is None:
        lambda_event = BEST["lambda_event"]
    if lambda_texture is None:
        lambda_texture = BEST["lambda_texture"]
    if lambda_proxy_texture is None:
        lambda_proxy_texture = BEST["lambda_proxy_texture"]
    if smooth_texture is None:
        smooth_texture = BEST["smooth_texture"]
    if smooth_event is None:
        smooth_event = BEST["smooth_event"]

    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]

    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]

    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]

    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]

    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]

    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior

def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    groups_full = meta_full["filename"].to_numpy()

    row_id_to_idx = {r: i for i, r in enumerate(sc_clean["row_id"])}
    aligned_indices = [row_id_to_idx[r] for r in meta_full["row_id"]]
    Y_ALIGNED = Y_SC[aligned_indices]

    y_strat = np.argmax(Y_ALIGNED, axis=1)
    unique_classes, counts = np.unique(y_strat, return_counts=True)
    rare_classes = unique_classes[counts < n_splits]
    y_strat[np.isin(y_strat, rare_classes)] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)

    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)

    splits = list(sgkf.split(scores_full_raw, y_strat, groups=groups_full))
    iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)

    for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())

        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]

        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx],
            sites=meta_full.iloc[va_idx]["site"].to_numpy(),
            hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )

        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold

    assert (fold_id >= 0).all()
    return oof_base, oof_prior, fold_id

def build_all_class_features_vectorized(Z, raw_scores, prior_scores, base_scores, valid_classes, n_windows=12):
    """
    A function that constructs all 14 types of features for all classes in one go, without using a for loop.
    Output tensor shape: (V: number of effective classes, N: number of samples, D+14)
    """
    N, D = Z.shape
    V = len(valid_classes)

    raw = raw_scores[:, valid_classes].T
    prior = prior_scores[:, valid_classes].T
    base = base_scores[:, valid_classes].T

    n_files = N // n_windows
    base_view = base.reshape(V, n_files, n_windows)

    prev_base = np.concatenate([base_view[:, :, :1], base_view[:, :, :-1]], axis=2).reshape(V, N)
    next_base = np.concatenate([base_view[:, :, 1:], base_view[:, :, -1:]], axis=2).reshape(V, N)
    mean_base = np.repeat(base_view.mean(axis=2), n_windows, axis=1)
    max_base = np.repeat(base_view.max(axis=2), n_windows, axis=1)
    std_base = np.repeat(base_view.std(axis=2), n_windows, axis=1)

    diff_mean = base - mean_base
    diff_prev = base - prev_base
    diff_next = base - next_base

    interact_rp = raw * prior
    interact_rb = raw * base
    interact_pb = prior * base

    scalar_feats = np.stack([
        raw, prior, base, prev_base, next_base,
        mean_base, max_base, std_base,
        diff_mean, diff_prev, diff_next,
        interact_rp, interact_rb, interact_pb
    ], axis=-1)

    Z_expanded = np.broadcast_to(Z, (V, N, D))

    X_all = np.concatenate([Z_expanded, scalar_feats], axis=-1)
    return X_all.astype(np.float32)

class VectorizedMLPProbes(nn.Module):
    """
    A class that combines multiple scikit-learn MLPClassifier classes into a single PyTorch model.
    """
    def __init__(self, probe_models, device="cpu"):
        super().__init__()
        self.valid_classes = sorted(list(probe_models.keys()))
        self.V = len(self.valid_classes)

        if self.V == 0:
            return

        sample_clf = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample_clf.coefs_)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for layer_idx in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[layer_idx] for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[layer_idx] for c in self.valid_classes], axis=0)

            self.weights.append(nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))

        self.to(device)

    def forward(self, x):

        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)

        return h.squeeze(-1)

def get_vectorized_mlp_scores(Z, raw, prior, base, probe_models, alpha_p, n_windows=12, device="cpu"):
    """
    A wrapper function that wraps all of the above vectorization processes
    """
    mlp_scores = base.copy()
    if len(probe_models) == 0:
        return mlp_scores

    valid_classes = sorted(list(probe_models.keys()))

    X_all = build_all_class_features_vectorized(Z, raw, prior, base, valid_classes, n_windows)

    vec_probe = VectorizedMLPProbes(probe_models, device=device)
    vec_probe.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_all, dtype=torch.float32, device=device)
        preds = vec_probe(X_tensor).cpu().numpy()

    preds_t = preds.T
    base_valid = base[:, valid_classes]

    mlp_scores[:, valid_classes] = (1.0 - alpha_p) * base_valid + alpha_p * preds_t
    return mlp_scores

class LoadedMLPProbe:
    def __init__(self, coefs_, intercepts_):
        self.coefs_ = [np.asarray(coef, dtype=np.float32) for coef in coefs_]
        self.intercepts_ = [np.asarray(intercept, dtype=np.float32) for intercept in intercepts_]


class LoadedStandardScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean_) / self.scale_).astype(np.float32)


class LoadedPCA:
    def __init__(self, mean, components, explained_variance):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.components_ = np.asarray(components, dtype=np.float32)
        self.explained_variance_ = np.asarray(explained_variance, dtype=np.float32)
        total = float(self.explained_variance_.sum())
        self.explained_variance_ratio_ = self.explained_variance_ / max(total, 1e-12)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)


def load_final_prior_tables(bundle_meta):
    tables = bundle_meta["final_prior_tables"]
    sh_to_i = {}
    for key, value in tables["sh_to_i"].items():
        site, hour = key.split("||", 1)
        sh_to_i[(site, int(hour))] = int(value)
    return {
        "global_p": np.asarray(tables["global_p"], dtype=np.float32),
        "site_to_i": {str(k): int(v) for k, v in tables["site_to_i"].items()},
        "site_n": np.asarray(tables["site_n"], dtype=np.float32),
        "site_p": np.asarray(tables["site_p"], dtype=np.float32),
        "hour_to_i": {int(k): int(v) for k, v in tables["hour_to_i"].items()},
        "hour_n": np.asarray(tables["hour_n"], dtype=np.float32),
        "hour_p": np.asarray(tables["hour_p"], dtype=np.float32),
        "sh_to_i": sh_to_i,
        "sh_n": np.asarray(tables["sh_n"], dtype=np.float32),
        "sh_p": np.asarray(tables["sh_p"], dtype=np.float32),
    }


def load_probe_models(bundle_meta, bundle_arrays):
    probe_models = {}
    for cls_idx in bundle_meta.get("probe_class_indices", []):
        coefs_ = []
        intercepts_ = []
        layer_idx = 0
        while True:
            coef_key = f"probe_{int(cls_idx)}_coef_{layer_idx}"
            inter_key = f"probe_{int(cls_idx)}_intercept_{layer_idx}"
            if coef_key not in bundle_arrays or inter_key not in bundle_arrays:
                break
            coefs_.append(bundle_arrays[coef_key])
            intercepts_.append(bundle_arrays[inter_key])
            layer_idx += 1
        if coefs_:
            probe_models[int(cls_idx)] = LoadedMLPProbe(coefs_, intercepts_)
    return probe_models

class SelectiveSSM(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]

class TemporalCrossAttention(nn.Module):
    """Multi-head cross-attention between temporal windows.
    Captures non-local patterns (e.g., dawn chorus onset, counter-singing)
    that sequential SSM may miss."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x

class ProtoSSMv2(nn.Module):

    def __init__(self, d_input=1536, d_model=192, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12,
                 dropout=0.2, n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

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

        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    else:
        group_map = {label: "Unknown" for label in primary_labels}

    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = []
    for label in primary_labels:
        grp = group_map.get(label, "Unknown")
        class_to_group.append(grp_to_idx.get(grp, 0))
    return len(groups), class_to_group, grp_to_idx

def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    n_sites = len(sites) + 1
    return site_to_idx, n_sites

def reshape_to_files(flat_array, meta_df, n_windows=None):
    if n_windows is None:
        n_windows = N_WINDOWS

    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows,\
        f"Expected {n_files * n_windows} rows, got {len(flat_array)}"

    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files

def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites = meta_df["site"].to_numpy()
    hours = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            sid = site_to_idx.get(sites[row], 0)
            site_ids[fi] = min(sid, n_sites_max - 1)
            hour_ids[fi] = int(hours[row]) % 24
    return site_ids, hour_ids

class ResidualSSM(nn.Module):

    def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                 n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.d_model = d_model
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

        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass_scores, site_ids=None, hours=None):

        B, T, _ = emb.shape

        x = torch.cat([emb, first_pass_scores], dim=-1)
        h = self.input_proj(x)

        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            hour_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
            h = h + meta.unsqueeze(1)

        h = h + self.pos_enc[:, :T, :]

        residual = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
        h = self.ssm_drop(h)
        h = self.ssm_norm(h + residual)

        correction = self.output_head(h)
        return correction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def adaptive_delta_smooth(probs, n_windows, base_alpha=0.20):
    n_files = probs.shape[0] // n_windows
    result = probs.copy()
    view = result.reshape(n_files, n_windows, -1)
    p_view = probs.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = p_view[:, i, :].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p_view[:, i-1, :] + p_view[:, i+1, :]) / 2.0
        view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
    return result.reshape(probs.shape)

# 3. Config Variables
REPO_ROOT = Path(__file__).resolve().parent.parent
MODE = "submit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_WALL_START = time.time()
BASE = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models" / "perch_v2_cpu" / "1"
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
DEVICE = torch.device("cpu")
LOGS = {}
CFG = {}
ONNX_PERCH_PATH = Path(os.environ.get("PERCH_ONNX_PATH", REPO_ROOT / "models" / "perch_onnx" / "perch_v2_finetuned.onnx"))
USE_ONNX_PERCH = ONNX_PERCH_PATH.exists()
NO_LABEL_INDEX = 0
MANUAL_SCIENTIFIC_NAME_MAP = {}
MAPPED_MASK = np.array([], dtype=bool)
MAPPED_POS = np.array([], dtype=np.int32)
UNMAPPED_POS = np.array([], dtype=np.int32)
MAPPED_BC_INDICES = np.array([], dtype=np.int32)
CLASS_NAME_MAP = {}
TEXTURE_TAXA = {"Amphibia", "Insecta"}
ACTIVE_CLASSES = []
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
SELECTED_PROXY_TARGETS = []
BUNDLE_DIR = Path(os.environ.get("PORTED_BUNDLE_DIR", REPO_ROOT / "outputs" / "ported_bundle_old"))
PROXY_REDUCE_CACHE = BUNDLE_DIR / "proxy_reduce_grid.json"
ProtoSSM_PATH = None
ProtoSSM_JSON = None
grid_results = None
BEST_PROBE = None
ENSEMBLE_WEIGHT_PROTO = 0.5
ResidualSSM_PATH = None
res_model = None
CORRECTION_WEIGHT = 0.0
PER_CLASS_THRESHOLDS = np.array([], dtype=np.float32)
T_AVES = 1.0
T_TEXTURE = 1.0

# 4. Execution
BEST = {}
PROXY_REDUCE_CACHE = BUNDLE_DIR / "proxy_reduce_grid.json"
LOGS = {}
_WALL_START = time.time()
seed_everything(1891)
assert MODE == "submit"
print("MODE =", MODE)
warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()
print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:", MODEL_DIR.exists())

if MODE == "submit":
    bundle_meta_path = BUNDLE_DIR / "bundle_meta.json"
    bundle_arrays_path = BUNDLE_DIR / "bundle_arrays.npz"
    proto_path = BUNDLE_DIR / "proto_ssm.safetensors"
    res_path = BUNDLE_DIR / "residual_ssm.safetensors"

    if not bundle_meta_path.exists():
        raise FileNotFoundError(f"Bundle metadata not found: {bundle_meta_path}")
    if not bundle_arrays_path.exists():
        raise FileNotFoundError(f"Bundle arrays not found: {bundle_arrays_path}")
    if not proto_path.exists():
        raise FileNotFoundError(f"ProtoSSM weights not found: {proto_path}")

    print(f"Loading portable bundle from: {BUNDLE_DIR}")
    with open(bundle_meta_path, "r") as f:
        bundle = json.load(f)
    bundle_arrays = np.load(bundle_arrays_path, allow_pickle=False)

    PRIMARY_LABELS = list(bundle["primary_labels"])
    N_CLASSES = int(bundle["n_classes"])
    N_WINDOWS = int(bundle.get("n_windows", N_WINDOWS))
    CFG = restore_cfg_paths(bundle.get("cfg", {}))
    CFG["full_cache_input_dir"] = BUNDLE_DIR
    CFG["full_cache_work_dir"] = BUNDLE_DIR
    BEST = CFG.get("best_fusion", {})
    BEST_PROBE = bundle.get("best_probe", {})
    ENSEMBLE_WEIGHT_PROTO = float(bundle.get("ensemble_weight_proto", ENSEMBLE_WEIGHT_PROTO))
    CORRECTION_WEIGHT = float(bundle.get("correction_weight", CORRECTION_WEIGHT))
    PER_CLASS_THRESHOLDS = np.asarray(bundle_arrays["per_class_thresholds"], dtype=np.float32)
    probe_models = load_probe_models(bundle, bundle_arrays)
    emb_scaler = LoadedStandardScaler(bundle["emb_scaler"]["mean"], bundle["emb_scaler"]["scale"])
    emb_pca = LoadedPCA(
        bundle["emb_pca"]["mean"],
        bundle["emb_pca"]["components"],
        bundle["emb_pca"]["explained_variance"],
    )
    site_to_idx = {str(k): int(v) for k, v in bundle["site_to_idx"].items()}
    final_prior_tables = load_final_prior_tables(bundle)
    CLASS_NAME_MAP = bundle.get("class_name_map", {})
    TEXTURE_TAXA = set(bundle.get("textured_taxa", ["Amphibia", "Insecta"]))
    fam_to_idx = {str(k): int(v) for k, v in bundle.get("family_index", {}).items()}
    class_to_family = np.asarray(bundle.get("class_to_family", []), dtype=np.int64)

    MAPPED_POS = np.asarray(bundle.get("mapped_pos", []), dtype=np.int32)
    MAPPED_BC_INDICES = np.asarray(bundle.get("mapped_bc_indices", []), dtype=np.int32)
    selected_proxy_pos_to_bc = {
        int(k): np.asarray(v, dtype=np.int32)
        for k, v in bundle.get("selected_proxy_pos_to_bc", {}).items()
    }
    idx_active_texture = np.asarray(bundle.get("idx_active_texture", []), dtype=np.int32)
    idx_active_event = np.asarray(bundle.get("idx_active_event", []), dtype=np.int32)
    idx_mapped_active_texture = np.asarray(bundle.get("idx_mapped_active_texture", []), dtype=np.int32)
    idx_mapped_active_event = np.asarray(bundle.get("idx_mapped_active_event", []), dtype=np.int32)
    idx_selected_proxy_active_texture = np.asarray(bundle.get("idx_selected_proxy_active_texture", []), dtype=np.int32)
    idx_selected_prioronly_active_texture = np.asarray(bundle.get("idx_selected_prioronly_active_texture", []), dtype=np.int32)
    idx_selected_prioronly_active_event = np.asarray(bundle.get("idx_selected_prioronly_active_event", []), dtype=np.int32)
    idx_unmapped_inactive = np.asarray(bundle.get("idx_unmapped_inactive", []), dtype=np.int32)
    proxy_reduce = CFG.get("proxy_reduce", "max")
    proxy_reduce_path = BUNDLE_DIR / "proxy_reduce_grid.json"
    if proxy_reduce_path.exists():
        try:
            proxy_reduce_data = json.loads(proxy_reduce_path.read_text())
            proxy_reduce = str(proxy_reduce_data.get("best_proxy_reduce", proxy_reduce))
        except Exception:
            pass

    if USE_ONNX_PERCH:
        print("Using ONNX Perch for bundle inference")
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = 4
        ONNX_SESSION = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so, providers=["CPUExecutionProvider"])
        ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
        ONNX_OUTPUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    else:
        birdclassifier = tf.saved_model.load(str(MODEL_DIR))
        infer_fn = birdclassifier.signatures["serving_default"]

    test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
    if len(test_paths) == 0:
        n_dry = int(CFG.get("dryrun_n_files", 20))
        print(f"test_soundscapes not found, running dry-run on first {n_dry} train soundscapes")
        test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n_dry]
        if len(test_paths) == 0:
            raise RuntimeError("No files found in test_soundscapes or train_soundscapes for dry-run inference")

    meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
        test_paths,
        batch_files=CFG["batch_files"],
        verbose=CFG["verbose"],
        proxy_reduce=proxy_reduce,
    )
    emb_test_files, _ = reshape_to_files(emb_test, meta_test)
    logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)
    test_site_ids = np.array([site_to_idx.get(s, 0) for s in meta_test.groupby("filename")["site"].first().values], dtype=np.int64)
    test_hours = meta_test.groupby("filename")["hour_utc"].first().astype(int).values
    emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
    logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)
    test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
    test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)

    proto_cfg = CFG["proto_ssm"]
    n_sites_model = int(proto_cfg.get("n_sites", max(1, len(site_to_idx))))
    model = ProtoSSMv2(
        d_input=emb_test_files.shape[-1],
        d_model=proto_cfg["d_model"],
        d_state=proto_cfg["d_state"],
        n_ssm_layers=proto_cfg["n_ssm_layers"],
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        n_sites=n_sites_model,
        dropout=proto_cfg["dropout"],
        meta_dim=proto_cfg.get("meta_dim", 16),
        use_cross_attn=proto_cfg.get("use_cross_attn", True),
        cross_attn_heads=proto_cfg.get("cross_attn_heads", 4),
    ).to(DEVICE)
    model.init_family_head(max(1, len(fam_to_idx)), class_to_family)
    model.load_state_dict(load_file(str(proto_path), device=str(DEVICE)))
    model.eval()

    tta_shifts = CFG.get("tta_shifts", [0])
    if len(tta_shifts) > 1:
        proto_scores = temporal_shift_tta(
            emb_test_files, logits_test_files, model,
            test_site_ids, test_hours, shifts=tta_shifts
        )
    else:
        with torch.no_grad():
            proto_out, _, _ = model(
                emb_test_tensor,
                logits_test_tensor,
                site_ids=test_site_tensor,
                hours=test_hour_tensor,
            )
            proto_scores = proto_out.numpy()
    proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

    test_base_scores, test_prior_scores = fuse_scores_with_tables(
        scores_test_raw,
        sites=meta_test["site"].to_numpy(),
        hours=meta_test["hour_utc"].to_numpy(),
        tables=final_prior_tables,
    )
    emb_test_scaled = emb_scaler.transform(emb_test)
    Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)
    alpha_p = float(BEST_PROBE["alpha"])
    mlp_scores = get_vectorized_mlp_scores(
        Z_TEST,
        scores_test_raw,
        test_prior_scores,
        test_base_scores,
        probe_models,
        alpha_p,
        n_windows=N_WINDOWS,
        device=DEVICE,
    )

    final_test_scores = (
        ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
        (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
    ).astype(np.float32)

    res_model = None
    if res_path is not None and Path(res_path).exists() and CORRECTION_WEIGHT > 0:
        res_cfg = CFG["residual_ssm"]
        res_model = ResidualSSM(
            d_input=emb_test_files.shape[-1],
            d_scores=N_CLASSES,
            d_model=res_cfg.get("d_model", 64),
            d_state=res_cfg.get("d_state", 8),
            n_classes=N_CLASSES,
            n_windows=N_WINDOWS,
            n_sites=n_sites_model,
            dropout=res_cfg.get("dropout", 0.15),
            meta_dim=8,
        ).to(DEVICE)
        res_model.load_state_dict(load_file(str(res_path), device=str(DEVICE)))
        res_model.eval()
        with torch.no_grad():
            first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
            first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)
            test_correction = res_model(
                emb_test_tensor,
                first_pass_test_t,
                site_ids=test_site_tensor,
                hours=test_hour_tensor,
            ).numpy()
        final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction.reshape(-1, N_CLASSES).astype(np.float32)

    temp_cfg = CFG["temperature"]
    T_AVES = temp_cfg["aves"]
    T_TEXTURE = temp_cfg["texture"]
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
    for ci, label in enumerate(PRIMARY_LABELS):
        if CLASS_NAME_MAP.get(label, "Aves") in TEXTURE_TAXA:
            class_temperatures[ci] = T_TEXTURE

    probs = sigmoid(final_test_scores / class_temperatures[None, :])
    top_k = CFG.get("file_level_top_k", 0)
    if top_k > 0:
        probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
        probs = np.clip(probs, 0.0, 1.0)
    if CFG.get("rank_aware_scale", False):
        probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=CFG.get("rank_aware_power", 0.5))
        probs = np.clip(probs, 0.0, 1.0)
    alpha = CFG.get("delta_shift_alpha", 0.0)
    if alpha > 0:
        probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=alpha)
        probs = np.clip(probs, 0.0, 1.0)

    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)
    submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", meta_test["row_id"].values)
    submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv from portable bundle")
    print("Submission shape:", submission.shape)

    # Train-style evaluation when labels are available (e.g., dry-run on train soundscapes).
    try:
        labels_path = BASE / "train_soundscapes_labels.csv"
        if labels_path.exists():
            lbl_df = pd.read_csv(labels_path)
            lbl_df["primary_label"] = lbl_df["primary_label"].astype(str)
            sc_clean_eval = (
                lbl_df
                .groupby(["filename", "start", "end"])["primary_label"]
                .apply(union_labels)
                .reset_index(name="label_list")
            )
            sc_clean_eval["start_sec"] = pd.to_timedelta(sc_clean_eval["start"]).dt.total_seconds().astype(int)
            sc_clean_eval["end_sec"] = pd.to_timedelta(sc_clean_eval["end"]).dt.total_seconds().astype(int)
            sc_clean_eval["row_id"] = (
                sc_clean_eval["filename"].str.replace(".ogg", "", regex=False)
                + "_" + sc_clean_eval["end_sec"].astype(str)
            )

            # Match train script semantics: evaluate only fully-labeled 12-window files.
            windows_per_file = sc_clean_eval.groupby("filename").size()
            full_files = set(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
            eval_file_set = set(meta_test["filename"].unique().tolist())
            sc_eval = sc_clean_eval[
                sc_clean_eval["filename"].isin(eval_file_set)
                & sc_clean_eval["filename"].isin(full_files)
            ].copy()

            sub_eval = submission[submission["row_id"].isin(sc_eval["row_id"])].copy()
            n_matched = len(sub_eval)
            if n_matched > 0:
                label_to_idx_local = {c: i for i, c in enumerate(PRIMARY_LABELS)}
                row_to_labels = dict(zip(sc_eval["row_id"], sc_eval["label_list"]))
                y_true = np.zeros((n_matched, N_CLASSES), dtype=np.uint8)
                y_pred = sub_eval[PRIMARY_LABELS].to_numpy(dtype=np.float32)

                for i, rid in enumerate(sub_eval["row_id"].tolist()):
                    for cls_name in row_to_labels.get(rid, []):
                        idx = label_to_idx_local.get(cls_name)
                        if idx is not None:
                            y_true[i, idx] = 1

                if (y_true.sum(axis=0) > 0).any():
                    auc_window = macro_auc_skip_empty(y_true, y_pred)
                    print(f"Macro ROC AUC (train-style window-level): {auc_window:.6f} | rows={n_matched}")

                    if n_matched % N_WINDOWS == 0:
                        y_true_file = y_true.reshape(-1, N_WINDOWS, N_CLASSES).max(axis=1)
                        y_pred_file = y_pred.reshape(-1, N_WINDOWS, N_CLASSES).max(axis=1)
                        if (y_true_file.sum(axis=0) > 0).any():
                            auc_file = macro_auc_skip_empty(y_true_file, y_pred_file)
                            print(f"Macro ROC AUC (train-style file-level max): {auc_file:.6f} | files={len(y_true_file)}")
                else:
                    print("Macro ROC AUC unavailable: matched rows contain no positive labels")
            else:
                print("Macro ROC AUC unavailable: no overlap with fully-labeled eval rows")
        else:
            print("Macro ROC AUC unavailable: train_soundscapes_labels.csv not found")
    except Exception as e:
        print(f"Macro ROC AUC skipped due to error: {e}")
