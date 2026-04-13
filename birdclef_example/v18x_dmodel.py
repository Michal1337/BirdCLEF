#!/usr/bin/env python
# coding: utf-8

# ## BirdCLEF 2026 — ProtoSSM Inference Pipeline
# Install dependencies and run inference.

# In[ ]:


def _ipython_system(cmd: str) -> None:
    """Run a shell command only when executed inside a notebook/IPython."""
    try:
        ipy = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return
    ipy.system(cmd)


# Install TF 2.20 (required for Perch v2 StableHLO compatibility) in Kaggle notebooks.
_ipython_system(
    "pip install -q --no-deps "
    "/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl"
)
_ipython_system(
    "pip install -q --no-deps "
    "/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)


# In[ ]:


# Imports and constants
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import json
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

_WALL_START = time.time()


# In[ ]:


# Path discovery and constants (local repo defaults)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Base data directory (expects taxonomy.csv, sample_submission.csv, train/test_soundscapes)
BASE = Path(os.environ.get("BIRDCLEF_DATA", REPO_ROOT / "data"))

# Perch model directory (saved_model format with assets/labels.csv)
MODEL_DIR = Path(
    os.environ.get(
        "PERCH_MODEL_DIR",
        REPO_ROOT / "models" / "perch_v2" / "1",
    )
)

# Cache dir: search local candidates first
CACHE_DIR = None
for candidate in [
    Path(os.environ.get("PERCH_CACHE_DIR", "")),
    REPO_ROOT / "data" / "perch_cache",
    REPO_ROOT / "data" / "perch_meta",
    REPO_ROOT / "perch_cache",
]:
    if str(candidate).strip() == "":
        continue
    if candidate.exists() and (candidate / "full_perch_meta.parquet").exists():
        CACHE_DIR = candidate
        break

if CACHE_DIR is None:
    # Fallback: try to find it inside the repo
    for p in REPO_ROOT.rglob("full_perch_meta.parquet"):
        CACHE_DIR = p.parent
        break

if CACHE_DIR is None:
    raise FileNotFoundError(
        "Cache directory not found. Set PERCH_CACHE_DIR to a folder containing "
        "full_perch_meta.parquet and full_perch_arrays.npz."
    )
print(f"BASE = {BASE}")
print(f"MODEL_DIR = {MODEL_DIR}")
print(f"CACHE_DIR = {CACHE_DIR}")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
N_CLASSES = 234

DEVICE = torch.device("cpu")


# In[ ]:


# V18 CFG
CFG = {
    "verbose": False,
    "batch_files": 1,
    "proxy_reduce": "max",
    "full_cache_input_dir": CACHE_DIR,
    "best_fusion": {
        "lambda_event": 0.45,
        "lambda_texture": 1.1,
        "lambda_proxy_texture": 0.9,
        "smooth_texture": 0.35,
        "smooth_event": 0.15,
    },
    "proto_ssm": {
        "d_model": 384,
        "d_state": 32,
        "n_ssm_layers": 4,
        "dropout": 0.12,
        "n_prototypes": 2,
        "n_sites": 20,
        "meta_dim": 24,
        "use_cross_attn": True,
        "cross_attn_heads": 8,
    },
    "proto_ssm_train": {
        "n_epochs": 80,
        "lr": 8e-4,
        "weight_decay": 1e-3,
        "val_ratio": 0.15,
        "patience": 20,
        "pos_weight_cap": 25.0,
        "distill_weight": 0.15,
        "proto_margin": 0.15,
        "label_smoothing": 0.03,
        "oof_n_splits": 5,
        "mixup_alpha": 0.4,
        "focal_gamma": 2.5,
        "swa_start_frac": 0.65,
        "swa_lr": 4e-4,
        "use_cosine_restart": True,
        "restart_period": 20,
    },
    "residual_ssm": {
        "d_model": 128,
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.1,
        "correction_weight": 0.35,
        "n_epochs": 40,
        "lr": 8e-4,
        "patience": 12,
    },
    "temperature": {
        "aves": 1.10,
        "texture": 0.95,
    },
    "file_level_top_k": 2,
    "tta_shifts": [0, 1, -1, 2, -2],
    "rank_aware_scale": True,
    "rank_aware_power": 0.4,
    "delta_shift_alpha": 0.20,
    "threshold_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "probe_backend": "mlp",
    "mlp_params": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },
    "frozen_best_probe": {
        "pca_dim": 128,
        "min_pos": 5,
        "C": 0.75,
        "alpha": 0.52,
    },
}

BEST = CFG["best_fusion"]
ENSEMBLE_WEIGHT_PROTO = 0.5


# In[ ]:


# Utility functions: parsing, metrics, smoothing, V16/V17
# Regex for parsing filenames
# ─────────────────────────────────────────────────────────────────────────────

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics and helper utilities
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# V16/V17 utilities
# ─────────────────────────────────────────────────────────────────────────────

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, reduction="mean"):
    """Focal loss for multi-label classification."""
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


def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1]):
    """TTA by circular-shifting the 12-window embedding sequence."""
    all_preds = []
    model.eval()

    for shift in shifts:
        if shift == 0:
            e = emb_files
            l = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            l = np.roll(logits_files, shift, axis=1)

        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(l, dtype=torch.float32),
                site_ids=torch.tensor(site_ids, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
            pred = out.numpy()

        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def rank_aware_scaling(scores, n_windows=12, power=0.5):
    """V17: Scale each window by (file_max)^power."""
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(file_max, power)
    scaled = view * scale
    return scaled.reshape(N, C)


def delta_shift_smooth(scores, n_windows=12, alpha=0.15):
    """V17: Temporal smoothing across windows."""
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = scores.reshape(n_files, n_windows, C)
    prev_view = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_view = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_view + next_view)
    return smoothed.reshape(N, C)


def optimize_per_class_thresholds(oof_scores, y_true, n_windows=12, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """V17: Find optimal decision threshold per class from OOF predictions."""
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


# In[ ]:


# Perch inference engine
# ─────────────────────────────────────────────────────────────────────────────
# Perch inference engine
# ─────────────────────────────────────────────────────────────────────────────

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


def _infer_one_file(path_and_row):
    """Process a single file: load audio + run XLA inference. Thread-safe: reads from
    immutable globals (infer_fn, MAPPED_BC_INDICES, etc.) and writes to no shared state.
    Returns (file_row_start, logits_12, emb_12, row_meta) where row_meta is a dict.
    """
    path, file_row = path_and_row
    y = read_soundscape_60s(path)
    x = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = outputs["label"].numpy().astype(np.float32, copy=False)
    emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
    meta = parse_soundscape_filename(path.name)
    stem = path.stem
    return file_row, logits, emb, {
        "row_ids": [f"{stem}_{t}" for t in range(5, 65, 5)],
        "filename": path.name,
        "site": meta["site"],
        "hour": int(meta["hour_utc"]),
    }


def infer_perch_with_embeddings(paths, batch_files=16, verbose=True, proxy_reduce="max"):
    # batch_files is kept for API compat; with n_workers=2 we submit 2 files concurrently.
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    # n_workers=2: two concurrent TF calls share the 4 cores, each ~5% faster than
    # sequential because TF releases the GIL and two XLA kernels overlap on different
    # core pairs. More than 2 workers causes contention with no further gain.
    n_workers = 2
    file_row_pairs = [(path, i * N_WINDOWS) for i, path in enumerate(paths)]

    pbar = tqdm(total=n_files, desc="Perch batches") if verbose else None
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_infer_one_file, item): item for item in file_row_pairs}
        for fut in as_completed(futures):
            file_row, logits, emb, meta_d = fut.result()
            row_end = file_row + N_WINDOWS
            row_ids[file_row:row_end] = meta_d["row_ids"]
            filenames[file_row:row_end] = meta_d["filename"]
            sites[file_row:row_end] = meta_d["site"]
            hours[file_row:row_end] = meta_d["hour"]
            scores[file_row:row_end, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
            embeddings[file_row:row_end] = emb
            # Selected frog proxies
            for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
                sub = logits[:, bc_idx_arr]
                proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
                scores[file_row:row_end, pos] = proxy_score.astype(np.float32)
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()

    del logits, emb

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings


# In[ ]:


# Prior tables and score fusion
# ─────────────────────────────────────────────────────────────────────────────
# Prior tables
# ─────────────────────────────────────────────────────────────────────────────

def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)

    global_p = Y_prior.mean(axis=0).astype(np.float32)

    # Site
    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)

    for s in site_keys:
        i = site_to_i[s]
        mask = prior_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_prior[mask].mean(axis=0)

    # Hour
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)

    for h in hour_keys:
        i = hour_to_i[h]
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_prior[mask].mean(axis=0)

    # Site-hour
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

    # mapped active
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]

    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]

    # selected frog proxies
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]

    # prior-only active unmapped
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]

    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]

    # inactive unmapped
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior


# In[ ]:


# Embedding probe feature builder
# ─────────────────────────────────────────────────────────────────────────────
# Embedding probes
# ─────────────────────────────────────────────────────────────────────────────

def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """
    emb_proj: (n, d)
    raw_col, prior_col, base_col: (n,)
    returns: (n, d + 13)
    """
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)

    diff_mean = base_col - mean_base
    diff_prev = base_col - prev_base
    diff_next = base_col - next_base

    feats = np.concatenate([
        emb_proj,
        raw_col[:, None],
        prior_col[:, None],
        base_col[:, None],
        prev_base[:, None],
        next_base[:, None],
        mean_base[:, None],
        max_base[:, None],
        std_base[:, None],
        diff_mean[:, None],
        diff_prev[:, None],
        diff_next[:, None],
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1)

    return feats.astype(np.float32, copy=False)


# In[ ]:


# ProtoSSM + ResidualSSM architecture
# ─────────────────────────────────────────────────────────────────────────────
# ProtoSSM architecture
# ─────────────────────────────────────────────────────────────────────────────

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
    """Multi-head cross-attention between temporal windows."""

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

        # 1. Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # 3. Metadata embeddings
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # 4. Bidirectional SSM layers
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

        # 4b. Cross-attention after SSM
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        # 5. Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        # 6. Per-class calibration bias
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        # 7. Per-class gated fusion with Perch logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # 8. Taxonomic auxiliary head
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

        # Project embeddings
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        # Add metadata embeddings
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        # Cross-attention for non-local temporal patterns
        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        # Prototypical cosine similarity + class bias
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# ResidualSSM architecture
# ─────────────────────────────────────────────────────────────────────────────

class ResidualSSM(nn.Module):
    """Lightweight SSM that takes first-pass scores + embeddings and predicts corrections."""

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



# In[ ]:


# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

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


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows, \
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


def mixup_files(emb, logits, labels, site_ids, hours, families, alpha=0.3):
    """File-level mixup augmentation for ProtoSSM training."""
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours, families

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)

    perm = np.random.permutation(n)

    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]

    families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None

    return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix


def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                           site_ids_train=None, hours_train=None,
                           emb_val=None, logits_val=None, labels_val=None,
                           site_ids_val=None, hours_val=None,
                           file_families_train=None, file_families_val=None,
                           cfg=None, verbose=True):
    """Train a single ProtoSSM v4 model with mixup, focal loss, and SWA."""
    if cfg is None:
        cfg = CFG["proto_ssm_train"]

    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    swa_start_frac = cfg.get("swa_start_frac", 1.0)
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    labels_np = labels_train.copy()

    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None

    fam_v = torch.tensor(file_families_val, dtype=torch.float32) if (has_val and file_families_val is not None) else None

    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                emb_train, logits_train, labels_np,
                site_ids_train, hours_train, file_families_train,
                alpha=mixup_alpha,
            )
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
            fam_mix = file_families_train

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
        fam_tr = torch.tensor(fam_mix, dtype=torch.float32) if fam_mix is not None else None

        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out, labels_tr,
                gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :],
            )
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr,
                pos_weight=pos_weight[None, None, :]
            )

        loss_distill = F.mse_loss(species_out, logits_tr)
        loss = loss_main + cfg["distill_weight"] * loss_distill

        if family_out is not None and fam_tr is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
            loss = loss + 0.1 * loss_family

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v,
                    pos_weight=pos_weight[None, None, :]
                )
                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}")

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  Applying SWA (averaged {swa_count} checkpoints)")
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} max={alphas.max():.3f}")
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")

    return model, history


# In[ ]:


# Phase 1: Dataset setup
print("Phase 1: Dataset setup...")
t0 = time.time()

taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES, f"Expected {N_CLASSES} classes, got {len(PRIMARY_LABELS)}"

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

sc_clean = (
    soundscape_labels
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)

sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = (
    sc_clean["filename"].str.replace(".ogg", "", regex=False)
    + "_" + sc_clean["end_sec"].astype(str)
)

meta_sc = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta_sc], axis=1)

# Fully-labeled files
windows_per_file = sc_clean.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

# Multi-hot label matrix
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

full_truth = (
    sc_clean[sc_clean["file_fully_labeled"]]
    .sort_values(["filename", "end_sec"])
    .reset_index(drop=False)
)

Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]

# Load cache
cache_meta_path = CACHE_DIR / "full_perch_meta.parquet"
cache_npz_path = CACHE_DIR / "full_perch_arrays.npz"

assert cache_meta_path.exists(), f"Cache not found: {cache_meta_path}"
assert cache_npz_path.exists(), f"Cache not found: {cache_npz_path}"

meta_full = pd.read_parquet(cache_meta_path)
arr = np.load(cache_npz_path)
scores_full_raw = arr["scores_full_raw"].astype(np.float32)
emb_full = arr["emb_full"].astype(np.float32)
print(emb_full.shape)

# Align truth to cached order
full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

print(f"  Done. Cached files: {len(meta_full['filename'].unique())}")
print(f"  Y_FULL: {Y_FULL.shape}, active classes: {int((Y_FULL.sum(axis=0) > 0).sum())}")
print(f"  Phase 1: {time.time() - t0:.1f}s")


# In[ ]:


# Phase 2: Load models
print("Phase 2: Load models...")
t0 = time.time()

birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)

NO_LABEL_INDEX = len(bc_labels)
MANUAL_SCIENTIFIC_NAME_MAP = {}

taxonomy_copy = taxonomy.copy()
taxonomy_copy["scientific_name_lookup"] = taxonomy_copy["scientific_name"].replace(MANUAL_SCIENTIFIC_NAME_MAP)

bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

mapping = taxonomy_copy.merge(
    bc_lookup[["scientific_name_lookup", "bc_index"]],
    on="scientific_name_lookup",
    how="left"
)

mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

CLASS_NAME_MAP = taxonomy_copy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

idx_active_texture = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
    dtype=np.int32
)
idx_active_event = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
    dtype=np.int32
)

idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]

idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]

idx_unmapped_inactive = np.array(
    [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES],
    dtype=np.int32
)

# Build automatic genus proxies for unmapped non-sonotypes
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
].copy()

def get_genus_hits(scientific_name):
    genus = str(scientific_name).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
    ].copy()
    return genus, hits

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    target = row["primary_label"]
    sci = row["scientific_name"]
    genus, hits = get_genus_hits(sci)
    if len(hits) > 0:
        proxy_map[target] = {
            "target_scientific_name": sci,
            "genus": genus,
            "bc_indices": hits["bc_index"].astype(int).tolist(),
            "proxy_scientific_names": hits["scientific_name"].tolist(),
        }

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
SELECTED_PROXY_TARGETS = sorted([
    t for t in proxy_map.keys()
    if CLASS_NAME_MAP.get(t) in PROXY_TAXA
])

selected_proxy_pos_to_bc = {
    label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
    for target in SELECTED_PROXY_TARGETS
}

idx_selected_proxy_active_texture = np.intersect1d(
    np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32),
    idx_active_texture
)
idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture,
    np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32))
idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event,
    np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32))

# XLA warm-up
print("  XLA warm-up...", end=" ", flush=True)
_n_warmup = CFG["batch_files"] * N_WINDOWS
_x_warmup = np.zeros((_n_warmup, WINDOW_SAMPLES), dtype=np.float32)
_ = infer_fn(inputs=tf.convert_to_tensor(_x_warmup))
del _x_warmup, _
gc.collect()
print("done")

print(f"  Done. Mapped: {MAPPED_MASK.sum()}/{N_CLASSES}, proxy targets: {len(SELECTED_PROXY_TARGETS)}")
print(f"  Phase 2: {time.time() - t0:.1f}s")


# In[ ]:


# Phase 3: Perch inference on test files
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

RUN_INFERENCE = len(test_paths) > 0
if not RUN_INFERENCE:
    print("Hidden test not mounted. Dry-run on first 20 train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:20]
else:
    print(f"Hidden test files: {len(test_paths)}")

print(f"Phase 3: Perch inference on {len(test_paths)} files...")
t0 = time.time()

meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
    test_paths,
    batch_files=CFG["batch_files"],
    verbose=True,
    proxy_reduce=CFG["proxy_reduce"],
)

print(f"  Done. meta_test: {meta_test.shape}, scores: {scores_test_raw.shape}")
print(f"  Phase 3: {time.time() - t0:.1f}s")


# In[ ]:


# Phase 4: Train ProtoSSM
print("Phase 4: Train ProtoSSM...")
t0 = time.time()

# Reshape cached data to file-level
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)

# Build taxonomy groups, site mapping
n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)

site_to_idx, n_sites_mapped = build_site_mapping(meta_full)
n_sites_cfg = CFG["proto_ssm"]["n_sites"]

site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)

# Build per-file family labels
file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
    for ci in active_classes:
        file_families[fi, class_to_family[ci]] = 1.0

# Train final ProtoSSM on all cached data
ssm_cfg = CFG["proto_ssm"]
model = ProtoSSMv2(
    d_input=emb_full.shape[1],
    d_model=ssm_cfg["d_model"],
    d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"],
    n_sites=ssm_cfg["n_sites"],
    meta_dim=ssm_cfg["meta_dim"],
    use_cross_attn=ssm_cfg.get("use_cross_attn", True),
    cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
).to(DEVICE)

emb_flat_tensor = torch.tensor(emb_full, dtype=torch.float32)
labels_flat_tensor = torch.tensor(Y_FULL, dtype=torch.float32)
model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
model.init_family_head(n_families, class_to_family)

print(f"  ProtoSSM parameters: {model.count_parameters():,}")

model, train_history = train_proto_ssm_single(
    model,
    emb_files, logits_files, labels_files.astype(np.float32),
    site_ids_train=site_ids_all, hours_train=hours_all,
    cfg=CFG["proto_ssm_train"],
    verbose=True,
)

print(f"  Phase 4: {time.time() - t0:.1f}s")


# In[ ]:


# Phase 5: Train MLP probes
print("Phase 5: Train MLP probes...")
t0 = time.time()

# Fit final prior tables on all labeled soundscapes
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

# Fit embedding scaler + PCA
emb_scaler = StandardScaler()
emb_full_scaled = emb_scaler.fit_transform(emb_full)

n_comp = min(
    int(CFG["frozen_best_probe"]["pca_dim"]),
    emb_full_scaled.shape[0] - 1,
    emb_full_scaled.shape[1]
)
emb_pca = PCA(n_components=n_comp)
Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)

# Build OOF base/prior for probe training features
groups_full = meta_full["filename"].to_numpy()
gkf5 = GroupKFold(n_splits=5)
oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(gkf5.split(scores_full_raw, groups=groups_full)):
    tr_idx = np.sort(tr_idx)
    va_idx = np.sort(va_idx)
    val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
    prior_mask = ~sc_clean["filename"].isin(val_files).values
    prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
    Y_prior_fold = Y_SC[prior_mask]
    tables_fold = fit_prior_tables(prior_df_fold, Y_prior_fold)
    va_base, va_prior = fuse_scores_with_tables(
        scores_full_raw[va_idx],
        sites=meta_full.iloc[va_idx]["site"].to_numpy(),
        hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
        tables=tables_fold,
    )
    oof_base[va_idx] = va_base
    oof_prior[va_idx] = va_prior

# Train final MLP probes
PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= int(CFG["frozen_best_probe"]["min_pos"]))[0].astype(np.int32)
probe_models = {}

for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training MLP probes"):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior[:, cls_idx],
        base_col=oof_base[:, cls_idx],
    )
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > n_pos:
        repeat = max(1, n_neg // n_pos)
        pos_idx = np.where(y == 1)[0]
        X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
    else:
        X_bal, y_bal = X_cls, y
    clf = MLPClassifier(**CFG["mlp_params"])
    clf.fit(X_bal, y_bal)
    probe_models[cls_idx] = clf

print(f"  Done. MLP probes trained: {len(probe_models)}")
print(f"  Phase 5: {time.time() - t0:.1f}s")


# In[ ]:


# Phase 6: Score fusion on test files
print("Phase 6: Score fusion...")
t0 = time.time()

# ProtoSSM inference on test files with TTA
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

test_site_ids, test_hours = get_file_metadata(
    meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"]
)

model.eval()
tta_shifts = CFG.get("tta_shifts", [0])
if len(tta_shifts) > 1:
    proto_scores = temporal_shift_tta(
        emb_test_files, logits_test_files, model,
        test_site_ids, test_hours, shifts=tta_shifts
    )
else:
    with torch.no_grad():
        emb_t = torch.tensor(emb_test_files, dtype=torch.float32)
        logits_t = torch.tensor(logits_test_files, dtype=torch.float32)
        site_t = torch.tensor(test_site_ids, dtype=torch.long)
        hour_t = torch.tensor(test_hours, dtype=torch.long)
        proto_out, _, _ = model(emb_t, logits_t, site_ids=site_t, hours=hour_t)
        proto_scores = proto_out.numpy()

proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

# Prior-fused base scores for test
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# MLP probe scores on test
emb_test_scaled = emb_scaler.transform(emb_test)
Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)

mlp_scores = test_base_scores.copy()
for cls_idx, clf in probe_models.items():
    X_cls_test = build_class_features(
        Z_TEST,
        raw_col=scores_test_raw[:, cls_idx],
        prior_col=test_prior_scores[:, cls_idx],
        base_col=test_base_scores[:, cls_idx],
    )
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls_test).astype(np.float32)
    alpha_p = float(CFG["frozen_best_probe"]["alpha"])
    mlp_scores[:, cls_idx] = (1.0 - alpha_p) * test_base_scores[:, cls_idx] + alpha_p * pred

# Ensemble fusion
final_test_scores = (
    ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
    (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
).astype(np.float32)

print(f"  Ensemble scores: {final_test_scores.shape}")


# In[ ]:


# ResidualSSM: second-pass boosting
res_cfg = CFG["residual_ssm"]

# Train ResidualSSM on cached data
model.eval()
with torch.no_grad():
    emb_train_t = torch.tensor(emb_files, dtype=torch.float32)
    logits_train_t = torch.tensor(logits_files, dtype=torch.float32)
    site_train_t = torch.tensor(site_ids_all, dtype=torch.long)
    hour_train_t = torch.tensor(hours_all, dtype=torch.long)
    proto_train_out, _, _ = model(emb_train_t, logits_train_t,
                                   site_ids=site_train_t, hours=hour_train_t)
    proto_train_scores = proto_train_out.numpy()

# Get prior-fused base for MLP on training data
train_base_scores, train_prior_scores = fuse_scores_with_tables(
    scores_full_raw,
    sites=meta_full["site"].to_numpy(),
    hours=meta_full["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)
mlp_train_scores_flat = train_base_scores.copy()
for cls_idx, clf in probe_models.items():
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=train_prior_scores[:, cls_idx],
        base_col=train_base_scores[:, cls_idx],
    )
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls).astype(np.float32)
    alpha_p = float(CFG["frozen_best_probe"]["alpha"])
    mlp_train_scores_flat[:, cls_idx] = (1 - alpha_p) * train_base_scores[:, cls_idx] + alpha_p * pred

mlp_train_scores_files, _ = reshape_to_files(mlp_train_scores_flat, meta_full)

first_pass_files = (
    ENSEMBLE_WEIGHT_PROTO * proto_train_scores +
    (1 - ENSEMBLE_WEIGHT_PROTO) * mlp_train_scores_files
).astype(np.float32)

labels_float = labels_files.astype(np.float32)
first_pass_probs = 1.0 / (1.0 + np.exp(-first_pass_files))
residuals = labels_float - first_pass_probs

res_model = ResidualSSM(
    d_input=emb_full.shape[1],
    d_scores=N_CLASSES,
    d_model=res_cfg["d_model"],
    d_state=res_cfg["d_state"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=res_cfg["dropout"],
    n_sites=CFG["proto_ssm"]["n_sites"],
    meta_dim=8,
).to(DEVICE)

n_files_cached = len(file_list)
n_val = max(1, int(n_files_cached * 0.15))
perm = torch.randperm(n_files_cached, generator=torch.Generator().manual_seed(123))
val_i = perm[:n_val].numpy()
train_i = perm[n_val:].numpy()

emb_tr_r = torch.tensor(emb_files[train_i], dtype=torch.float32)
fp_tr_r = torch.tensor(first_pass_files[train_i], dtype=torch.float32)
res_tr_r = torch.tensor(residuals[train_i], dtype=torch.float32)
site_tr_r = torch.tensor(site_ids_all[train_i], dtype=torch.long)
hour_tr_r = torch.tensor(hours_all[train_i], dtype=torch.long)

emb_va_r = torch.tensor(emb_files[val_i], dtype=torch.float32)
fp_va_r = torch.tensor(first_pass_files[val_i], dtype=torch.float32)
res_va_r = torch.tensor(residuals[val_i], dtype=torch.float32)
site_va_r = torch.tensor(site_ids_all[val_i], dtype=torch.long)
hour_va_r = torch.tensor(hours_all[val_i], dtype=torch.long)

res_optimizer = torch.optim.AdamW(res_model.parameters(), lr=res_cfg["lr"], weight_decay=1e-3)
res_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    res_optimizer, max_lr=res_cfg["lr"],
    epochs=res_cfg["n_epochs"], steps_per_epoch=1,
    pct_start=0.1, anneal_strategy='cos'
)

best_val_loss_r = float('inf')
best_state_r = None
wait_r = 0

for epoch in range(res_cfg["n_epochs"]):
    res_model.train()
    correction = res_model(emb_tr_r, fp_tr_r, site_ids=site_tr_r, hours=hour_tr_r)
    loss_r = F.mse_loss(correction, res_tr_r)

    res_optimizer.zero_grad()
    loss_r.backward()
    torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
    res_optimizer.step()
    res_scheduler.step()

    res_model.eval()
    with torch.no_grad():
        val_corr = res_model(emb_va_r, fp_va_r, site_ids=site_va_r, hours=hour_va_r)
        val_loss_r = F.mse_loss(val_corr, res_va_r)

    if val_loss_r.item() < best_val_loss_r:
        best_val_loss_r = val_loss_r.item()
        best_state_r = {k: v.clone() for k, v in res_model.state_dict().items()}
        wait_r = 0
    else:
        wait_r += 1

    if wait_r >= res_cfg["patience"]:
        break

if best_state_r is not None:
    res_model.load_state_dict(best_state_r)

CORRECTION_WEIGHT = res_cfg["correction_weight"]

# Apply ResidualSSM to test files
emb_test_t = torch.tensor(emb_test_files, dtype=torch.float32)
first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
fp_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)
test_site_t = torch.tensor(test_site_ids, dtype=torch.long)
test_hour_t = torch.tensor(test_hours, dtype=torch.long)

res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        emb_test_t, fp_test_t,
        site_ids=test_site_t, hours=test_hour_t
    ).numpy()

test_correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction_flat

print(f"  ResidualSSM correction applied (weight={CORRECTION_WEIGHT})")
print(f"  Phase 6: {time.time() - t0:.1f}s")


# In[ ]:


# Phase 7: Post-processing + Submission
print("Phase 7: Post-processing...")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# Per-taxon temperature scaling
temp_cfg = CFG["temperature"]
T_AVES = temp_cfg["aves"]
T_TEXTURE = temp_cfg["texture"]

class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
for ci, label in enumerate(PRIMARY_LABELS):
    cn = CLASS_NAME_MAP.get(label, "Aves")
    if cn in TEXTURE_TAXA:
        class_temperatures[ci] = T_TEXTURE

scaled_scores = final_test_scores / class_temperatures[None, :]
probs = sigmoid(scaled_scores)

# File-level confidence scaling
top_k = CFG.get("file_level_top_k", 0)
if top_k > 0:
    probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
    probs = np.clip(probs, 0.0, 1.0)

# Rank-aware post-processing
if CFG.get("rank_aware_scale", False):
    power = CFG.get("rank_aware_power", 0.5)
    probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=power)
    probs = np.clip(probs, 0.0, 1.0)

# Delta shift smoothing
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

alpha_ds = CFG.get("delta_shift_alpha", 0.0)
if alpha_ds > 0:
    probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=alpha_ds)
    probs = np.clip(probs, 0.0, 1.0)

# Per-class threshold sharpening (default 0.5)
PER_CLASS_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float32)
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)

# --- Build submission ---
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)

wall_time = time.time() - _WALL_START
print(f"\nSaved submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Final score range: {probs.min():.6f} to {probs.max():.6f}")
print(f"Final mean: {probs.mean():.4f}")
print(f"Total wall time: {wall_time:.1f}s ({wall_time/60:.1f}min)")
print(submission.iloc[:3, :8])
