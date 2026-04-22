#!/usr/bin/env python3
"""ResidualSSM config sweep.

Phase 1 (once per fold): trains LightProtoSSM + torch probes, computes
first_pass logits and calibrates per-class thresholds.

Phase 2 (once per residual config × fold): trains ResidualSSM and evaluates
all post-residual stages: residual_plus_temp, post_rank, post_smooth,
final_after_threshold.

Usage:
    python residual_ssm_config_sweep.py
    python residual_ssm_config_sweep.py --n-splits 3 --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    from birdclef_example.proto_ssm_config_sweep import (
        N_WINDOWS,
        LightProtoSSM,
        SelectiveSSM,
        macro_auc,
        sigmoid,
        train_light_proto_ssm,
        union_labels,
    )
except ModuleNotFoundError:
    from proto_ssm_config_sweep import (
        N_WINDOWS,
        LightProtoSSM,
        SelectiveSSM,
        macro_auc,
        sigmoid,
        train_light_proto_ssm,
        union_labels,
    )

SEED = 1337
STAGE_NAMES = [
    "residual_plus_temp",
    "post_rank",
    "post_smooth",
    "final_after_threshold",
]
TEXTURE_TAXA = {"Amphibia", "Insecta"}

RANK_POWER_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
SMOOTH_ALPHA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

MODE = "train"
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

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_class_freq_weights(Y: np.ndarray, cap: float = 10.0) -> np.ndarray:
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq = pos_count / Y.shape[0]
    w = np.clip(1.0 / (freq ** 0.5), 1.0, cap)
    return (w / w.mean()).astype(np.float32)


def build_sequential_features(
    scores_col: np.ndarray, n_windows: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = scores_col.reshape(-1, n_windows)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    return (
        prev.reshape(-1),
        nxt.reshape(-1),
        np.repeat(x.mean(axis=1), n_windows),
        np.repeat(x.max(axis=1), n_windows),
        np.repeat(x.std(axis=1), n_windows),
    )


# ── Probe MLP ──────────────────────────────────────────────────────────────────

class TorchProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple, dropout: float):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [1]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch_probes(
    emb: np.ndarray,
    scores_raw: np.ndarray,
    Y: np.ndarray,
    cfg: dict,
) -> Tuple[dict, StandardScaler, object]:
    min_pos = int(cfg.get("min_pos", 3))
    pca_dim = int(cfg.get("pca_dim", 0))
    max_rows = int(cfg.get("max_rows", 3000))
    hidden_dims = tuple(cfg.get("hidden_dims", (128, 64)))
    dropout = float(cfg.get("dropout", 0.1))
    epochs = int(cfg.get("epochs", 80))
    batch_size = int(cfg.get("batch_size", 512))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    patience = int(cfg.get("patience", 10))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    standardize = bool(cfg.get("standardize_features", True))
    seed = int(cfg.get("seed", SEED))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb).astype(np.float32)
    if pca_dim > 0:
        n_comp = min(int(pca_dim), emb_s.shape[1] - 1)
        pca = PCA(n_components=n_comp, random_state=seed)
        z = pca.fit_transform(emb_s).astype(np.float32)
    else:
        pca = None
        z = emb_s

    n_classes = Y.shape[1]
    class_weights = build_class_freq_weights(Y, cap=10.0)
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"  Training torch probes for {len(active)} species ...")

    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    probe_models: dict = {}

    for ci in tqdm(active, desc="  Probes", leave=False):
        yc = Y[:, ci].astype(np.float32)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        prev, nxt, mean_, max_, std_ = build_sequential_features(scores_raw[:, ci])
        x = np.hstack([
            z,
            scores_raw[:, ci:ci + 1],
            prev[:, None], nxt[:, None], mean_[:, None], max_[:, None], std_[:, None],
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
        va_i, tr_i = perm[:n_val], perm[n_val:]

        x_tr, y_tr = x_bal[tr_i], y_bal[tr_i]
        x_va, y_va = x_bal[va_i], y_bal[va_i]
        if standardize:
            mu = x_tr.mean(axis=0, keepdims=True)
            sd = np.where(x_tr.std(axis=0, keepdims=True) < 1e-6, 1.0,
                          x_tr.std(axis=0, keepdims=True))
            x_tr = (x_tr - mu) / sd
            x_va = (x_va - mu) / sd
        else:
            mu = np.zeros((1, x_bal.shape[1]), dtype=np.float32)
            sd = np.ones((1, x_bal.shape[1]), dtype=np.float32)

        model = TorchProbeMLP(x_bal.shape[1], hidden_dims, dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCEWithLogitsLoss()
        x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        x_va_t = torch.tensor(x_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)

        best_val, best_state, wait = float("inf"), None, 0
        for _ in range(epochs):
            model.train()
            order = torch.randperm(x_tr_t.shape[0])
            for s in range(0, x_tr_t.shape[0], batch_size):
                idx = order[s:s + batch_size]
                loss = crit(model(x_tr_t[idx]), y_tr_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(x_va_t), y_va_t).item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        probe_models[int(ci)] = {"model": model.cpu(), "mu": mu, "sd": sd}

    print(f"  Trained {len(probe_models)} probes")
    return probe_models, scaler, pca


def predict_torch_probe_logits(
    emb_test: np.ndarray,
    scores_test: np.ndarray,
    probe_models: dict,
    scaler: StandardScaler,
    pca: object,
) -> np.ndarray:
    z = scaler.transform(emb_test).astype(np.float32)
    if pca is not None:
        z = pca.transform(z).astype(np.float32)
    out = scores_test.copy()
    for ci, payload in probe_models.items():
        prev, nxt, mean_, max_, std_ = build_sequential_features(scores_test[:, ci])
        x = np.hstack([
            z,
            scores_test[:, ci:ci + 1],
            prev[:, None], nxt[:, None], mean_[:, None], max_[:, None], std_[:, None],
        ]).astype(np.float32)
        x = (x - payload["mu"]) / payload["sd"]
        with torch.no_grad():
            out[:, ci] = payload["model"](torch.tensor(x, dtype=torch.float32)).numpy()
    return out


def blend_probe_logits(
    base_scores: np.ndarray, probe_logits: np.ndarray, alpha: float
) -> np.ndarray:
    return (1.0 - alpha) * base_scores + alpha * probe_logits


# ── Prior ──────────────────────────────────────────────────────────────────────

def build_prior_tables(sc_df: pd.DataFrame, Y_labels: np.ndarray) -> dict:
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    return dict(
        global_p=global_p,
        site_to_i=site_to_i, site_p=site_p, site_n=site_n,
        hour_to_i=hour_to_i, hour_p=hour_p, hour_n=hour_n,
    )


def apply_prior(
    scores: np.ndarray,
    sites: np.ndarray,
    hours: np.ndarray,
    tables: dict,
    lambda_prior: float = 0.4,
) -> np.ndarray:
    eps = 1e-4
    n = len(scores)
    out = scores.copy()
    p = np.tile(tables["global_p"], (n, 1))
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    p = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)


# ── Post-processing ────────────────────────────────────────────────────────────

def calibrate_and_optimize_thresholds(
    oof_probs: np.ndarray,
    Y_FULL: np.ndarray,
    threshold_grid: list,
    n_windows: int = N_WINDOWS,
) -> np.ndarray:
    n_cls = oof_probs.shape[1]
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files = oof_probs.shape[0] // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    for c in range(n_cls):
        y_true = file_y[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(file_oof[:, c], y_true)
            y_cal = ir.transform(file_oof[:, c])
        except Exception:
            y_cal = file_oof[:, c]
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
    return thresholds


def apply_per_class_thresholds(
    scores: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    scaled = np.copy(scores)
    for c in range(scores.shape[1]):
        t = thresholds[c]
        above = scores[:, c] > t
        scaled[above, c] = 0.5 + 0.5 * (scores[above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def rank_aware_scaling(
    probs: np.ndarray, n_windows: int = N_WINDOWS, power: float = 0.4
) -> np.ndarray:
    view = probs.reshape(-1, n_windows, probs.shape[1])
    scale = np.power(view.max(axis=1, keepdims=True), power)
    return (view * scale).reshape(probs.shape)


def adaptive_delta_smooth(
    probs: np.ndarray, n_windows: int = N_WINDOWS, base_alpha: float = 0.20
) -> np.ndarray:
    result = probs.copy()
    view = probs.reshape(-1, n_windows, probs.shape[1])
    out = result.reshape(-1, n_windows, probs.shape[1])
    for t in range(n_windows):
        conf = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        if t == 0:
            nb = (view[:, t, :] + view[:, t + 1, :]) / 2.0
        elif t == n_windows - 1:
            nb = (view[:, t - 1, :] + view[:, t, :]) / 2.0
        else:
            nb = (view[:, t - 1, :] + view[:, t + 1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * nb
    return result


# ── ResidualSSM (configurable architecture) ───────────────────────────────────

class ResidualSSM(nn.Module):
    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = N_WINDOWS,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ):
        super().__init__()
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

    def forward(
        self,
        emb: torch.Tensor,
        first_pass: torch.Tensor,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_proj(torch.cat([emb, first_pass], dim=-1)) + self.pos_enc[:, :emb.shape[1], :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([
                self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                self.hour_emb(hours.clamp(0, 23)),
            ], dim=-1))
            h = h + meta.unsqueeze(1)
        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([h_f, h_b], dim=-1)))
        h = self.ssm_norm(h + res)
        return self.output_head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_residual_ssm(
    emb_full: np.ndarray,
    first_pass_flat: np.ndarray,
    Y_full: np.ndarray,
    site_ids: np.ndarray,
    hour_ids: np.ndarray,
    n_classes: int,
    n_windows: int = N_WINDOWS,
    d_model: int = 64,
    d_state: int = 8,
    dropout: float = 0.1,
    n_sites: int = 20,
    meta_dim: int = 8,
    n_epochs: int = 30,
    patience: int = 8,
    lr: float = 1e-3,
    correction_weight: float = 0.30,
    verbose: bool = False,
) -> Tuple[nn.Module, float]:
    n_files = len(emb_full) // n_windows
    emb_f = emb_full.reshape(n_files, n_windows, -1)
    fp_f = first_pass_flat.reshape(n_files, n_windows, -1)
    lab_f = Y_full.reshape(n_files, n_windows, -1).astype(np.float32)
    fp_prob = sigmoid(fp_f)
    residuals = (lab_f - fp_prob).astype(np.float32)

    n_val = max(1, int(n_files * 0.15))
    gen = torch.Generator()
    gen.manual_seed(SEED)
    perm = torch.randperm(n_files, generator=gen).numpy()
    val_i, train_i = perm[:n_val], perm[n_val:]

    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    fp_t = torch.tensor(fp_f, dtype=torch.float32)
    res_t = torch.tensor(residuals, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    torch.manual_seed(SEED)
    model = ResidualSSM(
        d_input=emb_full.shape[1],
        d_scores=n_classes,
        d_model=d_model,
        d_state=d_state,
        n_classes=n_classes,
        n_windows=n_windows,
        dropout=dropout,
        n_sites=n_sites,
        meta_dim=meta_dim,
    )
    print(f"    ResidualSSM params: {model.count_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos",
    )
    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(n_epochs):
        model.train()
        loss = F.mse_loss(
            model(emb_t[train_i], fp_t[train_i],
                  site_ids=site_t[train_i], hours=hour_t[train_i]),
            res_t[train_i],
        )
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(
                model(emb_t[val_i], fp_t[val_i],
                      site_ids=site_t[val_i], hours=hour_t[val_i]),
                res_t[val_i],
            ).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose:
                print(f"    Early stop ep {ep + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"    ResidualSSM best val MSE={best_loss:.6f}")
    return model, correction_weight


# ── Data loading ───────────────────────────────────────────────────────────────

def build_training_data(
    cache_meta: Path,
    cache_npz: Path,
    sample_sub: Path,
    sound_labels: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int, List[str]]:
    sub = pd.read_csv(sample_sub)
    primary_labels = sub.columns[1:].tolist()
    n_classes = len(primary_labels)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    sl = pd.read_csv(sound_labels)
    sc = (
        sl.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    y_sc = np.zeros((len(sc), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                y_sc[i, label_to_idx[lbl]] = 1

    wins = sc.groupby("filename").size()
    full_files = sorted(wins[wins == N_WINDOWS].index.tolist())
    full_rows = (
        sc[sc["filename"].isin(full_files)]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )

    meta_tr = pd.read_parquet(cache_meta)
    arr = np.load(cache_npz)
    sc_tr = arr["scores_full_raw"].astype(np.float32)
    emb_tr = arr["emb_full"].astype(np.float32)
    y_aligned = y_sc[
        full_rows.set_index("row_id").loc[meta_tr["row_id"], "index"].to_numpy()
    ].astype(np.float32)

    return emb_tr, sc_tr, y_aligned, meta_tr, n_classes, primary_labels


def build_temperatures(
    primary_labels: List[str], taxonomy_path: Path
) -> np.ndarray:
    taxonomy = pd.read_csv(taxonomy_path)
    cls_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    temps = np.ones(len(primary_labels), dtype=np.float32)
    for ci, lbl in enumerate(primary_labels):
        cls = cls_map.get(lbl, "Aves")
        temps[ci] = 0.95 if cls in TEXTURE_TAXA else 1.10
    return temps


# ── Config loading ─────────────────────────────────────────────────────────────

def load_residual_configs(configs_py: Path) -> List[Dict]:
    if not configs_py.exists():
        raise FileNotFoundError(f"Missing configs file: {configs_py}")
    spec = importlib.util.spec_from_file_location("residual_ssm_sweep_configs", str(configs_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "RESIDUAL_SSM_SWEEP_CONFIGS"):
        raise AttributeError(f"{configs_py} must define RESIDUAL_SSM_SWEEP_CONFIGS")
    configs = getattr(module, "RESIDUAL_SSM_SWEEP_CONFIGS")
    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("RESIDUAL_SSM_SWEEP_CONFIGS must be a non-empty list")
    for i, cfg in enumerate(configs):
        if "name" not in cfg:
            raise ValueError(f"Config at index {i} is missing 'name'")
    return configs


# ── Phase 1: build per-fold cache ─────────────────────────────────────────────

def _fold_site_hour_ids(
    meta: pd.DataFrame, fnames: np.ndarray, site2i: dict, n_sites: int
) -> Tuple[np.ndarray, np.ndarray]:
    site_ids = np.array(
        [
            min(site2i.get(str(meta.loc[meta["filename"] == fn, "site"].iloc[0]), 0), n_sites - 1)
            for fn in fnames
        ],
        dtype=np.int64,
    )
    hour_ids = np.array(
        [int(meta.loc[meta["filename"] == fn, "hour_utc"].iloc[0]) % 24 for fn in fnames],
        dtype=np.int64,
    )
    return site_ids, hour_ids


def build_fold_cache(
    emb_tr: np.ndarray,
    sc_tr: np.ndarray,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    n_classes: int,
    temperatures: np.ndarray,
    proto_cfg: dict,
    mlp_cfg: dict,
    lambda_prior: float,
    mlp_alpha_blend: float,
    ensemble_w: float,
    threshold_grid: List[float],
    n_splits: int,
) -> List[dict]:
    """Train SSM + probes once per fold, cache everything needed for residual sweep."""
    file_meta = meta_tr.drop_duplicates("filename").reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)

    proto_n_sites = int(proto_cfg.get("n_sites", 20))

    fold_cache = []
    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])
        tr_mask = meta_tr["filename"].isin(tr_fnames).values
        va_mask = meta_tr["filename"].isin(va_fnames).values

        emb_tr_f = emb_tr[tr_mask]; sc_tr_f = sc_tr[tr_mask]
        y_tr_f = y_tr[tr_mask];     meta_tr_f = meta_tr[tr_mask].reset_index(drop=True)
        emb_va_f = emb_tr[va_mask]; sc_va_f = sc_tr[va_mask]
        y_va_f = y_tr[va_mask];     meta_va_f = meta_tr[va_mask].reset_index(drop=True)

        print(f"\n[Fold {fold}/{n_splits}] Training ProtoSSM ...")
        torch.manual_seed(SEED); np.random.seed(SEED)
        proto_model, site2i, _ = train_light_proto_ssm(
            emb_full=emb_tr_f, scores_full=sc_tr_f, y_full=y_tr_f, meta_full=meta_tr_f,
            emb_val=emb_va_f, scores_val=sc_va_f, y_val=y_va_f, meta_val=meta_va_f,
            n_classes=n_classes,
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
            use_swa=bool(proto_cfg.get("use_swa", False)),
            swa_start_frac=float(proto_cfg.get("swa_start_frac", 1.1)),
            swa_lr=float(proto_cfg.get("swa_lr", 0.0)),
            use_cross_attn=bool(proto_cfg.get("use_cross_attn", True)),
            cross_attn_heads=int(proto_cfg.get("cross_attn_heads", 4)),
            run_tag=f"fold{fold}",
            verbose=False,
        )

        # Proto inference on val
        n_va = len(emb_va_f) // N_WINDOWS
        va_fnames_list = meta_va_f.drop_duplicates("filename")["filename"].tolist()
        va_site_ids, va_hour_ids = _fold_site_hour_ids(meta_va_f, va_fnames_list, site2i, proto_n_sites)
        proto_model.eval()
        with torch.no_grad():
            proto_va = (
                proto_model(
                    torch.tensor(emb_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                    torch.tensor(sc_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                    site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                    hours=torch.tensor(va_hour_ids, dtype=torch.long),
                ).numpy().reshape(-1, n_classes)
            )

        # Proto inference on train (no TTA for speed)
        n_tr = len(emb_tr_f) // N_WINDOWS
        tr_fnames_list = meta_tr_f.drop_duplicates("filename")["filename"].tolist()
        tr_site_ids, tr_hour_ids = _fold_site_hour_ids(meta_tr_f, tr_fnames_list, site2i, proto_n_sites)
        with torch.no_grad():
            proto_tr = (
                proto_model(
                    torch.tensor(emb_tr_f.reshape(n_tr, N_WINDOWS, -1), dtype=torch.float32),
                    torch.tensor(sc_tr_f.reshape(n_tr, N_WINDOWS, -1), dtype=torch.float32),
                    site_ids=torch.tensor(tr_site_ids, dtype=torch.long),
                    hours=torch.tensor(tr_hour_ids, dtype=torch.long),
                ).numpy().reshape(-1, n_classes)
            )

        print(f"[Fold {fold}/{n_splits}] Training probes ...")
        probe_models, emb_scaler, emb_pca = train_torch_probes(emb_tr_f, sc_tr_f, y_tr_f, mlp_cfg)

        prior_tables = build_prior_tables(meta_tr_f[["site", "hour_utc"]], y_tr_f)

        # Compute first_pass with fixed lambda/alpha/ensemble
        sc_va_prior = apply_prior(sc_va_f, meta_va_f["site"].to_numpy(),
                                  meta_va_f["hour_utc"].to_numpy(), prior_tables, lambda_prior)
        probe_logits_va = predict_torch_probe_logits(emb_va_f, sc_va_prior, probe_models, emb_scaler, emb_pca)
        sc_va_mlp = blend_probe_logits(sc_va_prior, probe_logits_va, mlp_alpha_blend)
        first_pass_va = ensemble_w * proto_va + (1.0 - ensemble_w) * sc_va_mlp

        sc_tr_prior = apply_prior(sc_tr_f, meta_tr_f["site"].to_numpy(),
                                  meta_tr_f["hour_utc"].to_numpy(), prior_tables, lambda_prior)
        probe_logits_tr = predict_torch_probe_logits(emb_tr_f, sc_tr_prior, probe_models, emb_scaler, emb_pca)
        sc_tr_mlp = blend_probe_logits(sc_tr_prior, probe_logits_tr, mlp_alpha_blend)
        first_pass_tr = ensemble_w * proto_tr + (1.0 - ensemble_w) * sc_tr_mlp

        # Calibrate thresholds on training first_pass (fixed for all residual configs)
        print(f"[Fold {fold}/{n_splits}] Calibrating thresholds ...")
        thresholds_fold = calibrate_and_optimize_thresholds(
            sigmoid(first_pass_tr), y_tr_f, threshold_grid,
        )

        # Log first_pass AUC for reference
        fp_auc = macro_auc(y_va_f, sigmoid(first_pass_va))
        print(f"[Fold {fold}/{n_splits}] first_pass val AUC = {fp_auc:.6f}")

        fold_cache.append({
            "fold": fold,
            "va_mask": va_mask,
            "y_tr_f": y_tr_f,
            "first_pass_va": first_pass_va,   # (n_va_rows, n_classes) logits
            "first_pass_tr": first_pass_tr,   # (n_tr_rows, n_classes) logits
            "emb_tr_f": emb_tr_f,
            "emb_va_f": emb_va_f,
            "tr_site_ids": tr_site_ids,
            "tr_hour_ids": tr_hour_ids,
            "va_site_ids": va_site_ids,
            "va_hour_ids": va_hour_ids,
            "thresholds_fold": thresholds_fold,
            "first_pass_auc": fp_auc,
        })
        print(f"[Fold {fold}/{n_splits}] Cache built — val files={len(va_fnames)}")

    return fold_cache


# ── Phase 2: residual config sweep ────────────────────────────────────────────

def run_residual_sweep(
    fold_cache: List[dict],
    y_tr: np.ndarray,
    n_classes: int,
    temperatures: np.ndarray,
    configs: List[dict],
    rank_power_grid: List[float],
    smooth_alpha_grid: List[float],
    verbose: bool = False,
) -> List[dict]:
    n_rows = len(y_tr)
    results = []

    for cfg in configs:
        cfg_name = str(cfg["name"])
        print(f"\n{'─'*60}")
        print(f"ResidualSSM config: {cfg_name}")
        t0 = time.time()

        # Phase A: train residual model per fold once, cache resid_probs
        per_fold_resid: List[dict] = []
        for fd in fold_cache:
            fold = fd["fold"]
            n_va = fd["emb_va_f"].shape[0] // N_WINDOWS
            print(f"  [Fold {fold}] Training ResidualSSM ...")

            res_model, corr_w = train_residual_ssm(
                emb_full=fd["emb_tr_f"],
                first_pass_flat=fd["first_pass_tr"],
                Y_full=fd["y_tr_f"],
                site_ids=fd["tr_site_ids"],
                hour_ids=fd["tr_hour_ids"],
                n_classes=n_classes,
                n_windows=N_WINDOWS,
                d_model=int(cfg.get("d_model", 64)),
                d_state=int(cfg.get("d_state", 8)),
                dropout=float(cfg.get("dropout", 0.1)),
                n_sites=int(cfg.get("n_sites", 20)),
                meta_dim=int(cfg.get("meta_dim", 8)),
                n_epochs=int(cfg.get("n_epochs", 30)),
                patience=int(cfg.get("patience", 8)),
                lr=float(cfg.get("lr", 1e-3)),
                correction_weight=float(cfg.get("correction_weight", 0.30)),
                verbose=verbose,
            )

            fp_va_seq = fd["first_pass_va"].reshape(n_va, N_WINDOWS, -1)
            res_model.eval()
            with torch.no_grad():
                va_corr = res_model(
                    torch.tensor(fd["emb_va_f"].reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                    torch.tensor(fp_va_seq, dtype=torch.float32),
                    site_ids=torch.tensor(fd["va_site_ids"], dtype=torch.long),
                    hours=torch.tensor(fd["va_hour_ids"], dtype=torch.long),
                ).numpy().reshape(-1, n_classes)

            final_scores = fd["first_pass_va"] + corr_w * va_corr
            final_scores = final_scores / temperatures[None, :]
            resid_probs = sigmoid(final_scores)

            per_fold_resid.append({
                "va_mask": fd["va_mask"],
                "resid_probs": resid_probs,
                "thresholds_fold": fd["thresholds_fold"],
            })

        # residual_plus_temp OOF AUC — invariant across (rp, sa) grid
        oof_resid = np.zeros((n_rows, n_classes), dtype=np.float32)
        for pfd in per_fold_resid:
            oof_resid[pfd["va_mask"]] = pfd["resid_probs"]
        resid_auc = round(float(macro_auc(y_tr, oof_resid)), 6)

        n_combos = len(rank_power_grid) * len(smooth_alpha_grid)
        print(
            f"  Residual trained; sweeping {n_combos} (rank_power × smooth_alpha) combos  "
            f"(residual_plus_temp OOF AUC={resid_auc:.6f})"
        )

        # Phase B: cheap sweep over (rank_power, smooth_alpha)
        cfg_results: List[dict] = []
        for rp in rank_power_grid:
            rank_per_fold = [
                rank_aware_scaling(pfd["resid_probs"], N_WINDOWS, rp)
                for pfd in per_fold_resid
            ]
            oof_rank = np.zeros((n_rows, n_classes), dtype=np.float32)
            for pfd, rank_probs in zip(per_fold_resid, rank_per_fold):
                oof_rank[pfd["va_mask"]] = rank_probs
            rank_auc = round(float(macro_auc(y_tr, oof_rank)), 6)

            for sa in smooth_alpha_grid:
                oof_smooth = np.zeros((n_rows, n_classes), dtype=np.float32)
                oof_final = np.zeros((n_rows, n_classes), dtype=np.float32)
                for pfd, rank_probs in zip(per_fold_resid, rank_per_fold):
                    smooth_probs = np.clip(
                        adaptive_delta_smooth(rank_probs, N_WINDOWS, sa), 0.0, 1.0
                    )
                    thresh_probs = apply_per_class_thresholds(
                        smooth_probs, pfd["thresholds_fold"]
                    )
                    oof_smooth[pfd["va_mask"]] = smooth_probs
                    oof_final[pfd["va_mask"]] = thresh_probs
                smooth_auc = round(float(macro_auc(y_tr, oof_smooth)), 6)
                final_auc = round(float(macro_auc(y_tr, oof_final)), 6)

                cfg_results.append({
                    "name": cfg_name,
                    "rank_power": float(rp),
                    "smooth_alpha": float(sa),
                    "metric_residual_plus_temp": resid_auc,
                    "metric_post_rank": rank_auc,
                    "metric_post_smooth": smooth_auc,
                    "metric_final_after_threshold": final_auc,
                })

        elapsed = round(time.time() - t0, 1)
        for r in cfg_results:
            r["elapsed_sec"] = elapsed
        results.extend(cfg_results)

        best = max(cfg_results, key=lambda r: r["metric_final_after_threshold"])
        print(
            f"[{cfg_name}] best  rp={best['rank_power']:.2f}  sa={best['smooth_alpha']:.2f}"
            f"  final={best['metric_final_after_threshold']:.6f}  ({elapsed:.1f}s)"
        )

    results.sort(key=lambda r: r["metric_final_after_threshold"], reverse=True)
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    base = Path(".")
    p = argparse.ArgumentParser(description="Sweep ResidualSSM configs.")
    p.add_argument("--cache-meta",       type=Path, default=base / "data" / "perch_cache_finetuned" / "full_perch_meta.parquet")
    p.add_argument("--cache-npz",        type=Path, default=base / "data" / "perch_cache_finetuned" / "full_perch_arrays.npz")
    p.add_argument("--sample-sub",       type=Path, default=base / "data" / "sample_submission.csv")
    p.add_argument("--sound-labels",     type=Path, default=base / "data" / "train_soundscapes_labels.csv")
    p.add_argument("--taxonomy",         type=Path, default=base / "data" / "taxonomy.csv")
    p.add_argument("--configs-py",       type=Path, default=base / "birdclef_example" / "residual_ssm_sweep_configs.py")
    p.add_argument("--output-json",      type=Path, default=base / "outputs" / "residual_ssm_sweep" / "results.json")
    p.add_argument("--output-csv",       type=Path, default=base / "outputs" / "residual_ssm_sweep" / "results.csv")
    p.add_argument("--n-splits",         type=int, default=5)
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


def results_to_table(results: List[dict]) -> pd.DataFrame:
    cols = [
        "name", "rank_power", "smooth_alpha",
        "metric_residual_plus_temp", "metric_post_rank",
        "metric_post_smooth", "metric_final_after_threshold",
        "elapsed_sec",
    ]
    rows = [{k: r.get(k) for k in cols} for r in results]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    for path in [args.cache_meta, args.cache_npz, args.sample_sub, args.sound_labels]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    configs = load_residual_configs(args.configs_py)
    print(f"Loaded {len(configs)} residual SSM configs")

    emb_tr, sc_tr, y_tr, meta_tr, n_classes, primary_labels = build_training_data(
        args.cache_meta, args.cache_npz, args.sample_sub, args.sound_labels,
    )
    print(f"Data: emb={emb_tr.shape}  scores={sc_tr.shape}  y={y_tr.shape}  classes={n_classes}")

    temperatures = (
        build_temperatures(primary_labels, args.taxonomy)
        if args.taxonomy.exists()
        else np.ones(n_classes, dtype=np.float32)
    )

    proto_cfg = CFG["proto_ssm_train"]
    mlp_cfg = CFG["mlp_params"]
    threshold_grid = CFG["threshold_grid"]
    lambda_prior = float(CFG["lambda_prior"])
    mlp_alpha_blend = float(mlp_cfg.get("alpha_blend", 0.15))
    ensemble_w = float(CFG["ensemble_w"])

    print(f"\n{'='*60}")
    print("Phase 1: building per-fold cache (SSM + probes + first_pass) ...")
    fold_cache = build_fold_cache(
        emb_tr, sc_tr, y_tr, meta_tr, n_classes, temperatures,
        proto_cfg, mlp_cfg,
        lambda_prior=lambda_prior,
        mlp_alpha_blend=mlp_alpha_blend,
        ensemble_w=ensemble_w,
        threshold_grid=threshold_grid,
        n_splits=args.n_splits,
    )

    oof_fp = np.zeros((len(y_tr), n_classes), dtype=np.float32)
    for fd in fold_cache:
        oof_fp[fd["va_mask"]] = sigmoid(fd["first_pass_va"])
    fp_oof_auc = macro_auc(y_tr, oof_fp)
    per_fold_str = "  ".join(f"f{fd['fold']}={fd['first_pass_auc']:.6f}" for fd in fold_cache)
    print(f"\nPhase 1 first_pass OOF AUC = {fp_oof_auc:.6f}  ({per_fold_str})")

    print(f"\n{'='*60}")
    print(
        f"Phase 2: sweeping {len(configs)} residual SSM configs × "
        f"{len(RANK_POWER_GRID)} rank_power × {len(SMOOTH_ALPHA_GRID)} smooth_alpha ..."
    )
    results = run_residual_sweep(
        fold_cache, y_tr, n_classes, temperatures, configs,
        rank_power_grid=RANK_POWER_GRID,
        smooth_alpha_grid=SMOOTH_ALPHA_GRID,
        verbose=args.verbose,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    results_to_table(results).to_csv(args.output_csv, index=False, float_format="%.6f")

    print(f"\n{'='*60}")
    print(f"Saved {len(results)} results → {args.output_json}")
    top_n = min(20, len(results))
    print(f"\nTop {top_n} (by final_after_threshold OOF AUC):")
    for rank, r in enumerate(results[:top_n], 1):
        print(
            f"  {rank:2d}. {r['name']:35s}"
            f"  rp={r['rank_power']:.2f}  sa={r['smooth_alpha']:.2f}"
            f"  resid={r['metric_residual_plus_temp']:.6f}"
            f"  rank={r['metric_post_rank']:.6f}"
            f"  smooth={r['metric_post_smooth']:.6f}"
            f"  final={r['metric_final_after_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
