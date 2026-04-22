#!/usr/bin/env python3

"""Standalone LightProtoSSM trainer with configurable sweeps.

Ports the core `train_light_proto_ssm` logic from `sota_oof.py` into a separate
script so you can test multiple architecture/training configs quickly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


N_WINDOWS = 12


def union_labels(series: pd.Series) -> List[str]:
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        a = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_model, -1)
        )
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, dim = x.shape
        xz = self.in_proj(x)
        x_ssm, _z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :timesteps].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        a = -torch.exp(self.A_log)
        b = self.B_proj(x_conv)
        c = self.C_proj(x_conv)
        h = torch.zeros(bsz, dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(timesteps):
            dA = torch.exp(a[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * b[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * c[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_state: int,
        n_classes: int,
        n_windows: int,
        dropout: float,
        n_sites: int,
        meta_dim: int,
        n_ssm_layers: int,
        use_cross_attn: bool,
        cross_attn_heads: int,
    ):
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

        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(self.n_ssm_layers)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(self.n_ssm_layers)])
        self.ssm_merge = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(self.n_ssm_layers)]
        )
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_ssm_layers)])
        self.drop = nn.Dropout(dropout)

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
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_ssm_layers)])

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> None:
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _b, timesteps, _d = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :timesteps, :]
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
            return alpha * sim + (1.0 - alpha) * perch_logits
        return sim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_light_proto_ssm(
    emb_full: np.ndarray,
    scores_full: np.ndarray,
    y_full: np.ndarray,
    meta_full: pd.DataFrame,
    emb_val: np.ndarray | None,
    scores_val: np.ndarray | None,
    y_val: np.ndarray | None,
    meta_val: pd.DataFrame | None,
    n_classes: int,
    n_epochs: int = 40,
    patience: int = 8,
    lr: float = 1e-3,
    n_sites: int = 20,
    d_model: int = 128,
    d_state: int = 16,
    n_ssm_layers: int = 2,
    dropout: float = 0.15,
    meta_dim: int = 16,
    distill_weight: float = 0.15,
    pos_weight_cap: float = 25.0,
    use_swa: bool = True,
    swa_start_frac: float = 0.65,
    swa_lr: float = 4e-4,
    use_cross_attn: bool = True,
    cross_attn_heads: int = 2,
    run_tag: str | None = None,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, int], float]:
    n_files = len(emb_full) // N_WINDOWS
    emb_f = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f = y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fnames = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].dropna().astype(str).unique())
    site2i = {s: i + 1 for i, s in enumerate(sites_u)}
    site_ids = np.array(
        [
            min(site2i.get(str(meta_full.loc[meta_full["filename"] == fn, "site"].iloc[0]), 0), n_sites - 1)
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
        n_classes=n_classes,
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
        torch.tensor(y_full, dtype=torch.float32),
    )

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
        emb_val_f = emb_val.reshape(n_val_files, N_WINDOWS, -1)
        log_val_f = scores_val.reshape(n_val_files, N_WINDOWS, -1)
        lab_val_f = y_val.reshape(n_val_files, N_WINDOWS, -1).astype(np.float32)

        fnames_val = meta_val["filename"].unique()
        site_ids_val = np.array(
            [
                min(site2i.get(str(meta_val.loc[meta_val["filename"] == fn, "site"].iloc[0]), 0), n_sites - 1)
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

        emb_val_t = torch.tensor(emb_val_f, dtype=torch.float32)
        log_val_t = torch.tensor(log_val_f, dtype=torch.float32)
        lab_val_t = torch.tensor(lab_val_f, dtype=torch.float32)
        site_val_t = torch.tensor(site_ids_val, dtype=torch.long)
        hour_val_t = torch.tensor(hour_ids_val, dtype=torch.long)

    pos_cnt = lab_t.sum(dim=(0, 1))
    total = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1.0)).clamp(max=float(pos_weight_cap))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_auc = float("-inf")
    best_state = None
    wait = 0

    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start = int(n_epochs * swa_start_frac)
        swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=swa_lr)
    else:
        swa_model = None
        swa_start = n_epochs + 1
        swa_sched = None

    for ep in range(n_epochs):
        model.train()
        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
        train_loss = F.binary_cross_entropy_with_logits(
            out, lab_t, pos_weight=pos_weight[None, None, :]
        ) + distill_weight * F.mse_loss(out, log_t)
        opt.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if use_swa and ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        train_loss_val = float(train_loss.item())

        model.eval()
        if has_val:
            with torch.no_grad():
                out_val = model(emb_val_t, log_val_t, site_ids=site_val_t, hours=hour_val_t)
                val_loss = F.binary_cross_entropy_with_logits(
                    out_val, lab_val_t, pos_weight=pos_weight[None, None, :]
                ) + distill_weight * F.mse_loss(out_val, log_val_t)
                val_loss_val = float(val_loss.item())
                val_probs = sigmoid(out_val.detach().cpu().numpy().reshape(-1, n_classes))
                val_auc = macro_auc(y_val, val_probs)
        else:
            val_loss_val = float("nan")
            val_auc = float("nan")

        if np.isfinite(val_auc):
            monitor_auc = float(val_auc)
        else:
            monitor_auc = -train_loss_val

        if monitor_auc > best_val_auc:
            best_val_auc = monitor_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        prefix = f"[{run_tag}] " if run_tag else ""
        if (ep + 1) % 5 == 0:
            print(
                f"{prefix}epoch={ep + 1:03d}/{n_epochs:03d} "
                f"train_loss={train_loss_val:.5f} val_loss={val_loss_val:.5f} "
                f"val_auc={val_auc:.6f} best_val_auc={best_val_auc:.6f} wait={wait}"
            )

        if wait >= patience:
            print(f"{prefix}Early stop at epoch {ep + 1}")
            break

    if use_swa and ep >= swa_start:
        # Same behavior as original; kept intentionally.
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)

    model.eval()
    return model, site2i, best_val_auc


def build_training_data(
    cache_meta_path: Path,
    cache_npz_path: Path,
    sample_submission_path: Path,
    sound_labels_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    sample_sub = pd.read_csv(sample_submission_path)
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    soundscape_labels = pd.read_csv(sound_labels_path)
    sc = (
        soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
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

    windows_per_file = sc.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    full_rows = sc[sc["filename"].isin(full_files)].sort_values(["filename", "end_sec"]).reset_index(drop=False)

    meta_tr = pd.read_parquet(cache_meta_path)
    arr = np.load(cache_npz_path)
    sc_tr = arr["scores_full_raw"].astype(np.float32)
    emb_tr = arr["emb_full"].astype(np.float32)

    y_full_aligned = y_sc[
        full_rows.set_index("row_id").loc[meta_tr["row_id"], "index"].to_numpy()
    ]
    return emb_tr, sc_tr, y_full_aligned.astype(np.float32), meta_tr, n_classes


def load_configs_from_python(configs_py: Path) -> List[Dict[str, object]]:
    if not configs_py.exists():
        raise FileNotFoundError(f"Missing configs python file: {configs_py}")

    spec = importlib.util.spec_from_file_location("proto_ssm_sweep_configs", str(configs_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import configs module from {configs_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PROTO_SSM_SWEEP_CONFIGS"):
        raise AttributeError(
            f"{configs_py} must define PROTO_SSM_SWEEP_CONFIGS (list of dicts)"
        )

    configs = getattr(module, "PROTO_SSM_SWEEP_CONFIGS")
    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("PROTO_SSM_SWEEP_CONFIGS must be a non-empty list")
    for i, cfg in enumerate(configs):
        if not isinstance(cfg, dict):
            raise TypeError(f"Config at index {i} is not a dict")
        if "name" not in cfg:
            raise ValueError(f"Config at index {i} is missing required field 'name'")
    return configs


def run_config_sweep(
    emb_tr: np.ndarray,
    sc_tr: np.ndarray,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    n_classes: int,
    configs: Sequence[Dict[str, object]],
    n_splits: int,
    verbose: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    groups = meta_tr["filename"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    for cfg in configs:
        cfg_name = str(cfg.get("name", "unnamed"))
        t0 = time.time()
        oof_scores = np.zeros((len(sc_tr), n_classes), dtype=np.float32)
        fold_best_val_aucs: List[float] = []
        param_counts: List[int] = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(sc_tr, groups=groups), 1):
            emb_fold_tr = emb_tr[tr_idx]
            sc_fold_tr = sc_tr[tr_idx]
            y_fold_tr = y_tr[tr_idx]
            meta_fold_tr = meta_tr.iloc[tr_idx].reset_index(drop=True)

            emb_fold_va = emb_tr[va_idx]
            sc_fold_va = sc_tr[va_idx]
            meta_fold_va = meta_tr.iloc[va_idx].reset_index(drop=True)

            model, site2i, best_val_auc = train_light_proto_ssm(
                emb_full=emb_fold_tr,
                scores_full=sc_fold_tr,
                y_full=y_fold_tr,
                meta_full=meta_fold_tr,
                emb_val=emb_fold_va,
                scores_val=sc_fold_va,
                y_val=y_tr[va_idx],
                meta_val=meta_fold_va,
                n_classes=n_classes,
                n_epochs=int(cfg.get("n_epochs", 40)),
                patience=int(cfg.get("patience", 8)),
                lr=float(cfg.get("lr", 1e-3)),
                n_sites=int(cfg.get("n_sites", 20)),
                d_model=int(cfg.get("d_model", 128)),
                d_state=int(cfg.get("d_state", 16)),
                n_ssm_layers=int(cfg.get("n_ssm_layers", 2)),
                dropout=float(cfg.get("dropout", 0.15)),
                meta_dim=int(cfg.get("meta_dim", 16)),
                distill_weight=float(cfg.get("distill_weight", 0.15)),
                pos_weight_cap=float(cfg.get("pos_weight_cap", 25.0)),
                use_swa=bool(cfg.get("use_swa", True)),
                swa_start_frac=float(cfg.get("swa_start_frac", 0.65)),
                swa_lr=float(cfg.get("swa_lr", 4e-4)),
                use_cross_attn=bool(cfg.get("use_cross_attn", True)),
                cross_attn_heads=int(cfg.get("cross_attn_heads", 2)),
                run_tag=f"{cfg_name} fold {fold}",
                verbose=verbose,
            )
            fold_best_val_aucs.append(float(best_val_auc))
            param_counts.append(int(count_trainable_params(model)))

            n_va_files = len(emb_fold_va) // N_WINDOWS
            va_fnames = meta_fold_va.drop_duplicates("filename")["filename"].tolist()
            va_site_ids = np.array(
                [
                    min(
                        site2i.get(str(meta_fold_va.loc[meta_fold_va["filename"] == fn, "site"].iloc[0]), 0),
                        int(cfg.get("n_sites", 20)) - 1,
                    )
                    for fn in va_fnames
                ],
                dtype=np.int64,
            )
            va_hour_ids = np.array(
                [
                    int(meta_fold_va.loc[meta_fold_va["filename"] == fn, "hour_utc"].iloc[0]) % 24
                    for fn in va_fnames
                ],
                dtype=np.int64,
            )

            model.eval()
            with torch.no_grad():
                out = model(
                    torch.tensor(emb_fold_va.reshape(n_va_files, N_WINDOWS, -1), dtype=torch.float32),
                    torch.tensor(sc_fold_va.reshape(n_va_files, N_WINDOWS, -1), dtype=torch.float32),
                    site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                    hours=torch.tensor(va_hour_ids, dtype=torch.long),
                ).numpy().reshape(-1, n_classes)

            probs_va = sigmoid(out)
            oof_scores[va_idx] = probs_va
            fold_auc = macro_auc(y_tr[va_idx], probs_va)
            print(
                f"[{cfg_name}] fold {fold}/{n_splits} auc={fold_auc:.6f} "
                f"best_val_auc={best_val_auc:.6f}"
            )

        oof_auc = macro_auc(y_tr, oof_scores)
        elapsed = time.time() - t0

        result = {
            "name": cfg_name,
            "oof_macro_auc": float(oof_auc),
            "mean_best_val_auc": float(np.mean(fold_best_val_aucs)) if fold_best_val_aucs else float("nan"),
            "fold_best_val_aucs": [float(x) for x in fold_best_val_aucs],
            "elapsed_sec": float(elapsed),
            "params": int(np.max(param_counts)) if param_counts else 0,
            "config": dict(cfg),
        }
        results.append(result)
        print(
            f"[{cfg_name}] oof_auc={oof_auc:.6f} mean_best_val_auc={result['mean_best_val_auc']:.6f} "
            f"params={result['params']:,} time={elapsed:.1f}s"
        )

    results.sort(key=lambda r: r["oof_macro_auc"], reverse=True)
    return results


def parse_args() -> argparse.Namespace:
    base = Path(".")
    parser = argparse.ArgumentParser(description="Sweep LightProtoSSM configs on cached training arrays.")
    parser.add_argument("--cache-meta", type=Path, default=base / "data" / "perch_cache_finetuned" / "full_perch_meta.parquet")
    parser.add_argument("--cache-npz", type=Path, default=base / "data" / "perch_cache_finetuned" / "full_perch_arrays.npz")
    parser.add_argument("--sample-submission", type=Path, default=base / "data" / "sample_submission.csv")
    parser.add_argument("--sound-labels", type=Path, default=base / "data" / "train_soundscapes_labels.csv")
    parser.add_argument(
        "--configs-py",
        type=Path,
        default=base / "birdclef_example" / "proto_ssm_sweep_configs.py",
        help="Python file defining PROTO_SSM_SWEEP_CONFIGS.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=base / "outputs" / "experiments_ssm" / "proto_ssm_sweep_results.json")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=base / "outputs" / "experiments_ssm" / "proto_ssm_sweep_results.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def results_to_table(results: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in results:
        cfg = dict(r.get("config", {}))
        row: Dict[str, object] = {
            "name": r.get("name", ""),
            "oof_macro_auc": r.get("oof_macro_auc", float("nan")),
            "mean_best_val_auc": r.get("mean_best_val_auc", float("nan")),
            "elapsed_sec": r.get("elapsed_sec", float("nan")),
            "params": r.get("params", 0),
            "fold_best_val_aucs": ";".join(
                f"{float(x):.6f}" for x in r.get("fold_best_val_aucs", [])
            ),
            "n_folds": len(r.get("fold_best_val_aucs", [])),
        }
        for k, v in cfg.items():
            row[f"cfg_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


SEED = 1337


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args = parse_args()
    for p in [args.cache_meta, args.cache_npz, args.sample_submission, args.sound_labels]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")

    configs = load_configs_from_python(args.configs_py)

    emb_tr, sc_tr, y_tr, meta_tr, n_classes = build_training_data(
        cache_meta_path=args.cache_meta,
        cache_npz_path=args.cache_npz,
        sample_submission_path=args.sample_submission,
        sound_labels_path=args.sound_labels,
    )

    print(f"Loaded: emb={emb_tr.shape} scores={sc_tr.shape} y={y_tr.shape}")
    print(f"Running {len(configs)} configs with {args.n_splits}-fold GroupKFold OOF...")

    results = run_config_sweep(
        emb_tr=emb_tr,
        sc_tr=sc_tr,
        y_tr=y_tr,
        meta_tr=meta_tr,
        n_classes=n_classes,
        configs=configs,
        n_splits=args.n_splits,
        verbose=args.verbose,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    results_to_table(results).to_csv(args.output_csv, index=False)
    print(f"Saved results: {args.output_json}")
    print(f"Saved results: {args.output_csv}")
    if len(results) > 0:
        best = results[0]
        print(
            f"Best: {best['name']} oof_auc={best['oof_macro_auc']:.6f} "
            f"mean_best_val_auc={best['mean_best_val_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
