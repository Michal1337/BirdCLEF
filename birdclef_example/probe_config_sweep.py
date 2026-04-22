#!/usr/bin/env python3

"""Standalone torch probe training/config sweep with 5-fold GroupKFold OOF.

Ports the probe path from sota_oof_cache.py into an isolated script so you can
benchmark alternative probe settings quickly.

This script intentionally keeps only the PyTorch probe path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from birdclef_example.probe_sweep_configs import (
        PROBE_SWEEP_CONFIGS as DEFAULT_PROBE_SWEEP_CONFIGS,
    )
except ModuleNotFoundError:
    from probe_sweep_configs import PROBE_SWEEP_CONFIGS as DEFAULT_PROBE_SWEEP_CONFIGS

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
    if keep.sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def build_class_freq_weights(y: np.ndarray, cap: float = 10.0) -> np.ndarray:
    total = y.shape[0]
    pos_count = y.sum(axis=0).astype(np.float32) + 1.0
    freq = pos_count / total
    weights = 1.0 / (freq**0.5)
    weights = np.clip(weights, 1.0, cap)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def build_sequential_features(scores_col: np.ndarray, n_windows: int = N_WINDOWS):
    n = len(scores_col)
    if n % n_windows != 0:
        raise ValueError(f"Expected rows multiple of {n_windows}, got {n}")
    x = scores_col.reshape(-1, n_windows)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean = np.repeat(x.mean(axis=1), n_windows)
    max_ = np.repeat(x.max(axis=1), n_windows)
    std = np.repeat(x.std(axis=1), n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def load_cache(meta_path: Path, npz_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_df = pd.read_parquet(meta_path)
    arr = np.load(npz_path)

    score_key = None
    for k in ("scores_full_raw", "scores_full", "scores"):
        if k in arr:
            score_key = k
            break
    if score_key is None:
        raise KeyError(f"Could not find scores key in {npz_path}. Found: {list(arr.keys())}")

    emb_key = None
    for k in ("emb_full", "embeddings", "emb"):
        if k in arr:
            emb_key = k
            break
    if emb_key is None:
        raise KeyError(f"Could not find embedding key in {npz_path}. Found: {list(arr.keys())}")

    scores = arr[score_key].astype(np.float32)
    emb = arr[emb_key].astype(np.float32)
    return meta_df, scores, emb


def build_aligned_targets(
    meta_df: pd.DataFrame,
    sample_submission_path: Path,
    soundscape_labels_path: Path,
) -> Tuple[np.ndarray, List[str]]:
    sample_sub = pd.read_csv(sample_submission_path)
    soundscape_labels = pd.read_csv(soundscape_labels_path)

    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

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

    y_sc = np.zeros((len(sc), len(primary_labels)), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                y_sc[i, label_to_idx[lbl]] = 1

    windows_per_file = sc.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    full_rows = (
        sc[sc["filename"].isin(full_files)]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )

    if "row_id" not in meta_df.columns:
        raise KeyError("meta parquet must include row_id for target alignment")

    idx_map = full_rows.set_index("row_id")["index"]
    missing = [r for r in meta_df["row_id"].tolist() if r not in idx_map.index]
    if missing:
        raise ValueError(
            f"{len(missing)} cache row_id values not present in labels; first missing: {missing[0]}"
        )

    aligned_idx = idx_map.loc[meta_df["row_id"]].to_numpy()
    y_aligned = y_sc[aligned_idx]
    return y_aligned, primary_labels


def fit_embedding_features(
    emb: np.ndarray,
    pca_dim: int,
) -> Tuple[StandardScaler, PCA | None, np.ndarray]:
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb).astype(np.float32)
    if pca_dim <= 0:
        return scaler, None, emb_s
    n_components = min(int(pca_dim), emb_s.shape[1] - 1)
    if n_components <= 0:
        return scaler, None, emb_s
    pca = PCA(n_components=n_components)
    z = pca.fit_transform(emb_s).astype(np.float32)
    return scaler, pca, z


def transform_embedding_features(
    emb: np.ndarray,
    scaler: StandardScaler,
    pca: PCA | None,
) -> np.ndarray:
    emb_s = scaler.transform(emb).astype(np.float32)
    if pca is None:
        return emb_s
    return pca.transform(emb_s).astype(np.float32)


class TorchProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [1]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch_probes(
    emb: np.ndarray,
    scores_raw: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
):
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
    seed = int(cfg.get("seed", 1337))

    scaler, pca, z = fit_embedding_features(emb, pca_dim)

    class_weights = build_class_freq_weights(y, cap=10.0)
    active = np.where(y.sum(axis=0) >= min_pos)[0]

    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    probe_models: Dict[int, Dict[str, Any]] = {}

    for ci in active:
        yc = y[:, ci].astype(np.float32)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        x = np.hstack(
            [
                z,
                scores_raw[:, ci : ci + 1],
                prev[:, None],
                next_[:, None],
                mean[:, None],
                max_[:, None],
                std[:, None],
            ]
        ).astype(np.float32)

        n_pos = int(yc.sum())
        n_neg = len(yc) - n_pos
        pos_idx = np.where(yc == 1)[0]

        w = float(class_weights[ci])
        repeat = max(1, int(round(w * n_neg / max(n_pos, 1))))
        repeat = min(repeat, 8)
        if n_pos * repeat + len(yc) > max_rows:
            repeat = max(1, (max_rows - len(yc)) // max(n_pos, 1))

        x_bal = np.vstack([x, np.tile(x[pos_idx], (repeat, 1))]).astype(np.float32)
        y_bal = np.concatenate([yc, np.ones(n_pos * repeat, dtype=np.float32)])

        n = len(y_bal)
        n_val = max(1, int(round(n * val_fraction)))
        perm = rng.permutation(n)
        va_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        x_tr = x_bal[tr_idx]
        y_tr = y_bal[tr_idx]
        x_va = x_bal[va_idx]
        y_va = y_bal[va_idx]

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

        best_val = float("inf")
        best_state = None
        wait = 0

        for _ep in range(epochs):
            model.train()
            order = torch.randperm(x_tr_t.shape[0], device=device)
            for start in range(0, x_tr_t.shape[0], batch_size):
                idx = order[start : start + batch_size]
                logits = model(x_tr_t[idx])
                loss = criterion(logits, y_tr_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                va_logits = model(x_va_t)
                val_loss = criterion(va_logits, y_va_t).item()

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

        probe_models[int(ci)] = {
            "model": model.cpu(),
            "mu": mu.astype(np.float32),
            "sd": sd.astype(np.float32),
        }

    return probe_models, scaler, pca


def predict_torch_probe_logits(
    emb_test: np.ndarray,
    scores_test: np.ndarray,
    probe_models: Dict[int, Dict[str, Any]],
    scaler: StandardScaler,
    pca: PCA | None,
) -> np.ndarray:
    z_test = transform_embedding_features(emb_test, scaler, pca)
    probe_logits = scores_test.copy()

    for ci, payload in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        x_test = np.hstack(
            [
                z_test,
                scores_test[:, ci : ci + 1],
                prev[:, None],
                next_[:, None],
                mean[:, None],
                max_[:, None],
                std[:, None],
            ]
        ).astype(np.float32)

        mu = payload["mu"]
        sd = payload["sd"]
        x_test = (x_test - mu) / sd

        model = payload["model"]
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32)).numpy().astype(np.float32)

        probe_logits[:, ci] = logits

    return probe_logits


def blend_probe_logits(
    base_scores: np.ndarray,
    probe_logits: np.ndarray,
    alpha_blend: float,
) -> np.ndarray:
    return (1.0 - alpha_blend) * base_scores + alpha_blend * probe_logits


def resolve_alpha_blends(cfg: Dict[str, Any]) -> List[float]:
    if "alpha_blends" in cfg and isinstance(cfg["alpha_blends"], (list, tuple)):
        vals = [float(x) for x in cfg["alpha_blends"]]
    else:
        vals = [float(cfg.get("alpha_blend", 0.4))]
    vals = [float(np.clip(v, 0.0, 1.0)) for v in vals]
    return sorted(set(vals))


def run_oof_probe_config(
    cfg: Dict[str, Any],
    emb: np.ndarray,
    scores_raw: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    n_splits: int,
) -> List[Dict[str, Any]]:
    method = str(cfg.get("method", "torch")).lower()
    cfg_name = str(cfg.get("name", "unnamed"))
    if method != "torch":
        raise ValueError(
            f"Config '{cfg_name}' has method={method}; only 'torch' is supported."
        )

    if "filename" not in meta_df.columns:
        raise KeyError("meta parquet must include filename for GroupKFold")

    groups = meta_df["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    alpha_blends = resolve_alpha_blends(cfg)
    oof_probs_by_alpha = {
        alpha: np.zeros_like(scores_raw, dtype=np.float32) for alpha in alpha_blends
    }
    fold_scores_by_alpha = {alpha: [] for alpha in alpha_blends}

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores_raw, groups=groups), 1):
        emb_tr, emb_va = emb[tr_idx], emb[va_idx]
        sc_tr, sc_va = scores_raw[tr_idx], scores_raw[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        probe_models, emb_scaler, emb_pca = train_torch_probes(
            emb_tr,
            sc_tr,
            y_tr,
            cfg,
        )
        probe_logits_va = predict_torch_probe_logits(
            emb_va,
            sc_va,
            probe_models,
            emb_scaler,
            emb_pca,
        )

        fold_log = []
        for alpha in alpha_blends:
            logits_va = blend_probe_logits(sc_va, probe_logits_va, alpha)
            probs_va = sigmoid(logits_va)
            oof_probs_by_alpha[alpha][va_idx] = probs_va
            fold_auc = macro_auc(y_va, probs_va)
            fold_scores_by_alpha[alpha].append(fold_auc)
            fold_log.append(f"a={alpha:.2f}:{fold_auc:.6f}")

        print(
            f"[{cfg_name}] fold {fold}/{n_splits} method={method} rows={len(va_idx)} "
            + " | ".join(fold_log)
        )

    elapsed = time.time() - t0
    results: List[Dict[str, Any]] = []
    for alpha in alpha_blends:
        overall_auc = macro_auc(y, oof_probs_by_alpha[alpha])
        alpha_tag = f"ab{int(round(alpha * 100)):03d}"
        cfg_eval = dict(cfg)
        cfg_eval["alpha_blend_eval"] = float(alpha)
        results.append(
            {
                "config_name": f"{cfg_name}_{alpha_tag}",
                "base_config_name": cfg_name,
                "method": method,
                "alpha_blend": float(alpha),
                "oof_auc": float(overall_auc),
                "fold_auc_mean": float(np.nanmean(fold_scores_by_alpha[alpha])),
                "fold_auc_std": float(np.nanstd(fold_scores_by_alpha[alpha])),
                "fold_auc": [float(x) for x in fold_scores_by_alpha[alpha]],
                "runtime_sec": float(elapsed),
                "config": cfg_eval,
            }
        )

    return results


def load_probe_configs(config_path: Path | None) -> List[Dict[str, Any]]:
    if config_path is None:
        return list(DEFAULT_PROBE_SWEEP_CONFIGS)

    spec = importlib.util.spec_from_file_location("probe_sweep_cfg", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PROBE_SWEEP_CONFIGS"):
        raise AttributeError(f"{config_path} must define PROBE_SWEEP_CONFIGS")
    return list(module.PROBE_SWEEP_CONFIGS)


def export_results(results: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda r: r["oof_auc"], reverse=True)

    json_path = out_dir / "probe_oof_sweep_results.json"
    csv_path = out_dir / "probe_oof_sweep_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(ranked, 1):
        flat = {
            "rank": rank,
            "config_name": row["config_name"],
            "oof_auc": round(float(row["oof_auc"]), 6),
            "fold_auc_mean": round(float(row["fold_auc_mean"]), 6),
            "fold_auc_std": round(float(row["fold_auc_std"]), 6),
            "runtime_sec": round(float(row["runtime_sec"]), 2),
        }
        rows.append(flat)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path, ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OOF config sweep")
    parser.add_argument(
        "--cache-meta",
        type=Path,
        default=Path("data/perch_cache_finetuned/full_perch_meta.parquet"),
    )
    parser.add_argument(
        "--cache-npz",
        type=Path,
        default=Path("data/perch_cache_finetuned/full_perch_arrays.npz"),
    )
    parser.add_argument(
        "--sample-sub",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument(
        "--soundscape-labels",
        type=Path,
        default=Path("data/train_soundscapes_labels.csv"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional custom python config file defining PROBE_SWEEP_CONFIGS",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/probe_sweep"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="If >0, run only first N configs",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Loading cache arrays...")
    meta_df, scores_raw, emb = load_cache(args.cache_meta, args.cache_npz)

    print("Building aligned targets...")
    y_aligned, primary_labels = build_aligned_targets(
        meta_df,
        args.sample_sub,
        args.soundscape_labels,
    )

    if not (len(meta_df) == len(scores_raw) == len(emb) == len(y_aligned)):
        raise ValueError("Length mismatch between meta/scores/emb/y")

    configs = load_probe_configs(args.config_path)
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    print(
        f"Rows={len(meta_df)} Classes={len(primary_labels)} Configs={len(configs)} Splits={args.n_splits}"
    )

    results = []
    for i, cfg in enumerate(configs, 1):
        cfg = dict(cfg)
        cfg.setdefault("seed", args.seed)
        cfg.setdefault("method", "torch")
        name = cfg.get("name", f"cfg_{i:03d}")
        method = cfg.get("method", "torch")
        print(f"\n[{i}/{len(configs)}] {name} ({method})")

        run_results = run_oof_probe_config(
            cfg=cfg,
            emb=emb,
            scores_raw=scores_raw,
            y=y_aligned,
            meta_df=meta_df,
            n_splits=args.n_splits,
        )
        for run_result in run_results:
            print(
                f"[{run_result['config_name']}] OOF macro-AUC={run_result['oof_auc']:.6f} "
                f"(fold mean={run_result['fold_auc_mean']:.6f})"
            )
            results.append(run_result)

        json_path, csv_path, ranked = export_results(results, args.out_dir)
        print(
            f"Saved partial results ({len(results)}/{len(configs)}): {json_path} and {csv_path}"
        )
        print(
            f"Current best: {ranked[0]['config_name']} ({ranked[0]['method']}) AUC={ranked[0]['oof_auc']:.6f}"
        )

    json_path, csv_path, ranked = export_results(results, args.out_dir)
    print("\nSweep complete")
    print(f"Best config: {ranked[0]['config_name']} ({ranked[0]['method']})")
    print(f"Best OOF macro-AUC: {ranked[0]['oof_auc']:.6f}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
