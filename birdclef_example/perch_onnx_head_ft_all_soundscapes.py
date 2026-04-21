#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import parse_primary_labels  # noqa: E402


SAMPLE_RATE = 32000
WINDOW_SEC = 5
N_WINDOWS = 12
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
FILE_SAMPLES = SAMPLE_RATE * 60
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SOUNDSCAPE_DIR = DEFAULT_DATA_DIR / "train_soundscapes"
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "perch_onnx" / "perch_v2.onnx"
DEFAULT_PERCH_LABELS = REPO_ROOT / "models" / "perch_onnx" / "labels.csv"
DEFAULT_PERCH_LABELS_FALLBACK = REPO_ROOT / "models" / "perch_v2_cpu" / "1" / "assets" / "labels.csv"
DEFAULT_SAVE_PATH = REPO_ROOT / "outputs" / "experiments_ft" / "perch_all_soundscapes_head.pt"
DEFAULT_METRICS_JSON = REPO_ROOT / "outputs" / "experiments_ft" / "perch_all_soundscapes_metrics.json"
DEFAULT_NO_SIGNAL_CSV = REPO_ROOT / "outputs" / "experiments_ft" / "perch_all_soundscapes_no_signal_per_class.csv"
DEFAULT_SWEEP_CSV = REPO_ROOT / "outputs" / "experiments_ft" / "perch_all_soundscapes_sweep.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def union_labels(series: pd.Series) -> List[str]:
    labels = set()
    for value in series:
        labels.update(parse_primary_labels(value))
    return sorted(labels)


def read_soundscape_60s(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SAMPLE_RATE}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32, copy=False)


def build_truth_windows(sound_labels_csv: Path, soundscape_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    soundscape_labels = pd.read_csv(sound_labels_csv)
    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

    grouped = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(int)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)

    grouped = grouped[grouped["filename"].map(lambda x: (soundscape_dir / x).exists())].copy()
    grouped = grouped.sort_values(["filename", "end_sec"]).reset_index(drop=True)

    available_files = sorted(grouped["filename"].drop_duplicates().tolist())
    return grouped, available_files


def load_perch_labels(perch_labels_csv: Path) -> pd.DataFrame:
    perch_labels = pd.read_csv(perch_labels_csv).reset_index().rename(columns={"index": "perch_index"})
    if "scientific_name" not in perch_labels.columns:
        if "inat2024_fsd50k" in perch_labels.columns:
            perch_labels = perch_labels.rename(columns={"inat2024_fsd50k": "scientific_name"})
        else:
            raise ValueError(
                f"Perch labels CSV {perch_labels_csv} is missing scientific_name/inat2024_fsd50k column."
            )
    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)
    perch_labels["perch_index"] = perch_labels["perch_index"].astype(int)
    return perch_labels[["perch_index", "scientific_name"]]


def build_no_signal_labels(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> List[str]:
    no_label_index = len(perch_labels_df)
    mapping = taxonomy_df.merge(perch_labels_df, on="scientific_name", how="left")
    mapping["perch_index"] = mapping["perch_index"].fillna(no_label_index).astype(int)

    label_to_perch = mapping.set_index("primary_label")["perch_index"].to_dict()
    perch_indices = np.array([int(label_to_perch.get(label, no_label_index)) for label in primary_labels], dtype=np.int32)
    mapped_mask = perch_indices != no_label_index
    unmapped_positions = np.where(~mapped_mask)[0].astype(np.int32)

    class_name_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
    taxonomy_by_label = taxonomy_df.set_index("primary_label")
    proxy_map_raw: Dict[int, np.ndarray] = {}
    for idx in unmapped_positions:
        label = primary_labels[int(idx)]
        if label not in taxonomy_by_label.index:
            continue
        sci = str(taxonomy_by_label.at[label, "scientific_name"])
        genus = sci.split()[0]
        hits = perch_labels_df[
            perch_labels_df["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map_raw[int(idx)] = hits["perch_index"].astype(np.int32).to_numpy()

    proxy_map = {
        idx: perch_idxs
        for idx, perch_idxs in proxy_map_raw.items()
        if class_name_map.get(primary_labels[idx]) in PROXY_TAXA
    }
    no_signal_positions = [int(idx) for idx in unmapped_positions if int(idx) not in proxy_map]
    return [primary_labels[i] for i in no_signal_positions]


def extract_spatial_embeddings(
    onnx_path: Path,
    soundscape_paths: Sequence[Path],
    batch_files: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, np.ndarray]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "8"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    if "spatial_embedding" not in out_map:
        raise RuntimeError(f"ONNX output 'spatial_embedding' not found. Available: {list(out_map.keys())}")

    paths = [Path(p) for p in soundscape_paths]
    row_ids: List[str] = []
    filenames: List[str] = []
    spatial_chunks: List[np.ndarray] = []

    iterator = range(0, len(paths), batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(len(paths) + batch_files - 1) // batch_files, desc="Perch extract")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths = paths[0:batch_files]
        future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

        for start in iterator:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            x_pos = 0
            for i, path in enumerate(batch_paths):
                audio = batch_audio[i]
                x[x_pos:x_pos + N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids.extend([f"{path.stem}_{sec}" for sec in range(5, 65, 5)])
                filenames.extend([path.name] * N_WINDOWS)
                x_pos += N_WINDOWS

            outs = session.run(None, {input_name: x})
            spatial_chunks.append(outs[out_map["spatial_embedding"]].astype(np.float32, copy=False))

    meta = pd.DataFrame({"row_id": row_ids, "filename": filenames})
    spatial_all = np.concatenate(spatial_chunks, axis=0)
    return meta, spatial_all


class RandomPerchSpatialHead(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 1536, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(emb_dim, n_classes * 4))
        self.alpha = nn.Parameter(torch.empty(1, n_classes, 4))
        self.bias = nn.Parameter(torch.zeros(1, n_classes))
        self.eps = float(eps)
        self.n_classes = int(n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.alpha, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)

    def forward(self, spatial_embedding: torch.Tensor) -> torch.Tensor:
        x = spatial_embedding / torch.sqrt((spatial_embedding * spatial_embedding).sum(dim=-1, keepdim=True) + self.eps)
        z = torch.matmul(x, self.weight)
        z = z.view(z.shape[0], 16, 4, self.n_classes, 4)
        z = torch.amax(z, dim=(1, 2))
        logits = (z * self.alpha).sum(dim=2) + self.bias
        return logits


def build_target_matrix(label_lists: Sequence[Sequence[str]], labels_subset: Sequence[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels_subset)}
    y = np.zeros((len(label_lists), len(labels_subset)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for label in labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                y[i, idx] = 1.0
    return y


def macro_auc_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, int]:
    pos = y_true.sum(axis=0)
    keep = (pos > 0) & (pos < y_true.shape[0])
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(y_true[:, keep], probs[:, keep], average="macro")
    return float(auc), n


def per_class_auc_from_logits(y_true: np.ndarray, logits: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    probs = 1.0 / (1.0 + np.exp(-logits))
    rows: List[Dict[str, object]] = []
    n_rows = y_true.shape[0]
    for i, label in enumerate(labels):
        y_col = y_true[:, i]
        pos = int(y_col.sum())
        neg = int(n_rows - pos)
        if pos == 0 or neg == 0:
            auc = float("nan")
            evaluable = False
        else:
            auc = float(roc_auc_score(y_col, probs[:, i]))
            evaluable = True
        rows.append({"label": str(label), "n_pos": pos, "n_neg": neg, "evaluable": bool(evaluable), "auc": auc})
    return pd.DataFrame(rows).sort_values(["evaluable", "auc"], ascending=[False, True]).reset_index(drop=True)


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def build_sample_weights(y_train: np.ndarray, sampler_power: float) -> np.ndarray:
    if sampler_power <= 0:
        return np.ones((len(y_train),), dtype=np.float64)
    class_pos = y_train.sum(axis=0).astype(np.float64) + 1.0
    inv_freq = 1.0 / class_pos
    sample_score = (y_train * inv_freq[None, :]).sum(axis=1)
    fallback = float(inv_freq.mean())
    sample_score = np.where(sample_score > 0, sample_score, fallback)
    weights = np.power(sample_score, sampler_power)
    weights = np.clip(weights, 1e-8, None)
    return weights.astype(np.float64, copy=False)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    loss_name: str,
    focal_gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if loss_name == "bce":
        return bce.mean()
    if loss_name == "focal":
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        focal_factor = torch.pow(1.0 - pt, float(focal_gamma))
        return (focal_factor * bce).mean()
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    loss_name: str,
    focal_gamma: float,
    sampler_power: float,
    pos_weight_cap: float,
    seed: int,
    fold_id: int,
    quiet: bool,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], List[Dict[str, float]], np.ndarray]:
    set_seed(seed + fold_id)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    sample_weights = build_sample_weights(y_train, sampler_power=sampler_power)
    if sampler_power > 0:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RandomPerchSpatialHead(n_classes=y_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_train_t = torch.from_numpy(y_train)
    pos = y_train_t.sum(dim=0)
    neg = y_train_t.shape[0] - pos
    pos_weight = (neg / (pos + 1.0)).clamp(max=float(pos_weight_cap)).to(device)

    best_auc = -np.inf
    best_val_loss = np.inf
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_logits = np.zeros((len(y_val), y_val.shape[1]), dtype=np.float32)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tqdm(train_loader, desc=f"Fold {fold_id:02d} epoch {epoch:02d} train", leave=False, disable=quiet):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = compute_loss(
                logits=logits,
                targets=yb,
                pos_weight=pos_weight,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        val_logits = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Fold {fold_id:02d} epoch {epoch:02d} val", leave=False, disable=quiet):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = compute_loss(
                    logits=logits,
                    targets=yb,
                    pos_weight=pos_weight,
                    loss_name=loss_name,
                    focal_gamma=focal_gamma,
                )
                va_losses.append(float(loss.item()))
                val_logits.append(logits.cpu().numpy())

        val_logits_np = np.concatenate(val_logits, axis=0)
        val_auc, n_eval = macro_auc_from_logits(y_val, val_logits_np)
        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        val_loss = float(np.mean(va_losses)) if va_losses else float("nan")

        history.append(
            {
                "fold": float(fold_id),
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auc_all": float(val_auc),
                "n_eval_classes_all": float(n_eval),
                "loss_name": str(loss_name),
                "focal_gamma": float(focal_gamma),
                "sampler_power": float(sampler_power),
            }
        )
        print(
            f"Fold {fold_id:02d} epoch {epoch:02d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | val_macro_auc_all={val_auc:.6f} ({n_eval} classes)"
        )

        auc_cmp = -np.inf if np.isnan(val_auc) else float(val_auc)
        if (auc_cmp > best_auc) or (auc_cmp == best_auc and val_loss < best_val_loss):
            best_auc = auc_cmp
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_logits = val_logits_np

    best_metrics = {
        "fold": float(fold_id),
        "best_val_macro_auc_all": None if not np.isfinite(best_auc) else float(best_auc),
        "best_val_loss": float(best_val_loss),
        "n_eval_classes_all_best": float(macro_auc_from_logits(y_val, best_val_logits)[1]),
        "n_train_rows": float(len(y_train)),
        "n_val_rows": float(len(y_val)),
        "loss_name": str(loss_name),
        "focal_gamma": float(focal_gamma),
        "sampler_power": float(sampler_power),
        "pos_weight_cap": float(pos_weight_cap),
    }
    return best_metrics, best_state, history, best_val_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train random head on frozen Perch features using all soundscape classes.")
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--soundscape-dir", type=Path, default=DEFAULT_SOUNDSCAPE_DIR)
    parser.add_argument("--sound-labels-csv", type=Path, default=DEFAULT_DATA_DIR / "train_soundscapes_labels.csv")
    parser.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_DATA_DIR / "taxonomy.csv")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--perch-labels-csv", type=Path, default=DEFAULT_PERCH_LABELS)
    parser.add_argument("--batch-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--sweep-lr", type=str, default="1e-3,5e-4")
    parser.add_argument("--sweep-weight-decay", type=str, default="1e-4")
    parser.add_argument("--sweep-loss", type=str, default="bce,focal")
    parser.add_argument("--sweep-focal-gamma", type=str, default="1.5,2.0")
    parser.add_argument("--sweep-sampler-power", type=str, default="0.0,1.0")
    parser.add_argument("--sweep-epochs", type=str, default="")
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--loss-name", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--sampler-power", type=float, default=0.0)
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--no-signal-per-class-csv", type=Path, default=DEFAULT_NO_SIGNAL_CSV)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    required_paths = [
        args.onnx_path,
        args.soundscape_dir,
        args.sound_labels_csv,
        args.taxonomy_csv,
        args.sample_submission,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    perch_labels_csv = args.perch_labels_csv
    if not perch_labels_csv.exists() and DEFAULT_PERCH_LABELS_FALLBACK.exists():
        perch_labels_csv = DEFAULT_PERCH_LABELS_FALLBACK
    if not perch_labels_csv.exists():
        raise FileNotFoundError(f"Missing Perch labels CSV: {args.perch_labels_csv}")

    sample_sub = pd.read_csv(args.sample_submission)
    primary_labels = sample_sub.columns[1:].tolist()
    taxonomy = pd.read_csv(args.taxonomy_csv)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    taxonomy["scientific_name"] = taxonomy["scientific_name"].astype(str)
    perch_labels = load_perch_labels(perch_labels_csv)

    no_signal_labels = build_no_signal_labels(taxonomy, perch_labels, primary_labels)
    if len(no_signal_labels) == 0:
        raise RuntimeError("No no-signal labels found for training.")

    truth_df, files = build_truth_windows(args.sound_labels_csv, args.soundscape_dir)
    if args.limit_files > 0:
        files = files[: args.limit_files]
        truth_df = truth_df[truth_df["filename"].isin(files)].copy()
        truth_df = truth_df.sort_values(["filename", "end_sec"]).reset_index(drop=True)
    if len(files) == 0:
        raise RuntimeError("No soundscape files with labels found.")

    soundscape_paths = [args.soundscape_dir / fn for fn in files]
    pred_meta, spatial_all = extract_spatial_embeddings(
        onnx_path=args.onnx_path,
        soundscape_paths=soundscape_paths,
        batch_files=args.batch_files,
        verbose=not args.quiet,
    )

    merged = pred_meta.merge(truth_df[["row_id", "label_list"]], on="row_id", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise RuntimeError("No labeled prediction rows after aligning predictions with truth.")

    # Train only on the 28 no-signal classes, but keep all soundscape rows/files.
    # Rows without these classes become all-zero targets (additional negatives).
    target_labels = list(no_signal_labels)
    y = build_target_matrix(merged["label_list"].tolist(), target_labels)
    target_pos = y.sum(axis=1)
    n_all_zero_rows = int((target_pos == 0).sum())
    groups = merged["filename"].astype(str).to_numpy()
    unique_files = np.unique(groups)
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")
    if len(unique_files) < args.n_splits:
        raise ValueError(
            f"Not enough unique files for GroupKFold: n_files={len(unique_files)}, n_splits={args.n_splits}"
        )

    if args.run_sweep:
        sweep_lrs = parse_float_list(args.sweep_lr)
        sweep_wds = parse_float_list(args.sweep_weight_decay)
        sweep_losses = parse_str_list(args.sweep_loss)
        sweep_gammas = parse_float_list(args.sweep_focal_gamma)
        sweep_sampler_powers = parse_float_list(args.sweep_sampler_power)
        if args.sweep_epochs.strip():
            sweep_epochs = [int(x) for x in parse_float_list(args.sweep_epochs)]
        else:
            sweep_epochs = [int(args.epochs)]
        configs = []
        for lr, wd, loss_name, sampler_power, ep in itertools.product(
            sweep_lrs,
            sweep_wds,
            sweep_losses,
            sweep_sampler_powers,
            sweep_epochs,
        ):
            gamma_values = [0.0] if loss_name == "bce" else sweep_gammas
            for gamma in gamma_values:
                configs.append(
                    {
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "loss_name": str(loss_name),
                        "focal_gamma": float(gamma),
                        "sampler_power": float(sampler_power),
                        "epochs": int(ep),
                    }
                )
    else:
        configs = [
            {
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "loss_name": str(args.loss_name),
                "focal_gamma": 0.0 if args.loss_name == "bce" else float(args.focal_gamma),
                "sampler_power": float(args.sampler_power),
                "epochs": int(args.epochs),
            }
        ]

    sweep_rows: List[Dict[str, Any]] = []
    best_run: Dict[str, Any] | None = None

    for cfg_idx, cfg in enumerate(configs, start=1):
        print(
            f"\n=== Config {cfg_idx}/{len(configs)} | "
            f"loss={cfg['loss_name']} gamma={cfg['focal_gamma']} sampler={cfg['sampler_power']} "
            f"lr={cfg['lr']} wd={cfg['weight_decay']} epochs={cfg['epochs']} ==="
        )
        splitter = GroupKFold(n_splits=args.n_splits)
        oof_logits = np.zeros_like(y, dtype=np.float32)
        oof_filled = np.zeros(y.shape[0], dtype=bool)
        fold_metrics: List[Dict[str, float]] = []
        all_history: List[Dict[str, float]] = []
        fold_states: List[Dict[str, torch.Tensor]] = []
        best_fold_state: Dict[str, torch.Tensor] | None = None
        best_fold_auc = -np.inf

        for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(np.arange(y.shape[0]), groups=groups), start=1):
            print(f"Fold {fold_id:02d} rows: train={len(tr_idx)} val={len(va_idx)}")
            x_train = spatial_all[tr_idx]
            y_train = y[tr_idx]
            x_val = spatial_all[va_idx]
            y_val = y[va_idx]

            metrics, state, history, val_logits_best = train_one_fold(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                batch_size=args.batch_size,
                epochs=int(cfg["epochs"]),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                loss_name=str(cfg["loss_name"]),
                focal_gamma=float(cfg["focal_gamma"]),
                sampler_power=float(cfg["sampler_power"]),
                pos_weight_cap=args.pos_weight_cap,
                seed=args.seed,
                fold_id=fold_id,
                quiet=args.quiet,
            )
            metrics["best_val_macro_auc_no_signal"] = float(metrics["best_val_macro_auc_all"])
            metrics["n_eval_classes_no_signal_best"] = float(metrics["n_eval_classes_all_best"])

            fold_metrics.append(metrics)
            all_history.extend(history)
            fold_states.append(state)
            oof_logits[va_idx] = val_logits_best
            oof_filled[va_idx] = True

            fold_auc_val = metrics["best_val_macro_auc_all"]
            fold_auc_cmp = -np.inf if fold_auc_val is None else float(fold_auc_val)
            if fold_auc_cmp > best_fold_auc:
                best_fold_auc = fold_auc_cmp
                best_fold_state = state

        if not np.all(oof_filled):
            missing = int((~oof_filled).sum())
            raise RuntimeError(f"OOF coverage incomplete, missing {missing} rows.")

        oof_auc_all, oof_n_eval_all = macro_auc_from_logits(y, oof_logits)
        oof_auc_no_signal, oof_n_eval_no_signal = oof_auc_all, oof_n_eval_all
        no_signal_per_class = per_class_auc_from_logits(
            y_true=y,
            logits=oof_logits,
            labels=target_labels,
        )
        sweep_rows.append(
            {
                "config_id": int(cfg_idx),
                "loss": str(cfg["loss_name"]),
                "focal_gamma": float(cfg["focal_gamma"]),
                "sampler_power": float(cfg["sampler_power"]),
                "lr": float(cfg["lr"]),
                "weight_decay": float(cfg["weight_decay"]),
                "epochs": int(cfg["epochs"]),
                "oof_macro_auc_all": float(oof_auc_all),
                "oof_n_eval_classes_all": int(oof_n_eval_all),
                "oof_macro_auc_no_signal": float(oof_auc_no_signal),
                "oof_n_eval_classes_no_signal": int(oof_n_eval_no_signal),
            }
        )
        run_payload = {
            "config": cfg,
            "oof_auc_all": float(oof_auc_all),
            "oof_n_eval_all": int(oof_n_eval_all),
            "oof_auc_no_signal": float(oof_auc_no_signal),
            "oof_n_eval_no_signal": int(oof_n_eval_no_signal),
            "fold_metrics": fold_metrics,
            "history": all_history,
            "fold_states": fold_states,
            "best_fold_state": best_fold_state,
            "no_signal_per_class": no_signal_per_class,
        }
        if best_run is None or float(oof_auc_all) > float(best_run["oof_auc_all"]):
            best_run = run_payload

    if best_run is None:
        raise RuntimeError("No config run completed.")

    if args.sweep_csv is not None:
        args.sweep_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sweep_rows).sort_values("oof_macro_auc_all", ascending=False).to_csv(args.sweep_csv, index=False)
        print(f"Saved sweep CSV: {args.sweep_csv}")

    best_cfg = best_run["config"]
    oof_auc_all = float(best_run["oof_auc_all"])
    oof_n_eval_all = int(best_run["oof_n_eval_all"])
    oof_auc_no_signal = float(best_run["oof_auc_no_signal"])
    oof_n_eval_no_signal = int(best_run["oof_n_eval_no_signal"])
    fold_metrics = best_run["fold_metrics"]
    all_history = best_run["history"]
    fold_states = best_run["fold_states"]
    best_fold_state = best_run["best_fold_state"]
    no_signal_per_class = best_run["no_signal_per_class"]

    print("\n=== Best Config ===")
    print(
        f"loss={best_cfg['loss_name']} gamma={best_cfg['focal_gamma']} sampler={best_cfg['sampler_power']} "
        f"lr={best_cfg['lr']} wd={best_cfg['weight_decay']} epochs={best_cfg['epochs']} | "
        f"OOF(target_28)={oof_auc_all:.6f} ({oof_n_eval_all})"
    )
    print(f"All-zero target rows (extra negatives): {n_all_zero_rows} / {len(y)}")

    if args.no_signal_per_class_csv is not None:
        args.no_signal_per_class_csv.parent.mkdir(parents=True, exist_ok=True)
        no_signal_per_class.to_csv(args.no_signal_per_class_csv, index=False)
        print(f"Saved no-signal per-class OOF CSV: {args.no_signal_per_class_csv}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        if best_fold_state is None:
            raise RuntimeError("No fold model state selected as best.")
        torch.save(
            {
                "state_dict": best_fold_state,
                "fold_states": fold_states,
                "primary_labels": primary_labels,
                "target_labels": target_labels,
                "no_signal_labels": no_signal_labels,
                "best_config": best_cfg,
                "sweep_rows": sweep_rows,
                "train_config": {
                    "epochs": int(best_cfg["epochs"]),
                    "batch_size": int(args.batch_size),
                    "lr": float(best_cfg["lr"]),
                    "weight_decay": float(best_cfg["weight_decay"]),
                    "loss": str(best_cfg["loss_name"]),
                    "focal_gamma": float(best_cfg["focal_gamma"]),
                    "sampler_power": float(best_cfg["sampler_power"]),
                    "pos_weight_cap": float(args.pos_weight_cap),
                    "n_splits": int(args.n_splits),
                    "seed": int(args.seed),
                },
                "fold_metrics": fold_metrics,
                "oof": {
                    "macro_auc_target_28": None if not np.isfinite(oof_auc_all) else float(oof_auc_all),
                    "n_eval_classes_target_28": int(oof_n_eval_all),
                },
                "no_signal_per_class_oof": no_signal_per_class.to_dict(orient="records"),
                "history": all_history,
            },
            args.save_path,
        )
        print(f"Saved checkpoint: {args.save_path}")

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "onnx_path": str(args.onnx_path),
            "perch_labels_csv": str(perch_labels_csv),
            "best_config": best_cfg,
            "sweep_rows": sweep_rows,
            "n_files": int(len(files)),
            "n_rows": int(len(merged)),
            "n_classes_total": int(len(primary_labels)),
            "n_classes_no_signal_target": int(len(no_signal_labels)),
            "n_all_zero_target_rows": int(n_all_zero_rows),
            "target_labels": target_labels,
            "fold_metrics": fold_metrics,
            "oof": {
                "macro_auc_target_28": None if not np.isfinite(oof_auc_all) else float(oof_auc_all),
                "n_eval_classes_target_28": int(oof_n_eval_all),
            },
            "no_signal_labels": no_signal_labels,
            "no_signal_per_class_oof_csv": (
                str(args.no_signal_per_class_csv) if args.no_signal_per_class_csv is not None else None
            ),
            "no_signal_per_class_oof": no_signal_per_class.to_dict(orient="records"),
            "history": all_history,
        }
        args.metrics_json.write_text(json.dumps(out, indent=2))
        print(f"Saved metrics JSON: {args.metrics_json}")


if __name__ == "__main__":
    main()
