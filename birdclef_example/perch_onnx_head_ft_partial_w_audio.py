#!/usr/bin/env python3

"""Fine-tune the original Perch ONNX classifier head in PyTorch.

This script does not add any layers.
It reuses the exact ONNX head tensors and optimizes only those parameters:
- dot_general6_reshaped_0  (projection weights)
- arith.constant62          (prototype weights)
- arith.constant61          (bias)

Training uses a masked BCE objective in Perch label space, implemented as BCE on
mapped Perch indices only (equivalent to a full-mask BCE with zeros elsewhere).
"""

from __future__ import annotations

import itertools
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import onnx
    from onnx import numpy_helper
except Exception as exc:
    raise RuntimeError(
        "This script requires the 'onnx' package to read ONNX initializers. "
        "Install it with: pip install onnx"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (  # noqa: E402
    parse_primary_labels,
    parse_secondary_labels,
)


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
SAMPLE_RATE = 32000
WINDOW_SEC = 5
N_WINDOWS = 12
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
FILE_SAMPLES = SAMPLE_RATE * 60

# Exact ONNX initializer names found in models/perch_onnx/perch_v2.onnx
HEAD_W_NAME = "jit(infer_fn)/MultiHeadClassifier/MultiHeadClassifier._call_model/heads_protopnet_logits/dot_general6_reshaped_0"
HEAD_ALPHA_NAME = "arith.constant62"
HEAD_BIAS_NAME = "arith.constant61"

# Paths/data config.
ONNX_PATH = REPO_ROOT / "models" / "perch_onnx" / "perch_v2.onnx"
MODEL_DIR = REPO_ROOT / "models" / "perch_v2_cpu" / "1"
DATA_DIR = REPO_ROOT / "data"
SOUNDSCAPE_DIR = REPO_ROOT / "data" / "train_soundscapes"
TRAIN_AUDIO_DIR = Path(os.environ.get("TRAIN_AUDIO_DIR", "/mnt/evafs/groups/re-com/mgromadzki/data/train_audio"))

# Fixed run settings.
BATCH_FILES = 8
VAL_SIZE = 0.2
SEED = 42
SAVE_HEAD_PATH: Path | None = REPO_ROOT / "outputs" / "experiments_ft" / "perch_onnx_head_ft_partial_w_audio_best.pt"

# Hyperparameters moved inside script.
HPARAMS_BASE = {
    "batch_size": 64,
    "epochs": 15,
    "lr": 1e-4,
    "weight_decay": 1e-6,
}

# Grid search over these values; include singletons to keep specific params fixed.
HPARAM_GRID = {
    "batch_size": [64, 128],
    "epochs": [15],
    "lr": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "weight_decay": [1e-6],
}

RUN_GRID_SEARCH = False
GRID_RESULTS_CSV: Path | None = REPO_ROOT / "outputs" / "eda" / "perch_onnx_head_ft_partial_w_audio_grid_results.csv"
W_TRAIN_MODE = "partial"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_soundscape_filename(name: str) -> Dict[str, object]:
    match = FNAME_RE.match(name)
    if not match:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }

    file_id, site, ymd, hms = match.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }


def union_labels(series: pd.Series) -> List[str]:
    labels = set()
    for value in series:
        labels.update(parse_primary_labels(value))
    return sorted(labels)


def read_audio_fixed_length(path: Path, target_samples: int) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SAMPLE_RATE}")
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        start = max(0, (len(y) - target_samples) // 2)
        y = y[start:start + target_samples]
    return y.astype(np.float32, copy=False)


def read_soundscape_60s(path: Path) -> np.ndarray:
    return read_audio_fixed_length(path, FILE_SAMPLES)


def load_onnx_head_tensors(onnx_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = onnx.load(str(onnx_path))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    if HEAD_W_NAME not in init or HEAD_ALPHA_NAME not in init or HEAD_BIAS_NAME not in init:
        raise RuntimeError("Could not find expected Perch head initializers in ONNX model.")

    w = init[HEAD_W_NAME].astype(np.float32, copy=False)       # [1536, 14795*4]
    alpha = init[HEAD_ALPHA_NAME].astype(np.float32, copy=False)  # [1, 14795, 4]
    bias = init[HEAD_BIAS_NAME].astype(np.float32, copy=False)    # [1, 14795]

    if w.ndim != 2 or alpha.ndim != 3 or bias.ndim != 2:
        raise RuntimeError(
            f"Unexpected head tensor shapes: W={w.shape}, alpha={alpha.shape}, bias={bias.shape}"
        )

    n_classes = bias.shape[1]
    if w.shape[1] != n_classes * 4:
        raise RuntimeError(f"Expected W second dim == n_classes*4, got W={w.shape}, n_classes={n_classes}")
    if alpha.shape[1] != n_classes or alpha.shape[2] != 4:
        raise RuntimeError(f"Unexpected alpha shape {alpha.shape} for n_classes={n_classes}")

    return w, alpha, bias


def get_genus_hits(scientific_name: str, perch_labels_df: pd.DataFrame) -> pd.DataFrame:
    genus = str(scientific_name).split()[0]
    return perch_labels_df[
        perch_labels_df["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\\s", na=False)
    ].copy()


def build_exact_mapping(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int, int]:
    # Match predict_ported_old.py: exact scientific-name join for mapped/unmapped split.
    no_label_index = len(perch_labels_df)
    taxonomy_local = taxonomy_df.copy()
    taxonomy_local["scientific_name_lookup"] = taxonomy_local["scientific_name"]
    perch_lookup = perch_labels_df.rename(columns={"scientific_name": "scientific_name_lookup"})
    mapping = taxonomy_local.merge(
        perch_lookup[["scientific_name_lookup", "perch_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    mapping["perch_index"] = mapping["perch_index"].fillna(no_label_index).astype(int)

    label_to_perch = mapping.set_index("primary_label")["perch_index"].to_dict()
    bc_indices = np.array([int(label_to_perch.get(lbl, no_label_index)) for lbl in primary_labels], dtype=np.int32)
    mapped_mask = bc_indices != no_label_index

    mapped_dict = {
        lbl: int(idx)
        for lbl, idx, is_mapped in zip(primary_labels, bc_indices.tolist(), mapped_mask.tolist())
        if is_mapped
    }

    class_name_map = taxonomy_local.set_index("primary_label")["class_name"].to_dict()
    proxy_taxa = {"Amphibia", "Insecta", "Aves"}
    unmapped_df = mapping[mapping["perch_index"] == no_label_index].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    proxy_counts = {"Amphibia": 0, "Aves": 0, "Insecta": 0}
    for _, row in unmapped_non_sonotype.iterrows():
        target = str(row["primary_label"])
        cls = class_name_map.get(target)
        if cls not in proxy_taxa:
            continue
        hits = get_genus_hits(str(row["scientific_name"]), perch_labels_df)
        if len(hits) > 0:
            proxy_counts[cls] += 1

    return mapped_dict, class_name_map, proxy_counts, int(mapped_mask.sum()), int((~mapped_mask).sum())


def build_row_truth(data_dir: Path, soundscape_dir: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    grouped = (
        labels_df.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(int)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)

    windows_per_file = grouped.groupby("filename").size()
    full_files = set(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    grouped = grouped[grouped["filename"].isin(full_files)].copy()
    grouped = grouped[grouped["filename"].map(lambda x: (soundscape_dir / x).exists())].copy()
    grouped = grouped.sort_values(["filename", "end_sec"]).reset_index(drop=True)
    return grouped


def build_train_audio_truth(data_dir: Path, train_audio_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    required_cols = {"filename", "primary_label"}
    missing_cols = required_cols.difference(train_df.columns)
    if missing_cols:
        raise ValueError(f"train.csv missing required columns: {sorted(missing_cols)}")

    if "secondary_labels" not in train_df.columns:
        train_df["secondary_labels"] = "[]"

    train_df = train_df.dropna(subset=["filename", "primary_label"]).copy()
    train_df["filename"] = train_df["filename"].astype(str)
    train_df = train_df[train_df["filename"].map(lambda x: (train_audio_dir / x).exists())].copy()

    if train_df.empty:
        raise RuntimeError(f"No train_audio files found under {train_audio_dir}")

    def _collect_labels(row: pd.Series) -> List[str]:
        labels = set(parse_primary_labels(row.get("primary_label")))
        labels.update(parse_secondary_labels(row.get("secondary_labels")))
        return sorted([lbl for lbl in labels if lbl])

    train_df["label_list"] = train_df.apply(_collect_labels, axis=1)
    train_df = train_df[train_df["label_list"].map(len) > 0].copy()
    # Keep both stem and basename; basename is used for robust join with meta_audio.filename.
    train_df["row_id"] = train_df["filename"].map(lambda x: Path(str(x)).stem)
    train_df["audio_basename"] = train_df["filename"].map(lambda x: Path(str(x)).name)
    return train_df[["row_id", "filename", "audio_basename", "label_list"]].reset_index(drop=True)


def extract_spatial_and_logits(
    onnx_path: Path,
    soundscape_paths: Sequence[Path],
    batch_files: int,
    windows_per_file: int,
    read_audio_fn,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "8"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    if "spatial_embedding" not in out_names or "label" not in out_names:
        raise RuntimeError(f"ONNX outputs missing required tensors, got: {out_names}")

    row_ids: List[str] = []
    filenames: List[str] = []
    sites: List[str] = []
    hours: List[int] = []

    spatial_chunks: List[np.ndarray] = []
    logits_chunks: List[np.ndarray] = []

    for start in tqdm(range(0, len(soundscape_paths), batch_files), desc="ONNX extract"):
        batch_paths = soundscape_paths[start:start + batch_files]
        x = np.empty((len(batch_paths) * windows_per_file, WINDOW_SAMPLES), dtype=np.float32)

        pos = 0
        for path in batch_paths:
            y = read_audio_fn(path)
            if windows_per_file == N_WINDOWS:
                x[pos:pos + windows_per_file] = y.reshape(windows_per_file, WINDOW_SAMPLES)
            elif windows_per_file == 1:
                x[pos] = y
            else:
                raise ValueError(f"Unsupported windows_per_file={windows_per_file}")
            meta = parse_soundscape_filename(path.name)
            if windows_per_file == N_WINDOWS:
                row_ids.extend([f"{path.stem}_{sec}" for sec in range(5, 65, 5)])
            else:
                row_ids.append(path.stem)
            filenames.extend([path.name] * windows_per_file)
            sites.extend([str(meta["site"])] * windows_per_file)
            hours.extend([int(meta["hour_utc"])] * windows_per_file)
            pos += windows_per_file

        outs = session.run(["spatial_embedding", "label"], {input_name: x})
        spatial_chunks.append(outs[0].astype(np.float32, copy=False))
        logits_chunks.append(outs[1].astype(np.float32, copy=False))

    meta_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        }
    )
    spatial_all = np.concatenate(spatial_chunks, axis=0)
    logits_all = np.concatenate(logits_chunks, axis=0)
    return meta_df, spatial_all, logits_all


class PerchOnnxHead(nn.Module):
    def __init__(self, w: np.ndarray, alpha: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(w.copy()))       # [1536, C*4]
        self.alpha = nn.Parameter(torch.from_numpy(alpha.copy()))    # [1, C, 4]
        self.bias = nn.Parameter(torch.from_numpy(bias.copy()))      # [1, C]
        self.eps = float(eps)
        self.n_classes = self.bias.shape[1]

    def forward(self, spatial_embedding: torch.Tensor) -> torch.Tensor:
        # Match ONNX: normalize last dim, project, reshape, max over spatial axes, weighted sum + bias.
        x = spatial_embedding / torch.sqrt((spatial_embedding * spatial_embedding).sum(dim=-1, keepdim=True) + self.eps)
        z = torch.matmul(x, self.weight)  # [B,16,4,C*4]
        z = z.view(z.shape[0], 16, 4, self.n_classes, 4)
        z = torch.amax(z, dim=(1, 2))
        logits = (z * self.alpha).sum(dim=2) + self.bias
        return logits


def masked_macro_auc(logits: np.ndarray, targets: np.ndarray) -> float:
    # ROC-AUC is only defined when both classes are present.
    pos = targets.sum(axis=0)
    keep = (pos > 0) & (pos < targets.shape[0])
    if not np.any(keep):
        return float("nan")
    probs = 1.0 / (1.0 + np.exp(-logits))
    return roc_auc_score(targets[:, keep], probs[:, keep], average="macro")

def build_full_target_matrix(
    label_lists: Sequence[Sequence[str]],
    primary_labels: Sequence[str],
) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    y_full = np.zeros((len(label_lists), len(primary_labels)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for label in labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                y_full[i, idx] = 1.0
    return y_full

def expand_hparam_grid(base: Dict[str, float | int], grid: Dict[str, List[float | int]]) -> List[Dict[str, float | int]]:
    keys = sorted(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(base)
        for k, v in zip(keys, values):
            cfg[k] = v
        combos.append(cfg)
    return combos


def build_weight_grad_mask(weight_shape: Tuple[int, int], class_indices: np.ndarray, classes_per_block: int = 4) -> np.ndarray:
    weight_rows, weight_cols = int(weight_shape[0]), int(weight_shape[1])
    mask = np.zeros((weight_rows, weight_cols), dtype=np.float32)
    class_indices = np.asarray(class_indices, dtype=np.int64)
    for cls_idx in class_indices.tolist():
        start = int(cls_idx) * classes_per_block
        end = start + classes_per_block
        if start < 0 or end > weight_cols:
            raise ValueError(f"Class index {cls_idx} is out of bounds for weight shape {weight_shape}")
        mask[:, start:end] = 1.0
    return mask


def train_and_eval_config(
    hparams: Dict[str, float | int],
    spatial_train: np.ndarray,
    target_train: np.ndarray,
    spatial_val: np.ndarray,
    target_val: np.ndarray,
    target_val_full: np.ndarray,
    mapped_idx_arr: np.ndarray,
    full_label_positions: np.ndarray,
    full_label_perch_indices: np.ndarray,
    val_base_logits: np.ndarray,
    w0: np.ndarray,
    alpha0: np.ndarray,
    bias0: np.ndarray,
    seed: int,
    trial_id: int,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    set_seed(seed + trial_id)

    batch_size = int(hparams["batch_size"])
    epochs = int(hparams["epochs"])
    lr = float(hparams["lr"])
    weight_decay = float(hparams["weight_decay"])

    x_train = torch.from_numpy(spatial_train)
    y_train = torch.from_numpy(target_train)
    x_val = torch.from_numpy(spatial_val)
    y_val = torch.from_numpy(target_val)

    model = PerchOnnxHead(w0, alpha0, bias0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mapped_idx_t = torch.from_numpy(mapped_idx_arr).to(device=device, dtype=torch.long)

    # Keep W partially trainable: only mapped-class column blocks receive gradients.
    w_grad_mask_np = build_weight_grad_mask(tuple(model.weight.shape), mapped_idx_arr, classes_per_block=4)
    w_grad_mask_t = torch.from_numpy(w_grad_mask_np).to(device=device)
    model.weight.register_hook(lambda grad: grad * w_grad_mask_t)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    best_key = (-np.inf, -np.inf, np.inf)
    best_state = {
        "weight": model.weight.detach().cpu().clone(),
        "alpha": model.alpha.detach().cpu().clone(),
        "bias": model.bias.detach().cpu().clone(),
    }
    best_metrics = {
        "best_epoch": 0.0,
        "train_loss": np.nan,
        "val_loss": np.nan,
        "val_mapped_auc": np.nan,
        "val_full_auc": np.nan,
        "val_full_mae_to_base": np.nan,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tqdm(train_loader, desc=f"Trial {trial_id:02d} epoch {epoch:02d} train", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits_full = model(xb)
            logits_mapped = logits_full.index_select(dim=1, index=mapped_idx_t)
            loss = F.binary_cross_entropy_with_logits(logits_mapped, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        va_logits_collect = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Trial {trial_id:02d} epoch {epoch:02d} val", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)
                logits_full = model(xb)
                logits_mapped = logits_full.index_select(dim=1, index=mapped_idx_t)
                loss = F.binary_cross_entropy_with_logits(logits_mapped, yb)
                va_losses.append(float(loss.item()))
                va_logits_collect.append(logits_mapped.cpu().numpy())

        va_logits = np.concatenate(va_logits_collect, axis=0)
        auc_mapped = masked_macro_auc(va_logits, target_val)

        with torch.no_grad():
            val_full_logits = model(x_val.to(device)).cpu().numpy()
        val_full_scores = np.zeros((len(val_full_logits), len(target_val_full[0])), dtype=np.float32)
        val_full_scores[:, full_label_positions] = val_full_logits[:, full_label_perch_indices]
        auc_full = masked_macro_auc(val_full_scores, target_val_full)
        mae_to_base = float(np.mean(np.abs(val_full_logits - val_base_logits)))

        train_loss = float(np.mean(tr_losses))
        val_loss = float(np.mean(va_losses))
        auc_mapped_for_compare = -np.inf if np.isnan(auc_mapped) else float(auc_mapped)
        auc_full_for_compare = -np.inf if np.isnan(auc_full) else float(auc_full)
        key = (auc_mapped_for_compare, auc_full_for_compare, -val_loss)
        if key > best_key:
            best_key = key
            best_state = {
                "weight": model.weight.detach().cpu().clone(),
                "alpha": model.alpha.detach().cpu().clone(),
                "bias": model.bias.detach().cpu().clone(),
            }
            best_metrics = {
                "best_epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mapped_auc": float(auc_mapped),
                "val_full_auc": float(auc_full),
                "val_full_mae_to_base": mae_to_base,
            }

        print(
            f"Trial {trial_id:02d} epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mapped_auc={auc_mapped:.6f} | "
            f"val_full_auc={auc_full:.6f} | "
            f"val_full_mae_to_base={mae_to_base:.6e}"
        )

    return best_metrics, best_state


def main() -> None:
    set_seed(SEED)

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    perch_labels = (
        pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "perch_index", "inat2024_fsd50k": "scientific_name"})
    )
    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)

    bird_to_perch_idx, _class_name_map, proxy_counts, mapped_n, unmapped_n = build_exact_mapping(
        taxonomy,
        perch_labels,
        primary_labels,
    )
    mapped_perch_indices = sorted(set(bird_to_perch_idx.values()))
    if not mapped_perch_indices:
        raise RuntimeError("No BirdCLEF labels could be mapped to Perch indices.")

    print(f"Proxy targets by class: {proxy_counts}")
    print(f"Mapped classes: {mapped_n} / {len(primary_labels)}")
    print(f"Unmapped classes: {unmapped_n}")
    print(f"Mapped BirdCLEF labels to Perch indices: {len(mapped_perch_indices)}")

    soundscape_truth = build_row_truth(DATA_DIR, SOUNDSCAPE_DIR)
    if soundscape_truth.empty:
        raise RuntimeError("No fully-labeled soundscape windows found.")

    train_audio_truth = build_train_audio_truth(DATA_DIR, TRAIN_AUDIO_DIR)

    active_primary_labels = sorted({lab for labs in soundscape_truth["label_list"] for lab in labs})
    full_label_lists = soundscape_truth["label_list"].tolist()
    y_full = build_full_target_matrix(full_label_lists, primary_labels)
    train_perch_indices = sorted(
        {
            int(bird_to_perch_idx[lab])
            for lab in active_primary_labels
            if lab in bird_to_perch_idx
        }
    )
    if not train_perch_indices:
        raise RuntimeError("No active train labels map to Perch indices.")

    full_label_positions = np.array([i for i, label in enumerate(primary_labels) if label in bird_to_perch_idx], dtype=np.int64)
    full_label_perch_indices = np.array([bird_to_perch_idx[label] for label in primary_labels if label in bird_to_perch_idx], dtype=np.int64)

    available_files = soundscape_truth["filename"].drop_duplicates().tolist()
    file_set = set(available_files)

    soundscape_truth = soundscape_truth[soundscape_truth["filename"].isin(file_set)].copy()
    soundscape_truth = soundscape_truth.sort_values(["filename", "end_sec"]).reset_index(drop=True)

    soundscape_paths = [SOUNDSCAPE_DIR / fname for fname in available_files]
    print(f"Using soundscape files: {len(soundscape_paths)}")
    print(f"Batch files (soundscapes): {BATCH_FILES}")

    meta_df, spatial_all, logits_base = extract_spatial_and_logits(
        ONNX_PATH,
        soundscape_paths,
        batch_files=BATCH_FILES,
        windows_per_file=N_WINDOWS,
        read_audio_fn=read_soundscape_60s,
    )

    truth_by_row = soundscape_truth[["row_id", "label_list"]].copy()
    merged = meta_df.merge(truth_by_row, on="row_id", how="inner")
    if merged.empty:
        raise RuntimeError("No aligned rows between ONNX extraction and truth labels.")

    row_idx = merged.index.to_numpy()
    spatial = spatial_all[row_idx]
    logits_base = logits_base[row_idx]

    mapped_idx_arr = np.array(train_perch_indices, dtype=np.int64)
    perch_col_to_local = {col: i for i, col in enumerate(train_perch_indices)}

    y_mapped = np.zeros((len(merged), len(train_perch_indices)), dtype=np.float32)
    for i, labels in enumerate(merged["label_list"].tolist()):
        for lab in labels:
            perch_idx = bird_to_perch_idx.get(lab)
            if perch_idx is None:
                continue
            local = perch_col_to_local.get(perch_idx)
            if local is not None:
                y_mapped[i, local] = 1.0

    y_full_val = build_full_target_matrix(merged["label_list"].tolist(), primary_labels)

    groups = merged["filename"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(np.arange(len(merged)), groups=groups))

    spatial_soundscape_train = spatial[tr_idx]
    spatial_val = spatial[va_idx]
    target_soundscape_train = y_mapped[tr_idx]
    target_val = y_mapped[va_idx]

    # Extra training source: train_audio + train.csv.
    train_audio_paths = [TRAIN_AUDIO_DIR / fname for fname in train_audio_truth["filename"].tolist()]
    print(f"Using train_audio files: {len(train_audio_paths)} from {TRAIN_AUDIO_DIR}")
    meta_audio, spatial_audio_all, _ = extract_spatial_and_logits(
        ONNX_PATH,
        train_audio_paths,
        batch_files=BATCH_FILES,
        windows_per_file=1,
        read_audio_fn=lambda p: read_audio_fixed_length(p, WINDOW_SAMPLES),
    )
    merged_audio = meta_audio.merge(
        train_audio_truth[["audio_basename", "label_list"]],
        left_on="filename",
        right_on="audio_basename",
        how="inner",
    )
    if merged_audio.empty:
        sample_meta = meta_audio["filename"].head(5).tolist()
        sample_truth = train_audio_truth["audio_basename"].head(5).tolist()
        raise RuntimeError(
            "No aligned rows between ONNX extraction and train_audio labels. "
            f"Example meta_audio filename: {sample_meta}; "
            f"example train_audio basename: {sample_truth}"
        )

    row_idx_audio = merged_audio.index.to_numpy()
    spatial_audio = spatial_audio_all[row_idx_audio]

    y_mapped_audio = np.zeros((len(merged_audio), len(train_perch_indices)), dtype=np.float32)
    for i, labels in enumerate(merged_audio["label_list"].tolist()):
        for lab in labels:
            perch_idx = bird_to_perch_idx.get(lab)
            if perch_idx is None:
                continue
            local = perch_col_to_local.get(perch_idx)
            if local is not None:
                y_mapped_audio[i, local] = 1.0

    spatial_train = np.concatenate([spatial_soundscape_train, spatial_audio], axis=0)
    target_train = np.concatenate([target_soundscape_train, y_mapped_audio], axis=0)

    w0, alpha0, bias0 = load_onnx_head_tensors(ONNX_PATH)

    print(f"Soundscape train rows: {len(tr_idx)} | Val rows: {len(va_idx)}")
    print(f"train_audio train rows: {len(spatial_audio)}")
    print(f"Total train rows: {len(spatial_train)} | Total val rows: {len(spatial_val)}")
    print(f"Head tensor shapes: W={tuple(w0.shape)}, alpha={tuple(alpha0.shape)}, bias={tuple(bias0.shape)}")
    print(f"Fine-tuned Perch classes (soundscapes only): {len(train_perch_indices)}")
    print(f"Mapped BirdCLEF labels (global): {len(mapped_perch_indices)}")
    print(f"Grid search enabled: {RUN_GRID_SEARCH}")
    print(f"W train mode: {W_TRAIN_MODE}")

    trial_configs = expand_hparam_grid(HPARAMS_BASE, HPARAM_GRID) if RUN_GRID_SEARCH else [dict(HPARAMS_BASE)]
    print(f"Hyperparameter trials: {len(trial_configs)}")

    trial_rows = []
    # Prefer mapped-space validation AUC (aligned with training objective),
    # then use full-space AUC and val loss as tie-breakers.
    best_key = (-np.inf, -np.inf, np.inf)
    best_trial = None
    best_state = None

    for i, cfg in enumerate(trial_configs, start=1):
        print(
            f"\n=== Trial {i}/{len(trial_configs)} === "
            f"batch_size={cfg['batch_size']} epochs={cfg['epochs']} lr={cfg['lr']} wd={cfg['weight_decay']}"
        )
        metrics, state = train_and_eval_config(
            hparams=cfg,
            spatial_train=spatial_train,
            target_train=target_train,
            spatial_val=spatial_val,
            target_val=target_val,
            target_val_full=y_full_val[va_idx],
            mapped_idx_arr=mapped_idx_arr,
            full_label_positions=full_label_positions,
            full_label_perch_indices=full_label_perch_indices,
            val_base_logits=logits_base[va_idx],
            w0=w0,
            alpha0=alpha0,
            bias0=bias0,
            seed=SEED,
            trial_id=i,
        )

        row = {
            "trial": i,
            "batch_size": int(cfg["batch_size"]),
            "epochs": int(cfg["epochs"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            **metrics,
        }
        trial_rows.append(row)

        auc_mapped = float(row["val_mapped_auc"]) if not np.isnan(row["val_mapped_auc"]) else -np.inf
        auc_full = float(row["val_full_auc"]) if not np.isnan(row["val_full_auc"]) else -np.inf
        key = (auc_mapped, auc_full, -float(row["val_loss"]))
        if key > best_key:
            best_key = key
            best_trial = row
            best_state = state

    if best_trial is None or best_state is None:
        raise RuntimeError("Grid search finished without a valid trial result.")

    print("\n=== Best trial ===")
    print(
        f"trial={best_trial['trial']} "
        f"batch_size={best_trial['batch_size']} epochs={best_trial['epochs']} "
        f"lr={best_trial['lr']} wd={best_trial['weight_decay']} "
        f"val_full_auc={best_trial['val_full_auc']:.6f} "
        f"val_mapped_auc={best_trial['val_mapped_auc']:.6f} "
        f"val_loss={best_trial['val_loss']:.6f}"
    )

    if GRID_RESULTS_CSV is not None:
        GRID_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trial_rows).sort_values("val_full_auc", ascending=False).to_csv(GRID_RESULTS_CSV, index=False)
        print(f"Saved grid search results: {GRID_RESULTS_CSV}")

    if SAVE_HEAD_PATH is not None:
        SAVE_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weight": best_state["weight"],
                "alpha": best_state["alpha"],
                "bias": best_state["bias"],
                "mapped_perch_indices": mapped_perch_indices,
                "train_perch_indices": train_perch_indices,
                "bird_to_perch_idx": bird_to_perch_idx,
                "best_trial": best_trial,
                "all_trials": trial_rows,
            },
            SAVE_HEAD_PATH,
        )
        print(f"Saved best fine-tuned head tensors: {SAVE_HEAD_PATH}")


if __name__ == "__main__":
    main()
