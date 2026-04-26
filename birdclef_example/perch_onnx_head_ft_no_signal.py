#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
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
DEFAULT_SAVE_PATH = REPO_ROOT / "outputs" / "experiments_ft" / "perch_no_signal_head.pt"
DEFAULT_METRICS_JSON = REPO_ROOT / "outputs" / "experiments_ft" / "perch_no_signal_head_metrics.json"
DEFAULT_PER_CLASS_OOF_CSV = REPO_ROOT / "outputs" / "experiments_ft" / "perch_no_signal_head_oof_per_class.csv"
DEFAULT_SWEEP_CSV = REPO_ROOT / "outputs" / "experiments_ft" / "perch_no_signal_head_sweep.csv"
DEFAULT_EXTRA_TRAIN_CSV = Path("/mnt/evafs/groups/re-com/mgromadzki/data/train.csv")
DEFAULT_EXTRA_TRAIN_AUDIO_DIR = Path("/mnt/evafs/groups/re-com/mgromadzki/data/train_audio")


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


def parse_secondary_labels_field(value: object) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if text == "" or text == "[]" or text.lower() == "nan":
        return []

    # Typical train.csv format is a Python-like list string, e.g. "['a', 'b']".
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError):
            pass

    # Fallback for delimiter-style strings.
    out = []
    for tok in re.split(r"[;,|]", text):
        tok = tok.strip().strip("'").strip('"')
        if tok:
            out.append(tok)
    return out


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


def read_audio_fixed_5s(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SAMPLE_RATE}")
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    elif len(y) > WINDOW_SAMPLES:
        start = max(0, (len(y) - WINDOW_SAMPLES) // 2)
        y = y[start:start + WINDOW_SAMPLES]
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


def build_no_signal_targets(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], List[str]]:
    no_label_index = len(perch_labels_df)
    mapping = taxonomy_df.merge(perch_labels_df, on="scientific_name", how="left")
    mapping["perch_index"] = mapping["perch_index"].fillna(no_label_index).astype(int)

    label_to_perch = mapping.set_index("primary_label")["perch_index"].to_dict()
    perch_indices = np.array([int(label_to_perch.get(label, no_label_index)) for label in primary_labels], dtype=np.int32)
    mapped_mask = perch_indices != no_label_index
    mapped_positions = np.where(mapped_mask)[0].astype(np.int32)

    class_name_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
    unmapped_positions = np.where(~mapped_mask)[0].astype(np.int32)
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

    no_signal_positions = np.array([idx for idx in unmapped_positions if int(idx) not in proxy_map], dtype=np.int32)
    no_signal_labels = [primary_labels[int(idx)] for idx in no_signal_positions]
    return mapped_positions, no_signal_positions, proxy_map, no_signal_labels


_FLAT_FALLBACK_SHAPE = (0, 1536)
_SPATIAL_FALLBACK_SHAPE = (0, 16, 4, 1536)


def _empty_fallback(output_name: str) -> np.ndarray:
    if output_name == "embedding":
        return np.zeros(_FLAT_FALLBACK_SHAPE, dtype=np.float32)
    if output_name == "spatial_embedding":
        return np.zeros(_SPATIAL_FALLBACK_SHAPE, dtype=np.float32)
    raise ValueError(f"Unknown output_name: {output_name}")


def extract_spatial_embeddings(
    onnx_path: Path,
    soundscape_paths: Sequence[Path],
    batch_files: int,
    verbose: bool,
    output_name: str = "spatial_embedding",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract Perch features for the soundscape val pool.

    `output_name` selects which ONNX output to pull:
      - "spatial_embedding" → (N, 16, 4, 1536)  pre-pooled time × frequency × channels
      - "embedding"         → (N, 1536)         the global-pooled vector the SSM uses
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "8"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    if output_name not in out_map:
        raise RuntimeError(f"ONNX output {output_name!r} not found. Available: {list(out_map.keys())}")

    paths = [Path(p) for p in soundscape_paths]
    row_ids: List[str] = []
    filenames: List[str] = []
    chunks: List[np.ndarray] = []

    iterator = range(0, len(paths), batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(len(paths) + batch_files - 1) // batch_files,
                        desc=f"Perch extract ({output_name})")

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
            chunks.append(outs[out_map[output_name]].astype(np.float32, copy=False))

    meta = pd.DataFrame({"row_id": row_ids, "filename": filenames})
    arr_all = np.concatenate(chunks, axis=0) if chunks else _empty_fallback(output_name)
    return meta, arr_all


def extract_spatial_embeddings_clips(
    onnx_path: Path,
    clip_paths: Sequence[Path],
    batch_size: int,
    verbose: bool,
    output_name: str = "spatial_embedding",
) -> np.ndarray:
    """Extract Perch features for the extra train.csv clips.

    `output_name` semantics match `extract_spatial_embeddings`.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    if output_name not in out_map:
        raise RuntimeError(f"ONNX output {output_name!r} not found. Available: {list(out_map.keys())}")

    paths = [Path(p) for p in clip_paths]
    chunks: List[np.ndarray] = []
    iterator = range(0, len(paths), batch_size)
    if verbose:
        iterator = tqdm(iterator, total=(len(paths) + batch_size - 1) // batch_size,
                        desc=f"Perch extra clips ({output_name})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths = paths[0:batch_size]
        future_audio = [io_executor.submit(read_audio_fixed_5s, p) for p in next_paths]

        for start in iterator:
            batch_paths = next_paths
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_size
            if next_start < len(paths):
                next_paths = paths[next_start:next_start + batch_size]
                future_audio = [io_executor.submit(read_audio_fixed_5s, p) for p in next_paths]

            x = np.stack(batch_audio, axis=0).astype(np.float32, copy=False)
            outs = session.run(None, {input_name: x})
            chunks.append(outs[out_map[output_name]].astype(np.float32, copy=False))

    if not chunks:
        return _empty_fallback(output_name)
    return np.concatenate(chunks, axis=0)


class RandomPerchSpatialHead(nn.Module):
    """L2-normed projection of (B, 16, 4, 1536) spatial embeddings into
    (B, n_classes, 4) per-class scores, max-pooled over the 16×4 spatial
    grid, then weighted-summed across the 4 sub-heads. Preserves time-
    frequency localization — the right inductive bias for sonotypes whose
    calls live in specific frequency bands.
    """

    INPUT_RANK = 4  # (B, 16, 4, 1536)

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


class RandomPerchFlatHead(nn.Module):
    """L2-normed linear head over the 1536-dim global-pooled Perch embedding
    (the same vector the SSM stack uses). Spatial localization is gone — the
    pre-pool max/mean already collapsed it. Compares against the spatial
    head as an A/B: simpler, ~16× less data, no time-frequency inductive
    bias.
    """

    INPUT_RANK = 2  # (B, 1536)

    def __init__(self, n_classes: int, emb_dim: int = 1536, eps: float = 1e-5,
                 hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.n_classes = int(n_classes)
        if hidden_dim and hidden_dim > 0:
            self.body = nn.Sequential(
                nn.Linear(emb_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), n_classes),
            )
        else:
            self.body = nn.Linear(emb_dim, n_classes)
        # Match the spatial head's L2-norm preconditioning so the loss/lr
        # regimes transfer between heads without retuning.
        self.bias = nn.Parameter(torch.zeros(1, n_classes))
        nn.init.zeros_(self.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = embedding / torch.sqrt((embedding * embedding).sum(dim=-1, keepdim=True) + self.eps)
        logits = self.body(x) + self.bias
        return logits


def make_head(head_type: str, n_classes: int) -> nn.Module:
    """Factory: pick the head class for the requested feature space.

    Each head class declares its expected `INPUT_RANK` so the trainer can
    sanity-check feature shapes before allocating tensors.
    """
    if head_type == "spatial":
        return RandomPerchSpatialHead(n_classes=n_classes)
    if head_type == "flat":
        return RandomPerchFlatHead(n_classes=n_classes)
    raise ValueError(f"Unknown head_type: {head_type!r} (expected 'spatial' or 'flat')")


def head_to_output_name(head_type: str) -> str:
    return "spatial_embedding" if head_type == "spatial" else "embedding"


def build_target_matrix(label_lists: Sequence[Sequence[str]], labels_subset: Sequence[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels_subset)}
    y = np.zeros((len(label_lists), len(labels_subset)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for label in labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                y[i, idx] = 1.0
    return y


def _r4(x) -> float | None:
    """Round-or-None: 4 decimals, NaN→None for JSON/CSV friendliness."""
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def macro_auc_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, int]:
    pos = y_true.sum(axis=0)
    keep = (pos > 0) & (pos < y_true.shape[0])
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(y_true[:, keep], probs[:, keep], average="macro")
    return float(auc), n


def taxa_auc_from_logits(
    y_true: np.ndarray, logits: np.ndarray, target_labels: Sequence[str],
    taxonomy_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Per-taxonomic-class macro AUC over the head's target_labels.

    Returns {taxon_name: {"macro_auc": float|None, "n_eval_classes": int}}.
    Skips classes with no positives (matches macro_auc_from_logits semantics
    + the BirdCLEF official metric).
    """
    label_to_taxon = dict(zip(
        taxonomy_df["primary_label"].astype(str),
        taxonomy_df["class_name"].astype(str),
    ))
    pos = y_true.sum(axis=0)
    out: Dict[str, Dict[str, Any]] = {}
    # Build per-taxon column-index list (within target_labels' columns)
    by_taxon: Dict[str, List[int]] = {}
    for ci, lb in enumerate(target_labels):
        by_taxon.setdefault(label_to_taxon.get(str(lb), "Unknown"), []).append(ci)
    probs = 1.0 / (1.0 + np.exp(-logits))
    for taxon, cols in by_taxon.items():
        cols_arr = np.array(cols, dtype=np.int64)
        keep = (pos[cols_arr] > 0) & (pos[cols_arr] < y_true.shape[0])
        n_eval = int(keep.sum())
        if n_eval == 0:
            out[taxon] = {"macro_auc": None, "n_eval_classes": 0}
            continue
        kept = cols_arr[keep]
        try:
            auc = float(roc_auc_score(y_true[:, kept], probs[:, kept], average="macro"))
        except ValueError:
            auc = float("nan")
        out[taxon] = {"macro_auc": _r4(auc), "n_eval_classes": n_eval}
    return out


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
        rows.append(
            {
                "label": str(label),
                "n_pos": pos,
                "n_neg": neg,
                "evaluable": bool(evaluable),
                "auc": _r4(auc),
            }
        )
    return pd.DataFrame(rows).sort_values(["evaluable", "auc"], ascending=[False, True],
                                          na_position="last").reset_index(drop=True)


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
    head_type: str = "spatial",
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

    model = make_head(head_type=head_type, n_classes=y_train.shape[1])
    if x_train.ndim != model.INPUT_RANK:
        raise ValueError(
            f"head_type={head_type!r} expects rank-{model.INPUT_RANK} features, "
            f"got x_train.shape={x_train.shape}. Re-extract embeddings with "
            f"output_name={head_to_output_name(head_type)!r}."
        )
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
        for xb, yb in tqdm(
            train_loader,
            desc=f"Fold {fold_id:02d} epoch {epoch:02d} train",
            leave=False,
            disable=quiet,
        ):
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
            for xb, yb in tqdm(
                val_loader,
                desc=f"Fold {fold_id:02d} epoch {epoch:02d} val",
                leave=False,
                disable=quiet,
            ):
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
                "fold": int(fold_id),
                "epoch": int(epoch),
                "train_loss": _r4(train_loss),
                "val_loss": _r4(val_loss),
                "val_macro_auc": _r4(val_auc),
                "n_eval_classes": int(n_eval),
            }
        )
        print(
            f"Fold {fold_id:02d} epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macro_auc={val_auc:.4f} ({n_eval} classes)"
        )

        auc_cmp = -np.inf if np.isnan(val_auc) else float(val_auc)
        if (auc_cmp > best_auc) or (auc_cmp == best_auc and val_loss < best_val_loss):
            best_auc = auc_cmp
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_logits = val_logits_np

    best_metrics = {
        "fold": int(fold_id),
        "best_val_macro_auc": _r4(None if not np.isfinite(best_auc) else best_auc),
        "best_val_loss": _r4(best_val_loss),
        "n_eval_classes_best": int(macro_auc_from_logits(y_val, best_val_logits)[1]),
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(y_val)),
        "loss_name": str(loss_name),
        "focal_gamma": _r4(focal_gamma),
        "sampler_power": _r4(sampler_power),
        "pos_weight_cap": _r4(pos_weight_cap),
    }
    return best_metrics, best_state, history, best_val_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a random head on frozen Perch features for no-signal classes.")
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--soundscape-dir", type=Path, default=DEFAULT_SOUNDSCAPE_DIR)
    parser.add_argument("--sound-labels-csv", type=Path, default=DEFAULT_DATA_DIR / "train_soundscapes_labels.csv")
    parser.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_DATA_DIR / "taxonomy.csv")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--perch-labels-csv", type=Path, default=DEFAULT_PERCH_LABELS)
    parser.add_argument("--batch-files", type=int, default=8)
    parser.add_argument("--extra-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--sweep-lr", type=str, default="1e-3,5e-4")
    parser.add_argument("--sweep-weight-decay", type=str, default="1e-4")
    parser.add_argument("--sweep-loss", type=str, default="bce,focal")
    parser.add_argument("--sweep-focal-gamma", type=str, default="1.5,2.0")
    parser.add_argument("--sweep-sampler-power", type=str, default="0.0,1.0")
    parser.add_argument("--sweep-epochs", type=str, default="20")
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--extra-train-csv", type=Path, default=DEFAULT_EXTRA_TRAIN_CSV)
    parser.add_argument("--extra-train-audio-dir", type=Path, default=DEFAULT_EXTRA_TRAIN_AUDIO_DIR)
    parser.add_argument("--extra-max-samples", type=int, default=0, help="Optional cap for extra train-audio samples.")
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--per-class-oof-csv", type=Path, default=DEFAULT_PER_CLASS_OOF_CSV)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--head-type", choices=["spatial", "flat"], default="spatial",
                        help="Which Perch ONNX feature to read + which head to train. "
                             "'spatial' = (16,4,1536) pre-pooled grid + RandomPerchSpatialHead "
                             "(default; preserves time-frequency localization, ~16× more memory). "
                             "'flat' = (1536,) global-pooled — same vector the SSM uses + "
                             "RandomPerchFlatHead (smaller, faster, no spatial bias).")
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

    _mapped_pos, no_signal_pos, proxy_map, no_signal_labels = build_no_signal_targets(
        taxonomy_df=taxonomy,
        perch_labels_df=perch_labels,
        primary_labels=primary_labels,
    )
    if len(no_signal_labels) == 0:
        raise RuntimeError("No no-signal classes found with current mapping/proxy definition.")

    truth_df, full_files = build_truth_windows(args.sound_labels_csv, args.soundscape_dir)
    if args.limit_files > 0:
        full_files = full_files[: args.limit_files]
        truth_df = truth_df[truth_df["filename"].isin(full_files)].copy()
        truth_df = truth_df.sort_values(["filename", "end_sec"]).reset_index(drop=True)
    if len(full_files) == 0:
        raise RuntimeError("No soundscape files with labels found.")

    head_type = str(args.head_type)
    onnx_output_name = head_to_output_name(head_type)
    print(f"[head] head_type={head_type!r}  pulling Perch ONNX output {onnx_output_name!r}")

    soundscape_paths = [args.soundscape_dir / fn for fn in full_files]
    pred_meta, spatial_all = extract_spatial_embeddings(
        onnx_path=args.onnx_path,
        soundscape_paths=soundscape_paths,
        batch_files=args.batch_files,
        verbose=not args.quiet,
        output_name=onnx_output_name,
    )

    pred_rows_before = len(pred_meta)
    merged = pred_meta.merge(truth_df[["row_id", "label_list"]], on="row_id", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise RuntimeError("No labeled prediction rows after aligning predictions with truth.")
    dropped_unlabeled = pred_rows_before - len(merged)
    if dropped_unlabeled > 0:
        print(f"Dropped unlabeled windows after alignment: {dropped_unlabeled}")

    target_labels = list(no_signal_labels)
    label_to_idx = {label: i for i, label in enumerate(target_labels)}
    y = build_target_matrix(merged["label_list"].tolist(), target_labels)

    soundscape_pos_per_class = y.sum(axis=0).astype(np.int64)
    n_soundscape_pos_classes = int((soundscape_pos_per_class > 0).sum())
    if n_soundscape_pos_classes == 0:
        raise RuntimeError("No positives for no-signal classes in selected soundscape data.")

    # Empty-extra fallback shape mirrors the Perch ONNX output rank we'll use.
    x_extra = _empty_fallback(onnx_output_name)
    y_extra = np.zeros((0, len(target_labels)), dtype=np.float32)
    n_extra_rows_raw = 0
    n_extra_rows_used = 0
    n_extra_missing_audio = 0

    if args.extra_train_csv.exists() and args.extra_train_audio_dir.exists():
        extra_df = pd.read_csv(args.extra_train_csv)
        extra_df["primary_label"] = extra_df["primary_label"].astype(str)
        extra_df["filename"] = extra_df["filename"].astype(str)
        extra_df = extra_df[extra_df["primary_label"].isin(set(target_labels))].copy()
        n_extra_rows_raw = int(len(extra_df))
        if n_extra_rows_raw > 0:
            extra_df["audio_path"] = extra_df["filename"].map(lambda p: args.extra_train_audio_dir / p)
            exists_mask = extra_df["audio_path"].map(lambda p: Path(p).exists())
            n_extra_missing_audio = int((~exists_mask).sum())
            extra_df = extra_df[exists_mask].copy()

            if args.extra_max_samples > 0 and len(extra_df) > args.extra_max_samples:
                extra_df = extra_df.sample(n=args.extra_max_samples, random_state=args.seed).reset_index(drop=True)

            n_extra_rows_used = int(len(extra_df))
            if n_extra_rows_used > 0:
                extra_paths = extra_df["audio_path"].tolist()
                x_extra = extract_spatial_embeddings_clips(
                    onnx_path=args.onnx_path,
                    clip_paths=extra_paths,
                    batch_size=args.extra_batch_size,
                    verbose=not args.quiet,
                    output_name=onnx_output_name,
                )
                y_extra = np.zeros((n_extra_rows_used, len(target_labels)), dtype=np.float32)
                extra_secondary = (
                    extra_df["secondary_labels"]
                    if "secondary_labels" in extra_df.columns
                    else pd.Series(["[]"] * n_extra_rows_used)
                )
                for i, (lbl, sec_raw) in enumerate(
                    zip(extra_df["primary_label"].tolist(), extra_secondary.tolist())
                ):
                    if lbl in label_to_idx:
                        y_extra[i, label_to_idx[lbl]] = 1.0
                    for sec_lbl in parse_secondary_labels_field(sec_raw):
                        if sec_lbl in label_to_idx:
                            y_extra[i, label_to_idx[sec_lbl]] = 1.0

    groups = merged["filename"].astype(str).to_numpy()
    unique_files = np.unique(groups)
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")
    if len(unique_files) < args.n_splits:
        raise ValueError(
            f"Not enough unique files for GroupKFold: n_files={len(unique_files)}, n_splits={args.n_splits}"
        )

    sweep_lrs = parse_float_list(args.sweep_lr)
    sweep_wds = parse_float_list(args.sweep_weight_decay)
    sweep_losses = parse_str_list(args.sweep_loss)
    sweep_gammas = parse_float_list(args.sweep_focal_gamma)
    sweep_sampler_powers = parse_float_list(args.sweep_sampler_power)
    sweep_epochs = [int(x) for x in parse_float_list(args.sweep_epochs)]
    configs = []
    for lr, wd, loss_name, gamma, sampler_power, epochs in itertools.product(
        sweep_lrs, sweep_wds, sweep_losses, sweep_gammas, sweep_sampler_powers, sweep_epochs
    ):
        if loss_name == "bce":
            gamma = 0.0
        configs.append(
            {
                "lr": float(lr),
                "weight_decay": float(wd),
                "loss_name": str(loss_name),
                "focal_gamma": float(gamma),
                "sampler_power": float(sampler_power),
                "epochs": int(epochs),
            }
        )

    est_train_soundscape = int(y.shape[0] - (y.shape[0] // args.n_splits))
    print(
        f"Source rows: soundscape_total={y.shape[0]} | "
        f"extra_train_audio_used={x_extra.shape[0]} "
        f"(raw_match={n_extra_rows_raw}, missing_audio={n_extra_missing_audio})"
    )
    print(
        f"Per fold (approx): soundscape_train≈{est_train_soundscape}, "
        f"soundscape_val≈{y.shape[0] // args.n_splits}, extra_train={x_extra.shape[0]}"
    )
    print(f"Configs to run: {len(configs)}")

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

        for fold_id, (tr_idx, va_idx) in enumerate(
            splitter.split(np.arange(y.shape[0]), groups=groups),
            start=1,
        ):
            print(
                f"Fold {fold_id:02d} source rows: "
                f"soundscape_train={len(tr_idx)} soundscape_val={len(va_idx)} extra_train={x_extra.shape[0]}"
            )
            x_train = spatial_all[tr_idx]
            y_train = y[tr_idx]
            x_val = spatial_all[va_idx]
            y_val = y[va_idx]

            if x_extra.shape[0] > 0:
                x_train_fold = np.concatenate([x_train, x_extra], axis=0)
                y_train_fold = np.concatenate([y_train, y_extra], axis=0)
            else:
                x_train_fold = x_train
                y_train_fold = y_train

            metrics, state, history, val_logits_best = train_one_fold(
                x_train=x_train_fold,
                y_train=y_train_fold,
                x_val=x_val,
                y_val=y_val,
                batch_size=args.batch_size,
                epochs=int(cfg["epochs"]),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                loss_name=str(cfg["loss_name"]),
                focal_gamma=float(cfg["focal_gamma"]),
                sampler_power=float(cfg["sampler_power"]),
                pos_weight_cap=float(args.pos_weight_cap),
                seed=args.seed,
                fold_id=fold_id,
                quiet=args.quiet,
                head_type=head_type,
            )
            metrics["n_train_soundscape_rows"] = int(len(tr_idx))
            metrics["n_train_extra_rows"] = int(x_extra.shape[0])
            metrics["head_type"] = head_type
            fold_metrics.append(metrics)
            all_history.extend(history)
            fold_states.append(state)

            oof_logits[va_idx] = val_logits_best
            oof_filled[va_idx] = True

            fold_auc_val = metrics["best_val_macro_auc"]
            fold_auc_cmp = -np.inf if fold_auc_val is None else float(fold_auc_val)
            if fold_auc_cmp > best_fold_auc:
                best_fold_auc = fold_auc_cmp
                best_fold_state = state

        if not np.all(oof_filled):
            missing = int((~oof_filled).sum())
            raise RuntimeError(f"OOF coverage incomplete, missing {missing} rows.")

        oof_auc, oof_n_eval = macro_auc_from_logits(y, oof_logits)
        oof_taxa = taxa_auc_from_logits(
            y_true=y, logits=oof_logits, target_labels=target_labels,
            taxonomy_df=taxonomy,
        )
        per_class_oof = per_class_auc_from_logits(y_true=y, logits=oof_logits, labels=target_labels)
        print(f"OOF macro AUC: {oof_auc:.4f} ({oof_n_eval} classes)")
        for taxon, m in sorted(oof_taxa.items()):
            print(f"  {taxon:<10s} oof_auc={m['macro_auc']}  (n_eval={m['n_eval_classes']})")

        run_payload = {
            "config": cfg,
            "oof_auc": _r4(oof_auc),
            "oof_n_eval": int(oof_n_eval),
            "oof_taxa": oof_taxa,
            "fold_metrics": fold_metrics,
            "history": all_history,
            "fold_states": fold_states,
            "best_fold_state": best_fold_state,
            "per_class_oof": per_class_oof,
        }
        # Sweep CSV row — `primary` mirrors the convention used by the SSM /
        # focal trainer summaries so a single sort across all sweep CSVs works.
        sweep_row = {
            "config_id": int(cfg_idx),
            "head_type": head_type,
            "loss": cfg["loss_name"],
            "focal_gamma": _r4(cfg["focal_gamma"]),
            "sampler_power": _r4(cfg["sampler_power"]),
            "lr": _r4(cfg["lr"]),
            "weight_decay": _r4(cfg["weight_decay"]),
            "epochs": int(cfg["epochs"]),
            "primary": _r4(oof_auc),
            "oof_macro_auc": _r4(oof_auc),
            "oof_n_eval_classes": int(oof_n_eval),
        }
        for taxon, m in oof_taxa.items():
            sweep_row[f"oof_auc_{taxon.lower()}"] = m["macro_auc"]
            sweep_row[f"oof_n_eval_{taxon.lower()}"] = m["n_eval_classes"]
        sweep_rows.append(sweep_row)

        # Best-run selection compares raw floats safely (None < anything via -inf)
        best_score = float("-inf") if best_run is None else (best_run["oof_auc"] or float("-inf"))
        cur_score = float(oof_auc) if np.isfinite(oof_auc) else float("-inf")
        if best_run is None or cur_score > best_score:
            best_run = run_payload

    if best_run is None:
        raise RuntimeError("No config run completed.")

    if args.sweep_csv is not None:
        args.sweep_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sweep_rows).sort_values("primary", ascending=False,
                                             na_position="last").to_csv(args.sweep_csv, index=False)
        print(f"Saved sweep CSV: {args.sweep_csv}")

    best_cfg = best_run["config"]
    oof_auc = float(best_run["oof_auc"]) if best_run["oof_auc"] is not None else float("nan")
    oof_n_eval = int(best_run["oof_n_eval"])
    oof_taxa = best_run["oof_taxa"]
    fold_metrics = best_run["fold_metrics"]
    all_history = best_run["history"]
    fold_states = best_run["fold_states"]
    best_fold_state = best_run["best_fold_state"]
    per_class_oof = best_run["per_class_oof"]

    evaluable_df = per_class_oof[per_class_oof["evaluable"]].copy()
    print("\n=== Best Config ===")
    print(
        f"loss={best_cfg['loss_name']} gamma={best_cfg['focal_gamma']} sampler={best_cfg['sampler_power']} "
        f"lr={best_cfg['lr']} wd={best_cfg['weight_decay']} epochs={best_cfg['epochs']} | "
        f"OOF={oof_auc:.4f} ({oof_n_eval} classes)"
    )
    print("Per-taxon OOF:")
    for taxon, m in sorted(oof_taxa.items()):
        print(f"  {taxon:<10s} oof_auc={m['macro_auc']}  (n_eval={m['n_eval_classes']})")
    print("Worst OOF classes:")
    if len(evaluable_df) == 0:
        print("  none")
    else:
        for _, row in evaluable_df.nsmallest(8, "auc").iterrows():
            print(f"  {row['label']}: auc={row['auc']:.4f} pos={int(row['n_pos'])} neg={int(row['n_neg'])}")
    print("Best OOF classes:")
    if len(evaluable_df) == 0:
        print("  none")
    else:
        for _, row in evaluable_df.nlargest(8, "auc").iterrows():
            print(f"  {row['label']}: auc={row['auc']:.4f} pos={int(row['n_pos'])} neg={int(row['n_neg'])}")

    if args.per_class_oof_csv is not None:
        args.per_class_oof_csv.parent.mkdir(parents=True, exist_ok=True)
        per_class_oof.to_csv(args.per_class_oof_csv, index=False)
        print(f"Saved per-class OOF CSV: {args.per_class_oof_csv}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        if best_fold_state is None:
            raise RuntimeError("No fold model state selected as best.")
        torch.save(
            {
                "state_dict": best_fold_state,
                "fold_states": fold_states,
                "target_labels": target_labels,
                "no_signal_labels_all": no_signal_labels,
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
                    "extra_train_csv": str(args.extra_train_csv),
                    "extra_train_audio_dir": str(args.extra_train_audio_dir),
                    "extra_batch_size": int(args.extra_batch_size),
                    "extra_max_samples": int(args.extra_max_samples),
                },
                "mapping_stats": {
                    "n_primary_labels": int(len(primary_labels)),
                    "n_no_signal_labels": int(len(no_signal_labels)),
                    "n_no_signal_positive_in_soundscapes": int(n_soundscape_pos_classes),
                    "n_proxy_labels": int(len(proxy_map)),
                    "n_extra_rows_raw": int(n_extra_rows_raw),
                    "n_extra_rows_used": int(n_extra_rows_used),
                    "n_extra_missing_audio": int(n_extra_missing_audio),
                },
                "fold_metrics": fold_metrics,
                "oof": {
                    "primary": _r4(None if not np.isfinite(oof_auc) else oof_auc),
                    "macro_auc": _r4(None if not np.isfinite(oof_auc) else oof_auc),
                    "n_eval_classes": int(oof_n_eval),
                    "by_taxon": oof_taxa,
                },
                "per_class_oof": per_class_oof.to_dict(orient="records"),
                "history": all_history,
            },
            args.save_path,
        )
        print(f"Saved best-fold no-signal head checkpoint (+all fold states): {args.save_path}")

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "onnx_path": str(args.onnx_path),
            "perch_labels_csv": str(perch_labels_csv),
            "best_config": best_cfg,
            "sweep_rows": sweep_rows,
            "n_files": int(len(full_files)),
            "n_rows": int(len(merged)),
            "n_no_signal_labels": int(len(no_signal_labels)),
            "n_no_signal_positive_in_soundscapes": int(n_soundscape_pos_classes),
            "n_extra_rows_raw": int(n_extra_rows_raw),
            "n_extra_rows_used": int(n_extra_rows_used),
            "n_extra_missing_audio": int(n_extra_missing_audio),
            "no_signal_labels": no_signal_labels,
            "target_labels": target_labels,
            "fold_metrics": fold_metrics,
            "oof": {
                "macro_auc": None if not np.isfinite(oof_auc) else float(oof_auc),
                "n_eval_classes": int(oof_n_eval),
            },
            "per_class_oof_csv": str(args.per_class_oof_csv) if args.per_class_oof_csv is not None else None,
            "per_class_oof": per_class_oof.to_dict(orient="records"),
            "history": all_history,
        }
        args.metrics_json.write_text(json.dumps(out, indent=2))
        print(f"Saved metrics JSON: {args.metrics_json}")

    print(
        f"No-signal labels (targeted): {len(target_labels)} | "
        f"with positives in soundscapes: {n_soundscape_pos_classes}"
    )
    print(
        f"Training rows by source: soundscape_train_per_fold=variable | "
        f"extra_train_audio={n_extra_rows_used} (raw matching csv rows={n_extra_rows_raw}, missing_audio={n_extra_missing_audio})"
    )
    print("Done.")


if __name__ == "__main__":
    main()
