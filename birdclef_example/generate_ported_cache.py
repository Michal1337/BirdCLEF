#!/usr/bin/env python3

"""Generate the Perch cache consumed by the ported BirdCLEF models.

This script reproduces the cache contract used by predict_ported.py:
- full_perch_meta.parquet
- full_perch_arrays.npz with scores_full_raw and emb_full

The score tensor is in BirdCLEF primary-label space, with exact Perch logits
copied into mapped classes and genus proxy scores optionally filled for selected
unmapped classes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import parse_primary_labels  # noqa: E402


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
SAMPLE_RATE = 32000
WINDOW_SEC = 5
N_WINDOWS = 12
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
FILE_SAMPLES = SAMPLE_RATE * 60
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "perch_onnx" / "perch_v2.onnx"
DEFAULT_TF_MODEL_DIR = REPO_ROOT / "models" / "perch_v2_cpu" / "1"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SOUNDSCAPE_DIR = REPO_ROOT / "data" / "train_soundscapes"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "perch_cache_finetuned"
DEFAULT_PERCH_LABELS_CSV = DEFAULT_TF_MODEL_DIR / "assets" / "labels.csv"
DEFAULT_TAXONOMY_CSV = DEFAULT_DATA_DIR / "taxonomy.csv"
DEFAULT_SAMPLE_SUBMISSION = DEFAULT_DATA_DIR / "sample_submission.csv"
DEFAULT_SOUND_LABELS = DEFAULT_DATA_DIR / "train_soundscapes_labels.csv"
DEFAULT_CACHE_DIR = Path(os.environ.get("PERCH_CACHE_DIR", DEFAULT_OUT_DIR))
DEFAULT_FINETUNED_ONNX_PATH = REPO_ROOT / "models" / "perch_onnx" / "perch_v2_finetuned.onnx"
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}


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


def union_labels(series: pd.Series) -> List[str]:
    labels = set()
    for value in series:
        labels.update(parse_primary_labels(value))
    return sorted(labels)


def build_full_files(sound_labels_csv: Path) -> List[str]:
    soundscape_labels = pd.read_csv(sound_labels_csv)
    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)
    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int).astype(str)
    windows_per_file = sc_clean.groupby("filename").size()
    return sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())


def build_truth_windows(sound_labels_csv: Path, soundscape_dir: Path) -> pd.DataFrame:
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
    return grouped


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
    n_eval = int(keep.sum())
    if n_eval == 0:
        return float("nan"), 0
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(y_true[:, keep], probs[:, keep], average="macro")
    return float(auc), n_eval


def load_perch_backend(backend: str, onnx_path: Path, tf_model_dir: Path):
    backend = backend.lower()
    if backend not in {"auto", "onnx", "tf"}:
        raise ValueError("backend must be one of: auto, onnx, tf")

    if backend in {"auto", "onnx"} and onnx_path.exists():
        import onnxruntime as ort  # imported lazily

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
        so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_map = {o.name: i for i, o in enumerate(session.get_outputs())}

        def infer(x: np.ndarray):
            outs = session.run(None, {input_name: x})
            return {
                "label": outs[output_map["label"]].astype(np.float32, copy=False),
                "embedding": outs[output_map["embedding"]].astype(np.float32, copy=False),
                "spatial_embedding": outs[output_map["spatial_embedding"]].astype(np.float32, copy=False),
            }

        return infer, "onnx"

    if backend in {"auto", "tf"}:
        import tensorflow as tf  # imported lazily

        birdclassifier = tf.saved_model.load(str(tf_model_dir))
        infer_fn = birdclassifier.signatures["serving_default"]

        def infer(x: np.ndarray):
            outputs = infer_fn(inputs=tf.convert_to_tensor(x))
            return {
                "label": outputs["label"].numpy().astype(np.float32, copy=False),
                "embedding": outputs["embedding"].numpy().astype(np.float32, copy=False),
                "spatial_embedding": outputs["spatial_embedding"].numpy().astype(np.float32, copy=False),
            }

        return infer, "tf"

    raise FileNotFoundError(
        f"Could not load backend {backend!r}: ONNX model missing at {onnx_path} and TF model not selected/available."
    )


def load_finetuned_backend(onnx_path: Path, expected_classes: int):
    """Load finetuned ONNX head with flexible output-name/shape handling.

    The finetuned export can have different top-layer dimensions and may not use
    the same output names as the base Perch model. This loader resolves a logits
    tensor dynamically at runtime.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"Finetuned ONNX model missing at {onnx_path}")

    import onnxruntime as ort  # imported lazily

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    preferred_names = (
        "label",
        "logits",
        "output",
        "predictions",
        "probabilities",
        "scores",
    )

    def infer(x: np.ndarray):
        outs = session.run(None, {input_name: x})
        name_to_arr = {name: arr for name, arr in zip(output_names, outs)}

        # 1) Prefer known output names.
        for name in preferred_names:
            arr = name_to_arr.get(name)
            if arr is None:
                continue
            if arr.ndim == 2 and (arr.shape[1] == expected_classes or expected_classes <= 0):
                return {"label": arr.astype(np.float32, copy=False)}

        # 2) Prefer any 2D output with expected class dimension.
        for arr in outs:
            if arr.ndim == 2 and arr.shape[1] == expected_classes:
                return {"label": arr.astype(np.float32, copy=False)}

        # 3) Fallback to first 2D output.
        for arr in outs:
            if arr.ndim == 2:
                return {"label": arr.astype(np.float32, copy=False)}

        shapes = [tuple(a.shape) for a in outs]
        raise RuntimeError(
            "Could not resolve finetuned logits output. "
            f"Expected 2D logits with {expected_classes} classes. "
            f"Available outputs: {list(zip(output_names, shapes))}"
        )

    return infer, "onnx_finetuned"


class RandomPerchSpatialHead(nn.Module):
    """Random projection head for Perch embeddings."""
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


def load_finetuned_pytorch(checkpoint_path: Path, n_classes: int):
    """Load finetuned head from PyTorch checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        n_classes: Number of no-signal classes
    
    Returns:
        (infer_fn, backend_name) tuple where infer_fn takes spatial embeddings 
        (batch, 16, 4, 1536) and returns logits (batch, n_classes)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint missing at {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Get the best fold state (or full model state if available)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "fold_states" in ckpt and len(ckpt["fold_states"]) > 0:
        state_dict = ckpt["fold_states"][0]
    else:
        state_dict = ckpt
    
    # Create model and load weights
    model = RandomPerchSpatialHead(n_classes=n_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    def infer(x: np.ndarray) -> dict:
        """Take spatial embeddings (batch*windows, 16, 4, 1536) and return logits."""
        with torch.no_grad():
            x_torch = torch.from_numpy(x).float()
            logits = model(x_torch).cpu().numpy()
        return {"label": logits.astype(np.float32)}
    
    return infer, "pytorch"


def build_exact_mapping(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray, List[str]]:
    """Build mapping exactly as training script does."""
    no_label_index = len(perch_labels_df)
    # Use exact same merge logic as training script: merge full perch_labels_df, not subset of columns.
    mapping = taxonomy_df.merge(perch_labels_df, on="scientific_name", how="left")
    mapping["perch_index"] = mapping["perch_index"].fillna(no_label_index).astype(int)

    label_to_perch = mapping.set_index("primary_label")["perch_index"].to_dict()
    bc_indices = np.array([int(label_to_perch.get(lbl, no_label_index)) for lbl in primary_labels], dtype=np.int32)
    mapped_mask = bc_indices != no_label_index
    mapped_positions = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc_indices = bc_indices[mapped_mask].astype(np.int32)

    class_name_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
    taxonomy_by_label = taxonomy_df.set_index("primary_label")
    unmapped_positions = np.where(~mapped_mask)[0].astype(np.int32)

    proxy_map_raw: Dict[int, np.ndarray] = {}
    for idx in unmapped_positions:
        lbl = primary_labels[int(idx)]
        if lbl not in taxonomy_by_label.index:
            continue
        sci = str(taxonomy_by_label.at[lbl, "scientific_name"])
        genus = sci.split()[0]
        hits = perch_labels_df[
            perch_labels_df["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map_raw[int(idx)] = hits["perch_index"].astype(np.int32).to_numpy()

    proxy_map: Dict[int, np.ndarray] = {}
    for idx, perch_idxs in proxy_map_raw.items():
        if class_name_map.get(primary_labels[idx]) in PROXY_TAXA:
            proxy_map[idx] = perch_idxs

    no_signal_positions = np.array([int(idx) for idx in unmapped_positions if int(idx) not in proxy_map], dtype=np.int32)
    no_signal_labels = [primary_labels[int(idx)] for idx in no_signal_positions]
    return mapped_positions, mapped_bc_indices, proxy_map, no_signal_positions, no_signal_labels


def infer_perch_with_embeddings(
    paths: Sequence[Path],
    infer_base_fn,
    infer_finetuned_fn,
    n_classes: int,
    mapped_positions: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_map: Dict[int, np.ndarray],
    no_signal_positions: np.ndarray,
    batch_files: int = 16,
    verbose: bool = True,
    proxy_reduce: str = "max",
    emb_dim: int = 1536,
):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, n_classes), dtype=np.float32)
    embeddings = np.zeros((n_rows, emb_dim), dtype=np.float32)

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

            outputs_base = infer_base_fn(x)
            base_logits = outputs_base["label"]
            emb = outputs_base["embedding"]
            spatial_emb = outputs_base["spatial_embedding"]

            scores[batch_row_start:write_row, mapped_positions] = base_logits[:, mapped_bc_indices]

            for pos, bc_idx_arr in proxy_map.items():
                sub = base_logits[:, bc_idx_arr]
                if proxy_reduce == "max":
                    proxy_score = sub.max(axis=1)
                elif proxy_reduce == "mean":
                    proxy_score = sub.mean(axis=1)
                else:
                    raise ValueError("proxy_reduce must be 'max' or 'mean'")
                scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

            if len(no_signal_positions) > 0:
                outputs_finetuned = infer_finetuned_fn(spatial_emb)
                no_signal_logits = outputs_finetuned["label"]
                if no_signal_logits.shape[1] != len(no_signal_positions):
                    raise ValueError(
                        "Finetuned output size mismatch for no-signal classes: "
                        f"expected {len(no_signal_positions)}, got {no_signal_logits.shape[1]}"
                    )
                scores[batch_row_start:write_row, no_signal_positions] = no_signal_logits.astype(np.float32, copy=False)

            embeddings[batch_row_start:write_row] = emb

            del x, outputs_base, base_logits, emb, spatial_emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        }
    )
    return meta_df, scores, embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the cache required by the ported BirdCLEF models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--soundscape-dir", type=Path, default=DEFAULT_SOUNDSCAPE_DIR)
    parser.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_TAXONOMY_CSV)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--sound-labels-csv", type=Path, default=DEFAULT_SOUND_LABELS)
    parser.add_argument("--perch-labels-csv", type=Path, default=DEFAULT_PERCH_LABELS_CSV)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments_ft" / "no_stage2_cfg2.pt",
        help="PyTorch checkpoint with finetuned head for no-signal classes.",
    )
    parser.add_argument("--tf-model-dir", type=Path, default=DEFAULT_TF_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--backend", choices=["auto", "onnx", "tf"], default="auto")
    parser.add_argument("--batch-files", type=int, default=16)
    parser.add_argument("--proxy-reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--limit-files", type=int, default=0, help="Optional debug limit for the number of full files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [
        args.taxonomy_csv,
        args.sample_submission,
        args.sound_labels_csv,
        args.perch_labels_csv,
        args.soundscape_dir,
        args.onnx_path,
        args.checkpoint,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_sub = pd.read_csv(args.sample_submission)
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    full_files = build_full_files(args.sound_labels_csv)
    if args.limit_files and args.limit_files > 0:
        full_files = full_files[: args.limit_files]

    taxonomy = pd.read_csv(args.taxonomy_csv)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    perch_labels = pd.read_csv(args.perch_labels_csv).reset_index().rename(columns={"index": "perch_index"})
    if "scientific_name" not in perch_labels.columns:
        if "inat2024_fsd50k" in perch_labels.columns:
            perch_labels = perch_labels.rename(columns={"inat2024_fsd50k": "scientific_name"})
        else:
            raise ValueError(
                f"Perch labels CSV {args.perch_labels_csv} is missing scientific_name/inat2024_fsd50k column."
            )
    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)

    # Build mapping using exact same logic as training script.
    mapped_positions, mapped_bc_indices, proxy_map, no_signal_positions, no_signal_labels = build_exact_mapping(
        taxonomy_df=taxonomy,
        perch_labels_df=perch_labels,
        primary_labels=primary_labels,
    )

    infer_base_fn, backend_used = load_perch_backend(args.backend, args.onnx_path, args.tf_model_dir)
    infer_finetuned_fn, finetuned_backend_used = load_finetuned_pytorch(
        args.checkpoint,
        n_classes=len(no_signal_positions),
    )
    print(f"Base backend: {backend_used}")
    print(f"Finetuned backend: {finetuned_backend_used}")
    print(f"Full files: {len(full_files)}")
    print(f"Mapped classes: {len(mapped_positions)}")
    print(f"Proxy targets: {len(proxy_map)}")
    print(f"No-signal targets (finetuned): {len(no_signal_positions)}")

    truth_df = build_truth_windows(args.sound_labels_csv, args.soundscape_dir)
    truth_df = truth_df[truth_df["filename"].isin(full_files)].copy()

    full_paths = [args.soundscape_dir / fn for fn in full_files]
    meta_df, scores_full_raw, emb_full = infer_perch_with_embeddings(
        full_paths,
        infer_base_fn=infer_base_fn,
        infer_finetuned_fn=infer_finetuned_fn,
        n_classes=n_classes,
        mapped_positions=mapped_positions,
        mapped_bc_indices=mapped_bc_indices,
        proxy_map=proxy_map,
        no_signal_positions=no_signal_positions,
        batch_files=args.batch_files,
        verbose=True,
        proxy_reduce=args.proxy_reduce,
    )

    out_meta = args.output_dir / "full_perch_meta.parquet"
    out_npz = args.output_dir / "full_perch_arrays.npz"
    meta_df.to_parquet(out_meta, index=False)
    np.savez_compressed(out_npz, scores_full_raw=scores_full_raw, emb_full=emb_full)

    merged = meta_df.merge(truth_df[["row_id", "label_list"]], on="row_id", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise RuntimeError("No labeled rows found after aligning generated cache with truth windows.")

    y_full = build_target_matrix(merged["label_list"].tolist(), primary_labels)
    score_frame = pd.DataFrame(scores_full_raw, index=meta_df["row_id"].to_numpy())
    merged_scores = score_frame.loc[merged["row_id"]].to_numpy(dtype=np.float32, copy=False)
    subset_specs = {
        "mapped": np.array(mapped_positions, dtype=np.int32),
        "proxy": np.array(sorted(proxy_map.keys()), dtype=np.int32),
        "no_signal": np.array(no_signal_positions, dtype=np.int32),
        "full": np.arange(len(primary_labels), dtype=np.int32),
    }
    metrics = {}
    for subset_name, subset_positions in subset_specs.items():
        auc, n_eval = macro_auc_from_logits(y_full[:, subset_positions], merged_scores[:, subset_positions])
        metrics[subset_name] = {
            "macro_auc": None if not np.isfinite(auc) else float(auc),
            "n_eval_classes": int(n_eval),
        }

    summary = {
        "base_backend": backend_used,
        "finetuned_backend": finetuned_backend_used,
        "base_onnx_path": str(args.onnx_path),
        "checkpoint": str(args.checkpoint),
        "full_files": len(full_files),
        "rows": int(len(meta_df)),
        "scores_shape": list(scores_full_raw.shape),
        "emb_shape": list(emb_full.shape),
        "mapped_classes": int(len(mapped_positions)),
        "proxy_targets": int(len(proxy_map)),
        "no_signal_targets": int(len(no_signal_positions)),
        "no_signal_labels": no_signal_labels,
        "metrics": metrics,
        "output_meta": str(out_meta),
        "output_npz": str(out_npz),
        "proxy_reduce": args.proxy_reduce,
    }
    (args.output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved cache to:")
    print("  ", out_meta)
    print("  ", out_npz)
    print("  ", args.output_dir / "cache_summary.json")
    print("Metrics:")
    for subset_name in ["mapped", "proxy", "no_signal", "full"]:
        info = metrics[subset_name]
        print(f"  {subset_name}: macro_auc={info['macro_auc']} n_eval_classes={info['n_eval_classes']}")


if __name__ == "__main__":
    main()
