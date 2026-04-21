#!/usr/bin/env python3

"""Run Perch ONNX inference with proxy mapping and report macro ROC-AUC."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score
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
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "eda" / "perch_proxy_inference_auc.json"


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

    windows_per_file = grouped.groupby("filename").size()
    full_files = set(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())

    grouped = grouped[grouped["filename"].isin(full_files)].copy()
    grouped = grouped[grouped["filename"].map(lambda x: (soundscape_dir / x).exists())].copy()
    grouped = grouped.sort_values(["filename", "end_sec"]).reset_index(drop=True)

    available_full_files = sorted(grouped["filename"].drop_duplicates().tolist())
    return grouped, available_full_files


def load_perch_labels(perch_labels_csv: Path) -> pd.DataFrame:
    perch_labels = pd.read_csv(perch_labels_csv).reset_index().rename(columns={"index": "perch_index"})

    if "scientific_name" not in perch_labels.columns:
        if "inat2024_fsd50k" in perch_labels.columns:
            perch_labels = perch_labels.rename(columns={"inat2024_fsd50k": "scientific_name"})
        else:
            raise ValueError(
                f"Perch labels CSV {perch_labels_csv} is missing scientific name column. "
                "Expected one of: scientific_name, inat2024_fsd50k"
            )

    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)
    perch_labels["perch_index"] = perch_labels["perch_index"].astype(int)
    return perch_labels[["perch_index", "scientific_name"]]


def build_mapping_context(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    no_label_index = len(perch_labels_df)

    mapping = taxonomy_df.merge(
        perch_labels_df,
        on="scientific_name",
        how="left",
    )
    mapping["perch_index"] = mapping["perch_index"].fillna(no_label_index).astype(int)

    label_to_perch = mapping.set_index("primary_label")["perch_index"].to_dict()
    perch_indices = np.array([int(label_to_perch.get(label, no_label_index)) for label in primary_labels], dtype=np.int32)

    mapped_mask = perch_indices != no_label_index
    mapped_positions = np.where(mapped_mask)[0].astype(np.int32)
    mapped_perch_indices = perch_indices[mapped_mask].astype(np.int32)

    class_name_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
    unmapped_positions = np.where(~mapped_mask)[0].astype(np.int32)
    taxonomy_by_label = taxonomy_df.set_index("primary_label")

    genus_hits_by_idx: Dict[int, np.ndarray] = {}
    for idx in unmapped_positions:
        target = primary_labels[int(idx)]
        if target not in taxonomy_by_label.index:
            continue
        scientific_name = str(taxonomy_by_label.at[target, "scientific_name"])
        genus = scientific_name.split()[0]

        hits = perch_labels_df[
            perch_labels_df["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            genus_hits_by_idx[int(idx)] = hits["perch_index"].astype(np.int32).to_numpy()

    class_lkp = taxonomy_df[["scientific_name", "class_name"]].drop_duplicates(subset=["scientific_name"])
    perch_with_class = perch_labels_df.merge(class_lkp, on="scientific_name", how="left")
    class_pool: Dict[str, np.ndarray] = {}
    class_df = perch_with_class.dropna(subset=["class_name"])
    for cls, grp in class_df.groupby("class_name"):
        class_pool[str(cls)] = np.unique(grp["perch_index"].astype(np.int32).to_numpy())

    class_hits_by_idx: Dict[int, np.ndarray] = {}
    for idx in unmapped_positions:
        target = primary_labels[int(idx)]
        class_name = class_name_map.get(target)
        if class_name in class_pool:
            class_hits_by_idx[int(idx)] = class_pool[str(class_name)]

    baseline_proxy_map = build_proxy_map_from_strategy(
        primary_labels=primary_labels,
        unmapped_positions=unmapped_positions,
        class_name_map=class_name_map,
        genus_hits_by_idx=genus_hits_by_idx,
        class_hits_by_idx=class_hits_by_idx,
        source="genus",
        filter_proxy_taxa=True,
        top_k=0,
    )

    baseline_stats = {
        "n_classes": int(len(primary_labels)),
        "n_mapped": int(mapped_positions.size),
        "n_unmapped": int((~mapped_mask).sum()),
        "n_proxy": int(len(baseline_proxy_map)),
        "n_still_without_signal": int((~mapped_mask).sum() - len(baseline_proxy_map)),
    }
    context = {
        "primary_labels": list(primary_labels),
        "class_name_map": class_name_map,
        "unmapped_positions": unmapped_positions,
        "genus_hits_by_idx": genus_hits_by_idx,
        "class_hits_by_idx": class_hits_by_idx,
        "baseline_proxy_map": baseline_proxy_map,
        "baseline_stats": baseline_stats,
        "n_classes": int(len(primary_labels)),
        "n_mapped": int(mapped_positions.size),
        "n_unmapped": int((~mapped_mask).sum()),
    }
    return mapped_positions, mapped_perch_indices, context


def build_proxy_map_from_strategy(
    primary_labels: Sequence[str],
    unmapped_positions: np.ndarray,
    class_name_map: Dict[str, str],
    genus_hits_by_idx: Dict[int, np.ndarray],
    class_hits_by_idx: Dict[int, np.ndarray],
    source: str,
    filter_proxy_taxa: bool,
    top_k: int,
) -> Dict[int, np.ndarray]:
    proxy_map: Dict[int, np.ndarray] = {}

    for raw_idx in unmapped_positions:
        idx = int(raw_idx)
        label = primary_labels[idx]

        if filter_proxy_taxa and class_name_map.get(label) not in PROXY_TAXA:
            continue

        genus_arr = genus_hits_by_idx.get(idx)
        class_arr = class_hits_by_idx.get(idx)

        if source == "genus":
            arr = genus_arr
        elif source == "class":
            arr = class_arr
        elif source == "genus_plus_class":
            parts = []
            if genus_arr is not None:
                parts.append(genus_arr)
            if class_arr is not None:
                parts.append(class_arr)
            if len(parts) == 0:
                arr = None
            elif len(parts) == 1:
                arr = parts[0]
            else:
                arr = np.unique(np.concatenate(parts))
        else:
            raise ValueError(f"Unsupported proxy source: {source}")

        if arr is None or arr.size == 0:
            continue

        if top_k > 0:
            arr = arr[:top_k]
        if arr.size == 0:
            continue

        proxy_map[idx] = arr.astype(np.int32, copy=False)

    return proxy_map


def build_proxy_approaches(baseline_reduce: str) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"baseline_genus_taxa_{baseline_reduce}",
            "source": "genus",
            "filter_proxy_taxa": True,
            "top_k": 0,
            "proxy_reduce": baseline_reduce,
        },
        {
            "name": "genus_taxa_mean",
            "source": "genus",
            "filter_proxy_taxa": True,
            "top_k": 0,
            "proxy_reduce": "mean",
        },
        {
            "name": "genus_taxa_top1_max",
            "source": "genus",
            "filter_proxy_taxa": True,
            "top_k": 1,
            "proxy_reduce": "max",
        },
        {
            "name": "genus_taxa_top3_max",
            "source": "genus",
            "filter_proxy_taxa": True,
            "top_k": 3,
            "proxy_reduce": "max",
        },
        {
            "name": "genus_taxa_top5_mean",
            "source": "genus",
            "filter_proxy_taxa": True,
            "top_k": 5,
            "proxy_reduce": "mean",
        },
        {
            "name": "genus_all_max",
            "source": "genus",
            "filter_proxy_taxa": False,
            "top_k": 0,
            "proxy_reduce": "max",
        },
        {
            "name": "genus_all_mean",
            "source": "genus",
            "filter_proxy_taxa": False,
            "top_k": 0,
            "proxy_reduce": "mean",
        },
        {
            "name": "class_taxa_top5_max",
            "source": "class",
            "filter_proxy_taxa": True,
            "top_k": 5,
            "proxy_reduce": "max",
        },
        {
            "name": "genus_plus_class_taxa_top5_max",
            "source": "genus_plus_class",
            "filter_proxy_taxa": True,
            "top_k": 5,
            "proxy_reduce": "max",
        },
        {
            "name": "genus_plus_class_taxa_top10_mean",
            "source": "genus_plus_class",
            "filter_proxy_taxa": True,
            "top_k": 10,
            "proxy_reduce": "mean",
        },
    ]


def run_perch_onnx(
    soundscape_paths: Sequence[Path],
    onnx_path: Path,
    n_perch_classes: int,
    batch_files: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, np.ndarray]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "8"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    if "label" not in output_map:
        raise RuntimeError(f"ONNX output 'label' not found. Available outputs: {list(output_map.keys())}")

    paths = [Path(p) for p in soundscape_paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    perch_logits = np.zeros((n_rows, n_perch_classes), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch ONNX")

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
                audio = batch_audio[i]
                x[x_pos:x_pos + N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids[write_row:write_row + N_WINDOWS] = [f"{path.stem}_{sec}" for sec in range(5, 65, 5)]
                filenames[write_row:write_row + N_WINDOWS] = path.name
                x_pos += N_WINDOWS
                write_row += N_WINDOWS

            outputs = session.run(None, {input_name: x})
            logits = outputs[output_map["label"]].astype(np.float32, copy=False)

            perch_logits[batch_row_start:write_row] = logits

            del x, outputs, logits, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames})
    return meta_df, perch_logits


def build_scores_with_proxy(
    perch_logits: np.ndarray,
    n_classes: int,
    mapped_positions: np.ndarray,
    mapped_perch_indices: np.ndarray,
    proxy_map: Dict[int, np.ndarray],
    proxy_reduce: str,
) -> np.ndarray:
    y_score = np.zeros((perch_logits.shape[0], n_classes), dtype=np.float32)
    y_score[:, mapped_positions] = perch_logits[:, mapped_perch_indices]

    for pos, perch_idx_arr in proxy_map.items():
        sub = perch_logits[:, perch_idx_arr]
        if proxy_reduce == "max":
            proxy_score = sub.max(axis=1)
        elif proxy_reduce == "mean":
            proxy_score = sub.mean(axis=1)
        else:
            raise ValueError("proxy_reduce must be 'max' or 'mean'")
        y_score[:, pos] = proxy_score.astype(np.float32, copy=False)

    return y_score


def build_target_matrix(label_lists: Sequence[Sequence[str]], primary_labels: Sequence[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(primary_labels)}
    y_true = np.zeros((len(label_lists), len(primary_labels)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for label in labels:
            idx = label_to_idx.get(label)
            if idx is not None:
                y_true[i, idx] = 1.0
    return y_true


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, int]:
    pos = y_true.sum(axis=0)
    keep = (pos > 0) & (pos < y_true.shape[0])
    n_kept = int(keep.sum())
    if n_kept == 0:
        return float("nan"), 0
    auc = roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
    return float(auc), n_kept


def build_cache_style_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mapped_positions: np.ndarray,
    proxy_positions: np.ndarray,
    no_signal_positions: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    subset_specs = {
        "mapped": mapped_positions,
        "proxy": proxy_positions,
        "no_signal": no_signal_positions,
        "full": np.arange(y_true.shape[1], dtype=np.int32),
    }
    metrics: Dict[str, Dict[str, Any]] = {}
    for subset_name, subset_positions in subset_specs.items():
        auc, n_eval = macro_auc(y_true[:, subset_positions], y_score[:, subset_positions])
        metrics[subset_name] = {
            "macro_auc": None if not np.isfinite(auc) else float(auc),
            "n_eval_classes": int(n_eval),
        }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perch ONNX inference + proxy mapping + macro ROC-AUC.")
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--soundscape-dir", type=Path, default=DEFAULT_SOUNDSCAPE_DIR)
    parser.add_argument("--sound-labels-csv", type=Path, default=DEFAULT_DATA_DIR / "train_soundscapes_labels.csv")
    parser.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_DATA_DIR / "taxonomy.csv")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--perch-labels-csv", type=Path, default=DEFAULT_PERCH_LABELS)
    parser.add_argument("--batch-files", type=int, default=8)
    parser.add_argument("--proxy-reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--limit-files", type=int, default=0, help="Optional debug limit for the number of full files.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--print-proxy-details",
        action="store_true",
        help="Print per-species proxy details for all evaluated approaches.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    truth_df, full_files = build_truth_windows(args.sound_labels_csv, args.soundscape_dir)
    if args.limit_files > 0:
        full_files = full_files[: args.limit_files]
        truth_df = truth_df[truth_df["filename"].isin(full_files)].copy()
        truth_df = truth_df.sort_values(["filename", "end_sec"]).reset_index(drop=True)

    if not full_files:
        raise RuntimeError("No fully-labeled soundscape files found for evaluation.")

    perch_labels = load_perch_labels(perch_labels_csv)
    perch_name_by_idx = perch_labels.set_index("perch_index")["scientific_name"].to_dict()
    mapped_positions, mapped_perch_indices, mapping_ctx = build_mapping_context(
        taxonomy_df=taxonomy,
        perch_labels_df=perch_labels,
        primary_labels=primary_labels,
    )

    soundscape_paths = [args.soundscape_dir / filename for filename in full_files]
    pred_meta, perch_logits = run_perch_onnx(
        soundscape_paths=soundscape_paths,
        onnx_path=args.onnx_path,
        n_perch_classes=len(perch_labels),
        batch_files=args.batch_files,
        verbose=not args.quiet,
    )

    truth_by_row = truth_df[["row_id", "label_list"]]
    merged = pred_meta.merge(truth_by_row, on="row_id", how="left", validate="one_to_one")
    if merged["label_list"].isna().any():
        missing = int(merged["label_list"].isna().sum())
        raise RuntimeError(f"Could not align {missing} prediction rows with truth labels.")

    y_true = build_target_matrix(merged["label_list"].tolist(), primary_labels)

    baseline_proxy_map = mapping_ctx["baseline_proxy_map"]
    baseline_y_score = build_scores_with_proxy(
        perch_logits=perch_logits,
        n_classes=len(primary_labels),
        mapped_positions=mapped_positions,
        mapped_perch_indices=mapped_perch_indices,
        proxy_map=baseline_proxy_map,
        proxy_reduce=args.proxy_reduce,
    )
    baseline_proxy_positions = np.array(sorted(baseline_proxy_map.keys()), dtype=np.int32)
    baseline_no_signal_positions = np.array(
        [int(idx) for idx in mapping_ctx["unmapped_positions"] if int(idx) not in baseline_proxy_map],
        dtype=np.int32,
    )
    cache_style_metrics = build_cache_style_metrics(
        y_true=y_true,
        y_score=baseline_y_score,
        mapped_positions=mapped_positions,
        proxy_positions=baseline_proxy_positions,
        no_signal_positions=baseline_no_signal_positions,
    )

    strategy_defs = build_proxy_approaches(args.proxy_reduce)
    approach_results: List[Dict[str, Any]] = []
    for strategy in strategy_defs:
        proxy_map = build_proxy_map_from_strategy(
            primary_labels=primary_labels,
            unmapped_positions=mapping_ctx["unmapped_positions"],
            class_name_map=mapping_ctx["class_name_map"],
            genus_hits_by_idx=mapping_ctx["genus_hits_by_idx"],
            class_hits_by_idx=mapping_ctx["class_hits_by_idx"],
            source=strategy["source"],
            filter_proxy_taxa=bool(strategy["filter_proxy_taxa"]),
            top_k=int(strategy["top_k"]),
        )

        y_score = build_scores_with_proxy(
            perch_logits=perch_logits,
            n_classes=len(primary_labels),
            mapped_positions=mapped_positions,
            mapped_perch_indices=mapped_perch_indices,
            proxy_map=proxy_map,
            proxy_reduce=str(strategy["proxy_reduce"]),
        )

        auc_all, eval_classes_all = macro_auc(y_true, y_score)
        auc_mapped, eval_classes_mapped = macro_auc(y_true[:, mapped_positions], y_score[:, mapped_positions])

        proxy_positions = np.array(sorted(proxy_map.keys()), dtype=np.int32)
        if proxy_positions.size > 0:
            auc_proxy, eval_classes_proxy = macro_auc(y_true[:, proxy_positions], y_score[:, proxy_positions])
            signal_positions = np.unique(np.concatenate([mapped_positions, proxy_positions]))
        else:
            auc_proxy, eval_classes_proxy = float("nan"), 0
            signal_positions = mapped_positions

        auc_signal, eval_classes_signal = macro_auc(y_true[:, signal_positions], y_score[:, signal_positions])

        proxy_details = []
        for pos in sorted(proxy_map.keys()):
            perch_idxs = proxy_map[pos]
            target_label = primary_labels[pos]
            taxonomy_row = taxonomy[taxonomy["primary_label"] == target_label]
            target_scientific_name = (
                str(taxonomy_row["scientific_name"].iloc[0]) if len(taxonomy_row) > 0 else target_label
            )
            proxy_details.append(
                {
                    "target_primary_label": target_label,
                    "target_idx": int(pos),
                    "target_scientific_name": target_scientific_name,
                    "n_perch_indices": int(len(perch_idxs)),
                    "perch_indices": [int(x) for x in perch_idxs.tolist()],
                    "perch_scientific_names": [str(perch_name_by_idx.get(int(x), f"idx_{int(x)}")) for x in perch_idxs],
                }
            )

        approach_results.append(
            {
                "name": str(strategy["name"]),
                "source": str(strategy["source"]),
                "proxy_reduce": str(strategy["proxy_reduce"]),
                "filter_proxy_taxa": bool(strategy["filter_proxy_taxa"]),
                "top_k": int(strategy["top_k"]),
                "mapping": {
                    "n_classes": int(mapping_ctx["n_classes"]),
                    "n_mapped": int(mapping_ctx["n_mapped"]),
                    "n_unmapped": int(mapping_ctx["n_unmapped"]),
                    "n_proxy": int(len(proxy_map)),
                    "n_still_without_signal": int(mapping_ctx["n_unmapped"] - len(proxy_map)),
                },
                "proxy_details": proxy_details,
                "metrics": {
                    "macro_auc_all_active": auc_all,
                    "macro_auc_mapped_active": auc_mapped,
                    "macro_auc_proxy_active": auc_proxy,
                    "macro_auc_signal_active": auc_signal,
                    "n_eval_classes_all": int(eval_classes_all),
                    "n_eval_classes_mapped": int(eval_classes_mapped),
                    "n_eval_classes_proxy": int(eval_classes_proxy),
                    "n_eval_classes_signal": int(eval_classes_signal),
                },
            }
        )

    def _sort_key(item: Dict[str, Any]) -> float:
        val = item["metrics"]["macro_auc_signal_active"]
        if isinstance(val, float) and np.isnan(val):
            return -np.inf
        return float(val)

    sorted_results = sorted(approach_results, key=_sort_key, reverse=True)
    baseline_name = f"baseline_genus_taxa_{args.proxy_reduce}"
    baseline_result = next((x for x in approach_results if x["name"] == baseline_name), approach_results[0])
    best_result = sorted_results[0]

    summary = {
        "onnx_path": str(args.onnx_path),
        "perch_labels_csv": str(perch_labels_csv),
        "proxy_reduce_baseline": args.proxy_reduce,
        "batch_files": int(args.batch_files),
        "n_files": int(len(full_files)),
        "n_rows": int(len(pred_meta)),
        "n_approaches": int(len(approach_results)),
        "baseline": baseline_result,
        "cache_style_metrics": cache_style_metrics,
        "best_by_signal_auc": {
            "name": best_result["name"],
            "macro_auc_signal_active": best_result["metrics"]["macro_auc_signal_active"],
        },
        "approaches": sorted_results,
        "mapping": baseline_result["mapping"],
        "metrics": baseline_result["metrics"],
    }

    print(f"Files used: {len(full_files)} | Rows: {len(pred_meta)} | Approaches: {len(approach_results)}")
    print("Results sorted by Macro ROC-AUC (mapped+proxy active):")
    for rank, item in enumerate(sorted_results, start=1):
        metrics = item["metrics"]
        mapping = item["mapping"]
        auc_proxy_val = metrics["macro_auc_proxy_active"]
        auc_proxy_str = f"{auc_proxy_val:.6f}" if not np.isnan(auc_proxy_val) else "n/a"
        print(
            f"{rank:2d}. {item['name']:<32s} "
            f"signal={metrics['macro_auc_signal_active']:.6f} "
            f"all={metrics['macro_auc_all_active']:.6f} "
            f"proxy={auc_proxy_str} "
            f"proxy_species={mapping['n_proxy']} "
            f"no_signal={mapping['n_still_without_signal']}"
        )

    print("\nBaseline (matching sota_full_oof.py mapping):")
    print(
        f"  {baseline_result['name']} | "
        f"signal={baseline_result['metrics']['macro_auc_signal_active']:.6f} | "
        f"proxy_species={baseline_result['mapping']['n_proxy']} | "
        f"no_signal={baseline_result['mapping']['n_still_without_signal']}"
    )
    print(
        f"Best approach: {best_result['name']} | "
        f"signal={best_result['metrics']['macro_auc_signal_active']:.6f}"
    )

    print("\nCache-style metrics (same subset logic as generate_ported_cache.py):")
    for subset_name in ["mapped", "proxy", "no_signal", "full"]:
        info = cache_style_metrics[subset_name]
        auc_val = info["macro_auc"]
        auc_str = f"{auc_val:.6f}" if auc_val is not None else "n/a"
        print(f"  {subset_name}: macro_auc={auc_str} n_eval_classes={info['n_eval_classes']}")

    print("\nBaseline proxy details:")
    baseline_proxy_details = baseline_result.get("proxy_details", [])
    if len(baseline_proxy_details) == 0:
        print("  none")
    else:
        for item in baseline_proxy_details:
            print(
                f"  {item['target_primary_label']} ({item['target_scientific_name']}): "
                f"{item['n_perch_indices']} Perch indices -> {item['perch_indices']}"
            )

    if args.print_proxy_details:
        print("\nProxy details by approach:")
        for item in sorted_results:
            print(f"- {item['name']}:")
            details = item.get("proxy_details", [])
            if len(details) == 0:
                print("  none")
                continue
            for d in details:
                print(
                    f"  {d['target_primary_label']} ({d['target_scientific_name']}): "
                    f"{d['n_perch_indices']} -> {d['perch_indices']}"
                )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary JSON: {args.output_json}")


if __name__ == "__main__":
    main()
