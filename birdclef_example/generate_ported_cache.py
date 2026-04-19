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
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "perch_onnx" / "perch_v2_finetuned_partial.onnx"
DEFAULT_TF_MODEL_DIR = REPO_ROOT / "models" / "perch_v2_cpu" / "1"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SOUNDSCAPE_DIR = REPO_ROOT / "data" / "train_soundscapes"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "perch_cache_finetuned_partial"
DEFAULT_PERCH_LABELS_CSV = DEFAULT_TF_MODEL_DIR / "assets" / "labels.csv"
DEFAULT_TAXONOMY_CSV = DEFAULT_DATA_DIR / "taxonomy.csv"
DEFAULT_SAMPLE_SUBMISSION = DEFAULT_DATA_DIR / "sample_submission.csv"
DEFAULT_SOUND_LABELS = DEFAULT_DATA_DIR / "train_soundscapes_labels.csv"
DEFAULT_CACHE_DIR = Path(os.environ.get("PERCH_CACHE_DIR", DEFAULT_OUT_DIR))


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
            }

        return infer, "tf"

    raise FileNotFoundError(
        f"Could not load backend {backend!r}: ONNX model missing at {onnx_path} and TF model not selected/available."
    )


def build_exact_mapping(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, np.ndarray]]:
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
    mapped_positions = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc_indices = bc_indices[mapped_mask].astype(np.int32)

    class_name_map = taxonomy_local.set_index("primary_label")["class_name"].to_dict()
    unmapped_df = mapping[mapping["perch_index"] == no_label_index].copy()
    unmapped_non_sonotype = unmapped_df[~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)].copy()

    proxy_taxa = {"Amphibia", "Insecta", "Aves"}
    proxy_map: Dict[int, np.ndarray] = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = str(row["primary_label"])
        if class_name_map.get(target) not in proxy_taxa:
            continue
        genus = str(row["scientific_name"]).split()[0]
        hits = perch_labels_df[
            perch_labels_df["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map[int(primary_labels.index(target))] = hits["perch_index"].astype(int).to_numpy()

    return mapped_positions, mapped_bc_indices, class_name_map, proxy_map


def infer_perch_with_embeddings(
    paths: Sequence[Path],
    infer_fn,
    n_classes: int,
    mapped_positions: np.ndarray,
    mapped_bc_indices: np.ndarray,
    proxy_map: Dict[int, np.ndarray],
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

            outputs = infer_fn(x)
            logits = outputs["label"]
            emb = outputs["embedding"]

            scores[batch_row_start:write_row, mapped_positions] = logits[:, mapped_bc_indices]

            for pos, bc_idx_arr in proxy_map.items():
                sub = logits[:, bc_idx_arr]
                if proxy_reduce == "max":
                    proxy_score = sub.max(axis=1)
                elif proxy_reduce == "mean":
                    proxy_score = sub.mean(axis=1)
                else:
                    raise ValueError("proxy_reduce must be 'max' or 'mean'")
                scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

            embeddings[batch_row_start:write_row] = emb

            del x, outputs, logits, emb, batch_audio
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
    parser.add_argument("--tf-model-dir", type=Path, default=DEFAULT_TF_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--backend", choices=["auto", "onnx", "tf"], default="auto")
    parser.add_argument("--batch-files", type=int, default=16)
    parser.add_argument("--proxy-reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--limit-files", type=int, default=0, help="Optional debug limit for the number of full files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.taxonomy_csv, args.sample_submission, args.sound_labels_csv, args.perch_labels_csv, args.soundscape_dir]:
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
    perch_labels = (
        pd.read_csv(args.perch_labels_csv)
        .reset_index()
        .rename(columns={"index": "perch_index", "inat2024_fsd50k": "scientific_name"})
    )
    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)

    mapped_positions, mapped_bc_indices, _class_name_map, proxy_map = build_exact_mapping(
        taxonomy_df=taxonomy,
        perch_labels_df=perch_labels,
        primary_labels=primary_labels,
    )

    infer_fn, backend_used = load_perch_backend(args.backend, args.onnx_path, args.tf_model_dir)
    print(f"Backend: {backend_used}")
    print(f"Full files: {len(full_files)}")
    print(f"Mapped classes: {len(mapped_positions)}")
    print(f"Proxy targets: {len(proxy_map)}")

    full_paths = [args.soundscape_dir / fn for fn in full_files]
    meta_df, scores_full_raw, emb_full = infer_perch_with_embeddings(
        full_paths,
        infer_fn=infer_fn,
        n_classes=n_classes,
        mapped_positions=mapped_positions,
        mapped_bc_indices=mapped_bc_indices,
        proxy_map=proxy_map,
        batch_files=args.batch_files,
        verbose=True,
        proxy_reduce=args.proxy_reduce,
    )

    out_meta = args.output_dir / "full_perch_meta.parquet"
    out_npz = args.output_dir / "full_perch_arrays.npz"
    meta_df.to_parquet(out_meta, index=False)
    np.savez_compressed(out_npz, scores_full_raw=scores_full_raw, emb_full=emb_full)

    summary = {
        "backend": backend_used,
        "full_files": len(full_files),
        "rows": int(len(meta_df)),
        "scores_shape": list(scores_full_raw.shape),
        "emb_shape": list(emb_full.shape),
        "output_meta": str(out_meta),
        "output_npz": str(out_npz),
        "proxy_reduce": args.proxy_reduce,
    }
    (args.output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved cache to:")
    print("  ", out_meta)
    print("  ", out_npz)
    print("  ", args.output_dir / "cache_summary.json")


if __name__ == "__main__":
    main()
