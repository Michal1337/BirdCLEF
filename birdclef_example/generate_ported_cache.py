#!/usr/bin/env python3

"""Generate a Perch-only cache for the BirdCLEF pipeline.

Pure Perch ONNX inference over fully-labeled soundscapes — no finetuned head
for no-signal classes, no genus-proxy logits for Aves/Amphibia/Insecta. The
score tensor stays in BirdCLEF primary-label space (234 cols) so it's a
drop-in for the existing cache contract:

    full_perch_meta.parquet                   row_id, filename, site, hour_utc
    full_perch_arrays.npz  scores_full_raw    (N, 234) raw Perch logits in
                                              BirdCLEF column order; classes
                                              not predicted by Perch are 0.
                           emb_full           (N, 1536) Perch embeddings.

After writing, runs an AUC sanity block (macro AUC over labeled-soundscape
rows for the 'mapped' subset and the full 234 cols) so you can quickly
spot a broken build vs the reference numbers.
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
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SOUNDSCAPE_DIR = DEFAULT_DATA_DIR / "train_soundscapes"
DEFAULT_PERCH_LABELS_CSV = REPO_ROOT / "models" / "perch_v2_cpu" / "1" / "assets" / "labels.csv"
DEFAULT_TAXONOMY_CSV = DEFAULT_DATA_DIR / "taxonomy.csv"
DEFAULT_SAMPLE_SUBMISSION = DEFAULT_DATA_DIR / "sample_submission.csv"
DEFAULT_SOUND_LABELS = DEFAULT_DATA_DIR / "train_soundscapes_labels.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "perch_cache_simple"


def parse_soundscape_filename(name: str) -> Dict[str, object]:
    m = FNAME_RE.match(name)
    if not m:
        return {"site": None, "hour_utc": -1}
    _, site, _ymd, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


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
    out: set[str] = set()
    for v in series:
        out.update(parse_primary_labels(v))
    return sorted(out)


def build_full_files(sound_labels_csv: Path) -> List[str]:
    """Return the sorted list of soundscape files that have all 12 windows labelled."""
    df = pd.read_csv(sound_labels_csv)
    df["primary_label"] = df["primary_label"].astype(str)
    grouped = (
        df.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels).reset_index(name="label_list")
    )
    counts = grouped.groupby("filename").size()
    return sorted(counts[counts == N_WINDOWS].index.tolist())


def build_truth_windows(sound_labels_csv: Path, soundscape_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(sound_labels_csv)
    df["primary_label"] = df["primary_label"].astype(str)
    grouped = (
        df.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels).reset_index(name="label_list")
    )
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(int)
    grouped["row_id"] = (
        grouped["filename"].str.replace(".ogg", "", regex=False)
        + "_" + grouped["end_sec"].astype(str)
    )
    grouped = grouped[grouped["filename"].map(lambda x: (soundscape_dir / x).exists())].copy()
    return grouped.sort_values(["filename", "end_sec"]).reset_index(drop=True)


def build_target_matrix(label_lists: Sequence[Sequence[str]],
                        primary_labels: Sequence[str]) -> np.ndarray:
    label_to_idx = {lb: i for i, lb in enumerate(primary_labels)}
    Y = np.zeros((len(label_lists), len(primary_labels)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for lb in labels:
            j = label_to_idx.get(lb)
            if j is not None:
                Y[i, j] = 1.0
    return Y


def macro_auc_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, int]:
    pos = y_true.sum(axis=0)
    keep = (pos > 0) & (pos < y_true.shape[0])
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    probs = 1.0 / (1.0 + np.exp(-logits))
    return float(roc_auc_score(y_true[:, keep], probs[:, keep], average="macro")), n


def load_perch_onnx(onnx_path: Path):
    """ONNX-only Perch loader. Returns (infer_fn, backend_name)."""
    import onnxruntime as ort

    if not onnx_path.exists():
        raise FileNotFoundError(f"Perch ONNX model not found: {onnx_path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    out_idx = {o.name: i for i, o in enumerate(sess.get_outputs())}
    if "label" not in out_idx or "embedding" not in out_idx:
        raise RuntimeError(
            f"Perch ONNX missing expected outputs. Found: {list(out_idx)}"
        )

    def infer(x: np.ndarray) -> Dict[str, np.ndarray]:
        outs = sess.run(None, {in_name: x})
        return {
            "label":     outs[out_idx["label"]].astype(np.float32, copy=False),
            "embedding": outs[out_idx["embedding"]].astype(np.float32, copy=False),
        }

    return infer, "onnx"


def build_direct_mapping(
    taxonomy_df: pd.DataFrame,
    perch_labels_df: pd.DataFrame,
    primary_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Direct scientific_name → perch_index mapping. Returns (mapped_positions,
    mapped_bc_indices) — positions in the BirdCLEF column space (0..234) and
    the matching column index in Perch's logit space. Unmapped classes are
    silently left out; the score tensor keeps them at 0.
    """
    no_label_index = len(perch_labels_df)
    merged = taxonomy_df.merge(perch_labels_df, on="scientific_name", how="left")
    merged["perch_index"] = merged["perch_index"].fillna(no_label_index).astype(int)
    label_to_perch = merged.set_index("primary_label")["perch_index"].to_dict()
    bc = np.array(
        [int(label_to_perch.get(lb, no_label_index)) for lb in primary_labels],
        dtype=np.int32,
    )
    mapped = bc != no_label_index
    return np.where(mapped)[0].astype(np.int32), bc[mapped].astype(np.int32)


def infer_perch(
    paths: Sequence[Path],
    infer_fn,
    n_classes: int,
    mapped_positions: np.ndarray,
    mapped_bc_indices: np.ndarray,
    batch_files: int = 16,
    emb_dim: int = 1536,
    verbose: bool = True,
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
        iterator = tqdm(
            iterator, total=(n_files + batch_files - 1) // batch_files,
            desc="Perch batches",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_pool:
        next_paths = paths[0:batch_files]
        future_audio = [io_pool.submit(read_soundscape_60s, p) for p in next_paths]

        for start in iterator:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < n_files:
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [io_pool.submit(read_soundscape_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            batch_row_start = write_row
            x_pos = 0
            for path, y in zip(batch_paths, batch_audio):
                x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                meta = parse_soundscape_filename(path.name)
                stem = path.stem
                row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[write_row:write_row + N_WINDOWS] = path.name
                sites[write_row:write_row + N_WINDOWS] = meta["site"]
                hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])
                x_pos += N_WINDOWS
                write_row += N_WINDOWS

            outs = infer_fn(x)
            scores[batch_row_start:write_row, mapped_positions] = (
                outs["label"][:, mapped_bc_indices]
            )
            embeddings[batch_row_start:write_row] = outs["embedding"]

            del x, outs, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })
    return meta_df, scores, embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--soundscape-dir", type=Path, default=DEFAULT_SOUNDSCAPE_DIR)
    p.add_argument("--taxonomy-csv", type=Path, default=DEFAULT_TAXONOMY_CSV)
    p.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    p.add_argument("--sound-labels-csv", type=Path, default=DEFAULT_SOUND_LABELS)
    p.add_argument("--perch-labels-csv", type=Path, default=DEFAULT_PERCH_LABELS_CSV)
    p.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-files", type=int, default=16)
    p.add_argument("--limit-files", type=int, default=0,
                   help="Optional debug limit on the number of full files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for path in [
        args.taxonomy_csv, args.sample_submission, args.sound_labels_csv,
        args.perch_labels_csv, args.soundscape_dir, args.onnx_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_sub = pd.read_csv(args.sample_submission)
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    taxonomy = pd.read_csv(args.taxonomy_csv)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    perch_labels = (
        pd.read_csv(args.perch_labels_csv)
        .reset_index().rename(columns={"index": "perch_index"})
    )
    if "scientific_name" not in perch_labels.columns:
        if "inat2024_fsd50k" in perch_labels.columns:
            perch_labels = perch_labels.rename(columns={"inat2024_fsd50k": "scientific_name"})
        else:
            raise ValueError(
                f"Perch labels CSV {args.perch_labels_csv} is missing "
                "scientific_name/inat2024_fsd50k column."
            )
    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)

    mapped_positions, mapped_bc_indices = build_direct_mapping(
        taxonomy_df=taxonomy, perch_labels_df=perch_labels,
        primary_labels=primary_labels,
    )

    full_files = build_full_files(args.sound_labels_csv)
    if args.limit_files and args.limit_files > 0:
        full_files = full_files[: args.limit_files]
    full_paths = [args.soundscape_dir / fn for fn in full_files]

    infer_fn, backend = load_perch_onnx(args.onnx_path)
    print(f"Backend: {backend}")
    print(f"Full files: {len(full_files)}")
    print(f"Mapped classes (direct): {len(mapped_positions)} / {n_classes}")
    print(f"Unmapped classes (left at 0): {n_classes - len(mapped_positions)}")

    meta_df, scores_full_raw, emb_full = infer_perch(
        full_paths, infer_fn=infer_fn, n_classes=n_classes,
        mapped_positions=mapped_positions, mapped_bc_indices=mapped_bc_indices,
        batch_files=args.batch_files, verbose=True,
    )

    out_meta = args.output_dir / "full_perch_meta.parquet"
    out_npz = args.output_dir / "full_perch_arrays.npz"
    meta_df.to_parquet(out_meta, index=False)
    np.savez_compressed(out_npz, scores_full_raw=scores_full_raw, emb_full=emb_full)

    # ── AUC sanity block ──────────────────────────────────────────────
    truth_df = build_truth_windows(args.sound_labels_csv, args.soundscape_dir)
    truth_df = truth_df[truth_df["filename"].isin(full_files)].copy()
    merged = meta_df.merge(
        truth_df[["row_id", "label_list"]], on="row_id", how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise RuntimeError("No labeled rows after aligning generated cache with truth.")
    Y = build_target_matrix(merged["label_list"].tolist(), primary_labels)
    score_frame = pd.DataFrame(scores_full_raw, index=meta_df["row_id"].to_numpy())
    merged_scores = score_frame.loc[merged["row_id"]].to_numpy(dtype=np.float32, copy=False)
    auc_mapped, n_mapped = macro_auc_from_logits(
        Y[:, mapped_positions], merged_scores[:, mapped_positions],
    )
    auc_full, n_full = macro_auc_from_logits(Y, merged_scores)

    summary = {
        "backend": backend,
        "onnx_path": str(args.onnx_path),
        "full_files": len(full_files),
        "rows": int(len(meta_df)),
        "scores_shape": list(scores_full_raw.shape),
        "emb_shape": list(emb_full.shape),
        "mapped_classes": int(len(mapped_positions)),
        "unmapped_classes": int(n_classes - len(mapped_positions)),
        "metrics": {
            "mapped": {"macro_auc": auc_mapped if np.isfinite(auc_mapped) else None,
                       "n_eval_classes": n_mapped},
            "full":   {"macro_auc": auc_full   if np.isfinite(auc_full)   else None,
                       "n_eval_classes": n_full},
        },
        "output_meta": str(out_meta),
        "output_npz":  str(out_npz),
    }
    (args.output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("Saved cache to:")
    print(f"  {out_meta}")
    print(f"  {out_npz}")
    print(f"  {args.output_dir / 'cache_summary.json'}")
    print("Metrics (raw Perch logits, sigmoid → macro AUC):")
    print(f"  mapped subset : auc={auc_mapped:.4f}  n_eval={n_mapped}")
    print(f"  full 234 cols : auc={auc_full:.4f}  n_eval={n_full}")


if __name__ == "__main__":
    main()
