#!/usr/bin/env python3

"""Simple Kaggle submission inference using a pretrained Perch ONNX model.

This script intentionally avoids additional modeling (no SSM/probes/ensembles).
It runs ONNX logits on 5-second windows and maps Perch classes to BirdCLEF classes
via exact scientific-name matching from taxonomy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
except Exception as exc:
    raise RuntimeError(
        "This script requires onnxruntime. Install it with: pip install onnxruntime"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import parse_primary_labels  # noqa: E402


SAMPLE_RATE = 32000
WINDOW_SEC = 5
N_WINDOWS = 12
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
FILE_SAMPLES = SAMPLE_RATE * 60

# In-script configuration.
ONNX_PATH = Path(os.environ.get("PERCH_ONNX_PATH", REPO_ROOT / "models" / "perch_onnx" / "perch_v2_finetuned.onnx"))
LABELS_CSV = REPO_ROOT / "models" / "perch_v2_cpu" / "1" / "assets" / "labels.csv"
TAXONOMY_CSV = REPO_ROOT / "data" / "taxonomy.csv"
SAMPLE_SUBMISSION_CSV = REPO_ROOT / "data" / "sample_submission.csv"
SOUNDSCAPE_DIR = REPO_ROOT / "data" / "test_soundscapes"
TRAIN_SOUNDSCAPE_DIR = REPO_ROOT / "data" / "train_soundscapes"
TRAIN_SOUNDSCAPE_LABELS_CSV = REPO_ROOT / "data" / "train_soundscapes_labels.csv"
OUTPUT_CSV = Path("submission.csv")
BATCH_FILES = 16
DRYRUN_N_FILES = 20
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
ORT_INTER_THREADS = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def load_mapping(
    taxonomy_csv: Path,
    labels_csv: Path,
    primary_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, int]:
    taxonomy = pd.read_csv(taxonomy_csv).copy()
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

    perch_labels = (
        pd.read_csv(labels_csv)
        .reset_index()
        .rename(columns={"index": "perch_index", "inat2024_fsd50k": "scientific_name"})
    )
    perch_labels["scientific_name"] = perch_labels["scientific_name"].astype(str)

    no_label_index = len(perch_labels)

    taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"].astype(str)
    perch_lookup = perch_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

    mapping = taxonomy.merge(
        perch_lookup[["scientific_name_lookup", "perch_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    mapping["perch_index"] = mapping["perch_index"].fillna(no_label_index).astype(int)

    label_to_perch = mapping.set_index("primary_label")["perch_index"].to_dict()
    perch_indices = np.array([int(label_to_perch.get(lbl, no_label_index)) for lbl in primary_labels], dtype=np.int32)

    mapped_mask = perch_indices != no_label_index
    mapped_positions = np.where(mapped_mask)[0].astype(np.int32)
    mapped_perch_indices = perch_indices[mapped_mask].astype(np.int32)
    return mapped_positions, mapped_perch_indices, int((~mapped_mask).sum())


def build_file_list_from_sample_submission(sample_submission: pd.DataFrame) -> List[str]:
    stems = (
        sample_submission["row_id"]
        .astype(str)
        .str.replace(r"_[0-9]+$", "", regex=True)
        .drop_duplicates()
        .tolist()
    )
    return [f"{stem}.ogg" for stem in stems]


def build_ort_session(onnx_path: Path, intra_threads: int, inter_threads: int):
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_threads)
    so.inter_op_num_threads = int(inter_threads)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_map = {o.name: i for i, o in enumerate(sess.get_outputs())}
    if "label" not in output_map:
        raise RuntimeError(f"ONNX output 'label' not found; outputs are {list(output_map.keys())}")
    return sess, input_name, output_map["label"]


def macro_auc_skip_empty(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    keep = np.where((y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < len(y_true)))[0]
    if len(keep) == 0:
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_prob[:, keep], average="macro"))


def build_train_truth_matrix(
    labels_csv: Path,
    primary_labels: Sequence[str],
    row_ids: Sequence[str],
) -> np.ndarray:
    labels_df = pd.read_csv(labels_csv)
    labels_df["primary_label"] = labels_df["primary_label"].astype(str)
    grouped = (
        labels_df.groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda s: sorted({lbl for raw in s for lbl in parse_primary_labels(raw)}))
        .reset_index(name="label_list")
    )
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(int)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)

    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    row_to_labels = dict(zip(grouped["row_id"].tolist(), grouped["label_list"].tolist()))

    y_true = np.zeros((len(row_ids), len(primary_labels)), dtype=np.float32)
    for i, rid in enumerate(row_ids):
        labels = row_to_labels.get(str(rid), [])
        for lab in labels:
            idx = label_to_idx.get(lab)
            if idx is not None:
                y_true[i, idx] = 1.0
    return y_true


def main() -> None:
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    if "row_id" not in sample_sub.columns:
        raise ValueError(f"Missing row_id column in {SAMPLE_SUBMISSION_CSV}")

    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    mapped_positions, mapped_perch_indices, n_unmapped = load_mapping(
        taxonomy_csv=TAXONOMY_CSV,
        labels_csv=LABELS_CSV,
        primary_labels=primary_labels,
    )

    sess, input_name, label_out_idx = build_ort_session(
        onnx_path=ONNX_PATH,
        intra_threads=ORT_INTRA_THREADS,
        inter_threads=ORT_INTER_THREADS,
    )

    test_files = sorted([p.name for p in SOUNDSCAPE_DIR.glob("*.ogg")]) if SOUNDSCAPE_DIR.exists() else []
    dryrun_mode = len(test_files) == 0

    if dryrun_mode:
        train_files = sorted([p.name for p in TRAIN_SOUNDSCAPE_DIR.glob("*.ogg")])
        if not train_files:
            raise FileNotFoundError(
                f"No test soundscapes found in {SOUNDSCAPE_DIR} and no train soundscapes found in {TRAIN_SOUNDSCAPE_DIR}."
            )
        file_list = train_files[:DRYRUN_N_FILES]
        active_soundscape_dir = TRAIN_SOUNDSCAPE_DIR
        print(f"No test soundscapes found. Dry-run mode on first {len(file_list)} train files.")
    else:
        file_list = build_file_list_from_sample_submission(sample_sub)
        active_soundscape_dir = SOUNDSCAPE_DIR

    pred_by_row: Dict[str, np.ndarray] = {}

    print(f"Files to score: {len(file_list)}")
    print(f"Mapped classes: {len(mapped_positions)} / {n_classes} | Unmapped classes (filled with 0): {n_unmapped}")

    for start in tqdm(range(0, len(file_list), BATCH_FILES), desc="ONNX inference"):
        batch_files = file_list[start:start + BATCH_FILES]
        batch_audio = np.empty((len(batch_files) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        available = []

        pos = 0
        for fname in batch_files:
            fpath = active_soundscape_dir / fname
            if not fpath.exists():
                for sec in range(5, 65, 5):
                    pred_by_row[f"{Path(fname).stem}_{sec}"] = np.zeros(n_classes, dtype=np.float32)
                continue

            y = read_soundscape_60s(fpath)
            batch_audio[pos:pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            available.append(fname)
            pos += N_WINDOWS

        if not available:
            continue

        logits = sess.run(None, {input_name: batch_audio[: len(available) * N_WINDOWS]})[label_out_idx]
        logits = logits.astype(np.float32, copy=False)

        scores = np.zeros((logits.shape[0], n_classes), dtype=np.float32)
        scores[:, mapped_positions] = logits[:, mapped_perch_indices]
        probs = sigmoid(scores)

        r = 0
        for fname in available:
            stem = Path(fname).stem
            for sec in range(5, 65, 5):
                pred_by_row[f"{stem}_{sec}"] = probs[r]
                r += 1

    if dryrun_mode:
        row_ids = [f"{Path(fname).stem}_{sec}" for fname in file_list for sec in range(5, 65, 5)]
        submission = pd.DataFrame({"row_id": row_ids})
    else:
        submission = sample_sub[["row_id"]].copy()

    out = np.zeros((len(submission), n_classes), dtype=np.float32)
    submission_row_ids = submission["row_id"].astype(str).tolist()
    for i, rid in enumerate(submission_row_ids):
        row_pred = pred_by_row.get(rid)
        if row_pred is not None:
            out[i] = row_pred

    for j, col in enumerate(primary_labels):
        submission[col] = out[:, j]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission: {OUTPUT_CSV}")

    if dryrun_mode:
        y_true = build_train_truth_matrix(
            labels_csv=TRAIN_SOUNDSCAPE_LABELS_CSV,
            primary_labels=primary_labels,
            row_ids=submission_row_ids,
        )
        auc = macro_auc_skip_empty(y_true, out)
        if np.isnan(auc):
            print("Dry-run macro AUC: nan (no evaluable classes)")
        else:
            print(f"Dry-run macro AUC on first {len(file_list)} train files: {auc:.6f}")


if __name__ == "__main__":
    main()
