"""Parse train.csv into a usable metadata frame and label matrices."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from birdclef.config.paths import TRAIN_AUDIO, TRAIN_CSV
from birdclef.data.soundscapes import label_to_idx, primary_labels

_SPLIT_RE = re.compile(r"[;,\s]+")


def _parse_primary(raw) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s or s == "[]":
        return []
    return [x for x in _SPLIT_RE.split(s) if x]


def _parse_secondary(raw) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw if x]
    s = str(raw).strip()
    if not s or s in {"[]", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed if x]
    except (ValueError, SyntaxError):
        pass
    return [x for x in _SPLIT_RE.split(s) if x]


def load_train_audio_meta() -> pd.DataFrame:
    """Return train_audio metadata with normalized label columns.

    Columns: filename (str, relative to TRAIN_AUDIO), abspath (Path),
             primary_labels (List[str]), secondary_labels (List[str]).
    Filters out rows whose audio file doesn't exist on disk.
    """
    df = pd.read_csv(TRAIN_CSV)
    fn_col = _pick_filename_col(df)
    df = df.rename(columns={fn_col: "filename"})
    df["primary_labels"] = df["primary_label"].apply(_parse_primary) if "primary_label" in df.columns else [[]] * len(df)
    df["secondary_labels"] = df["secondary_labels"].apply(_parse_secondary) if "secondary_labels" in df.columns else [[]] * len(df)
    df["abspath"] = df["filename"].apply(lambda f: TRAIN_AUDIO / str(f))
    mask = df["abspath"].apply(lambda p: Path(p).exists())
    missing = int((~mask).sum())
    if missing:
        print(f"[train_audio] warning: {missing} rows in train.csv have no file on disk, dropped")
    return df[mask].reset_index(drop=True)


def _pick_filename_col(df: pd.DataFrame) -> str:
    for c in ("filename", "filepath", "path", "audio_file"):
        if c in df.columns:
            return c
    raise KeyError(f"train.csv has none of filename/filepath/path/audio_file; columns={list(df.columns)}")


def build_train_audio_labels(df: pd.DataFrame) -> np.ndarray:
    """Multi-hot (N × n_classes) using primary ∪ secondary labels.

    Winners 2024 mask the secondary-label loss; we keep it in the target and
    let the loss-side masking decide.
    """
    idx = label_to_idx()
    n_classes = len(idx)
    Y = np.zeros((len(df), n_classes), dtype=np.uint8)
    for i, (prim, sec) in enumerate(zip(df["primary_labels"], df["secondary_labels"])):
        for lb in list(prim) + list(sec):
            j = idx.get(lb)
            if j is not None:
                Y[i, j] = 1
    return Y


def build_secondary_mask(df: pd.DataFrame) -> np.ndarray:
    """Mask where secondary-only positive labels are marked True.

    Used by the loss to optionally zero-out secondary-label positions.
    """
    idx = label_to_idx()
    n_classes = len(idx)
    M = np.zeros((len(df), n_classes), dtype=np.uint8)
    for i, (prim, sec) in enumerate(zip(df["primary_labels"], df["secondary_labels"])):
        primset = {lb for lb in prim if lb in idx}
        for lb in sec:
            j = idx.get(lb)
            if j is not None and lb not in primset:
                M[i, j] = 1
    return M


def class_counts(df: pd.DataFrame, include_secondary: bool = True) -> np.ndarray:
    """Per-class positive count across train_audio (for rare/frequent split)."""
    idx = label_to_idx()
    counts = np.zeros(len(idx), dtype=np.int64)
    for prim, sec in zip(df["primary_labels"], df["secondary_labels"] if include_secondary else [[]] * len(df)):
        for lb in list(prim) + list(sec):
            j = idx.get(lb)
            if j is not None:
                counts[j] += 1
    return counts
