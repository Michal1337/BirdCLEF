"""Parse train_soundscapes/ + its label CSV into per-window metadata.

A file is "fully labeled" when it has exactly N_WINDOWS=12 rows in the labels
CSV. Unlabeled files are still listed so pseudo-labeling and inference can
enumerate them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from birdclef.config.paths import (
    N_WINDOWS,
    SAMPLE_SUB,
    SCLABEL_CSV,
    SOUNDSCAPES,
    TAXONOMY,
    TEST_SC,
)

FNAME_RE = re.compile(
    r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg"
)


def parse_fname(name: str) -> dict:
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "date": "00000000", "hour_utc": -1}
    _, site, date, hms = m.groups()
    return {"site": site, "date": date, "hour_utc": int(hms[:2])}


def union_labels(series: Iterable) -> List[str]:
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


def primary_labels() -> List[str]:
    return pd.read_csv(SAMPLE_SUB).columns[1:].tolist()


def label_to_idx() -> dict:
    return {c: i for i, c in enumerate(primary_labels())}


def load_soundscape_meta() -> pd.DataFrame:
    """Return one row per 5s window of labeled soundscapes.

    Columns: row_id, filename, start, end, end_sec, site, date, hour_utc,
             label_list (List[str]), fully_labeled (bool)
    """
    sc_labels = pd.read_csv(SCLABEL_CSV)
    sc = (
        sc_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = (
        sc["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + sc["end_sec"].astype(str)
    )
    meta = sc["filename"].apply(parse_fname).apply(pd.Series)
    sc = pd.concat([sc, meta], axis=1)

    counts = sc.groupby("filename").size()
    full = set(counts[counts == N_WINDOWS].index)
    sc["fully_labeled"] = sc["filename"].isin(full)
    return sc.sort_values(["filename", "end_sec"]).reset_index(drop=True)


def build_label_matrix(sc_meta: pd.DataFrame) -> np.ndarray:
    """Multi-hot (rows × N_CLASSES) in the order of sc_meta."""
    idx = label_to_idx()
    n_classes = len(idx)
    Y = np.zeros((len(sc_meta), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc_meta["label_list"]):
        for lb in lbls:
            j = idx.get(lb)
            if j is not None:
                Y[i, j] = 1
    return Y


def list_soundscape_files(labeled_only: bool = False) -> List[Path]:
    paths = sorted(SOUNDSCAPES.glob("*.ogg"))
    if not labeled_only:
        return paths
    labeled = set(load_soundscape_meta().query("fully_labeled")["filename"])
    return [p for p in paths if p.name in labeled]


def list_test_files() -> List[Path]:
    return sorted(TEST_SC.glob("*.ogg"))


def list_unlabeled_soundscape_files() -> List[Path]:
    labeled = set(load_soundscape_meta()["filename"])
    return [p for p in sorted(SOUNDSCAPES.glob("*.ogg")) if p.name not in labeled]


def load_taxonomy() -> pd.DataFrame:
    return pd.read_csv(TAXONOMY)
