"""Perch v2 ONNX wrapper + genus proxy mapping.

Runs 60 s soundscape files as 12x5 s windows and returns (scores, embeddings)
aligned to the BirdCLEF 2026 label set.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
from pathlib import Path
from typing import List
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from birdclef.config.paths import (
    FILE_SAMPLES,
    N_WINDOWS,
    ONNX_PERCH_PATH,
    PERCH_PROXY_MAP,
    PERCH_TF_LABELS,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, primary_labels, load_taxonomy


def read_60s(path: Path | str) -> np.ndarray:
    y, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32)


def load_onnx_session(num_threads: int = 8):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(num_threads)
    sess = ort.InferenceSession(
        str(ONNX_PERCH_PATH), sess_options=so, providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE"}),
            "CPUExecutionProvider",
        ]
    )
    return sess


def build_label_mapping(bc_labels_csv: Path | None = None):
    """Resolve each BirdCLEF target species to a Perch logit index.

    Returns:
        MAPPED_POS      : np.int32[n_mapped] — positions in BirdCLEF label order
        MAPPED_BC_IDX   : np.int32[n_mapped] — corresponding Perch indices
        UNMAPPED_POS    : np.int32[n_unmapped]
        proxy_map       : dict[int, List[int]] — BirdCLEF_pos -> list of Perch idx
    """
    csv_path = Path(bc_labels_csv or PERCH_TF_LABELS)
    bc = (
        pd.read_csv(csv_path)
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    no_label = len(bc)
    tax = load_taxonomy()
    mapping = tax.merge(bc, on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label).astype(int)
    lbl2bc = mapping.set_index("primary_label")["bc_index"]
    labels = primary_labels()
    bc_indices = np.array([int(lbl2bc.loc[c]) for c in labels], dtype=np.int32)
    mapped_mask = bc_indices != no_label
    mapped_pos = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc = bc_indices[mapped_mask].astype(np.int32)
    unmapped_pos = np.where(~mapped_mask)[0].astype(np.int32)

    proxy_taxa = {"Amphibia", "Insecta", "Aves"}
    l2i = label_to_idx()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    unmapped_df = tax[tax["primary_label"].isin([labels[i] for i in unmapped_pos])]
    proxy_map: dict[int, List[int]] = {}
    for _, row in unmapped_df.iterrows():
        target = row["primary_label"]
        sci = str(row["scientific_name"])
        genus = sci.split()[0]
        hits = bc[bc["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)]
        if len(hits) > 0 and class_map.get(target) in proxy_taxa:
            proxy_map[l2i[target]] = hits["bc_index"].astype(int).tolist()

    return {
        "mapped_pos": mapped_pos,
        "mapped_bc": mapped_bc,
        "unmapped_pos": unmapped_pos,
        "proxy_map": proxy_map,
    }


def save_proxy_map(proxy_map: dict[int, List[int]]) -> None:
    PERCH_PROXY_MAP.parent.mkdir(parents=True, exist_ok=True)
    PERCH_PROXY_MAP.write_text(
        json.dumps({str(k): v for k, v in proxy_map.items()}, indent=2),
        encoding="utf-8",
    )


def load_proxy_map() -> dict[int, List[int]]:
    if not PERCH_PROXY_MAP.exists():
        return {}
    raw = json.loads(PERCH_PROXY_MAP.read_text(encoding="utf-8"))
    return {int(k): list(v) for k, v in raw.items()}


def run_perch(
    paths: List[Path],
    *,
    session=None,
    mapping: dict | None = None,
    batch_files: int = 16,
    verbose: bool = True,
):
    """Run Perch ONNX over a list of 60 s OGG paths.

    Returns (meta_df, scores[N*12, C], embs[N*12, 1536]).
    """
    from birdclef.data.soundscapes import parse_fname

    labels = primary_labels()
    n_classes = len(labels)
    if session is None:
        session = load_onnx_session()
    if mapping is None:
        mapping = build_label_mapping()
    mapped_pos = mapping["mapped_pos"]
    mapped_bc = mapping["mapped_bc"]
    proxy_map = mapping["proxy_map"]

    input_name = session.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(session.get_outputs())}

    paths = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    dates = np.empty(n_rows, dtype=object)
    hours = np.zeros(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, n_classes), dtype=np.float32)
    embs = np.zeros((n_rows, 1536), dtype=np.float32)

    wr = 0
    it = range(0, len(paths), batch_files)
    if verbose:
        it = tqdm(it, desc="Perch")
    with cf.ThreadPoolExecutor(max_workers=8) as io:
        next_paths = paths[0:batch_files]
        future = [io.submit(read_60s, p) for p in next_paths]
        for start in it:
            batch_paths = next_paths
            batch_audio = [f.result() for f in future]
            nxt = start + batch_files
            if nxt < len(paths):
                next_paths = paths[nxt : nxt + batch_files]
                future = [io.submit(read_60s, p) for p in next_paths]
            n = len(batch_paths)
            x = np.empty((n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr
            for bi, pth in enumerate(batch_paths):
                y = batch_audio[bi]
                meta = parse_fname(pth.name)
                stem = pth.stem
                x[bi * N_WINDOWS : (bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids[wr : wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr : wr + N_WINDOWS] = pth.name
                sites[wr : wr + N_WINDOWS] = meta["site"]
                dates[wr : wr + N_WINDOWS] = meta["date"]
                hours[wr : wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS
            outs = session.run(None, {input_name: x})
            logits = outs[out_map["label"]].astype(np.float32)
            emb = outs[out_map["embedding"]].astype(np.float32)
            scores[br:wr, mapped_pos] = logits[:, mapped_bc]
            embs[br:wr] = emb
            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.asarray(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)
    meta_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "date": dates,
            "hour_utc": hours,
        }
    )
    return meta_df, scores, embs
