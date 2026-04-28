"""Regenerate birdclef/cache/perch/ artifacts from raw soundscape OGGs.

Creates:
    cache/perch/meta.parquet      one row per 5-s window, stable sort
    cache/perch/arrays.npz        scores_full_raw, scores_proxy, emb_full
                                  (all float32). scores_full_raw holds
                                  ONLY direct sci_name → Perch logits (zeros
                                  at unmapped positions); scores_proxy holds
                                  the genus-max fill at proxy positions
                                  (zeros elsewhere). Disjoint by construction.
                                  Each consumer decides whether to add them
                                  via `apply_proxy_to_scores()`.
    cache/perch/labels.npy        multi-hot Y[N,C] uint8 (labeled rows only;
                                  unlabeled rows all-zero)
    cache/perch/proxy_map.json    unmapped-species genus proxy indices (kept
                                  for diagnostics; consumers don't need it
                                  to apply proxy since scores_proxy already
                                  has the fill values)

Runtime: hours on CPU. Use --dry-run-files to smoke-test first.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.config.paths import (
    PERCH_DIR,
    PERCH_META,
    PERCH_NPZ,
    PERCH_LABELS,
    SOUNDSCAPES,
    ensure_dirs,
)
from birdclef.data.soundscapes import (
    build_label_matrix,
    load_soundscape_meta,
    primary_labels,
)
from birdclef.models.perch import (
    build_label_mapping,
    load_onnx_session,
    run_perch,
    save_proxy_map,
)


def main(dry_run_files: int = 0) -> None:
    ensure_dirs()
    PERCH_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(SOUNDSCAPES.glob("*.ogg"))
    if dry_run_files:
        all_paths = all_paths[:dry_run_files]
    print(f"[perch_cache] {len(all_paths)} soundscape files")

    sc_meta = load_soundscape_meta()
    labeled_set = set(sc_meta[sc_meta["fully_labeled"]]["filename"])

    # Sort: labeled first, unlabeled second, filename ascending
    labeled = sorted([p for p in all_paths if p.name in labeled_set])
    unlabeled = sorted([p for p in all_paths if p.name not in labeled_set])
    ordered = labeled + unlabeled

    mapping = build_label_mapping()
    save_proxy_map(mapping["proxy_map"])
    print(f"[perch_cache] mapped {len(mapping['mapped_pos'])} species, "
          f"unmapped {len(mapping['unmapped_pos'])}, proxies {len(mapping['proxy_map'])}")

    session = load_onnx_session()
    t0 = time.time()
    meta_df, scores, embs, scores_proxy = run_perch(
        ordered, session=session, mapping=mapping, batch_files=32,
    )
    meta_df["is_labeled"] = meta_df["filename"].isin(labeled_set).astype("uint8")
    dt = time.time() - t0
    print(f"[perch_cache] Perch inference done in {dt/60:.1f} min. rows={len(meta_df)}")
    n_proxy_cells = int((scores_proxy != 0).sum())
    print(f"[perch_cache] proxy fill: {n_proxy_cells} non-zero cells across "
          f"{len(mapping['proxy_map'])} target columns")

    # Persist. scores_full_raw and scores_proxy are stored separately so
    # downstream code can choose whether to merge them via
    # `apply_proxy_to_scores()`.
    meta_df.to_parquet(PERCH_META, index=False)
    np.savez_compressed(
        PERCH_NPZ,
        scores_full_raw=scores,
        scores_proxy=scores_proxy,
        emb_full=embs,
    )

    # Build labels aligned to meta_df rows
    labels_n = len(primary_labels())
    Y = np.zeros((len(meta_df), labels_n), dtype=np.uint8)
    sc_y = build_label_matrix(sc_meta)
    sc_by_rowid = dict(zip(sc_meta["row_id"], range(len(sc_meta))))
    for i, rid in enumerate(meta_df["row_id"].values):
        j = sc_by_rowid.get(rid)
        if j is not None:
            Y[i] = sc_y[j]
    np.save(PERCH_LABELS, Y)
    active = int((Y.sum(axis=0) > 0).sum())
    print(f"[perch_cache] labels.npy saved; active classes on labeled rows: {active}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run-files", type=int, default=0)
    args = p.parse_args()
    main(dry_run_files=args.dry_run_files)
