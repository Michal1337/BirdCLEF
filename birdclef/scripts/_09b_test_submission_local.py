"""Run the Kaggle inference template locally on a sample of train_soundscapes.

Catches inference bugs (dtype mismatch, missing keys, wrong shape) without
burning a Kaggle daily submission quota. When the chosen files are labeled
(V-anchor or fully_labeled), it also computes macro AUC against ground truth
so you can compare to training-time numbers.

Defaults: 13 V-anchor soundscapes, write to outputs/submit/local_test/. The
13 V-anchor files are exactly what the SED trainer reported `vanchor auc=...`
on, so the AUC printed here should match those numbers within ~0.005 (the
difference is float16 ONNX vs float32 PyTorch).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from birdclef.config.paths import (
    OUTPUT_ROOT,
    SAMPLE_SUB,
    SOUNDSCAPES,
)
from birdclef.data.soundscapes import (
    build_label_matrix,
    load_soundscape_meta,
    primary_labels,
)
from birdclef.data.splits import load_v_anchor
from birdclef.eval.metrics import compute_stage_metrics


def _pick_files(source: str, n_files: int) -> List[Path]:
    if source == "anchor":
        names = load_v_anchor()
        if not names:
            raise SystemExit("V-anchor file list is empty. Run scripts/_02_build_splits.py first.")
        files = [SOUNDSCAPES / n for n in names]
    elif source == "labeled":
        sc = load_soundscape_meta()
        names = sorted(set(sc[sc["fully_labeled"]]["filename"]))
        files = [SOUNDSCAPES / n for n in names]
    elif source == "all":
        files = sorted(SOUNDSCAPES.glob("*.ogg"))
    else:
        raise SystemExit(f"unknown --source: {source}")
    files = [p for p in files if p.exists()]
    if n_files > 0:
        files = files[:n_files]
    if not files:
        raise SystemExit(f"No soundscape files found for source={source}")
    return files


def _stage_files(files: List[Path], stage_dir: Path) -> None:
    """Symlink (Linux) / copy (Windows fallback) chosen files into a flat directory.

    Inference template globs `*.ogg` from a single dir, so we mirror just the
    selected subset to avoid running on the full ~10k corpus.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = stage_dir / f.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(f.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(f, dst)


def _score_v_anchor_csv(out_csv: Path, n_files_in_test: int) -> dict:
    """If the test pool overlapped labeled soundscapes, score against ground truth.

    Returns metrics dict (or empty dict if nothing scoreable).
    """
    sub = pd.read_csv(out_csv)
    label_cols = [c for c in sub.columns if c != "row_id"]
    if not label_cols:
        return {}

    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].copy()
    if sc.empty:
        return {}
    sc["row_id"] = (
        sc["filename"].str.replace(".ogg", "", regex=False)
        + "_" + sc["end_sec"].astype(int).astype(str)
    )
    Y_all = build_label_matrix(sc)
    sc = sc.assign(_y_idx=np.arange(len(sc)))

    # Inner-join on row_id
    merged = sub.merge(sc[["row_id", "filename", "site", "hour_utc", "_y_idx"]],
                       on="row_id", how="inner")
    if merged.empty:
        return {}

    P = merged[label_cols].to_numpy(dtype=np.float32)
    Y = Y_all[merged["_y_idx"].to_numpy()]
    meta = merged[["site", "hour_utc"]].reset_index(drop=True)
    m = compute_stage_metrics(Y, P, meta)
    m["n_files_scored"] = int(merged["filename"].nunique())
    m["n_rows_scored"] = int(len(merged))
    m["n_files_in_test_pool"] = int(n_files_in_test)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sed-onnx", nargs="+", required=True,
                    help="LOCAL paths to SED ONNX files (one or more).")
    ap.add_argument("--perch-onnx", default=None,
                    help="Optional local Perch ONNX path. None = skip Perch.")
    ap.add_argument("--recipe", default=None,
                    help="Path to recipe JSON. Omit for uniform-weight blend.")
    ap.add_argument("--source", default="anchor",
                    choices=["anchor", "labeled", "all"],
                    help=("'anchor' = V-anchor files (13, exact match to SED logs), "
                          "'labeled' = all fully-labeled soundscapes, "
                          "'all' = every soundscape OGG (slow)."))
    ap.add_argument("--n-files", type=int, default=20,
                    help="Cap on files. 0 = no cap (use whole source). "
                         "For source='anchor', defaults to 20 but caps at the "
                         "13 V-anchor files automatically.")
    ap.add_argument("--out-dir", default=str(OUTPUT_ROOT / "submit" / "local_test"),
                    help="Where to write submission.csv + staged files.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir / "stage_oggs"
    out_csv = out_dir / "submission.csv"

    # Pick + stage files
    files = _pick_files(args.source, args.n_files)
    print(f"[local-test] source={args.source}  n_files={len(files)}")
    if stage_dir.exists():
        # Wipe stage to avoid mixing runs
        shutil.rmtree(stage_dir)
    _stage_files(files, stage_dir)
    print(f"[local-test] staged into {stage_dir}")

    # Recipe (optional)
    recipe = None
    if args.recipe and Path(args.recipe).exists():
        recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        print(f"[local-test] loaded recipe from {args.recipe}")

    # Run the same inference template Kaggle would use
    from birdclef.submit.inference_template import run_submission

    t0 = time.time()
    run_submission(
        test_dir=stage_dir,
        sample_sub_csv=SAMPLE_SUB,
        perch_onnx=args.perch_onnx,
        sed_onnx_paths=args.sed_onnx,
        recipe=recipe,
        output_csv=out_csv,
    )
    elapsed_min = (time.time() - t0) / 60.0

    # Score against ground truth if any of the staged files happen to be labeled
    metrics = _score_v_anchor_csv(out_csv, n_files_in_test=len(files))
    if metrics:
        print()
        print(f"[local-test] === scored {metrics['n_files_scored']} files "
              f"({metrics['n_rows_scored']} rows) ===")
        for k in ("macro_auc", "site_auc_std", "rare_auc", "frequent_auc"):
            v = metrics.get(k)
            if v is None:
                continue
            try:
                print(f"  {k:18s} = {float(v):.4f}")
            except (TypeError, ValueError):
                print(f"  {k:18s} = {v}")
        per_site = metrics.get("per_site_auc", {})
        if per_site:
            print(f"  per_site_auc       = {{ {', '.join(f'{k}:{v:.3f}' for k, v in per_site.items())} }}")
    else:
        print("[local-test] no labeled rows in test pool — skipping AUC")

    print()
    print(f"[local-test] DONE  csv={out_csv}  rows={pd.read_csv(out_csv).shape[0]}  "
          f"elapsed={elapsed_min:.2f}m")
    print(f"  per-file inference cost ≈ {elapsed_min*60/max(1, len(files)):.2f}s")
    print(f"  ETA for full Kaggle test ({len(files)} → ~700 files) ≈ "
          f"{elapsed_min * 700 / max(1, len(files)):.1f}m")


if __name__ == "__main__":
    main()
