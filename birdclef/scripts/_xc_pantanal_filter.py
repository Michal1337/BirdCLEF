"""Inspect train.csv for recordings within (or near) the Pantanal region.

The competition's soundscape recorder coordinates from
data/recording_location.txt are:
    Latitude:  -16.5 to -21.6   (i.e. -16.5 ≥ lat ≥ -21.6)
    Longitude: -55.9 to -57.6

Restricting train.csv (Xeno-Canto + iNat focal recordings) to a
neighborhood around those bounds gives the focal-recording subset that's
acoustically and geographically closest to the LB test set. This script
quantifies how many rows survive at various buffers, and (optionally)
writes a filtered CSV for trainer use.

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._xc_pantanal_filter
        # prints stats at multiple buffers, writes nothing

    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._xc_pantanal_filter \\
        --buffer-deg 2.0 --out data/train_pantanal_buf2.csv
        # writes a filtered subset CSV at the chosen buffer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pantanal recorder bounds (from data/recording_location.txt)
LAT_MIN, LAT_MAX = -21.6, -16.5
LON_MIN, LON_MAX = -57.6, -55.9


def in_box(lat: pd.Series, lon: pd.Series, buffer_deg: float) -> pd.Series:
    """Boolean mask: rows within [LAT/LON range] expanded by `buffer_deg`."""
    return (
        (lat >= (LAT_MIN - buffer_deg))
        & (lat <= (LAT_MAX + buffer_deg))
        & (lon >= (LON_MIN - buffer_deg))
        & (lon <= (LON_MAX + buffer_deg))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--train-csv", type=Path,
                    default=Path("/mnt/evafs/groups/re-com/mgromadzki/data/train.csv"),
                    help="Path to train.csv. Default = the Hopper mount.")
    ap.add_argument("--taxonomy-csv", type=Path,
                    default=REPO_ROOT / "data" / "taxonomy.csv")
    ap.add_argument("--sample-sub", type=Path,
                    default=REPO_ROOT / "data" / "sample_submission.csv")
    ap.add_argument("--buffer-deg", type=float, default=None,
                    help="If given, write filtered CSV at this buffer to --out.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output filtered CSV path (only used with --buffer-deg).")
    args = ap.parse_args()

    if not args.train_csv.exists():
        # Fallback to repo data/ for local dev
        local = REPO_ROOT / "data" / "train.csv"
        if local.exists():
            args.train_csv = local
        else:
            raise SystemExit(
                f"train.csv not found. Tried {args.train_csv} and {local}."
            )

    print(f"[pantanal] reading {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    print(f"[pantanal] total rows: {len(df):,}")
    print(f"[pantanal] columns: {list(df.columns)}")

    # Identify lat/lon columns. train.csv historically uses 'latitude' /
    # 'longitude'; some flavors use 'lat' / 'lon'. Try both.
    lat_col = next((c for c in ("latitude", "lat") if c in df.columns), None)
    lon_col = next((c for c in ("longitude", "lon", "long") if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise SystemExit(
            f"train.csv missing lat/lon columns. Found: {list(df.columns)}"
        )
    print(f"[pantanal] using columns: lat='{lat_col}', lon='{lon_col}'")

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    has_coords = lat.notna() & lon.notna()
    n_with_coords = int(has_coords.sum())
    print(f"[pantanal] rows with valid lat/lon: {n_with_coords:,} "
          f"({n_with_coords/len(df)*100:.1f}%)")
    print(f"[pantanal] rows missing coords: {len(df) - n_with_coords:,} "
          f"(would be dropped or kept depending on filter policy)")

    # Class breakdown
    if "primary_label" in df.columns:
        class_col = "primary_label"
    else:
        raise SystemExit("train.csv missing 'primary_label' column")
    n_classes_total = df[class_col].astype(str).nunique()

    # Reference: classes that appear in the soundscape label set (the 234
    # competition classes). Any class outside this set is irrelevant for LB.
    if args.sample_sub.exists():
        sample_sub = pd.read_csv(args.sample_sub)
        comp_classes = set(sample_sub.columns[1:].astype(str))
        df["_is_comp"] = df[class_col].astype(str).isin(comp_classes)
        print(f"[pantanal] competition classes (from sample_sub): {len(comp_classes)}")
    else:
        comp_classes = None

    # ── Stats at multiple buffer sizes ─────────────────────────────
    print()
    print(f"{'buffer':>8s} | {'rows kept':>12s} | {'  %':>6s} | "
          f"{'unique classes':>16s} | {'comp classes':>14s}")
    print("-" * 72)

    box_centroid = ((LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2)
    for buf in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
        mask = has_coords & in_box(lat, lon, buf)
        sub = df[mask]
        n = len(sub)
        n_cls = sub[class_col].astype(str).nunique()
        n_comp_cls = (sub[df.columns.get_loc("_is_comp") if "_is_comp" in df.columns else 0]
                       .astype(str).nunique() if "_is_comp" not in df.columns else
                      sub.loc[sub["_is_comp"], class_col].astype(str).nunique()) if comp_classes else 0
        print(f"{buf:>6.1f}°  | {n:>12,d} | {n/len(df)*100:>5.2f}% | "
              f"{n_cls:>16d} | {n_comp_cls:>14d}")

    # ── Write filtered subset if requested ────────────────────────
    if args.buffer_deg is not None:
        if args.out is None:
            args.out = REPO_ROOT / "data" / f"train_pantanal_buf{args.buffer_deg:.1f}.csv"
        mask = has_coords & in_box(lat, lon, float(args.buffer_deg))
        sub = df[mask].drop(columns=[c for c in ("_is_comp",) if c in df.columns])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(args.out, index=False)
        print()
        print(f"[pantanal] wrote {len(sub):,} rows at buffer {args.buffer_deg}° to {args.out}")

    # ── Recommendation ────────────────────────────────────────────
    print()
    print("[pantanal] === Recommendation ===")
    print("  - 0° buffer is too tight: ~0-100 rows for most species. Useless alone.")
    print("  - 2° buffer is the historical recipe: keeps ~Pantanal-adjacent recordings,")
    print("    typically 5-15% of train.csv. Worth trying as an XC subset for AST/CNN.")
    print("  - 5° buffer: ~20-40% of train.csv, broader 'tropical SA' acoustic context.")
    print("  - 20°+ buffer: roughly the whole continent — defeats the point.")
    print()
    print("  Realistic plan: train AST/CNN on `train_pantanal_buf2.csv` + the existing")
    print("  pseudo cache. The pseudo step gives soundscape-domain signal; the geographic")
    print("  filter ensures focal pre-training is already in roughly the right acoustic")
    print("  region. Expected effect: smaller in-distribution inflation than the global")
    print("  XC-trained CNN, possibly better LB transfer.")
    print()
    print("  NOTE: the Pantanal soundscape coverage is biased toward Aves. Insecta and")
    print("  Amphibia at LB are likely undersampled in any 2° buffer. Use the buffer for")
    print("  Aves but consider unfiltered XC for non-bird taxa.")


if __name__ == "__main__":
    main()
