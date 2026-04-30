"""Extract dates from soundscape filenames and check whether the
labeled subset is a tight temporal window vs a broad spread.

Filename format (from `birdclef.data.soundscapes.parse_fname`):
    BC2026_Train_<id>_<S##>_<YYYYMMDD>_<HHMMSS>.ogg

If the within-S22 adversarial AUC is 0.99 (the inspection-script result),
the discriminator is finding labeled/unlabeled differences WITHIN the
same recorder location. Time-of-recording is the most likely axis. This
script confirms or rules out that hypothesis by printing per-site date
distributions for labeled and unlabeled cache rows.

Output (per mixed site):
  - labeled date range (min, max, span days, n unique dates)
  - unlabeled date range
  - overlap: are labeled dates a subset of unlabeled dates?
  - per-month breakdown so you can see clusters
  - "temporal labeled-likeness" suggestion: which date windows of
    unlabeled rows are within ≤ N days of any labeled date

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10c_inspect_temporal_split
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._10c_inspect_temporal_split --window-days 14
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import OUTPUT_ROOT, PERCH_META

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_date_from_fname(name: str) -> pd.Timestamp | None:
    m = FNAME_RE.match(name)
    if not m:
        return None
    _id, _site, ymd, _hms = m.groups()
    return pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")


def fmt_date_range(dates: pd.Series) -> str:
    """Pretty-print min..max + span in days for a pd.Series of dates."""
    if dates.empty:
        return "(empty)"
    valid = dates.dropna()
    if valid.empty:
        return "(no valid dates)"
    lo, hi = valid.min(), valid.max()
    span = (hi - lo).days
    n_unique = valid.dt.normalize().nunique()
    return (f"{lo.date()} → {hi.date()}  span={span:>4d}d  "
            f"unique_dates={n_unique:>4d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--meta", type=Path, default=PERCH_META,
                    help="Path to the Perch cache meta.parquet.")
    ap.add_argument("--window-days", type=int, default=14,
                    help="Build a 'temporal labeled-likeness' score where "
                         "unlabeled rows within ≤ N days of any labeled "
                         "date count as 'temporally labeled-like'. Default 14.")
    ap.add_argument("--out-npz", type=Path,
                    default=OUTPUT_ROOT / "adversarial" / "temporal_likeness.npz",
                    help="Where to write per-row temporal-likeness scores.")
    args = ap.parse_args()

    if not args.meta.exists():
        raise SystemExit(f"meta not found at {args.meta}")

    print(f"[temp] loading {args.meta}")
    meta = pd.read_parquet(args.meta)
    if "is_labeled" not in meta.columns:
        raise SystemExit("meta has no 'is_labeled' column. Rebuild the cache.")

    # Parse dates from filenames
    print(f"[temp] parsing dates from {len(meta):,} filenames...")
    meta = meta.copy()
    meta["date"] = meta["filename"].astype(str).map(parse_date_from_fname)
    meta["is_labeled"] = meta["is_labeled"].astype(bool)
    n_unparsed = int(meta["date"].isna().sum())
    if n_unparsed:
        print(f"[temp] WARN: {n_unparsed} filenames didn't match the regex "
              f"(they'll be excluded from per-site stats)")
    parsed = meta.dropna(subset=["date"]).copy()
    print(f"[temp] parsed {len(parsed):,} dates "
          f"(global {fmt_date_range(parsed['date'])})")

    # ── Global labeled-vs-unlabeled date ranges ─────────────────────
    print()
    print("=== Global labeled vs unlabeled date distributions ===")
    print(f"  labeled  : {fmt_date_range(parsed.loc[parsed['is_labeled'], 'date'])}")
    print(f"  unlabeled: {fmt_date_range(parsed.loc[~parsed['is_labeled'], 'date'])}")
    overlap_dates = (
        set(parsed.loc[parsed["is_labeled"], "date"].dt.normalize())
        & set(parsed.loc[~parsed["is_labeled"], "date"].dt.normalize())
    )
    n_lab_dates = parsed.loc[parsed["is_labeled"], "date"].dt.normalize().nunique()
    n_unl_dates = parsed.loc[~parsed["is_labeled"], "date"].dt.normalize().nunique()
    print(f"  shared dates (labeled ∩ unlabeled): {len(overlap_dates)} / "
          f"{n_lab_dates} labeled dates, {n_unl_dates} unlabeled dates")

    # ── Per-site temporal split ─────────────────────────────────────
    print()
    print("=== Per-site temporal split (mixed sites only) ===")
    if "site" not in parsed.columns:
        raise SystemExit("meta has no 'site' column. Rebuild the cache.")

    rows = []
    for site, sub in parsed.groupby("site"):
        n_lab = int(sub["is_labeled"].sum())
        n_unl = int((~sub["is_labeled"]).sum())
        if n_lab == 0 or n_unl == 0:
            continue   # not a mixed site
        lab_dates = sub.loc[sub["is_labeled"], "date"]
        unl_dates = sub.loc[~sub["is_labeled"], "date"]
        lab_range = (lab_dates.min(), lab_dates.max())
        unl_range = (unl_dates.min(), unl_dates.max())
        lab_span = (lab_range[1] - lab_range[0]).days
        unl_span = (unl_range[1] - unl_range[0]).days

        # Are labeled dates a subset of the unlabeled date range?
        lab_inside_unl_range = bool(
            (lab_range[0] >= unl_range[0]) and (lab_range[1] <= unl_range[1])
        )
        # Hard date-set overlap (same calendar day appears in both)
        lab_dates_set = set(lab_dates.dt.normalize())
        unl_dates_set = set(unl_dates.dt.normalize())
        date_overlap = len(lab_dates_set & unl_dates_set)
        # How many unlabeled rows fall within the labeled date range?
        in_lab_range = int(((unl_dates >= lab_range[0]) &
                             (unl_dates <= lab_range[1])).sum())
        rows.append({
            "site": site,
            "n_lab": n_lab, "n_unl": n_unl,
            "lab_span_d": lab_span, "unl_span_d": unl_span,
            "lab_min": lab_range[0].date(), "lab_max": lab_range[1].date(),
            "unl_min": unl_range[0].date(), "unl_max": unl_range[1].date(),
            "lab_uniq_dates": lab_dates.dt.normalize().nunique(),
            "unl_uniq_dates": unl_dates.dt.normalize().nunique(),
            "shared_calendar_days": date_overlap,
            "lab_inside_unl_range": lab_inside_unl_range,
            "unl_rows_in_lab_range": in_lab_range,
            "unl_rows_in_lab_range_pct": (
                100.0 * in_lab_range / max(1, n_unl)),
        })

    if not rows:
        print("[temp] No sites have both labeled and unlabeled rows.")
        return

    df = pd.DataFrame(rows).sort_values("n_lab", ascending=False)
    print(df.to_string(index=False))

    print()
    # Pull out the headline numbers in human terms
    for _, r in df.iterrows():
        verdict = []
        if r["lab_span_d"] <= 30 and r["unl_span_d"] >= 90:
            verdict.append("LABELED IS TIGHT WINDOW (≤30d)")
        if r["unl_rows_in_lab_range_pct"] < 5.0:
            verdict.append("≤5% of unlabeled is in labeled date range")
        if r["shared_calendar_days"] == 0:
            verdict.append("NO CALENDAR-DAY OVERLAP")
        if verdict:
            print(f"  [{r['site']}] {' | '.join(verdict)}")

    # ── Build temporal-likeness scores for every cache row ─────────
    print()
    print(f"=== Temporal labeled-likeness (window={args.window_days}d) ===")
    lab_dates_global = parsed.loc[parsed["is_labeled"], "date"].dt.normalize().unique()
    lab_dates_global = pd.DatetimeIndex(sorted(lab_dates_global))
    print(f"  {len(lab_dates_global)} unique labeled dates globally")

    # For each row in meta (incl. unparsed), compute distance in days to nearest
    # labeled date. NaN dates → score = 0 (can't filter without info).
    n_rows = len(meta)
    scores = np.zeros(n_rows, dtype=np.float32)
    has_date = ~meta["date"].isna()
    if has_date.any():
        date_array = meta.loc[has_date, "date"].dt.normalize().to_numpy()
        # Vectorized nearest-label distance
        lab_arr = lab_dates_global.to_numpy()
        # Difference matrix would be O(N × M); tolerate it because lab is small
        dates_idx = np.searchsorted(lab_arr, date_array)
        # Distances to neighbour-left and neighbour-right of each insertion point
        left = np.where(dates_idx > 0, lab_arr[np.clip(dates_idx - 1, 0, len(lab_arr) - 1)], lab_arr[0])
        right = np.where(dates_idx < len(lab_arr), lab_arr[np.clip(dates_idx, 0, len(lab_arr) - 1)], lab_arr[-1])
        d_left = np.abs((date_array - left).astype("timedelta64[D]").astype(np.int64))
        d_right = np.abs((date_array - right).astype("timedelta64[D]").astype(np.int64))
        nearest_days = np.minimum(d_left, d_right).astype(np.float32)
        # Score: 1 if nearest labeled date ≤ window_days, else exponentially decay
        wd = float(args.window_days)
        score = np.where(nearest_days <= wd, 1.0, np.exp(-(nearest_days - wd) / max(wd, 1.0)))
        scores[has_date.to_numpy()] = score.astype(np.float32)

    # Summary stats
    n_within_window_unl = int(((scores >= 1.0) & (~meta["is_labeled"]).to_numpy()).sum())
    n_unl_total = int((~meta["is_labeled"]).sum())
    print(f"  unlabeled rows ≤ {args.window_days}d from any labeled date: "
          f"{n_within_window_unl:,} / {n_unl_total:,} "
          f"({n_within_window_unl/max(1, n_unl_total)*100:.1f}%)")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        temporal_likeness=scores,
        is_labeled=meta["is_labeled"].to_numpy().astype(np.uint8),
        filenames=meta["filename"].astype(str).to_numpy(),
        labeled_dates_yyyymmdd=np.array(
            [d.strftime("%Y%m%d") for d in lab_dates_global], dtype="U8",
        ),
        window_days=np.int32(args.window_days),
    )
    print(f"  wrote per-row temporal-likeness scores → {args.out_npz}")
    print()
    print("[temp] === Use suggestions ===")
    print("  - If labeled spans ≤30 days and ≤5% of unlabeled overlaps that range:")
    print("      filter pseudo to unlabeled rows with temporal_likeness >= 1.0")
    print("      (i.e., within window_days of a labeled date) → temporally aligned")
    print("      pseudo-labeling.")
    print("  - If labeled spans 100+ days and overlaps unlabeled broadly: time is")
    print("      NOT the primary axis. The 0.99 adv-AUC must come from equipment")
    print("      / mic / protocol differences. Filter doesn't help; pivot to")
    print("      acquiring data with matching equipment, or accept the gap.")


if __name__ == "__main__":
    main()
