"""Download Xeno-Canto recordings for a list of weak classes.

Stages files into `data/train_audio_xc/<primary_label>/` and writes a
parallel CSV (`data/train_audio_xc.csv`) with the same schema as
`data/train.csv`. The focal trainer can then concat the two CSVs at
load time via `--include-xc`.

Targeted at the Group A/B weak Amphibia + thin Aves from the per-class
audit (outputs/eda/per_class_audit.csv). Default target list lives in
`DEFAULT_TARGETS` at the top of this file — edit there or pass
`--targets` to override.

API auth (REQUIRED for v3 since 2025):
    Get a key at https://xeno-canto.org/account → "API access token"
    Pass via `--api-key <KEY>` or env var `XENO_CANTO_API_KEY`.
    Without a key, the script falls back to the (deprecated) v2 endpoint
    which may or may not work.

Rate limit: 1 req/s by request — script sleeps 1s between API calls.

Run:
    export XENO_CANTO_API_KEY=...
    python -m birdclef.scripts._xc_fetch --max-per-species 80
    python -m birdclef.scripts._xc_fetch --targets bunibi1 strher2 --max-per-species 50
    python -m birdclef.scripts._xc_fetch --dry-run    # query only, no download

Output:
    data/train_audio_xc/<primary_label>/<XCID>.<ext>
    data/train_audio_xc.csv

To use in training, run focal trainer with --include-xc.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.data.soundscapes import load_taxonomy


DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR  = DEFAULT_DATA_DIR / "train_audio_xc"
DEFAULT_OUT_CSV  = DEFAULT_DATA_DIR / "train_audio_xc.csv"
DEFAULT_TAXONOMY = DEFAULT_DATA_DIR / "taxonomy.csv"

# Default target list — weak Amphibia/Aves from the audit. Add/remove freely.
# The actual species names are looked up from data/taxonomy.csv.
DEFAULT_TARGETS = [
    # Group A — low/no focal Amphibia (audit ssm_auc <0.7 OR Perch underperforms)
    "22961",   # Leptodactylus podicipinus  (n_train_csv=6)
    "25073",   # Chiasmocleis mehelyi       (n_train_csv=0)
    "1491113", # Adenomera guarani          (n_train_csv=0)
    "517063",  # Pithecopus azureus         (n_train_csv=0)
    "24321",   # Scinax acuminatus          (n_train_csv=2)
    "22967",   # Leptodactylus elenae       (n_train_csv=8)
    "67252",   # Trachycephalus typhonius   (n_train_csv=6)
    "326272",  # Physalaemus biligonigerus  (n_train_csv=28, but weak)
    # Group B — birds with thin focal coverage in train.csv
    "nacnig1", # Chordeiles nacunda         (n_train_csv=18)
    "litnig1", # Setopagis parvula          (already 109 but val weak)
]

API_V3 = "https://www.xeno-canto.org/api/3/recordings"
API_V2 = "https://www.xeno-canto.org/api/2/recordings"


def _query_xc(scientific_name: str, page: int = 1, sleep: float = 1.0,
              quality: str = "A,B", api_key: str | None = None) -> Dict[str, Any]:
    """One paginated Xeno-Canto query.

    Uses v3 (with `key` param) if `api_key` provided, else falls back to
    v2 which historically required no auth. `quality=A,B` filters to
    highest-rated recordings (A=best..E=worst) — what most BirdCLEF top
    solutions use.

    v3 ONLY accepts tag-based queries. We split the binomial into
    `gen:` + `sp:` tags. Free-text queries return 400. v2 still accepts
    quoted free text.
    """
    parts = str(scientific_name).split()
    if api_key:
        # v3: tag syntax. genus + species; if there's only one token treat
        # it as genus (rare: catches taxa named to genus only).
        if len(parts) >= 2:
            q = f'gen:"{parts[0]}" sp:"{parts[1]}" q:{quality}'
        else:
            q = f'gen:"{parts[0]}" q:{quality}'
        params = {"query": q, "page": page, "key": api_key}
        url = API_V3
    else:
        # v2: free-text
        q = f'"{scientific_name}" q:{quality}'
        params = {"query": q, "page": page}
        url = API_V2
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 401:
        raise RuntimeError(
            "XC API 401 Unauthorized. Get a key at https://xeno-canto.org/account "
            "and pass via --api-key or env XENO_CANTO_API_KEY."
        )
    if r.status_code != 200:
        raise RuntimeError(f"XC API status {r.status_code}: {r.text[:200]}")
    time.sleep(sleep)
    return r.json()


def _all_results(scientific_name: str, sleep: float = 1.0,
                 quality: str = "A,B", max_pages: int = 20,
                 api_key: str | None = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        data = _query_xc(scientific_name, page=page, sleep=sleep,
                         quality=quality, api_key=api_key)
        recs = data.get("recordings", []) or []
        out.extend(recs)
        npages = int(data.get("numPages", 1))
        if page >= npages:
            break
        page += 1
    return out


def _file_url(rec: Dict[str, Any]) -> str | None:
    """Pick the primary audio URL from a recording entry."""
    # XC API v3 may return .file (direct) or .file-name + paths via 'sono'.
    # Prefer the explicit `file` field which is the playable URL.
    url = rec.get("file") or ""
    if url and not url.startswith("http"):
        url = "https:" + url if url.startswith("//") else "https://" + url.lstrip("/")
    return url or None


def _ext_for(rec: Dict[str, Any], url: str) -> str:
    """Best-guess extension for the downloaded file."""
    fn = rec.get("file-name", "") or ""
    if "." in fn:
        return fn.rsplit(".", 1)[-1].lower()
    parsed = urllib.parse.urlparse(url).path
    if "." in parsed:
        return parsed.rsplit(".", 1)[-1].lower()
    return "mp3"


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"    DOWNLOAD FAIL {url}: {e}")
        try:
            (dest.with_suffix(dest.suffix + ".part")).unlink(missing_ok=True)
        except Exception:
            pass
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=None,
                    help="primary_label values to fetch. Default = "
                         "DEFAULT_TARGETS list inside this script.")
    ap.add_argument("--auto-targets", action="store_true",
                    help="Override --targets: auto-pick all classes with ZERO focal "
                         "recordings in train.csv AND a real scientific name "
                         "(skips the 25 sonotypes — they have no XC equivalent). "
                         "These are the highest-leverage XC fetch targets — the "
                         "focal CNN currently cannot learn them at all.")
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_DATA_DIR / "train.csv",
                    help="Used by --auto-targets to find classes missing focal data.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    ap.add_argument("--max-per-species", type=int, default=80,
                    help="Cap per-species recordings (after quality filter).")
    ap.add_argument("--quality", default="A,B",
                    help="XC quality grades to include (comma-sep). Default 'A,B'.")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds between API calls (politeness).")
    ap.add_argument("--api-key", default=None,
                    help="Xeno-Canto API key. Alternatively set env XENO_CANTO_API_KEY. "
                         "Required for the v3 endpoint as of 2025; v2 fallback may break "
                         "without warning.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Query only, count results, no downloads or CSV write.")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("XENO_CANTO_API_KEY")
    if api_key:
        print(f"[xc] using v3 endpoint with API key (last 4 chars: ...{api_key[-4:]})")
    else:
        print("[xc] no API key — falling back to v2 endpoint (may not work). "
              "Get a key at https://xeno-canto.org/account")

    if not args.taxonomy.exists():
        raise SystemExit(f"Missing taxonomy: {args.taxonomy}")
    tax = pd.read_csv(args.taxonomy).set_index("primary_label")

    if args.auto_targets:
        # Classes absent from train.csv AND with a real scientific name
        # (i.e. binomial Latin name, not a "son01" sonotype placeholder)
        if not args.train_csv.exists():
            raise SystemExit(f"--auto-targets requires {args.train_csv}")
        focal_classes = set(pd.read_csv(args.train_csv)["primary_label"].astype(str).unique())
        targets = []
        for lb, row in tax.iterrows():
            lb = str(lb)
            if lb in focal_classes:
                continue  # already has focal data
            sci = str(row.get("scientific_name", ""))
            if not sci or sci.lower().startswith("insect son"):
                continue  # no real binomial — sonotype, untraceable
            targets.append(lb)
        print(f"[xc] --auto-targets selected {len(targets)} classes "
              f"(no focal in train.csv, has scientific name):")
        for t in targets:
            print(f"  - {t:<14s} {tax.loc[t, 'scientific_name']}  ({tax.loc[t, 'class_name']})")
    else:
        targets = args.targets or DEFAULT_TARGETS
        targets = [str(t) for t in targets]

    rows: List[Dict[str, Any]] = []
    grand_total = 0
    for primary_label in targets:
        if primary_label not in tax.index:
            print(f"[xc] WARN: {primary_label!r} not in taxonomy.csv — skipping")
            continue
        sci = str(tax.loc[primary_label, "scientific_name"])
        common = str(tax.loc[primary_label, "common_name"])
        cls    = str(tax.loc[primary_label, "class_name"])
        print(f"\n[xc] === {primary_label}  {sci}  ({cls} — {common}) ===")
        try:
            recs = _all_results(sci, sleep=args.sleep, quality=args.quality,
                                api_key=api_key)
        except Exception as e:
            print(f"[xc] query failed for {sci}: {e}")
            continue
        # Quality fallback: many South-American taxa have few A/B grade
        # recordings. If the strict filter returns nothing, retry once with
        # an extra grade looser (A,B → A,B,C) and warn so you can spot
        # quality-degraded fetches in the per-row CSV (rating column).
        if not recs and "C" not in args.quality:
            relaxed = (args.quality + ",C").strip(",")
            print(f"[xc]   0 results at q:{args.quality}, retrying with q:{relaxed}")
            try:
                recs = _all_results(sci, sleep=args.sleep, quality=relaxed,
                                    api_key=api_key)
            except Exception as e:
                print(f"[xc] retry failed for {sci}: {e}")
                continue

        # Sort by rating + length (prefer rated A then B; mid-length 5-90s)
        def _sort_key(r: Dict[str, Any]) -> tuple:
            grade = r.get("q", "Z")
            length_s = 0
            try:
                ll = str(r.get("length", "0:00"))
                parts = ll.split(":")
                length_s = sum(int(p) * 60 ** i for i, p in enumerate(reversed(parts)))
            except Exception:
                pass
            mid_pref = abs(length_s - 30)  # prefer ~30s clips
            return (grade, mid_pref)

        recs = sorted(recs, key=_sort_key)
        recs = recs[: int(args.max_per_species)]
        print(f"[xc]   {len(recs)} recordings to take (cap {args.max_per_species})")

        if args.dry_run:
            grand_total += len(recs)
            continue

        species_dir = args.out_dir / primary_label
        species_dir.mkdir(parents=True, exist_ok=True)

        for rec in recs:
            url = _file_url(rec)
            if not url:
                continue
            xc_id = str(rec.get("id", "")).strip()
            if not xc_id:
                continue
            ext = _ext_for(rec, url)
            fname_rel = f"{primary_label}/XC{xc_id}.{ext}"
            dest = args.out_dir / fname_rel
            ok = _download(url, dest)
            if not ok:
                continue
            rows.append({
                "primary_label":     primary_label,
                "secondary_labels":  "[]",
                "type":              "[]",
                "latitude":          rec.get("lat") or None,
                "longitude":         rec.get("lng") or None,
                "scientific_name":   sci,
                "common_name":       common,
                "class_name":        cls,
                "inat_taxon_id":     primary_label if primary_label.isdigit() else "",
                "author":            rec.get("rec", "") or "",
                "license":           rec.get("lic", "") or "",
                "rating":            rec.get("q", "") or "",
                "url":               url,
                "filename":          fname_rel,
                "collection":        "XC",
            })
        grand_total += len(rows)

    if args.dry_run:
        print(f"\n[xc] DRY-RUN total recordings available across {len(targets)} targets: {grand_total}")
        return

    if not rows:
        print("[xc] no recordings downloaded")
        return

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_csv.exists():
        # Merge with prior runs, dedup by (primary_label, filename)
        prev = pd.read_csv(args.out_csv)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["primary_label", "filename"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)

    print()
    print(f"[xc] downloaded {len(rows)} new recordings into {args.out_dir}")
    print(f"[xc] wrote merged CSV ({len(df)} total rows): {args.out_csv}")
    print()
    print("To use in training, concat both CSVs in train_ddp_focal.main():")
    print("  train_csv_df = pd.concat([")
    print("      pd.read_csv('data/train.csv'),")
    print("      pd.read_csv('data/train_audio_xc.csv'),")
    print("  ], ignore_index=True)")


if __name__ == "__main__":
    main()
