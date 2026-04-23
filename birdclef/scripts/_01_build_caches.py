"""Entry point: build the perch cache and/or the train_audio waveform cache.

Both stages can be run independently and resumed.
"""
from __future__ import annotations

import argparse

from birdclef.cache.build_perch_cache import main as perch_main
from birdclef.cache.build_waveform_cache import main as wave_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["perch", "waveform", "all"], default="all")
    ap.add_argument("--dry-run-files", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.stage in ("perch", "all"):
        perch_main(dry_run_files=args.dry_run_files)
    if args.stage in ("waveform", "all"):
        wave_main(dry_run_files=args.dry_run_files, resume=not args.no_resume)


if __name__ == "__main__":
    main()
