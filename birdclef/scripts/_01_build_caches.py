"""Entry point: build the perch / train_audio / soundscape caches.

Each stage can be run independently. The waveform stage supports --resume;
the perch and soundscape stages always rebuild from scratch (idempotent
within their output directory).
"""
from __future__ import annotations

import argparse

from birdclef.cache.build_perch_cache import main as perch_main
from birdclef.cache.build_soundscape_cache import main as soundscape_main
from birdclef.cache.build_waveform_cache import main as wave_main


STAGES = ("perch", "waveform", "soundscape", "all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--dry-run-files", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true",
                    help="Disable resume on the waveform stage (other stages always rebuild).")
    ap.add_argument("--num-threads", type=int, default=8,
                    help="Decode threads for the soundscape stage.")
    args = ap.parse_args()
    if args.stage in ("perch", "all"):
        perch_main(dry_run_files=args.dry_run_files)
    if args.stage in ("waveform", "all"):
        wave_main(dry_run_files=args.dry_run_files, resume=not args.no_resume)
    if args.stage in ("soundscape", "all"):
        soundscape_main(dry_run_files=args.dry_run_files, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
