"""Rebuild the keep_mask of an existing pseudo-label round without re-running
the teacher. Useful when the initial tau/topk choice produced a too-permissive
or too-strict mask.

Also supports switching which tensor is used as the emitted target
(`final` → `probs`  or  `first_pass` → better-calibrated pre-post-proc output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdclef.config.paths import PSEUDO_DIR
from birdclef.train.pseudo_label import _apply_confidence_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--topk-per-species", type=int, default=0)
    ap.add_argument(
        "--emit", choices=["final", "first_pass"], default="final",
        help="Which tensor to write as `probs` in the rebuilt npz. "
             "`first_pass` is better-calibrated for soft-target BCE.",
    )
    ap.add_argument(
        "--inplace", action="store_true",
        help="Overwrite round{N}/probs.npz. Default writes round{N}/probs.tau{tau}.npz.",
    )
    args = ap.parse_args()

    rdir = PSEUDO_DIR / f"round{int(args.round)}"
    src = rdir / "probs.npz"
    if not src.exists():
        raise SystemExit(f"No such file: {src}")
    arrs = dict(np.load(src))
    keys = list(arrs.keys())
    print(f"[refilter] loaded {src}  keys={keys}  shapes={ {k: arrs[k].shape for k in keys} }")

    if args.emit == "first_pass" and "first_pass" not in arrs:
        raise SystemExit("first_pass array missing from this round; regen with the SSM teacher.")

    emit_probs = arrs[args.emit].astype(np.float32)
    first_pass = arrs.get("first_pass", arrs.get("probs")).astype(np.float32)
    final = arrs.get("probs").astype(np.float32) if "probs" in arrs else emit_probs

    print(f"[refilter] emitting `{args.emit}` as probs  "
          f"mean={emit_probs.mean():.4f}  p99={np.percentile(emit_probs, 99):.4f}")
    keep_mask = _apply_confidence_filter(emit_probs, args.tau, args.topk_per_species)
    print(f"[refilter] tau={args.tau}  topk={args.topk_per_species}  "
          f"keep_fraction={float(keep_mask.mean()):.6f}")

    out_path = src if args.inplace else rdir / f"probs.tau{args.tau:.2f}.npz"
    # We always store:
    #   probs      = tensor the consumer uses as soft-target
    #   keep_mask  = the re-built mask
    #   final, first_pass = both original tensors for provenance
    np.savez_compressed(
        out_path,
        probs=emit_probs,
        keep_mask=keep_mask,
        final=final,
        first_pass=first_pass,
    )
    print(f"[refilter] wrote {out_path}")

    info_path = rdir / "info.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info = {}
    info["refilter"] = {
        "tau": args.tau,
        "topk_per_species": args.topk_per_species,
        "emit": args.emit,
        "keep_fraction": float(keep_mask.mean()),
        "output": str(out_path.name),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[refilter] updated {info_path}")


if __name__ == "__main__":
    main()
