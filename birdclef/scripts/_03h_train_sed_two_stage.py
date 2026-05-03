"""Two-stage fold-aware SED training: broad pretrain + soundscape finetune.

Stage 1 — pretrain on the full pool (focal `train_audio` + labeled
fold-train soundscapes + pseudo round 2) for N1 epochs. Same recipe as
the standard `train_sed_ddp` run; uses the BASELINE config + pseudo_round.

Stage 2 — finetune on ALL SOUNDSCAPES (labeled fold-train GT + pseudo
round 2 unlabeled) — no focal `train_audio`. Warm-started from stage 1's
best.pt with a smaller LR. Backbone optionally frozen so only the heads
specialize to in-distribution Pantanal calls.

Why this recipe: stage 1 builds broad species discrimination from focal
+ pseudo. Stage 2 strips focal, leaving only the in-distribution
soundscape pool (~10k files × 12 windows ≈ 120k samples). The student
specializes its predictions to soundscape acoustics without forgetting
species features. Labeled-only stage 2 was too small (~564 rows → ~4
steps/epoch) to do meaningful finetuning; pseudo expands the pool to
real training scale.

Output:
    MODEL_ROOT/sed/<config>_stage1/fold{f}/best.pt    ← broad-pretrain
    MODEL_ROOT/sed/<config>_stage2/fold{f}/best.pt    ← finetuned (USE THIS)
    MODEL_ROOT/sed/<config>_stage{1,2}/fold{f}/train_history.jsonl

Usage:
    PYTHONIOENCODING=utf-8 BIRDCLEF_PRELOAD_CACHE=1 \\
        torchrun --standalone --nproc_per_node=2 \\
        -m birdclef.scripts._03h_train_sed_two_stage \\
        --config sed_b0_dual --fold 0 \\
        --stage1-epochs 40 --stage2-epochs 50 --stage2-lr 1e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import MODEL_ROOT
from birdclef.train.train_sed_ddp import _build_cfg, train_one_fold


def _is_main_rank() -> bool:
    """Rank-0 of the (possibly DDP) process group, for logging only."""
    import os
    return int(os.environ.get("RANK", "0")) == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Base SED config name (used as prefix for both stages' "
                         "ckpt dirs: <config>_stage1, <config>_stage2).")
    ap.add_argument("--fold", type=int, required=True,
                    help="Fold index 0..n_splits-1. Stage 2 reuses the same fold.")
    ap.add_argument("--stage1-epochs", type=int, default=40,
                    help="Epochs for broad pretrain. Default 40.")
    ap.add_argument("--stage1-pseudo-round", type=int, default=2,
                    help="Pseudo round for stage 1. 0/None for no pseudo. Default 2.")
    ap.add_argument("--stage2-epochs", type=int, default=20,
                    help="Epochs for soundscape-finetune. Stage 2 pool is now "
                         "labeled GT + pseudo (~120k samples, ~937 steps/epoch at "
                         "batch=64×2 ranks). Default 20 epochs ≈ 18.7k steps — "
                         "comfortable for head finetuning.")
    ap.add_argument("--stage2-lr", type=float, default=1e-4,
                    help="Stage-2 learning rate. Lower than stage 1 (1e-3) since "
                         "we're finetuning, not training from scratch.")
    ap.add_argument("--stage2-pseudo-round", type=int, default=2,
                    help="Pseudo round for stage 2. Default 2 (uses round 2 to "
                         "expand the soundscape pool). Set to 0 / None to use "
                         "labeled fold-train ONLY (legacy small-pool recipe; not "
                         "recommended).")
    ap.add_argument("--stage2-batch-size", type=int, default=None,
                    help="Override batch size for stage 2 (default: same as stage 1).")
    ap.add_argument("--stage2-warmup-frac", type=float, default=0.0,
                    help="Stage-2 warmup fraction. Default 0.0 — no warmup needed "
                         "when warm-starting from a converged backbone.")
    ap.add_argument("--stage2-eval-every-n-steps", type=int, default=500,
                    help="Eval frequency (steps) for stage 2. With the expanded "
                         "soundscape pool (~937 steps/epoch), the BASELINE default "
                         "of 500 fires ~37 times across 20 epochs — fine.")
    ap.add_argument("--stage2-freeze-backbone", action="store_true", default=True,
                    help="Stage 2: freeze the backbone (timm CNN). Only the "
                         "attention pool + clip head + framewise head are "
                         "trained. Keeps backbone BN running stats frozen so "
                         "the small finetune dataset doesn't drift them. "
                         "Default ON — disable with --no-stage2-freeze-backbone.")
    ap.add_argument("--no-stage2-freeze-backbone", action="store_false",
                    dest="stage2_freeze_backbone",
                    help="Train the full model in stage 2 (legacy behavior).")
    ap.add_argument("--stage2-only", action="store_true",
                    help="Skip stage 1 (assumes <stage1-from>/fold{f}/best.pt "
                         "already exists). Useful for re-running stage 2 with "
                         "different LR / epoch count / freeze settings without "
                         "re-pretraining. Stage 1 must have finished first.")
    ap.add_argument("--stage1-from", type=str, default=None,
                    help="Override the config name to look for stage-1 "
                         "checkpoints under. Default: <config>_stage1. Use "
                         "this to reuse fold checkpoints from a prior "
                         "training run, e.g. `--stage1-from sed_b0_dual` "
                         "loads from `MODEL_ROOT/sed/sed_b0_dual/fold{k}/best.pt`. "
                         "Combine with --stage2-only to skip retraining stage 1.")
    ap.add_argument("--stage1-ckpt", type=str, default=None,
                    help="Direct override of the stage-1 checkpoint path. "
                         "Use `{fold}` as a placeholder for the fold index, "
                         "e.g. `--stage1-ckpt path/to/fold{fold}/best.pt`. "
                         "Takes precedence over --stage1-from if both given.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="k=v pairs (JSON-parsed) applied to BOTH stages' configs.")
    args = ap.parse_args()

    # Parse common overrides into a dict
    from birdclef.train.train_sed_ddp import parse_overrides
    common_overrides = parse_overrides(args.override)

    # Stage 1 dir name. If --stage1-from is given (e.g. to reuse the
    # existing `sed_b0_dual` fold checkpoints), use it; otherwise default
    # to `<config>_stage1`.
    s1_name = args.stage1_from if args.stage1_from else f"{args.config}_stage1"
    s2_name = f"{args.config}_stage2"

    # ── Stage 1: broad pretrain ─────────────────────────────────────────
    if not args.stage2_only:
        s1_overrides = {
            **common_overrides,
            "epochs": int(args.stage1_epochs),
            "use_train_audio": True,
        }
        if args.stage1_pseudo_round and args.stage1_pseudo_round > 0:
            s1_overrides["pseudo_round"] = int(args.stage1_pseudo_round)
        s1_cfg = _build_cfg(s1_name, s1_overrides)

        if _is_main_rank():
            print()
            print("=" * 78)
            print(f"STAGE 1 — broad pretrain  config={s1_name}  fold={args.fold}  "
                  f"epochs={s1_overrides['epochs']}  pseudo_round={s1_overrides.get('pseudo_round')}")
            print("=" * 78)
        train_one_fold(s1_cfg, fold=int(args.fold))

    # ── Stage 2: labeled fine-tune, warm-started from stage 1 ───────────
    if args.stage1_ckpt:
        s1_ckpt = Path(args.stage1_ckpt.format(fold=int(args.fold)))
    else:
        s1_ckpt = (Path(MODEL_ROOT) / "sed" / s1_name
                   / f"fold{int(args.fold)}" / "best.pt")
    if not s1_ckpt.exists():
        raise SystemExit(
            f"Stage 1 ckpt not found at {s1_ckpt}. "
            "Run without --stage2-only first, or fix the path."
        )

    s2_overrides = {
        **common_overrides,
        "epochs": int(args.stage2_epochs),
        "use_train_audio": False,            # no focal in stage 2
        # pseudo_round: default 2 → expand soundscape pool with pseudo
        # rows. Set --stage2-pseudo-round 0 to revert to labeled-only.
        "pseudo_round": (int(args.stage2_pseudo_round)
                         if args.stage2_pseudo_round and args.stage2_pseudo_round > 0
                         else None),
        "lr": float(args.stage2_lr),
        "warmup_frac": float(args.stage2_warmup_frac),
        "eval_every_n_steps": int(args.stage2_eval_every_n_steps),
        "freeze_backbone": bool(args.stage2_freeze_backbone),
        "init_from": str(s1_ckpt),
    }
    if args.stage2_batch_size is not None:
        s2_overrides["batch_size"] = int(args.stage2_batch_size)
    s2_cfg = _build_cfg(s2_name, s2_overrides)

    if _is_main_rank():
        print()
        print("=" * 78)
        pool_desc = ("labeled+pseudo soundscapes" if s2_overrides["pseudo_round"]
                     else "labeled soundscapes only")
        print(f"STAGE 2 — soundscape finetune ({pool_desc})  "
              f"config={s2_name}  fold={args.fold}  "
              f"epochs={s2_overrides['epochs']}  lr={s2_overrides['lr']:.2e}")
        print(f"  freeze_backbone={s2_overrides['freeze_backbone']}  "
              f"pseudo_round={s2_overrides['pseudo_round']}  "
              f"eval_every_n_steps={s2_overrides['eval_every_n_steps']}")
        print(f"  init_from={s1_ckpt}")
        print("=" * 78)
    train_one_fold(s2_cfg, fold=int(args.fold))

    if _is_main_rank():
        s2_ckpt = (Path(MODEL_ROOT) / "sed" / s2_name
                   / f"fold{int(args.fold)}" / "best.pt")
        print()
        print(f"DONE — final ckpt: {s2_ckpt}")
        print(f"  stage 1: {s1_ckpt}")
        print(f"  stage 2: {s2_ckpt}  ← USE THIS for ONNX export and stitched-OOF eval")


if __name__ == "__main__":
    main()
