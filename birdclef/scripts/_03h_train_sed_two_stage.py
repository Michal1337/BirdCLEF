"""Two-stage fold-aware SED training: broad pretrain + labeled finetune.

Stage 1 — pretrain on the full pool (focal `train_audio` + labeled
fold-train soundscapes + pseudo round 2) for N1 epochs. Same recipe as
the standard `train_sed_ddp` run; uses the BASELINE config + pseudo_round.

Stage 2 — finetune on labeled fold-train soundscapes ONLY for N2 epochs,
warm-started from stage 1's best.pt with a smaller LR. Hypothesis:
specializing the broad-pretrained backbone on in-distribution data shifts
features toward what test_soundscapes actually looks like.

Why two stages: the soundscape-only training-from-scratch fails (mode
collapse, fold OOF=0.51) because there's not enough species diversity in
labeled soundscapes alone. But starting from a broad-pretrained backbone
that already discriminates 234 species, the labeled fold-train pool can
nudge it toward Pantanal-specific calling patterns without collapsing.

Stage 2 is small (47 fold-train files × 12 windows = 564 rows). At
batch=64 × 2 ranks, that's ~4 steps/epoch. Run for many "epochs" (default
50) to get meaningful step count (~200 finetune steps total). Smaller
batch and longer schedule are valid if you want more gradient updates;
override via `--stage2-epochs` and `--stage2-batch-size`.

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
    ap.add_argument("--stage2-epochs", type=int, default=50,
                    help="Epochs for labeled-only finetune. With ~564 fold-train "
                         "rows and batch=64×2 ranks, ~4 steps/epoch — so 50 epochs "
                         "≈ 200 finetune steps. Increase if you want more.")
    ap.add_argument("--stage2-lr", type=float, default=1e-4,
                    help="Stage-2 learning rate. Lower than stage 1 (1e-3) since "
                         "we're finetuning, not training from scratch.")
    ap.add_argument("--stage2-batch-size", type=int, default=None,
                    help="Override batch size for stage 2 (default: same as stage 1). "
                         "Smaller batch (e.g. 16) gives more gradient steps per epoch.")
    ap.add_argument("--stage2-warmup-frac", type=float, default=0.0,
                    help="Stage-2 warmup fraction. Default 0.0 — no warmup needed "
                         "when warm-starting from a converged backbone.")
    ap.add_argument("--stage2-only", action="store_true",
                    help="Skip stage 1 (assumes <config>_stage1/fold{f}/best.pt "
                         "already exists). Useful for re-running stage 2 with "
                         "different LR / epoch count without re-pretraining.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="k=v pairs (JSON-parsed) applied to BOTH stages' configs.")
    args = ap.parse_args()

    # Parse common overrides into a dict
    from birdclef.train.train_sed_ddp import parse_overrides
    common_overrides = parse_overrides(args.override)

    s1_name = f"{args.config}_stage1"
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
        "use_train_audio": False,
        "pseudo_round": None,                # labeled-only finetune
        "lr": float(args.stage2_lr),
        "warmup_frac": float(args.stage2_warmup_frac),
        "init_from": str(s1_ckpt),
    }
    if args.stage2_batch_size is not None:
        s2_overrides["batch_size"] = int(args.stage2_batch_size)
    s2_cfg = _build_cfg(s2_name, s2_overrides)

    if _is_main_rank():
        print()
        print("=" * 78)
        print(f"STAGE 2 — labeled finetune  config={s2_name}  fold={args.fold}  "
              f"epochs={s2_overrides['epochs']}  lr={s2_overrides['lr']:.2e}")
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
