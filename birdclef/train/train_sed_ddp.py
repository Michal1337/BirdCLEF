"""DDP SED trainer.

Launch with:
    torchrun --standalone --nproc_per_node=<N> \
        -m birdclef.train.train_sed_ddp --config sed_v2s --fold 0

Single-process fallback (no torchrun): runs with world_size=1.

Reads train_audio from the shared memmap produced by scripts/01_build_caches.py.
Writes per-fold checkpoint under MODEL_ROOT/sed/<name>/fold{k}/best.pt and EMA
weights alongside. A periodic fold-val evaluation runs every
`eval_every_n_steps` on rank 0; checkpoint selection uses fold_val.macro_auc
minus a site_auc_std penalty. After all folds train, run
`scripts/_03b_stitched_oof_sed.py` for the cross-fold metric.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Dict, List

# Make `import birdclef.*` work when torchrun launches this file as a script
# (`torchrun path/to/train_sed_ddp.py`) in addition to the module form
# (`torchrun -m birdclef.train.train_sed_ddp`).
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from birdclef.config.paths import MODEL_ROOT, N_WINDOWS, SR, SOUNDSCAPES, WINDOW_SAMPLES, ensure_dirs
from birdclef.utils.seed import seed_everything
from birdclef.config.sed_configs import BASELINE as SED_BASELINE
from birdclef.data.augment import WaveformAug, background_mix, mixup
from birdclef.data.datasets import SEDTrainDataset
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta, primary_labels
from birdclef.data.splits import load_folds
from birdclef.models.losses import build_loss
from birdclef.models.sed import SED, SEDConfig
from birdclef.eval.metrics import compute_stage_metrics


def _dist_ctx():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    return ddp, world_size, rank, local_rank


def _is_main(rank: int) -> bool:
    return rank == 0


def _setup_ddp(ddp: bool):
    if not ddp or dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def _teardown_ddp(ddp: bool):
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


def _build_cfg(name: str, overrides: dict) -> dict:
    cfg = dict(SED_BASELINE)
    cfg["name"] = name
    cfg.update(overrides)
    return cfg


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        saved = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        return saved

    def restore(self, model: torch.nn.Module, saved: Dict[str, torch.Tensor]) -> None:
        for n, p in model.named_parameters():
            if n in saved:
                p.data.copy_(saved[n])


def _windowize(waveform: np.ndarray, n_windows: int = N_WINDOWS, win: int = WINDOW_SAMPLES) -> np.ndarray:
    """60 s -> [12, win] float32 batch."""
    if waveform.shape[0] < n_windows * win:
        waveform = np.pad(waveform, (0, n_windows * win - waveform.shape[0]))
    else:
        waveform = waveform[: n_windows * win]
    return waveform.reshape(n_windows, win)


@torch.no_grad()
def _eval_on_soundscape_files(
    model: torch.nn.Module,
    device: torch.device,
    filenames,
    max_files: int = 0,
) -> Dict:
    """Evaluate model on a set of labeled soundscape filenames.

    Returns metrics dict from eval.metrics.compute_stage_metrics, plus
    "n_files" and "n_windows" for logging.
    """
    import soundfile as sf

    target_names = set(str(f) for f in filenames)
    if not target_names:
        return {"macro_auc": float("nan"), "n_files": 0, "n_windows": 0}
    sc_meta = load_soundscape_meta()
    sc_meta = sc_meta[sc_meta["fully_labeled"]]
    rows = sc_meta[sc_meta["filename"].isin(target_names)]
    files = sorted(set(rows["filename"]))
    if max_files:
        files = files[:max_files]
    idx = label_to_idx()
    n_cls = len(idx)

    y_true = []
    y_score = []
    meta_rows = []
    model.eval()
    for fn in files:
        p = SOUNDSCAPES / fn
        try:
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        except Exception:
            continue
        if y.ndim == 2:
            y = y.mean(axis=1)
        wins = _windowize(y)
        x = torch.from_numpy(wins).to(device)
        from birdclef.models.sed import dual_head_predict
        out = model(x)
        probs = dual_head_predict(out).cpu().numpy()
        sub = rows[rows["filename"] == fn].sort_values("end_sec")
        if len(sub) < N_WINDOWS:
            continue
        for _, r in sub.iterrows():
            y_row = np.zeros(n_cls, dtype=np.uint8)
            for lb in r["label_list"]:
                j = idx.get(lb)
                if j is not None:
                    y_row[j] = 1
            y_true.append(y_row)
            meta_rows.append({"site": r["site"], "hour_utc": int(r["hour_utc"])})
        y_score.append(probs)
    if not y_true:
        return {"macro_auc": float("nan"), "n_files": 0, "n_windows": 0}
    Y = np.stack(y_true)
    P = np.concatenate(y_score, axis=0)
    if P.shape[0] != Y.shape[0]:
        n = min(P.shape[0], Y.shape[0])
        Y, P = Y[:n], P[:n]
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True).iloc[: len(Y)]
    m = compute_stage_metrics(Y, P, meta_df)
    m["n_files"] = int(len(files))
    m["n_windows"] = int(Y.shape[0])
    return m


def _fold_val_filenames(fold: int, n_splits: int = 5) -> list:
    """Return filenames that are held out in `fold` of the file-level split."""
    from birdclef.data.splits import load_folds

    folds_df = load_folds(n_splits=int(n_splits))
    return folds_df.loc[folds_df["fold"] == int(fold), "filename"].astype(str).tolist()


@torch.no_grad()
def _fold_val_eval(
    model: torch.nn.Module, device: torch.device, fold: int,
    n_splits: int = 5, max_files: int = 0,
) -> Dict:
    """Evaluate on the val files. Rank-0 only.

    - Fold-aware mode (fold=0..n_splits-1): evaluate on that fold's
      held-out val files (honest OOF).
    - Full-data mode (fold=None): evaluate on ALL labeled soundscapes
      (in-sample fit, LEAKED — the model was trained on these rows).
      Useful for collapse detection and confirming the model reached
      the memorization ceiling. NOT a generalization metric.
    """
    if fold is None:
        sc_meta = load_soundscape_meta()
        all_labeled = sc_meta[sc_meta["fully_labeled"]]["filename"].astype(str).unique().tolist()
        return _eval_on_soundscape_files(
            model, device, all_labeled, max_files=max_files,
        )
    return _eval_on_soundscape_files(
        model, device, _fold_val_filenames(fold, n_splits=n_splits),
        max_files=max_files,
    )


def _combine_eval_summary(fold_m: Dict) -> Dict:
    """Flatten to what we want in printed logs and saved JSON."""
    def _g(d, k, default=float("nan")):
        v = d.get(k, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    return {
        "fold_val": {
            "macro_auc": _g(fold_m, "macro_auc"),
            "site_auc_std": _g(fold_m, "site_auc_std"),
            "rare_auc": _g(fold_m, "rare_auc"),
            "frequent_auc": _g(fold_m, "frequent_auc"),
            "n_files": int(fold_m.get("n_files", 0) or 0),
            "n_windows": int(fold_m.get("n_windows", 0) or 0),
        },
    }


def build_sed_model(cfg: dict, device: torch.device) -> SED:
    sed_cfg = SEDConfig(
        backbone=cfg["backbone"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], f_min=cfg["f_min"], f_max=cfg["f_max"],
        freq_mask_param=cfg["freq_mask_param"], time_mask_param=cfg["time_mask_param"],
        specaugment_masks=cfg["specaugment_masks"], spec_noise_std=cfg["spec_noise_std"],
    )
    return SED(sed_cfg).to(device)


def train_one_fold(cfg: dict, fold: int | None, dry_run_steps: int = 0) -> dict:
    ddp, world_size, rank, local_rank = _dist_ctx()
    _setup_ddp(ddp)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Per-rank seed derived from the config — so cfg["seed"]=N gives
    # reproducible runs, and each DDP rank still gets a distinct stream.
    base_seed = int(cfg.get("seed", 42))
    if fold is not None:
        base_seed += int(fold) * 37
    seed_everything(base_seed + rank)
    if _is_main(rank):
        ensure_dirs()

    model = build_sed_model(cfg, device)
    model_raw = model
    # Optional warm-start from a previous fold's best.pt — useful for salvaging
    # folds that fell into the mode-collapse attractor (val AUC stuck at 0.5).
    # Pre-loaded weights give the optimizer a non-trivial starting point that's
    # already past the all-zero region.
    init_from = cfg.get("init_from")
    if init_from:
        ckpt_path = Path(init_from)
        if not ckpt_path.exists():
            raise SystemExit(f"init_from checkpoint not found: {ckpt_path}")
        if _is_main(rank):
            print(f"[sed:{cfg['name']} f{fold}] WARM-START from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("state_dict", state)
        # Load with strict=False so a slight architecture variation (e.g. the
        # framewise head added later) doesn't break the warm-start.
        missing, unexpected = model_raw.load_state_dict(sd, strict=False)
        if _is_main(rank):
            if missing:
                print(f"  missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"  unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        # If the ckpt has EMA shadow, prefer those weights as init too — they're
        # smoother than the raw state_dict at the moment of save.
        ema_shadow = state.get("ema") or {}
        if ema_shadow and _is_main(rank):
            print(f"  applying EMA shadow ({len(ema_shadow)} params)")
        if ema_shadow:
            with torch.no_grad():
                for n, p in model_raw.named_parameters():
                    if n in ema_shadow:
                        p.data.copy_(ema_shadow[n].to(p.device))

    # Optional: freeze the backbone (timm CNN). Used for stage-2 finetune
    # in the two-stage recipe — only the heads + attention pool train, so
    # 200 finetune steps on 564 rows can specialize the heads without
    # disturbing the broad-pretrained features. BN running stats are also
    # frozen (backbone.eval() in the train loop) so the small finetune
    # dataset can't drift them.
    if bool(cfg.get("freeze_backbone", False)):
        for p in model_raw.backbone.parameters():
            p.requires_grad = False
        if _is_main(rank):
            n_total = sum(p.numel() for p in model_raw.parameters())
            n_trainable = sum(p.numel() for p in model_raw.parameters()
                              if p.requires_grad)
            print(f"[sed:{cfg['name']} f{fold}] freeze_backbone=True — "
                  f"trainable: {n_trainable:,} / {n_total:,} "
                  f"({100 * n_trainable / max(1, n_total):.1f}%)")

    if ddp:
        # `find_unused_parameters=True` when freezing — DDP needs to know
        # the backbone has no gradients flowing back.
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=bool(cfg.get("freeze_backbone", False)),
        )
    ema = EMA(model_raw, decay=float(cfg["ema_decay"]))

    # Loss
    loss_kind = cfg["loss"]
    if loss_kind == "focal_bce":
        loss_fn = build_loss("focal_bce",
                             alpha=float(cfg["focal_alpha"]), gamma=float(cfg["focal_gamma"]),
                             label_smoothing=float(cfg["label_smoothing"]))
    elif loss_kind == "bce_focal_mean":
        loss_fn = build_loss("bce_focal_mean",
                             focal_alpha=float(cfg["focal_alpha"]), focal_gamma=float(cfg["focal_gamma"]),
                             label_smoothing=float(cfg["label_smoothing"]))
    else:
        loss_fn = build_loss("bce_posw", label_smoothing=float(cfg["label_smoothing"]))

    # Dataset / loader. With V-anchor abandoned, training pool is every
    # labeled non-val-fold soundscape file.
    n_splits = int(cfg.get("n_splits", 5))
    dataset = SEDTrainDataset(
        fold=fold, n_splits=n_splits,
        soundscape_fraction=float(cfg["soundscape_fraction"]),
        first_window_prob=float(cfg["first_window_prob"]),
        window_seconds=int(cfg["window_seconds"]),
        pseudo_round=cfg.get("pseudo_round"),
        seed=base_seed + rank,
        use_train_audio=bool(cfg.get("use_train_audio", True)),
    )
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=base_seed)
    else:
        sampler = None
    # num_workers tuning matters a lot once pseudo-label mode is on: each
    # soundscape sample blocks on sf.read of a full 60 s OGG, so with too
    # few workers the GPU starves. `persistent_workers=True` avoids the
    # respawn cost between epochs; `prefetch_factor` overlaps decode with
    # compute.
    num_workers = int(cfg.get("num_workers", 8))
    prefetch_factor = int(cfg.get("prefetch_factor", 4)) if num_workers > 0 else None
    loader_kwargs = dict(
        batch_size=int(cfg["batch_size"]),
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    wav_aug = WaveformAug().to(device)

    # Only optimize parameters with requires_grad=True (omits frozen backbone
    # when `freeze_backbone` is set).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, int(cfg["epochs"]) * steps_per_epoch)
    warmup_steps = int(total_steps * float(cfg["warmup_frac"]))
    if _is_main(rank):
        eff_batch = (
            int(cfg["batch_size"]) * max(1, world_size)
            * int(cfg.get("grad_accum", 1))
        )
        print(
            f"[sed:{cfg['name']} f{fold}] schedule  "
            f"epochs={int(cfg['epochs'])}  "
            f"steps/epoch={steps_per_epoch}  "
            f"total_steps={total_steps}  "
            f"warmup_steps={warmup_steps}  "
            f"({float(cfg['warmup_frac']) * 100:.1f}%)  "
            f"eff_batch={eff_batch} "
            f"(per_rank={int(cfg['batch_size'])} × ranks={max(1, world_size)} "
            f"× grad_accum={int(cfg.get('grad_accum', 1))})  "
            f"dataset_len={len(dataset)}"
        )
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    # AMP dtype — bf16 on H100/A100 has the same tensor-core throughput as
    # fp16 but better numerical range, so we skip the GradScaler entirely
    # (bf16 doesn't need loss scaling). Default remains fp16 for backward
    # compat with older cards.
    amp_dtype_name = str(cfg.get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    use_grad_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        enabled=bool(cfg["amp"]) and torch.cuda.is_available() and use_grad_scaler
    )

    step = 0
    best_primary = float("-inf")
    best_path = None
    # In full-data mode (fold=None) different seeds need different ckpt dirs
    # so multi-seed runs don't clobber each other. Fold-aware mode keeps the
    # original `fold{k}` layout.
    if fold is None:
        ckpt_dir = (Path(MODEL_ROOT) / "sed" / cfg["name"]
                    / f"fulldata_seed{int(cfg.get('seed', 42))}")
    else:
        ckpt_dir = Path(MODEL_ROOT) / "sed" / cfg["name"] / f"fold{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_eval = {}
    t0 = time.time()

    # Per-fold training history — JSONL, append on rank 0 only. One line per
    # log event ({"kind": "step" | "eval" | "final", ...}). Easy to tail
    # during training, easy to parse afterwards.
    history_path = ckpt_dir / "train_history.jsonl"
    history_fh = None
    if _is_main(rank):
        history_fh = open(history_path, "w", encoding="utf-8")
        # Header line: capture the resolved config + schedule for reproducibility
        header = {
            "kind": "header",
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()
                       if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
            "fold": fold,
            "n_splits": n_splits,
            "world_size": int(world_size),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
            "dataset_len": int(len(dataset)),
            "amp_dtype": amp_dtype_name,
            "started_at": time.time(),
        }
        history_fh.write(json.dumps(header, default=str) + "\n")
        history_fh.flush()
        print(f"[sed:{cfg['name']} f{fold}] training history → {history_path}")

    def _log_history(record: dict) -> None:
        """Append one JSON record to train_history.jsonl. Rank-0 only."""
        if history_fh is None:
            return
        history_fh.write(json.dumps(record, default=str) + "\n")
        history_fh.flush()

    # Initial eval at step 0 — captures the warm-start baseline whenever
    # `init_from` is set (stage 2 finetune, fold-1 salvage, etc.). Lets us
    # compare "stage 1's primary" to "stage 2's final primary" directly,
    # and saves a best.pt for the warm-start state so even a regression
    # leaves us with a working artifact.
    if _is_main(rank) and init_from:
        saved = ema.copy_to(model_raw)
        fold_m = _fold_val_eval(model_raw, device, fold, n_splits=n_splits)
        ema.restore(model_raw, saved)
        summary = _combine_eval_summary(fold_m)
        last_eval = summary
        f_mauc = summary["fold_val"]["macro_auc"]
        f_std = summary["fold_val"]["site_auc_std"] or 0.0
        primary = f_mauc if not math.isnan(f_mauc) else float("-inf")
        eval_label = "in_sample" if fold is None else "fold_val"
        fold_tag = (f"fulldata seed={cfg.get('seed', 42)}" if fold is None
                    else f"f{fold}")
        print(
            f"[sed:{cfg['name']} {fold_tag}] eval step=0 (warm-start init)  "
            f"{eval_label} auc={f_mauc:.4f} "
            f"(n={summary['fold_val']['n_files']} files)  "
            f"site_std={f_std:.4f}  primary={primary:.4f}"
        )
        _log_history({
            "kind": "eval", "step": 0, "epoch": -1,
            "eval_kind": eval_label, "warm_start": True,
            "fold_val_macro_auc": float(f_mauc) if not math.isnan(f_mauc) else None,
            "site_auc_std": float(f_std),
            "rare_auc": summary["fold_val"].get("rare_auc"),
            "frequent_auc": summary["fold_val"].get("frequent_auc"),
            "primary": float(primary) if not math.isinf(primary) else None,
            "n_val_files": int(summary["fold_val"].get("n_files", 0) or 0),
            "n_val_windows": int(summary["fold_val"].get("n_windows", 0) or 0),
            "elapsed_min": 0.0,
        })
        if primary > best_primary:
            best_primary = primary
            best_path = ckpt_dir / "best.pt"
            torch.save({"state_dict": model_raw.state_dict(),
                        "ema": ema.shadow,
                        "cfg": cfg,
                        "step": 0,
                        "metrics": summary,
                        "primary": primary,
                        "warm_start": True},
                       best_path)
            print(f"[sed:{cfg['name']} {fold_tag}] saved warm-start ckpt as best.pt "
                  f"(primary={primary:.4f}) — training must beat this to overwrite")
            _log_history({
                "kind": "best_ckpt", "step": 0,
                "eval_kind": eval_label, "warm_start": True,
                "primary": float(primary), "fold_val_macro_auc": float(f_mauc),
                "site_auc_std": float(f_std),
                "path": str(best_path),
            })

    for epoch in range(int(cfg["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            # Dataset always returns 3-tuple (wav, y, loss_mask). loss_mask is
            # all ones for samples without pseudo-labels; only pseudo rows use
            # it to gate confidence-filtered positions.
            if len(batch) == 3:
                wav, y, loss_mask = batch
            else:
                wav, y = batch
                loss_mask = torch.ones_like(y)
            wav = wav.to(device, non_blocking=True).unsqueeze(1)
            y = y.to(device, non_blocking=True).float()
            loss_mask = loss_mask.to(device, non_blocking=True).float()
            if model.training is False:
                model.train()
                # If backbone is frozen, keep it in eval mode so BN running
                # stats don't drift on the small finetune dataset.
                if bool(cfg.get("freeze_backbone", False)):
                    model_raw.backbone.eval()
            if float(cfg["mixup_alpha"]) > 0:
                wav, y = mixup(wav, y, alpha=float(cfg["mixup_alpha"]),
                               mode=str(cfg["mixup_mode"]))
                # Mixup blends two samples but the loss mask for position j
                # should fire if either source wanted supervision there.
                perm = torch.arange(loss_mask.size(0), device=loss_mask.device)
                loss_mask = torch.maximum(loss_mask, loss_mask[perm.flip(0)])
            wav = wav_aug(wav)
            for g in opt.param_groups:
                g["lr"] = float(cfg["lr"]) * lr_at(step)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=amp_dtype,
                enabled=bool(cfg["amp"]) and torch.cuda.is_available(),
            ):
                out = model(wav.squeeze(1))
                from birdclef.models.sed import dual_head_loss
                loss = dual_head_loss(out, y, loss_fn, loss_mask=loss_mask,
                                      frame_weight=float(cfg.get("frame_weight", 0.5)))
            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                scaler.step(opt)
                scaler.update()
            else:
                # bf16 path: no loss scaling needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                opt.step()
            ema.update(model_raw)

            if _is_main(rank) and step % 50 == 0:
                cur_lr = float(opt.param_groups[0]["lr"])
                cur_loss = float(loss.item())
                print(f"[sed:{cfg['name']} f{fold}] step={step} loss={cur_loss:.4f} "
                      f"lr={cur_lr:.2e}")
                _log_history({
                    "kind": "step", "step": int(step), "epoch": int(epoch),
                    "loss": cur_loss, "lr": cur_lr,
                    "elapsed_min": (time.time() - t0) / 60.0,
                })

            # Periodic eval (rank 0 only): fold-val only. V-anchor was
            # abandoned — see plan file. Best checkpoint = highest
            # fold_val.macro_auc minus site_auc_std penalty.
            if _is_main(rank) and step > 0 and step % int(cfg["eval_every_n_steps"]) == 0:
                saved = ema.copy_to(model_raw)
                fold_m = _fold_val_eval(model_raw, device, fold, n_splits=n_splits)
                ema.restore(model_raw, saved)
                summary = _combine_eval_summary(fold_m)
                last_eval = summary
                f_mauc = summary["fold_val"]["macro_auc"]
                f_std = summary["fold_val"]["site_auc_std"] or 0.0
                # Primary metric for best-checkpoint selection = val macro AUC.
                # In fold-aware mode this is honest OOF; in full-data mode this
                # is in-sample fit (leaked but useful for collapse detection
                # and memorization-ceiling check).
                primary = f_mauc if not math.isnan(f_mauc) else float("-inf")
                eval_label = "in_sample" if fold is None else "fold_val"
                fold_tag = (f"fulldata seed={cfg.get('seed', 42)}" if fold is None
                            else f"f{fold}")
                print(
                    f"[sed:{cfg['name']} {fold_tag}] eval step={step}  "
                    f"{eval_label} auc={f_mauc:.4f} "
                    f"(n={summary['fold_val']['n_files']} files)  "
                    f"site_std={f_std:.4f}  primary={primary:.4f}"
                )
                _log_history({
                    "kind": "eval", "step": int(step), "epoch": int(epoch),
                    "eval_kind": eval_label,    # "in_sample" or "fold_val"
                    "fold_val_macro_auc": float(f_mauc) if not math.isnan(f_mauc) else None,
                    "site_auc_std": float(f_std),
                    "rare_auc": summary["fold_val"].get("rare_auc"),
                    "frequent_auc": summary["fold_val"].get("frequent_auc"),
                    "primary": float(primary) if not math.isinf(primary) else None,
                    "n_val_files": int(summary["fold_val"].get("n_files", 0) or 0),
                    "n_val_windows": int(summary["fold_val"].get("n_windows", 0) or 0),
                    "best_primary_so_far": float(best_primary) if not math.isinf(best_primary) else None,
                    "elapsed_min": (time.time() - t0) / 60.0,
                })
                if primary > best_primary:
                    best_primary = primary
                    best_path = ckpt_dir / "best.pt"
                    torch.save({"state_dict": model_raw.state_dict(),
                                "ema": ema.shadow,
                                "cfg": cfg,
                                "step": step,
                                "metrics": summary,
                                "primary": primary},
                               best_path)
                    print(f"[sed:{cfg['name']} {fold_tag}] ↑ new best primary={primary:.4f}  "
                          f"({eval_label}={f_mauc:.4f})")
                    _log_history({
                        "kind": "best_ckpt", "step": int(step),
                        "eval_kind": eval_label,
                        "primary": float(primary), "fold_val_macro_auc": float(f_mauc),
                        "site_auc_std": float(f_std),
                        "path": str(best_path),
                    })
            step += 1
            if dry_run_steps and step >= dry_run_steps:
                break
        if dry_run_steps and step >= dry_run_steps:
            break

    # Final eval + save (full file count for fold val)
    if _is_main(rank):
        saved = ema.copy_to(model_raw)
        fold_m = _fold_val_eval(model_raw, device, fold, n_splits=n_splits, max_files=0)
        ema.restore(model_raw, saved)
        summary = _combine_eval_summary(fold_m)
        f_mauc = summary["fold_val"]["macro_auc"]
        f_std = summary["fold_val"]["site_auc_std"] or 0.0
        # Final primary = fold-val macro AUC (matches checkpoint-selection metric).
        final_primary = f_mauc if not math.isnan(f_mauc) else float("-inf")
        print(
            f"[sed:{cfg['name']} f{fold}] FINAL  "
            f"fold_val auc={f_mauc:.4f} "
            f"(n={summary['fold_val']['n_files']})  "
            f"site_std={f_std:.4f}  primary={final_primary:.4f}"
        )
        # In full-data mode (fold=None) there's no held-out fold-val, so the
        # primary is NaN/-inf and best-ckpt selection during training never
        # fired. Save the FINAL (EMA-applied) state as best.pt — that's the
        # deployable artifact. In fold-aware mode, this branch only fires if
        # the final eval beats the best seen during training, which is rare
        # but valid.
        if fold is None or final_primary > best_primary:
            best_primary = final_primary if final_primary > best_primary else best_primary
            best_path = ckpt_dir / "best.pt"
            # Write EMA shadow as the "deployed" weights (the model is in
            # `saved`-restored state right now, so re-snapshot EMA before save).
            ema_saved = ema.copy_to(model_raw)
            torch.save({"state_dict": model_raw.state_dict(),
                        "ema": ema.shadow, "cfg": cfg, "step": step,
                        "metrics": summary, "primary": final_primary},
                       best_path)
            ema.restore(model_raw, ema_saved)
            if fold is None:
                print(f"[sed:{cfg['name']} fulldata seed={cfg.get('seed', 42)}] "
                      f"saved final ckpt → {best_path}")
        (ckpt_dir / "final_metrics.json").write_text(
            json.dumps({
                "final": summary,
                "best_primary": best_primary,
                "runtime_min": (time.time() - t0) / 60.0,
            }, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[sed:{cfg['name']} f{fold}] DONE best_primary={best_primary:.4f} "
              f"runtime={(time.time()-t0)/60:.1f}m")
        _log_history({
            "kind": "final", "step": int(step),
            "fold_val_macro_auc": float(f_mauc) if not math.isnan(f_mauc) else None,
            "site_auc_std": float(f_std),
            "rare_auc": summary["fold_val"].get("rare_auc"),
            "frequent_auc": summary["fold_val"].get("frequent_auc"),
            "primary": float(final_primary) if not math.isinf(final_primary) else None,
            "best_primary": float(best_primary) if not math.isinf(best_primary) else None,
            "n_val_files": int(summary["fold_val"].get("n_files", 0) or 0),
            "n_val_windows": int(summary["fold_val"].get("n_windows", 0) or 0),
            "runtime_min": (time.time() - t0) / 60.0,
        })
        if history_fh is not None:
            history_fh.close()
    _teardown_ddp(ddp)
    # Caller side reads `metrics.global` for the sweep summary; we surface
    # the fold_val summary there so a single-fold run produces a usable row
    # in the sweep CSV (stitched OOF requires running all folds — see
    # _03b_stitched_oof_sed.py).
    return {"metrics": {
        "fold_val": (last_eval or {}).get("fold_val", {}),
        "global": (last_eval or {}).get("fold_val", {}),
        "per_fold": {},
    }}


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Name key in sed_configs; honoured verbatim")
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold index 0..n_splits-1 for fold-aware training, "
                         "or -1 for full-data (no holdout — trains on every "
                         "labeled file, skips fold-val eval). Use full-data "
                         "for actual LB deployment after fold-aware OOF "
                         "validates the recipe.")
    ap.add_argument("--dry-run-steps", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[],
                    help="k=v pairs (JSON-parsed) to override config")
    args = ap.parse_args()
    overrides = parse_overrides(args.override)

    cfg = _build_cfg(args.config, overrides)
    # --fold -1 → full-data mode (no holdout). Pass fold=None into the trainer
    # so SEDTrainDataset skips the fold filter and the fold-val eval returns
    # NaN AUC (no eval performed).
    fold_arg = None if int(args.fold) < 0 else int(args.fold)
    train_one_fold(cfg, fold=fold_arg, dry_run_steps=int(args.dry_run_steps))


if __name__ == "__main__":
    main()
