"""DDP SED trainer.

Launch with:
    torchrun --standalone --nproc_per_node=<N> \
        -m birdclef.train.train_sed_ddp --config sed_v2s --fold 0

Single-process fallback (no torchrun): runs with world_size=1.

Reads train_audio from the shared memmap produced by scripts/01_build_caches.py.
Writes per-fold checkpoint under MODEL_ROOT/sed/<name>/fold{k}/best.pt and EMA
weights alongside. A minimal V-anchor evaluation runs every
`eval_every_n_steps` on rank 0.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from birdclef.config.paths import MODEL_ROOT, N_WINDOWS, SR, SOUNDSCAPES, WINDOW_SAMPLES, ensure_dirs
from birdclef.config.sed_configs import BASELINE as SED_BASELINE
from birdclef.data.augment import WaveformAug, background_mix, mixup
from birdclef.data.datasets import SEDTrainDataset
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta, primary_labels
from birdclef.data.splits import load_v_anchor
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


def _v_anchor_file_paths() -> List[Path]:
    files = load_v_anchor()
    return [SOUNDSCAPES / f for f in files]


def _windowize(waveform: np.ndarray, n_windows: int = N_WINDOWS, win: int = WINDOW_SAMPLES) -> np.ndarray:
    """60 s -> [12, win] float32 batch."""
    if waveform.shape[0] < n_windows * win:
        waveform = np.pad(waveform, (0, n_windows * win - waveform.shape[0]))
    else:
        waveform = waveform[: n_windows * win]
    return waveform.reshape(n_windows, win)


@torch.no_grad()
def _v_anchor_eval(model: torch.nn.Module, device: torch.device, max_files: int = 0) -> Dict:
    """Evaluate model on V-anchor files, returning metrics dict. Rank-0 only."""
    import soundfile as sf

    sc_meta = load_soundscape_meta()
    sc_meta = sc_meta[sc_meta["fully_labeled"]]
    anchor_names = set(load_v_anchor())
    rows = sc_meta[sc_meta["filename"].isin(anchor_names)]
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
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
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
        return {"macro_auc": float("nan")}
    Y = np.stack(y_true)
    P = np.concatenate(y_score, axis=0)
    if P.shape[0] != Y.shape[0]:
        n = min(P.shape[0], Y.shape[0])
        Y, P = Y[:n], P[:n]
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True).iloc[: len(Y)]
    return compute_stage_metrics(Y, P, meta_df)


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
    torch.manual_seed(42 + rank)
    if _is_main(rank):
        ensure_dirs()

    model = build_sed_model(cfg, device)
    model_raw = model
    if ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False)
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

    # Dataset / loader
    dataset = SEDTrainDataset(
        fold=fold, exclude_v_anchor=True,
        soundscape_fraction=float(cfg["soundscape_fraction"]),
        first_window_prob=float(cfg["first_window_prob"]),
        window_seconds=int(cfg["window_seconds"]),
        pseudo_round=cfg.get("pseudo_round"),
    )
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset, batch_size=int(cfg["batch_size"]),
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=True,
    )

    wav_aug = WaveformAug().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    total_steps = max(1, int(cfg["epochs"]) * max(1, len(loader)))
    warmup_steps = int(total_steps * float(cfg["warmup_frac"]))
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    scaler = torch.amp.GradScaler(enabled=bool(cfg["amp"]) and torch.cuda.is_available())

    step = 0
    best_primary = float("-inf")
    best_path = None
    ckpt_dir = Path(MODEL_ROOT) / "sed" / cfg["name"] / f"fold{fold if fold is not None else -1}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_eval = {}
    t0 = time.time()

    for epoch in range(int(cfg["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for wav, y in loader:
            wav = wav.to(device, non_blocking=True).unsqueeze(1)
            y = y.to(device, non_blocking=True).float()
            if model.training is False:
                model.train()
            if float(cfg["mixup_alpha"]) > 0:
                wav, y = mixup(wav, y, alpha=float(cfg["mixup_alpha"]),
                               mode=str(cfg["mixup_mode"]))
            wav = wav_aug(wav)
            # Set LR
            for g in opt.param_groups:
                g["lr"] = float(cfg["lr"]) * lr_at(step)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                     enabled=bool(cfg["amp"]) and torch.cuda.is_available()):
                logits = model(wav.squeeze(1))
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            ema.update(model_raw)

            if _is_main(rank) and step % 50 == 0:
                print(f"[sed:{cfg['name']} f{fold}] step={step} loss={loss.item():.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

            # Periodic V-anchor eval (rank 0 only)
            if _is_main(rank) and step > 0 and step % int(cfg["eval_every_n_steps"]) == 0:
                saved = ema.copy_to(model_raw)
                m = _v_anchor_eval(model_raw, device, max_files=16)
                ema.restore(model_raw, saved)
                mauc = m.get("macro_auc", float("nan"))
                site_std = m.get("site_auc_std", 0.0) or 0.0
                primary = (mauc - 1.0 * site_std) if not math.isnan(mauc) else float("-inf")
                last_eval = m
                if primary > best_primary:
                    best_primary = primary
                    best_path = ckpt_dir / "best.pt"
                    torch.save({"state_dict": model_raw.state_dict(),
                                "ema": ema.shadow,
                                "cfg": cfg,
                                "step": step,
                                "metrics": m,
                                "primary": primary},
                               best_path)
                    print(f"[sed:{cfg['name']} f{fold}] new best primary={primary:.4f}")
            step += 1
            if dry_run_steps and step >= dry_run_steps:
                break
        if dry_run_steps and step >= dry_run_steps:
            break

    # Final eval + save
    if _is_main(rank):
        saved = ema.copy_to(model_raw)
        final_m = _v_anchor_eval(model_raw, device, max_files=0)
        ema.restore(model_raw, saved)
        final_primary = (final_m.get("macro_auc", float("nan")) -
                         (final_m.get("site_auc_std", 0.0) or 0.0))
        if final_primary > best_primary:
            best_primary = final_primary
            best_path = ckpt_dir / "best.pt"
            torch.save({"state_dict": model_raw.state_dict(),
                        "ema": ema.shadow, "cfg": cfg, "step": step,
                        "metrics": final_m, "primary": final_primary},
                       best_path)
        (ckpt_dir / "final_metrics.json").write_text(
            json.dumps({"final": final_m, "best_primary": best_primary,
                        "runtime_min": (time.time() - t0) / 60.0}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[sed:{cfg['name']} f{fold}] DONE best_primary={best_primary:.4f} "
              f"runtime={(time.time()-t0)/60:.1f}m")
    _teardown_ddp(ddp)
    return {"metrics": {"v_anchor": last_eval, "global": last_eval, "per_fold": {}}}


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
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dry-run-steps", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[],
                    help="k=v pairs (JSON-parsed) to override config")
    args = ap.parse_args()
    overrides = parse_overrides(args.override)
    cfg = _build_cfg(args.config, overrides)
    train_one_fold(cfg, fold=args.fold, dry_run_steps=int(args.dry_run_steps))


if __name__ == "__main__":
    main()
