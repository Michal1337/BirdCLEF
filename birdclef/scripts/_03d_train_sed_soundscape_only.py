"""Train and evaluate the dual-head SED on labeled train_soundscapes only.

Small self-contained probe — NO fold split, NO train_audio (focal), NO
pseudo-labels. Trains the new dual-head SED on every labeled soundscape
row (~708 rows ≈ 59 files × 12 windows) and scores in-sample on those
same files. Direct apples-to-apples comparison vs:
    - Tucker's distilled SED (leaked)        : 0.985 macro_auc
    - Our SSM (in-sample fit, _07c probe)    : 0.9856 macro_auc

If the dual-head SED reaches ≈ 0.985 on this probe, our architecture
port is structurally equivalent to Tucker's bundle. Lower would suggest
a missing piece (pretrained weights, augment recipe, etc.).

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._03d_train_sed_soundscape_only
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._03d_train_sed_soundscape_only \\
        --epochs 30 --batch-size 32 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef.config.paths import (
    FILE_SAMPLES,
    MODEL_ROOT,
    N_WINDOWS,
    OUTPUT_ROOT,
    SOUNDSCAPE_INDEX,
    SOUNDSCAPE_NPY,
    SOUNDSCAPES,
    SR,
    WINDOW_SAMPLES,
)
from birdclef.data.soundscapes import label_to_idx, load_soundscape_meta
from birdclef.eval.metrics import compute_stage_metrics
from birdclef.models.sed import SED, SEDConfig, dual_head_loss, dual_head_predict
from birdclef.utils.seed import seed_everything


def _open_soundscape_store():
    if not SOUNDSCAPE_NPY.exists() or not SOUNDSCAPE_INDEX.exists():
        return None, {}
    store = np.load(SOUNDSCAPE_NPY, mmap_mode="r")
    idx = pd.read_parquet(SOUNDSCAPE_INDEX)
    if "ok" in idx.columns:
        idx = idx[idx["ok"] == 1]
    row_by_file = dict(zip(idx["filename"].astype(str), idx["row_idx"].astype(int)))
    return store, row_by_file


class LabeledSoundscapeWindowDataset(Dataset):
    """Yields (waveform[5s], y[n_classes]) for every labeled (file, window).

    No fold split, no augmentation in __getitem__ (caller adds aug). The
    indexed rows are exactly the ones in `load_soundscape_meta(fully_labeled)`.
    """

    def __init__(self):
        self.store, self.row_by_file = _open_soundscape_store()
        sc = load_soundscape_meta()
        sc = sc[sc["fully_labeled"]].copy()
        sc = sc.sort_values(["filename", "end_sec"]).reset_index(drop=True)
        self.idx_map = label_to_idx()
        self.n_classes = len(self.idx_map)

        # Materialize one (file, window_idx, multihot) record per row
        items: list[tuple[str, int, np.ndarray]] = []
        for _, r in sc.iterrows():
            wi = int(r["end_sec"]) // 5 - 1
            if not (0 <= wi < N_WINDOWS):
                continue
            y = np.zeros(self.n_classes, dtype=np.float32)
            for lb in r["label_list"]:
                j = self.idx_map.get(lb)
                if j is not None:
                    y[j] = 1.0
            items.append((str(r["filename"]), wi, y))
        self.items = items
        # Cache per-file 60s waveform (small dataset → no need to re-decode)
        self._wav_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _read_60s(self, filename: str) -> np.ndarray:
        if filename in self._wav_cache:
            return self._wav_cache[filename]
        if self.store is not None and filename in self.row_by_file:
            row = int(self.row_by_file[filename])
            wav = np.asarray(self.store[row, :FILE_SAMPLES], dtype=np.float32)
        else:
            y, _sr = sf.read(str(SOUNDSCAPES / filename), dtype="float32",
                             always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            if y.shape[0] < FILE_SAMPLES:
                y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
            else:
                y = y[:FILE_SAMPLES]
            wav = y.astype(np.float32)
        self._wav_cache[filename] = wav
        return wav

    def __getitem__(self, idx: int):
        fn, wi, y = self.items[idx]
        wav60 = self._read_60s(fn)
        off = wi * WINDOW_SAMPLES
        x = wav60[off:off + WINDOW_SAMPLES]
        if x.shape[0] < WINDOW_SAMPLES:
            x = np.pad(x, (0, WINDOW_SAMPLES - x.shape[0]))
        return x.astype(np.float32), y.astype(np.float32)


def _focal_bce(logits, y, alpha: float = 0.25, gamma: float = 2.0,
               label_smoothing: float = 0.03) -> torch.Tensor:
    """Multilabel focal BCE loss with label smoothing."""
    if label_smoothing > 0:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * y + (1 - p) * (1 - y)
    a_t = alpha * y + (1 - alpha) * (1 - y)
    focal = a_t * (1 - p_t).clamp(min=1e-7).pow(gamma) * bce
    return focal.mean()


def _wrap_loss_for_dual_head(focal_alpha: float, focal_gamma: float,
                             label_smoothing: float):
    """Adapter so dual_head_loss can call it with (logits, y, loss_mask=None)."""
    def fn(logits, y, loss_mask=None):
        if loss_mask is not None:
            # All ones for this script; included for signature parity.
            pass
        return _focal_bce(logits, y, alpha=focal_alpha, gamma=focal_gamma,
                          label_smoothing=label_smoothing)
    return fn


@torch.no_grad()
def _evaluate_in_sample(model, dataset: LabeledSoundscapeWindowDataset,
                        device, batch_size: int = 64):
    """Predict on every labeled (file, window) row using dual-head agg.
    Returns (probs, y_true) aligned to dataset.items order.
    """
    model.eval()
    n = len(dataset)
    n_classes = dataset.n_classes
    P = np.zeros((n, n_classes), dtype=np.float32)
    Y = np.zeros((n, n_classes), dtype=np.uint8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    cursor = 0
    for batch_x, batch_y in loader:
        x = batch_x.to(device, non_blocking=True)
        out = model(x)
        p = dual_head_predict(out).cpu().numpy()
        bsz = p.shape[0]
        P[cursor:cursor + bsz] = p
        Y[cursor:cursor + bsz] = batch_y.numpy().astype(np.uint8)
        cursor += bsz
    return P, Y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--frame-weight", type=float, default=0.5,
                    help="Weight on the framewise (max-pool) loss in the "
                         "dual-head loss; 0=clip-only, 1=frame-only.")
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path,
                    default=OUTPUT_ROOT / "sed_inhouse_inSample",
                    help="Where to write metrics JSON + the trained checkpoint.")
    ap.add_argument("--config-name", type=str, default="sed_b0_dual_inhouse",
                    help="Subdir name under --out-dir + ckpt-dir under MODEL_ROOT/sed/.")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="Skip writing a best.pt checkpoint after training.")
    args = ap.parse_args()

    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sed-inhouse] device={device}")

    sed_cfg = SEDConfig()  # uses Tucker-matching defaults
    print(f"[sed-inhouse] SEDConfig: backbone={sed_cfg.backbone}  "
          f"n_mels={sed_cfg.n_mels}  hop={sed_cfg.hop_length}  "
          f"n_classes={sed_cfg.n_classes}")

    dataset = LabeledSoundscapeWindowDataset()
    print(f"[sed-inhouse] labeled (file, window) rows: {len(dataset):,}  "
          f"unique files: {len({i[0] for i in dataset.items}):,}  "
          f"classes: {dataset.n_classes}")
    if len(dataset) == 0:
        raise SystemExit("No labeled rows; check soundscape labels CSV.")

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(int(args.num_workers) > 0),
    )

    model = SED(sed_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sed-inhouse] model params: {n_params/1e6:.2f}M")

    base_loss = _wrap_loss_for_dual_head(
        focal_alpha=float(args.focal_alpha),
        focal_gamma=float(args.focal_gamma),
        label_smoothing=float(args.label_smoothing),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    total_steps = max(1, int(args.epochs) * max(1, len(loader)))
    warmup_steps = max(1, int(total_steps * float(args.warmup_frac)))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    print(f"[sed-inhouse] training: epochs={args.epochs}  "
          f"steps/epoch={len(loader)}  total_steps={total_steps}  "
          f"warmup_steps={warmup_steps}")

    step = 0
    t0 = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        running = 0.0
        n_seen = 0
        for batch_x, batch_y in loader:
            x = batch_x.to(device, non_blocking=True)
            y = batch_y.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = float(args.lr) * lr_at(step)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = dual_head_loss(out, y, base_loss,
                                  frame_weight=float(args.frame_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.shape[0]
            n_seen += x.shape[0]
            step += 1
        elapsed = (time.time() - t0) / 60.0
        avg = running / max(1, n_seen)
        print(f"[sed-inhouse] epoch {epoch + 1}/{args.epochs}  "
              f"loss={avg:.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"elapsed={elapsed:.1f}m")

    print(f"[sed-inhouse] training done in {(time.time() - t0)/60:.1f}m")

    # In-sample fit eval
    print(f"[sed-inhouse] running in-sample evaluation on training rows...")
    P, Y = _evaluate_in_sample(model, dataset, device,
                               batch_size=int(args.batch_size))
    # Build per-row site/hour meta for compute_stage_metrics
    sc = load_soundscape_meta()
    sc = sc[sc["fully_labeled"]].sort_values(["filename", "end_sec"]).reset_index(drop=True)
    fn_set = {i[0] for i in dataset.items}
    sc = sc[sc["filename"].isin(fn_set)].reset_index(drop=True)
    eval_meta = sc[["site", "hour_utc"]].reset_index(drop=True)
    if len(eval_meta) != len(P):
        # Should not happen given dataset construction, but guard regardless.
        raise SystemExit(f"Eval-meta rows ({len(eval_meta)}) != dataset rows ({len(P)})")

    metrics = compute_stage_metrics(Y, P, eval_meta)
    macro_auc = float(metrics.get("macro_auc", float("nan")))
    site_std = float(metrics.get("site_auc_std", float("nan")))

    summary = {
        "config_name": args.config_name,
        "n_rows": int(len(P)),
        "n_files": int(len(fn_set)),
        "n_classes": int(dataset.n_classes),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "frame_weight": float(args.frame_weight),
        "macro_auc_in_sample": macro_auc,
        "site_auc_std_in_sample": site_std,
        "rare_auc": metrics.get("rare_auc", float("nan")),
        "frequent_auc": metrics.get("frequent_auc", float("nan")),
        "stage_metrics_full": metrics,
        "sed_cfg": dict(
            backbone=sed_cfg.backbone, n_mels=sed_cfg.n_mels,
            hop_length=sed_cfg.hop_length, n_fft=sed_cfg.n_fft,
            f_min=sed_cfg.f_min, f_max=sed_cfg.f_max,
            n_classes=sed_cfg.n_classes,
        ),
        "n_params_M": round(n_params / 1e6, 3),
    }

    out_dir = Path(args.out_dir) / args.config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"in_sample_metrics_seed{args.seed}.json"
    out_npz = out_dir / f"in_sample_probs_seed{args.seed}.npz"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(out_npz, probs=P, y_true=Y)

    print()
    print(f"[sed-inhouse] === in-sample fit ({args.config_name}, seed={args.seed}) ===")
    print(f"[sed-inhouse]   macro_auc       = {macro_auc:.4f}")
    print(f"[sed-inhouse]   site_auc_std    = {site_std:.4f}")
    print(f"[sed-inhouse]   rare_auc        = {summary['rare_auc']}")
    print(f"[sed-inhouse]   frequent_auc    = {summary['frequent_auc']}")
    print(f"[sed-inhouse] wrote {out_json}")
    print(f"[sed-inhouse] wrote {out_npz}")
    print()
    print(f"[sed-inhouse] Compare to:")
    print(f"[sed-inhouse]   Tucker leaked OOF (memorized): 0.985")
    print(f"[sed-inhouse]   SSM in-sample fit (_07c):      0.9856")
    print(f"[sed-inhouse]   Target: ≈ 0.985 to confirm architectural parity.")

    if not args.no_ckpt:
        ckpt_dir = Path(MODEL_ROOT) / "sed" / args.config_name / "all_labeled_inSample"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "cfg": summary["sed_cfg"],
        }, ckpt_path)
        print(f"[sed-inhouse] wrote checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
