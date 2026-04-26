"""AST (Audio Spectrogram Transformer) trainer — fine-tunes an
AudioSet-pretrained transformer on train.csv and validates on the same
labeled-soundscape pool as `train_ddp_focal.py`.

Why this exists alongside `train_ddp_focal.py`:
    The focal CNN sweep is currently underperforming Perch on `val_auc_focal_seen`
    (~0.83 vs Perch's 0.87). Top BirdCLEF 2024/2025 solutions use audio-pretrained
    transformers (PaSST/BEATs/AST) as the third blend member next to Perch+SSM
    and a CNN. AST is the easiest entry point: HuggingFace transformers ships it
    as `MIT/ast-finetuned-audioset-10-10-0.4593`, AudioSet-pretrained on 2M
    YouTube clips × 527 sound classes (insects, amphibians, birds all included).

Key differences from `train_ddp_focal.py`:
    - Backbone is `ASTForAudioClassification` (vit-style, ~85M params).
    - Spectrogram preprocessing is kaldi-style fbank at 16kHz (what AST was
      pretrained on). `_AstFbankExtract` does this on-the-fly inside the model's
      forward — keeps `BirdCLEFDataset` unchanged.
    - LRs are an order of magnitude smaller (transformer fine-tuning), no
      separate head/backbone LR (single-LR AdamW). Cosine schedule with warmup.
    - Default batch size smaller (24 → 16 on the same GPU because AST is bigger).
    - Mixup off by default (transformer training is sensitive); spec-augment
      is implemented inside the AST module so we don't double-augment.

Same as `train_ddp_focal.py`:
    - Train data: full train.csv (35k focal recordings, 206 classes); optional
      `--include-xc` extends with Xeno-Canto.
    - Val data: all 66 labeled soundscapes (739 windows, 75 atomic classes).
    - Selection metric: `val_auc_focal_seen` (clean ranker, no sonotype noise).
    - Per-taxon AUC breakdown (Aves / Amphibia / Insecta / Mammalia / Reptilia)
      written to `metrics.csv` and `summary.json`.
    - All floats rounded to 4 decimals.
    - `experiments_summary.csv` sorted by `primary` so it stacks alongside
      the SSM and focal-CNN summaries.

Requires `transformers>=4.35`. Install: `pip install transformers`.

Invocation pattern matches `train_ddp_focal.py`:

    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \\
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
    torchrun --standalone --nproc_per_node=2 \\
        -m birdclef_example.train_ddp_ast --preload-train-audio

Outputs:
    birdclef_example/outputs/ast/<config_name>/
        best_model.pt
        label_map.json
        metrics.csv
        summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchaudio
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdclef_example.data import (
    BirdCLEFDataset,
    build_label_map,
    prepare_soundscape_metadata,
    prepare_train_audio_metadata,
)
from birdclef_example.train_ddp_focal import (  # noqa: E402  reuse helpers verbatim
    _build_focal_class_indices,
    _build_taxa_indices,
    _evaluate_full,
    _r4,
)
from birdclef_example.train_ddp_sota import (  # noqa: E402
    BCEFocalComboLoss,
    _cleanup_distributed,
    _get_distributed_context,
    _is_main_process,
    _print_main,
    _setup_distributed,
    build_rank_metadata_shard,
    build_sample_weights,
    train_one_epoch_ddp,
)
from birdclef_example.utils import is_better_score, save_model, set_seed  # noqa: E402


DEFAULT_AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
# AST AudioSet normalization stats from the original paper (Gong et al. 2021).
# Critical to match: changing these breaks transfer from the AudioSet weights.
_FBANK_MEAN = -4.2677
_FBANK_STD = 4.5689


def _interpolate_position_embeddings(
    old_pe: torch.Tensor, n_freq: int, n_time_old: int, n_time_new: int,
    n_special: int = 2,
) -> torch.Tensor:
    """2D bicubic interpolation of AST position embeddings along the time axis.

    AST checkpoint at max_length=1024 has position_embeddings of shape
    (1, 1214, 768) = (1, n_special + n_freq*n_time_old, embed_dim) where
    n_special=2 (CLS + distillation token), n_freq=12, n_time_old=101.

    When we change max_length (e.g. to 512 → n_time_new=50), the naive
    `from_pretrained` flow REINITIALIZES this Parameter randomly because
    of the size mismatch — throwing away the pretrained temporal priors.
    Interpolating preserves them.
    """
    embed_dim = old_pe.shape[-1]
    special = old_pe[:, :n_special, :]                     # (1, n_special, embed)
    patches = old_pe[:, n_special:, :]                     # (1, n_freq*n_time_old, embed)
    # Reshape to spatial grid: (1, embed, n_freq, n_time_old)
    grid = patches.transpose(1, 2).reshape(1, embed_dim, n_freq, n_time_old).float()
    new_grid = torch.nn.functional.interpolate(
        grid, size=(n_freq, n_time_new), mode="bicubic", align_corners=False,
    )                                                      # (1, embed, n_freq, n_time_new)
    new_patches = new_grid.reshape(1, embed_dim, n_freq * n_time_new).transpose(1, 2)
    return torch.cat([special, new_patches.to(old_pe.dtype)], dim=1)


class ASTSpectrogramClassifier(nn.Module):
    """AST wrapper: raw 32kHz waveform in, 234-class logits out.

    Computes kaldi-style fbank features at 16kHz inside `forward` so the
    rest of the training pipeline (BirdCLEFDataset, DDP, eval) doesn't
    have to know AST's preprocessing requirements. The fbank extraction
    has to match the AudioSet-pretraining recipe exactly — different
    settings break transfer.

    `max_length` controls the time-frame budget the model sees. For 5s
    audio at 16kHz with 10ms hop, ~498 frames; we round up to 512 and
    let the AST positional encoding interpolate to that length.
    """

    INPUT_RANK = 3  # (B, 1, T) raw waveform

    def __init__(
        self,
        n_classes: int,
        hf_model_name: str = DEFAULT_AST_CHECKPOINT,
        input_sample_rate: int = 32000,
        target_sample_rate: int = 16000,
        max_length: int = 512,
        num_mel_bins: int = 128,
        freq_mask_param: int = 24,
        time_mask_param: int = 96,
        specaugment_masks: int = 2,
    ):
        super().__init__()
        try:
            from transformers import ASTConfig, ASTForAudioClassification
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "transformers package required. Install: pip install transformers>=4.35"
            ) from e

        # 1) Pull the pretrained config + state_dict at the ORIGINAL max_length
        #    (1024 by default) so we get the full-size pretrained position
        #    embeddings (1, 1214, 768). Without this step `from_pretrained` at
        #    a different max_length silently REINITIALIZES the position
        #    embeddings — wiping out the AudioSet temporal priors.
        orig_cfg = ASTConfig.from_pretrained(hf_model_name)
        tmp_model = ASTForAudioClassification.from_pretrained(
            hf_model_name,
            num_labels=int(n_classes),
            ignore_mismatched_sizes=True,  # only the classifier head mismatches; PE matches
        )
        state = tmp_model.state_dict()
        del tmp_model

        # 2) Interpolate the position embeddings to our target time length.
        n_freq = (int(num_mel_bins) - orig_cfg.patch_size) // orig_cfg.frequency_stride + 1
        n_time_old = (orig_cfg.max_length - orig_cfg.patch_size) // orig_cfg.time_stride + 1
        n_time_new = (int(max_length) - orig_cfg.patch_size) // orig_cfg.time_stride + 1
        pe_key = "audio_spectrogram_transformer.embeddings.position_embeddings"
        if n_time_new != n_time_old and pe_key in state:
            state[pe_key] = _interpolate_position_embeddings(
                state[pe_key], n_freq=n_freq,
                n_time_old=n_time_old, n_time_new=n_time_new, n_special=2,
            )

        # 3) Build the actual model with the target config + load interpolated weights.
        cfg = ASTConfig.from_pretrained(hf_model_name)
        cfg.num_labels = int(n_classes)
        cfg.max_length = int(max_length)
        cfg.num_mel_bins = int(num_mel_bins)
        self.model = ASTForAudioClassification(cfg)
        # The classifier in `state` has fresh-init weights (right shape) from the
        # tmp_model build above. Loading non-strict is fine; remaining fresh-init
        # is just the patch embeddings shape if num_mel_bins changed, which it
        # didn't here. strict=True after popping classifier keys would also work.
        self.model.load_state_dict(state, strict=False)

        self.input_sample_rate = int(input_sample_rate)
        self.target_sample_rate = int(target_sample_rate)
        self.max_length = int(max_length)
        self.num_mel_bins = int(num_mel_bins)
        if input_sample_rate != target_sample_rate:
            self.resample = torchaudio.transforms.Resample(
                int(input_sample_rate), int(target_sample_rate),
            )
        else:
            self.resample = nn.Identity()

        # SpecAugment over the fbank — applied only when self.training.
        self.freq_mask = torchaudio.transforms.FrequencyMasking(int(freq_mask_param))
        self.time_mask = torchaudio.transforms.TimeMasking(int(time_mask_param))
        self.specaugment_masks = int(specaugment_masks)

    def _fbank_one(self, waveform_1xT: torch.Tensor) -> torch.Tensor:
        """Kaldi-style fbank for a single (1, T) waveform at 16kHz.

        AST's pretraining used these exact settings (10ms hop, 25ms window,
        128 mel bins, no dither). We must match.

        torchaudio.compliance.kaldi.fbank does NOT support bf16/fp16 inputs
        (raises 'Unsupported dtype BFloat16'). The caller `_compute_fbanks`
        wraps this in an autocast-disabled block + casts to float32.
        """
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_1xT,
            htk_compat=True,
            sample_frequency=self.target_sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,
            frame_length=25,
        )
        # AudioSet stats (paper). Half-stride normalization: (x - mean) / (2*std).
        fbank = (fbank - _FBANK_MEAN) / (_FBANK_STD * 2.0)
        n = fbank.size(0)
        if n < self.max_length:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, self.max_length - n))
        elif n > self.max_length:
            fbank = fbank[: self.max_length, :]
        return fbank  # (max_length, num_mel_bins)

    def _compute_fbanks(self, waveform_BxT: torch.Tensor) -> torch.Tensor:
        # kaldi.fbank requires float32 + isn't autocast-compatible; force fp32
        # context for the preprocessing stage. The model body still runs in
        # whatever dtype the outer autocast specifies.
        device_type = waveform_BxT.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            wav_f32 = waveform_BxT.float()
            out = [self._fbank_one(wav_f32[i : i + 1]) for i in range(wav_f32.size(0))]
            return torch.stack(out, dim=0)  # (B, max_length, num_mel_bins)

    def _apply_specaugment(self, fbank: torch.Tensor) -> torch.Tensor:
        # fbank: (B, T, F) — torchaudio masks expect (..., F, T), so transpose
        x = fbank.transpose(1, 2)  # (B, F, T)
        for _ in range(self.specaugment_masks):
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x.transpose(1, 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, 1, T) or (B, T) raw at input_sample_rate.
        wav = waveform.squeeze(1) if waveform.dim() == 3 else waveform   # (B, T)
        wav = self.resample(wav)                                          # (B, T_resampled)
        fbank = self._compute_fbanks(wav)                                 # (B, max_length, n_mels)
        if self.training and self.specaugment_masks > 0:
            fbank = self._apply_specaugment(fbank)
        return self.model(input_values=fbank).logits                      # (B, n_classes)


def build_ast_experiment_configs() -> List[Dict[str, Any]]:
    """Tight grid of AST hyperparameters.

    AST is bigger and converges differently than the timm CNNs, so the
    grid focuses on the knobs that matter most for transformer fine-tuning:
    base LR, weight decay, mixup strength, SpecAugment intensity, and a
    longer-epoch deep dive on the best base.
    """
    common = {
        "backbone_name": DEFAULT_AST_CHECKPOINT,
        "image_size": 128,        # used only in summary, not by the model itself
        "n_mels": 128,
        "n_fft": 0,                # AST uses kaldi.fbank (windowing baked in)
        "hop_length": 0,
        "max_length": 512,
        "freq_mask_param": 24,
        "time_mask_param": 96,
        "specaugment_masks": 2,
        "spec_noise_std": 0.0,
        "mixup_alpha": 0.2,
        "mixup_probability": 0.5,
        "label_smoothing": 0.05,
        "weight_decay": 1e-3,
        "warmup_epochs": 2,
        "min_lr_ratio": 0.05,
        "bce_weight": 0.7,
        "focal_weight": 0.3,
        "focal_gamma": 2.0,
        "sampling_power": 0.5,
        # Two-phase fine-tune: classifier head trains alone for the first N
        # epochs (backbone frozen → no risk of corrupting AudioSet weights
        # with noisy classifier gradients), then everything is unfrozen for
        # the remaining (epochs - frozen_warmup_epochs) epochs at the lower
        # full-fine-tune LR. Standard transformer fine-tuning recipe.
        "frozen_warmup_epochs": 2,
        "frozen_warmup_lr":     1e-3,
        # AST is single-LR; we still emit `lr_head` and `lr_backbone` so the
        # build_optimizer wrapper from train_ddp_sota works unchanged. Set both
        # to the same target LR (used in the unfrozen phase).
    }
    out: List[Dict[str, Any]] = []

    # Pass 1: LR sweep at default 15 epochs
    for lr in [1e-5, 3e-5, 5e-5, 1e-4]:
        cfg = dict(common)
        cfg.update({
            "name": f"ast_lr{lr:.0e}_e15",
            "lr_head": lr, "lr_backbone": lr,
            "epochs": 15,
        })
        out.append(cfg)

    # Pass 2: best-LR-likely (5e-5) deep dives
    base_lr = 5e-5
    for tag, overrides in [
        ("e30",            {"epochs": 30}),
        ("e30_mixupA10",   {"epochs": 30, "mixup_alpha": 1.0, "mixup_probability": 0.7}),
        ("e30_specheavy",  {"epochs": 30, "freq_mask_param": 32, "time_mask_param": 128,
                             "specaugment_masks": 3}),
        ("e30_lowwd",      {"epochs": 30, "weight_decay": 1e-4}),
        ("e30_smooth0",    {"epochs": 30, "label_smoothing": 0.0}),
    ]:
        cfg = dict(common)
        cfg.update({
            "name": f"ast_{tag}",
            "lr_head": base_lr, "lr_backbone": base_lr,
            "epochs": 15,  # default; overridden below
        })
        cfg.update(overrides)
        out.append(cfg)

    return out


def _select_experiments(name_filter: Optional[str]) -> List[Dict[str, Any]]:
    all_exp = build_ast_experiment_configs()
    if not name_filter:
        return all_exp
    matches = [e for e in all_exp if e["name"] == name_filter]
    if not matches:
        sub = [e for e in all_exp if name_filter in e["name"]]
        if not sub:
            available = ", ".join(e["name"] for e in all_exp[:5])
            raise SystemExit(
                f"No experiment matched --exp '{name_filter}'. "
                f"First 5 available: {available} ..."
            )
        return sub
    return matches


def _split_ast_params(model: nn.Module) -> tuple[list, list]:
    """Return (head_params, backbone_params) — works on both bare model and DDP.

    Head = `classifier.*` (the freshly-initialized 234-class linear).
    Backbone = everything else (pretrained AudioSet weights).
    """
    bare = model.module if isinstance(model, DDP) else model
    head, backbone = [], []
    for name, p in bare.named_parameters():
        # bare is ASTSpectrogramClassifier; classifier lives at .model.classifier.*
        if "classifier" in name:
            head.append(p)
        else:
            backbone.append(p)
    return head, backbone


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    bare = model.module if isinstance(model, DDP) else model
    for name, p in bare.named_parameters():
        if "classifier" not in name:
            p.requires_grad = bool(trainable)


def _build_optimizer_ast(model: nn.Module, lr: float, weight_decay: float):
    """Single-LR AdamW for the full-fine-tune phase.

    AST doesn't have a meaningful "head vs backbone" LR split after the freeze
    warmup is done — every parameter except the new Linear is pretrained and
    benefits from the same low LR for end-to-end fine-tuning.
    """
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr), weight_decay=float(weight_decay), betas=(0.9, 0.999),
    )


def _build_optimizer_head_only(model: nn.Module, lr: float, weight_decay: float):
    """AdamW on classifier params only — used for the frozen-backbone warmup.

    During this phase the backbone has requires_grad=False and DOESN'T appear
    in this optimizer's param list, so its weights stay byte-identical to
    the pretrained AudioSet checkpoint.
    """
    head_params, _ = _split_ast_params(model)
    return torch.optim.AdamW(
        head_params, lr=float(lr), weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )


def _cosine_warmup_scheduler(optimizer, epochs: int, warmup_epochs: int,
                             min_lr_ratio: float):
    """Linear warmup → cosine decay to min_lr_ratio."""
    warmup_epochs = max(1, int(warmup_epochs))
    epochs = max(warmup_epochs + 1, int(epochs))

    def lr_lambda(e: int) -> float:
        if e < warmup_epochs:
            return (e + 1) / warmup_epochs
        progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_one_experiment(
    experiment: Dict[str, Any],
    train_dataset: BirdCLEFDataset,
    val_dataset: Optional[BirdCLEFDataset],
    train_rank_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    label_map: Dict[str, int],
    focal_class_indices: np.ndarray,
    taxa_indices: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
    rank: int,
    local_rank: int,
    distributed: bool,
    use_bf16: bool,
    train_meta_len: int,
    train_rank_meta_len: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    grad_clip_norm: float,
) -> Dict[str, Any]:
    """One full training run of one AST config.

    Selection metric: `val_auc_focal_seen` — same as `train_ddp_focal.py`.
    """
    epochs = int(experiment.get("epochs", epochs))
    exp_name = experiment["name"]
    exp_output_dir = output_dir / exp_name
    if _is_main_process(rank):
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    sample_weights = build_sample_weights(
        train_rank_meta, label_map,
        power=float(experiment.get("sampling_power", 0.5)),
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if _is_main_process(rank) and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )

    model_config = {
        "n_classes": len(label_map),
        "hf_model_name": str(experiment.get("backbone_name", DEFAULT_AST_CHECKPOINT)),
        "input_sample_rate": 32000,
        "target_sample_rate": 16000,
        "max_length": int(experiment.get("max_length", 512)),
        "num_mel_bins": int(experiment.get("n_mels", 128)),
        "freq_mask_param": int(experiment.get("freq_mask_param", 24)),
        "time_mask_param": int(experiment.get("time_mask_param", 96)),
        "specaugment_masks": int(experiment.get("specaugment_masks", 2)),
    }
    model = ASTSpectrogramClassifier(**model_config).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    criterion = BCEFocalComboLoss(
        bce_weight=float(experiment["bce_weight"]),
        focal_weight=float(experiment["focal_weight"]),
        focal_gamma=float(experiment["focal_gamma"]),
        label_smoothing=float(experiment["label_smoothing"]),
    )

    # Phase 1: frozen-backbone warmup.
    # Classifier head trains alone with a higher LR so it converges quickly
    # before any gradient touches the AudioSet weights. Standard transformer
    # fine-tuning recipe — without this, noisy gradients from the freshly-init
    # classifier flow back into the backbone and degrade pretraining.
    frozen_warmup_epochs = int(experiment.get("frozen_warmup_epochs", 0))
    frozen_warmup_lr     = float(experiment.get("frozen_warmup_lr", 1e-3))
    if frozen_warmup_epochs > 0:
        _set_backbone_trainable(model, False)
        optimizer = _build_optimizer_head_only(
            model=model, lr=frozen_warmup_lr,
            weight_decay=float(experiment["weight_decay"]),
        )
        _print_main(rank, f"[phase 1] backbone FROZEN — head-only training for "
                          f"{frozen_warmup_epochs} epoch(s) at lr={frozen_warmup_lr:.0e}")
    else:
        _set_backbone_trainable(model, True)
        optimizer = _build_optimizer_ast(
            model=model,
            lr=float(experiment["lr_head"]),
            weight_decay=float(experiment["weight_decay"]),
        )
    # Phase-2 scheduler is built fresh after the unfreeze; for phase 1 we use
    # a constant-LR no-op so the loop's `scheduler.step()` is harmless.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    amp_enabled = use_bf16 and device.type == "cuda"

    _print_main(rank, "")
    _print_main(rank, f"=== {exp_name} ===")
    _print_main(rank, f"Backbone: {model_config['hf_model_name']} | max_length={model_config['max_length']}")
    _print_main(rank, f"DDP enabled: {distributed} (world_size="
                      f"{dist.get_world_size() if dist.is_initialized() else 1})")
    _print_main(rank, f"Train rows (focal): {train_meta_len}  (rank shard: {train_rank_meta_len})")
    _print_main(rank, f"Val rows (soundscape): {len(val_meta)}")
    _print_main(rank, f"Epochs: {epochs}  lr={experiment['lr_head']:.0e}  "
                      f"wd={experiment['weight_decay']:.0e}")

    best_metrics: Dict[str, Any] = {}
    best_val_loss = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        # Phase transition: at the first epoch AFTER frozen_warmup_epochs, unfreeze
        # the backbone and switch to the full-fine-tune optimizer + cosine schedule.
        # The schedule's epoch-budget is the remaining (epochs - frozen_warmup_epochs)
        # so the cosine still goes from peak LR → min_lr_ratio over the unfrozen run.
        if frozen_warmup_epochs > 0 and epoch == frozen_warmup_epochs + 1:
            _set_backbone_trainable(model, True)
            optimizer = _build_optimizer_ast(
                model=model,
                lr=float(experiment["lr_head"]),
                weight_decay=float(experiment["weight_decay"]),
            )
            scheduler = _cosine_warmup_scheduler(
                optimizer=optimizer,
                epochs=epochs - frozen_warmup_epochs,
                warmup_epochs=int(experiment["warmup_epochs"]),
                min_lr_ratio=float(experiment["min_lr_ratio"]),
            )
            n_unfrozen = sum(p.requires_grad for p in (model.module if isinstance(model, DDP) else model).parameters())
            _print_main(rank, f"[phase 2] backbone UNFROZEN at epoch {epoch} — "
                              f"full fine-tune for {epochs - frozen_warmup_epochs} epochs at "
                              f"peak lr={experiment['lr_head']:.0e} ({n_unfrozen} trainable params)")

        avg_train_loss = train_one_epoch_ddp(
            model=model, train_loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch_label=f"{exp_name} | ep {epoch}",
            rank=rank,
            mixup_alpha=float(experiment["mixup_alpha"]),
            mixup_probability=float(experiment["mixup_probability"]),
        )
        scheduler.step()
        if dist.is_initialized():
            dist.barrier()

        if _is_main_process(rank):
            eval_model = model.module if isinstance(model, DDP) else model
            assert val_loader is not None
            metrics = _evaluate_full(
                eval_model, val_loader, torch.nn.BCEWithLogitsLoss(),
                device, focal_class_indices=focal_class_indices,
                taxa_indices=taxa_indices, use_bf16=use_bf16,
            )

            row = {
                "config_name": exp_name, "epoch": epoch,
                "train_loss": _r4(avg_train_loss),
                **{k: _r4(v) for k, v in metrics.items()},
            }
            history.append(row)
            taxa_log = "  ".join(
                f"{taxon[:3].lower()}={row.get(f'val_auc_{taxon.lower()}'):.4f}"
                for taxon in sorted(taxa_indices)
                if row.get(f"val_auc_{taxon.lower()}") is not None
            )
            print(
                f"{exp_name} | ep {epoch:02d} | "
                f"train={row['train_loss']} | "
                f"loss={row['val_loss']} "
                f"focal_seen={row['val_auc_focal_seen']} "
                f"seen={row['val_auc_seen']} | {taxa_log}"
            )

            current_primary = metrics.get("val_auc_focal_seen", float("nan"))
            best_primary = best_metrics.get("val_auc_focal_seen", float("nan"))
            should_save = False
            if is_better_score(current_primary, best_primary):
                best_metrics = dict(metrics)
                best_val_loss = float(metrics.get("val_loss", float("inf")))
                should_save = True
            elif pd.isna(current_primary) and metrics.get("val_loss", float("inf")) < best_val_loss:
                best_val_loss = float(metrics.get("val_loss", float("inf")))
                should_save = True

            if should_save:
                best_epoch = epoch
                model_path = exp_output_dir / "best_model.pt"
                save_model(
                    {
                        "model_state": eval_model.state_dict(),
                        "label_map": label_map,
                        "model_config": model_config,
                        "epoch": epoch,
                        **{f"best_{k}": _r4(v) for k, v in metrics.items()},
                        "experiment": experiment,
                    },
                    model_path,
                )
                with open(exp_output_dir / "label_map.json", "w", encoding="utf-8") as f:
                    json.dump(label_map, f, indent=2, sort_keys=True)

        if dist.is_initialized():
            dist.barrier()

    summary: Dict[str, Any] = {
        "_kind": "ast_run",
        "config_name": exp_name,
        "architecture": "ast_audio_spectrogram_transformer",
        "backbone_name": model_config["hf_model_name"],
        "best_epoch": best_epoch,
        "primary": _r4(best_metrics.get("val_auc_focal_seen")),
        **{f"best_{k}": _r4(v) for k, v in best_metrics.items()},
        "n_focal_classes": int(len(focal_class_indices)),
        "experiment_config": experiment,
        "model_config": model_config,
        "history": history,
    }

    if _is_main_process(rank):
        if history:
            pd.DataFrame(history).to_csv(exp_output_dir / "metrics.csv", index=False)
        with open(exp_output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None,
                    help="Experiment name (or substring) from build_ast_experiment_configs(). "
                         "Default: sweep all configs.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="AST is ~5× larger than nfnet_l0 — default 16. "
                         "Bump to 24-32 if memory allows on H100/H200.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--preload-workers", type=int, default=8)
    ap.add_argument("--preload-train-audio", action="store_true", default=False)
    ap.add_argument("--preload-val-audio", action="store_true", default=True)
    ap.add_argument("--no-preload-val-audio", dest="preload_val_audio", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-bf16", action="store_true", default=True)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir. Default: birdclef_example/outputs/ast")
    ap.add_argument("--include-xc", action="store_true",
                    help="Also load Xeno-Canto recordings (data/train_audio_xc.csv).")
    args = ap.parse_args()

    distributed, world_size, rank, local_rank = _get_distributed_context()
    _setup_distributed(distributed)

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = Path(args.out_dir) if args.out_dir else \
        repo_root / "birdclef_example" / "outputs" / "ast"

    train_csv_path = data_dir / "train.csv"
    train_audio_dir = data_dir / "train_audio"
    soundscape_labels_path = data_dir / "train_soundscapes_labels.csv"
    train_soundscape_dir = data_dir / "train_soundscapes"
    taxonomy_path = data_dir / "taxonomy.csv"
    for p in (train_csv_path, train_audio_dir, soundscape_labels_path,
              train_soundscape_dir, taxonomy_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    set_seed(int(args.seed) + rank)
    if _is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    train_csv_df = pd.read_csv(train_csv_path)
    train_meta = prepare_train_audio_metadata(train_csv_df, train_audio_dir)
    if args.include_xc:
        xc_csv = data_dir / "train_audio_xc.csv"
        xc_audio_dir = data_dir / "train_audio_xc"
        if not xc_csv.exists():
            raise FileNotFoundError(
                f"--include-xc requested but {xc_csv} missing. "
                f"Build it: python -m birdclef.scripts._xc_fetch"
            )
        xc_df = pd.read_csv(xc_csv)
        xc_meta = prepare_train_audio_metadata(xc_df, xc_audio_dir)
        _print_main(rank, f"[setup] including Xeno-Canto: +{len(xc_meta)} rows")
        train_meta = pd.concat([train_meta, xc_meta], ignore_index=True)
    train_meta = train_meta[train_meta["audio_filepath"].apply(lambda p: Path(p).exists())]
    train_meta = train_meta.reset_index(drop=True)

    soundscape_df = pd.read_csv(soundscape_labels_path)
    val_meta = prepare_soundscape_metadata(soundscape_df, train_soundscape_dir)
    label_map = build_label_map(train_meta, taxonomy_path)
    if len(label_map) != 234:
        raise ValueError(f"Expected 234 classes from taxonomy, got {len(label_map)}.")

    focal_class_indices = _build_focal_class_indices(train_meta, label_map)
    taxa_indices = _build_taxa_indices(label_map, taxonomy_path)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    base_model_config = {"dropout": 0.10, "sample_rate": 32000, "duration": 5.0}

    selected_exps = _select_experiments(args.exp)
    _print_main(rank, "=== AST sweep plan ===")
    _print_main(rank, f"  experiments        : {len(selected_exps)}  (--exp filter: {args.exp or '<all>'})")
    _print_main(rank, f"  train rows (focal) : {len(train_meta)}")
    _print_main(rank, f"  val rows (sscape)  : {len(val_meta)}")
    _print_main(rank, f"  focal-trained cls  : {len(focal_class_indices)}  / {len(label_map)}")
    _print_main(rank, f"  preload_train      : {args.preload_train_audio}")
    _print_main(rank, f"  output root        : {output_dir}")
    _print_main(rank, "")

    train_rank_meta = build_rank_metadata_shard(train_meta, world_size=world_size, rank=rank)
    _print_main(rank, f"[setup] building train dataset (rank shard: {len(train_rank_meta)} rows)...")
    train_dataset = BirdCLEFDataset(
        metadata=train_rank_meta, label_map=label_map,
        sample_rate=int(base_model_config["sample_rate"]),
        duration=float(base_model_config["duration"]),
        training=True, preload_audio=bool(args.preload_train_audio),
        preload_workers=int(args.preload_workers),
    )
    val_dataset = None
    if _is_main_process(rank):
        _print_main(rank, f"[setup] building val dataset ({len(val_meta)} rows)...")
        val_dataset = BirdCLEFDataset(
            metadata=val_meta, label_map=label_map,
            sample_rate=int(base_model_config["sample_rate"]),
            duration=float(base_model_config["duration"]),
            training=False, preload_audio=bool(args.preload_val_audio),
            preload_workers=int(args.preload_workers),
        )
    if dist.is_initialized():
        dist.barrier()

    all_summaries: List[Dict[str, Any]] = []
    for experiment in selected_exps:
        try:
            summary = run_one_experiment(
                experiment=experiment,
                train_dataset=train_dataset, val_dataset=val_dataset,
                train_rank_meta=train_rank_meta, val_meta=val_meta,
                label_map=label_map,
                focal_class_indices=focal_class_indices,
                taxa_indices=taxa_indices,
                output_dir=output_dir, device=device,
                rank=rank, local_rank=local_rank, distributed=distributed,
                use_bf16=args.use_bf16,
                train_meta_len=len(train_meta),
                train_rank_meta_len=len(train_rank_meta),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                epochs=int(args.epochs),
                grad_clip_norm=float(args.grad_clip_norm),
            )
        except Exception as ex:
            if _is_main_process(rank):
                failure = {"experiment": experiment.get("name", "unknown"),
                           "status": "failed", "error": str(ex)}
                all_summaries.append(failure)
                print(f"[WARN] {failure['experiment']} FAILED: {ex}")
            if dist.is_initialized():
                dist.barrier()
            continue

        if _is_main_process(rank):
            lean = {k: v for k, v in summary.items() if k != "history"}
            all_summaries.append(lean)
            with open(output_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, sort_keys=True, default=str)
            df = pd.DataFrame(all_summaries)
            if "primary" in df.columns:
                df = df.sort_values(by=["primary"], ascending=[False], na_position="last")
            df.to_csv(output_dir / "experiments_summary.csv", index=False)

    _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
