"""Sweep grids for SED backbone training.

BASELINE matches the Tucker Arrants distilled-SED bundle architecture
(`models/sed_kaggle/sed_fold{0..4}.onnx`): EfficientNet-B0 backbone,
256-mel, hop=512. The model has a dual head (clip + framewise) — see
`birdclef.models.sed.SED` and `dual_head_loss` / `dual_head_predict`.
"""
from __future__ import annotations

BASELINE = dict(
    # Reproducibility — seed used by sweep runner + per-rank
    # (effective manual_seed = seed + rank in train_sed_ddp).
    seed=42,
    # Model — defaults track the Kaggle distilled-SED bundle.
    backbone="tf_efficientnet_b0.ns_jft_in1k",
    n_classes=234,
    dropout=0.30,
    sample_rate=32000,
    n_mels=256,
    n_fft=2048,
    hop_length=512,
    f_min=20,
    f_max=16000,
    freq_mask_param=24,
    time_mask_param=32,
    specaugment_masks=2,
    spec_noise_std=0.01,
    # Dual-head loss weight: how much the framewise (max-pool) loss
    # contributes vs the clip-level loss. 0.5 = equal split, matching the
    # public-notebook inference aggregation.
    frame_weight=0.5,
    # Training
    epochs=20,
    lr=1e-3,
    weight_decay=1e-4,
    warmup_frac=0.05,
    batch_size=64,              # per-rank; effective = N_GPUs * batch_size
    grad_accum=1,
    window_seconds=5,
    soundscape_fraction=0.5,
    first_window_prob=0.7,
    # Loss
    loss="focal_bce",
    focal_alpha=0.25,
    focal_gamma=2.0,
    label_smoothing=0.03,
    mask_secondary=True,
    # Augs
    mixup_alpha=0.5,
    mixup_mode="max",
    # DDP / precision
    amp=True,
    amp_dtype="fp16",      # "fp16" for V100/consumer, "bf16" for A100/H100 (no GradScaler)
    grad_clip=1.0,
    ema_decay=0.999,
    # Eval
    eval_every_n_steps=100,
    pseudo_round=None,
    # DataLoader throughput — soundscape OGG decoding is the hot path under
    # pseudo-label mode. Raise workers until GPU util saturates; prefetch
    # factor lets workers keep producing while GPU is busy.
    num_workers=8,
    prefetch_factor=4,
)


def _make(**kwargs):
    out = dict(BASELINE)
    out.update(kwargs)
    return out


SWEEP_SED_BASELINE = [_make(name="sed_b0_dual")]

SWEEP_SED_COMPARE = [
    _make(name="sed_b0_dual"),
    _make(name="sed_v2s_dual", backbone="tf_efficientnetv2_s.in21k_ft_in1k",
          n_mels=128, hop_length=320),
    _make(name="sed_nfnet_l0_dual", backbone="eca_nfnet_l0"),
    _make(name="sed_b0_highmix", mixup_alpha=0.8),
]
