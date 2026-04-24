"""Sweep grids for SED backbone training."""
from __future__ import annotations

BASELINE = dict(
    # Reproducibility — seed used by sweep runner + per-rank
    # (effective manual_seed = seed + rank in train_sed_ddp).
    seed=42,
    # Model
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
    n_classes=234,
    dropout=0.30,
    sample_rate=32000,
    n_mels=128,
    n_fft=2048,
    hop_length=320,
    f_min=20,
    f_max=16000,
    freq_mask_param=24,
    time_mask_param=32,
    specaugment_masks=2,
    spec_noise_std=0.01,
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
    grad_clip=1.0,
    ema_decay=0.999,
    # Eval
    eval_every_n_steps=100,
    pseudo_round=None,
)


def _make(**kwargs):
    out = dict(BASELINE)
    out.update(kwargs)
    return out


SWEEP_SED_BASELINE = [_make(name="sed_v2s")]

SWEEP_SED_COMPARE = [
    _make(name="sed_v2s"),
    _make(name="sed_nfnet_l0", backbone="eca_nfnet_l0"),
    _make(name="sed_v2s_highmix", mixup_alpha=0.8),
    _make(name="sed_v2s_bigmel", n_mels=256, hop_length=512),
]
