"""Sweep grids for the Perch+SSM head pipeline (train/train_ssm_head.py).

Each config is merged with BASELINE. The hparams-diff CSV auto-drops fields
that are constant across the sweep — so you only see the knobs that move.
"""
from __future__ import annotations

BASELINE = dict(
    # Base stack blends
    ensemble_w=0.50,
    lambda_prior=0.0,
    correction_weight=0.0,
    # Loss choice
    loss="focal_bce",
    focal_alpha=0.25,
    focal_gamma=2.0,
    label_smoothing=0.03,
    mask_secondary=False,          # soundscape labels have no primary/secondary split
    # TTA
    tta="window_roll",             # or "waveform_shift"
    tta_shifts=(0, 1, -1, 2, -2),
    tta_shifts_secs=(0.0, 1.25, -1.25, 2.5, -2.5),
    # Post-processing
    smoothing="gaussian",          # or "adaptive"
    smoothing_alpha=0.20,
    use_boost=True,
    boost_threshold=0.5,
    boost_lift=0.25,
    file_conf_top_k=2,
    file_conf_power=0.40,
    rank_power=0.40,
    # Thresholds
    threshold_grid=(0.50, ),
    # Training
    proto_n_epochs=80,
    proto_lr=8e-4,
    proto_patience=20,
    proto_pos_weight_cap=25.0,
    proto_distill_weight=0.15,
    proto_swa_start_frac=0.65,
    proto_swa_lr=4e-4,
    residual_n_epochs=40,
    residual_lr=8e-4,
    residual_patience=12,
    residual_d_model=128,
    residual_d_state=16,
    residual_dropout=0.10,
    mlp_pca_dim=64,
    mlp_alpha_blend=0.40,
    mlp_min_pos=5,
    n_sites_cap=20,
)


def _make(**kwargs):
    out = dict(BASELINE)
    out.update(kwargs)
    return out


SWEEP_BASELINE = [_make(name="baseline")]

SWEEP_CHEAP_WINS = [
    _make(name="baseline"),
    _make(name="focal_bce_waveform_shift", tta="waveform_shift"),
    _make(name="bce_posw_legacy", loss="bce_posw"),
    _make(name="no_boost", use_boost=False),
    _make(name="adaptive_smooth", smoothing="adaptive"),
    _make(name="gaussian_plus_boost",
          smoothing="gaussian", use_boost=True, tta="waveform_shift"),
    _make(name="bce_focal_mean", loss="bce_focal_mean"),
    _make(name="boost_strong", boost_lift=0.40, boost_threshold=0.4),
]
