"""Sweep grids for the Perch+SSM head pipeline (train/train_ssm_head.py).

BASELINE = LB_093.ipynb knobs (the 0.93 LB recipe). After the lb sweep
confirmed LB_093 wins on stitched/mean OOF too, it became the new default.

Each config is merged with BASELINE. The hparams-diff CSV auto-drops fields
that are constant across the sweep — so you only see the knobs that move.

Two sweeps:
    SWEEP_CHEAP_WINS — first-pass A/B around BASELINE (TTA, loss, boost on/off)
    SWEEP_BEST_SSM   — fine-grained exploration around BASELINE
"""
from __future__ import annotations

BASELINE = dict(
    seed=42,
    ensemble_w=0.50,
    lambda_prior=0.40,
    lambda_prior_texture=1.00,
    correction_weight=0.30,
    loss="bce_focal_mean",
    focal_alpha=0.25,
    focal_gamma=2.5,
    label_smoothing=0.03,
    mask_secondary=False,
    tta="window_roll",
    tta_shifts=(0, 1, -1, 2, -2),
    tta_shifts_secs=(0.0, 1.25, -1.25, 2.5, -2.5),
    smoothing="adaptive",
    smoothing_alpha=0.20,
    use_boost=False,
    boost_threshold=0.5,
    boost_lift=0.25,
    file_conf_top_k=2,
    file_conf_power=0.40,
    rank_power=0.40,
    threshold_grid=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    proto_n_epochs=40,
    proto_lr=1e-3,
    proto_patience=8,
    proto_pos_weight_cap=25.0,
    proto_distill_weight=0.15,
    proto_swa_start_frac=0.65,
    proto_swa_lr=4e-4,
    residual_n_epochs=30,
    residual_lr=1e-3,
    residual_patience=8,
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


SWEEP_CHEAP_WINS = [
    _make(name="baseline"),
    _make(name="waveform_shift_tta", tta="waveform_shift"),
    _make(name="bce_posw_legacy", loss="bce_posw"),
    _make(name="focal_bce_only", loss="focal_bce"),
    _make(name="gaussian_smooth", smoothing="gaussian"),
    _make(name="smooth_off", smoothing="none"),
    _make(name="boost_on", use_boost=True),
    _make(name="boost_strong", use_boost=True, boost_lift=0.40, boost_threshold=0.4),
    _make(name="threshold_single", threshold_grid=(0.50,)),
]


SWEEP_BEST_SSM = [
    _make(name="baseline"),
    # ensemble_w
    _make(name="ens_w_000_mlp_only", ensemble_w=0.00),
    _make(name="ens_w_015", ensemble_w=0.15),
    _make(name="ens_w_025", ensemble_w=0.25),
    _make(name="ens_w_075_proto_heavy", ensemble_w=0.75),
    # MLP probes (BASELINE pca=64)
    _make(name="mlp_alpha_060", mlp_alpha_blend=0.60),
    _make(name="mlp_alpha_080", mlp_alpha_blend=0.80),
    _make(name="mlp_pca_32", mlp_pca_dim=32),
    _make(name="mlp_pca_128", mlp_pca_dim=128),
    _make(name="mlp_minpos_3", mlp_min_pos=3),
    # ProtoSSM capacity (BASELINE epochs=40 patience=8)
    _make(name="proto_long", proto_n_epochs=80, proto_patience=20),
    _make(name="proto_distill_heavy", proto_distill_weight=0.30),
    # ResidualSSM (BASELINE corr=0.30 epochs=30)
    _make(name="corr_off", correction_weight=0.0),
    _make(name="corr_strong", correction_weight=0.50),
    _make(name="residual_long", residual_n_epochs=60, residual_patience=15),
    # Prior strength (BASELINE lambda=0.40, lambda_texture=1.00)
    _make(name="prior_off", lambda_prior=0.0, lambda_prior_texture=0.0),
    _make(name="prior_weak", lambda_prior=0.20, lambda_prior_texture=0.50),
    _make(name="prior_strong", lambda_prior=0.60, lambda_prior_texture=1.50),
    _make(name="prior_texture_off", lambda_prior_texture=0.40),
    _make(name="prior_texture_15", lambda_prior_texture=1.50),
    # Loss shape (BASELINE bce_focal_mean, focal_gamma=2.5)
    _make(name="focal_g15", focal_gamma=1.5),
    _make(name="focal_g35", focal_gamma=3.5),
    _make(name="focal_a010", focal_alpha=0.10),
    _make(name="focal_a050", focal_alpha=0.50),
    _make(name="ls_0", label_smoothing=0.0),
    _make(name="ls_10", label_smoothing=0.10),
    # Threshold grid (BASELINE 0.25..0.70 step 0.05)
    _make(name="threshold_single", threshold_grid=(0.50,)),
    _make(name="threshold_coarse", threshold_grid=(0.30, 0.50, 0.70)),
    # Stacked combos
    _make(name="mlp_heavy_combo",
          ensemble_w=0.25, mlp_alpha_blend=0.60),
    _make(name="prior_corr_off",
          lambda_prior=0.0, correction_weight=0.0),
    _make(name="proto_long_prior_strong",
          proto_n_epochs=80, proto_patience=20, lambda_prior=0.60),
]
