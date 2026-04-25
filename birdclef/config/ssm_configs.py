"""Sweep grids for the Perch+SSM head pipeline (train/train_ssm_head.py).

Each config is merged with BASELINE. The hparams-diff CSV auto-drops fields
that are constant across the sweep — so you only see the knobs that move.

Three sweeps:
    SWEEP_CHEAP_WINS — first-pass A/B (TTA, smoothing, loss, boost)
    SWEEP_BEST_SSM   — fine-grained exploration around BASELINE
    SWEEP_LB_093     — direct A/B: BASELINE vs the LB_093.ipynb port
"""
from __future__ import annotations

BASELINE = dict(
    seed=42,
    ensemble_w=0.50,
    lambda_prior=0.0,
    correction_weight=0.0,
    loss="focal_bce",
    focal_alpha=0.25,
    focal_gamma=2.0,
    label_smoothing=0.03,
    mask_secondary=False,
    tta="window_roll",
    tta_shifts=(0, 1, -1, 2, -2),
    tta_shifts_secs=(0.0, 1.25, -1.25, 2.5, -2.5),
    smoothing="gaussian",
    smoothing_alpha=0.20,
    use_boost=True,
    boost_threshold=0.5,
    boost_lift=0.25,
    file_conf_top_k=2,
    file_conf_power=0.40,
    rank_power=0.40,
    threshold_grid=(0.50,),
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
    mlp_pca_dim=128,
    mlp_alpha_blend=0.40,
    mlp_min_pos=5,
    n_sites_cap=20,
)


def _make(**kwargs):
    out = dict(BASELINE)
    out.update(kwargs)
    return out


# Knobs that don't map onto our config surface (silently dropped from LB_093):
#   mixup_alpha, use_cosine_restart, restart_period, proto_margin, val_ratio.
LB_093 = _make(
    name="lb_093_port",
    lambda_prior=0.4,
    correction_weight=0.30,
    loss="bce_focal_mean",
    focal_gamma=2.5,
    mlp_pca_dim=64,
    smoothing="adaptive",
    smoothing_alpha=0.20,
    use_boost=False,
    threshold_grid=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    proto_n_epochs=40,
    proto_lr=1e-3,
    proto_patience=8,
    residual_n_epochs=30,
    residual_lr=1e-3,
    residual_patience=8,
)


SWEEP_CHEAP_WINS = [
    _make(name="baseline"),
    _make(name="focal_bce_waveform_shift", tta="waveform_shift"),
    _make(name="bce_posw_legacy", loss="bce_posw"),
    _make(name="no_boost", use_boost=False),
    _make(name="adaptive_smooth", smoothing="adaptive"),
    _make(name="gaussian_plus_boost", smoothing="gaussian", use_boost=True,
          tta="waveform_shift"),
    _make(name="bce_focal_mean", loss="bce_focal_mean"),
    _make(name="boost_strong", boost_lift=0.40, boost_threshold=0.4),
]


SWEEP_BEST_SSM = [
    _make(name="baseline"),
    # ensemble_w
    _make(name="ens_w_000_mlp_only", ensemble_w=0.00),
    _make(name="ens_w_015", ensemble_w=0.15),
    _make(name="ens_w_025", ensemble_w=0.25),
    _make(name="ens_w_075_proto_heavy", ensemble_w=0.75),
    # MLP probes
    _make(name="mlp_alpha_060", mlp_alpha_blend=0.60),
    _make(name="mlp_alpha_080", mlp_alpha_blend=0.80),
    _make(name="mlp_pca_32", mlp_pca_dim=32),
    _make(name="mlp_pca_128", mlp_pca_dim=128),
    _make(name="mlp_minpos_3", mlp_min_pos=3),
    # ProtoSSM capacity
    _make(name="proto_short", proto_n_epochs=30, proto_patience=10),
    _make(name="proto_distill_heavy", proto_distill_weight=0.30),
    # Loss shape
    _make(name="focal_g1", focal_gamma=1.0),
    _make(name="focal_g3", focal_gamma=3.0),
    _make(name="focal_a010", focal_alpha=0.10),
    _make(name="focal_a050", focal_alpha=0.50),
    _make(name="ls_0", label_smoothing=0.0),
    _make(name="ls_10", label_smoothing=0.10),
    # Post-processing
    _make(name="boost_soft", boost_lift=0.15),
    _make(name="no_scaling", file_conf_power=0.0, rank_power=0.0),
    _make(name="smooth_off", smoothing="none"),
    # Stacked combos
    _make(name="mlp_heavy_combo",
          ensemble_w=0.25, mlp_alpha_blend=0.60, boost_lift=0.15),
    _make(name="mlp_only_clean",
          ensemble_w=0.0, file_conf_power=0.0, rank_power=0.0,
          smoothing="none", use_boost=False),
    _make(name="proto_short_mlp_heavy",
          proto_n_epochs=30, proto_patience=10,
          ensemble_w=0.25, mlp_alpha_blend=0.60),
]


SWEEP_LB_093 = [
    _make(name="baseline"),
    LB_093,
]
