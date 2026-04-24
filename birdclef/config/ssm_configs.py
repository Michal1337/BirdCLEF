"""Sweep grids for the Perch+SSM head pipeline (train/train_ssm_head.py).

Each config is merged with BASELINE. The hparams-diff CSV auto-drops fields
that are constant across the sweep — so you only see the knobs that move.

Current BASELINE winners (from cheap_wins):
    baseline: v_anchor=0.8730, site_std=0.1212, macro_oof=0.9160, primary=0.7518
Findings carried forward:
    loss="focal_bce"    (beats bce_posw and bce_focal_mean)
    tta="window_roll"   (beats waveform_shift at 46-file scale)
    smoothing="gaussian" + use_boost=True  (best post-processing)
    correction_weight=0 + threshold_grid=(0.5,)  (already neutralized)

SWEEP_BEST_SSM is the working sweep focused on turning baseline into the best
teacher for pseudo-labeling round 0. Keeps the winning "shape" of the pipeline
and explores only knobs that still have unexplored signal.
"""
from __future__ import annotations

BASELINE = dict(
    # Reproducibility — the sweep runner reseeds python/numpy/torch at the
    # start of each config with this value; per-fold seeds derive as seed+f+1.
    seed=42,
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
    smoothing="gaussian",          # or "adaptive" / "none"
    smoothing_alpha=0.20,
    use_boost=True,
    boost_threshold=0.5,
    boost_lift=0.25,
    file_conf_top_k=2,
    file_conf_power=0.40,
    rank_power=0.40,
    # Thresholds (identity when single-value grid)
    threshold_grid=(0.50,),
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
    _make(name="gaussian_plus_boost", smoothing="gaussian", use_boost=True,
          tta="waveform_shift"),
    _make(name="bce_focal_mean", loss="bce_focal_mean"),
    _make(name="boost_strong", boost_lift=0.40, boost_threshold=0.4),
]


# ── Target: best teacher for pseudo-labeling round 0 ───────────────────────
# Every config starts from the cheap_wins winner ("baseline"). Groups target
# the four remaining unexplored axes: (A) proto↔MLP blend, (B) MLP probe
# knobs, (C) ProtoSSM regularization, (D) loss shape, (E) post-processing,
# (F) combined best-guesses.
SWEEP_BEST_SSM = [
    # Reference
    _make(name="baseline"),

    # ── (A) ensemble_w: earlier sweeps showed MLP stage > proto stage ─────
    # proto_only was 0.73 vs prior_mlp 0.88 in the old sweep; reduce proto.
    _make(name="ens_w_000_mlp_only",  ensemble_w=0.00),
    _make(name="ens_w_015",           ensemble_w=0.15),
    _make(name="ens_w_025",           ensemble_w=0.25),
    _make(name="ens_w_075_proto_heavy", ensemble_w=0.75),

    # ── (B) MLP-probe regularization ──────────────────────────────────────
    # mlp_alpha_blend: within MLP branch, how much to trust the probe vs raw
    _make(name="mlp_alpha_060",  mlp_alpha_blend=0.60),
    _make(name="mlp_alpha_080",  mlp_alpha_blend=0.80),
    _make(name="mlp_pca_32",     mlp_pca_dim=32),      # stronger compression
    _make(name="mlp_pca_128",    mlp_pca_dim=128),     # more capacity
    _make(name="mlp_minpos_3",   mlp_min_pos=3),       # more active probes

    # ── (C) ProtoSSM capacity (less overfit on 46 files) ──────────────────
    _make(name="proto_short",        proto_n_epochs=30, proto_patience=10),
    _make(name="proto_distill_heavy", proto_distill_weight=0.30),

    # ── (D) loss shape around focal_bce ───────────────────────────────────
    _make(name="focal_g1",   focal_gamma=1.0),
    _make(name="focal_g3",   focal_gamma=3.0),
    _make(name="focal_a010", focal_alpha=0.10),
    _make(name="focal_a050", focal_alpha=0.50),
    _make(name="ls_0",       label_smoothing=0.0),    # no smoothing
    _make(name="ls_10",      label_smoothing=0.10),   # heavy smoothing

    # ── (E) post-processing around gaussian+boost ─────────────────────────
    _make(name="boost_soft",     boost_lift=0.15),
    _make(name="no_scaling",     file_conf_power=0.0, rank_power=0.0),
    _make(name="smooth_off",     smoothing="none"),

    # ── (F) stacked best-guess combos ─────────────────────────────────────
    _make(name="mlp_heavy_combo",          # proto weak + MLP trust high
          ensemble_w=0.25, mlp_alpha_blend=0.60, boost_lift=0.15),
    _make(name="mlp_only_clean",           # raw prior_mlp style output
          ensemble_w=0.0, file_conf_power=0.0, rank_power=0.0,
          smoothing="none", use_boost=False),
    _make(name="proto_short_mlp_heavy",    # less capacity + rely on MLP
          proto_n_epochs=30, proto_patience=10,
          ensemble_w=0.25, mlp_alpha_blend=0.60),
]
