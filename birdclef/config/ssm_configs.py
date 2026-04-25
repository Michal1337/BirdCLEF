"""Sweep grids for the Perch+SSM head pipeline (train/train_ssm_head.py).

Each config is merged with BASELINE. The hparams-diff CSV auto-drops fields
that are constant across the sweep — so you only see the knobs that move.

Current BASELINE numbers (legacy v_anchor era; rerun under stitched OOF when
the new metric is the primary ranker):
    baseline: site_std=0.1212, macro_oof=0.9160
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
    mlp_pca_dim=128,
    mlp_alpha_blend=0.40,
    mlp_min_pos=5,
    n_sites_cap=20,
)


def _make(**kwargs):
    out = dict(BASELINE)
    out.update(kwargs)
    return out


SWEEP_BASELINE = [_make(name="baseline")]


# ── LB_093 port ────────────────────────────────────────────────────────────
# Exact knobs from LB_093.ipynb (the load-bearing 0.93 LB notebook), mapped
# onto our config surface. See STRATEGY_V2.md §10 for the full reference
# table. Only fields where LB_093 differs from BASELINE are listed below;
# everything else (focal_alpha, label_smoothing, mlp_alpha_blend, etc.)
# inherits from BASELINE since the LB_093 values match.
#
# Knobs that DO NOT MAP onto our current surface (silently dropped):
#   - mixup_alpha=0.4  (ProtoSSM training-time mixup; not exposed)
#   - use_cosine_restart=True, restart_period=20
#   - proto_margin=0.15, val_ratio=0.15  (internal ProtoSSM regulariser)
# If sweeping shows LB_093 substantially beats BASELINE on stitched OOF,
# wire these into models/ssm.py and re-add them here.
LB_093 = _make(
    name="lb_093_port",
    # Site/hour prior: LB_093 lifts raw Perch logits with a 0.4-weighted
    # log-odds shift. BASELINE has prior=0 since it hurt v_anchor (which we
    # now know was a misleading metric).
    lambda_prior=0.4,
    # ResidualSSM: LB_093 trains it and applies at correction_weight=0.30.
    # BASELINE neutralised it (=0) for the same v_anchor reason.
    correction_weight=0.30,
    # Loss: LB_093 uses BCE + focal combo with focal_gamma=2.5. Closest
    # match in our registry is `bce_focal_mean` (50/50 BCE+Focal, 2024 #2).
    loss="bce_focal_mean",
    focal_gamma=2.5,
    # MLP probes: LB_093 uses smaller PCA but same alpha_blend / min_pos.
    mlp_pca_dim=64,
    # Post-processing: LB_093 chains
    #   file_confidence_scale(top_k=2, power=0.4)
    #   rank_aware_scaling(power=0.4)
    #   adaptive_delta_smooth(base_alpha=0.20)
    #   apply_per_class_thresholds (isotonic + grid [0.25..0.70])
    # No "hard soundscape boost" — that's a different 2024-winner trick we
    # added under our `use_boost`. LB_093 doesn't use it.
    smoothing="adaptive",
    smoothing_alpha=0.20,
    use_boost=False,
    threshold_grid=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    # Training schedules: LB_093 ProtoSSM call site overrides cfg defaults
    # to n_epochs=40, lr=1e-3, patience=8 (cell 25). ResidualSSM call site
    # uses n_epochs=30, lr=1e-3, patience=8.
    proto_n_epochs=40,
    proto_lr=1e-3,
    proto_patience=8,
    residual_n_epochs=30,
    residual_lr=1e-3,
    residual_patience=8,
)


# Direct A/B: LB_093 port vs current BASELINE under the new stitched-OOF
# metric. If LB_093 wins by ≥0.005 on stitched-OOF, the prior=0/corr=0
# cleanups were genuinely a regression and we should ship the LB_093 port
# as the new baseline. If they tie, BASELINE is preferred (simpler).
SWEEP_LB_093 = [
    _make(name="baseline"),
    LB_093,
]


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
