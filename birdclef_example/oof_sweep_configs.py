"""Predefined full-stack OOF sweep configurations.

Each entry is applied with CFG.update(sweep_cfg) before one sweep run.
Fold count comes from birdclef/splits/folds_site_date.parquet; change it via
    python -m birdclef.scripts._02_build_splits --n-splits <k>

Design (informed by the prior_off sweep win):
    no-cache: prior_off=0.8624 final; prior_mlp stage alone=0.8783
    cache:    prior_off=0.9019 final; prior_mlp stage alone=0.9167
    → site/hour prior hurts on 46 files. Threshold calibration hurts −0.01–0.03.
      Post-scaling + smoothing are roughly neutral-to-mild-positive.

New baseline is `prior_off`; sweep explores what else helps on top of it.

Sweep-config keys honoured by run_pipeline_oof_fullstack (CFG.get):
    ensemble_w, lambda_prior, correction_weight,
    mlp_alpha_blend, mlp_min_pos, mlp_pca_dim,
    file_conf_top_k, file_conf_power, rank_power, smooth_alpha,
    tta_shifts, threshold_grid,
    proto_n_epochs, proto_patience, proto_lr,
    residual_n_epochs, residual_patience, residual_lr
"""

# Every config here implicitly uses lambda_prior=0.0 via the "prior_off" base.
# If/when you change CFG.lambda_prior to 0.0 in the CFG dict itself, remove
# this field from each entry below.
_PRIOR_OFF = {"lambda_prior": 0.0}


def _cfg(name, **overrides):
    return {**_PRIOR_OFF, "name": name, **overrides}


OOF_SWEEP_CONFIGS = [
    # ── Group 0: reference points ───────────────────────────────────────────
    _cfg("prior_off_baseline"),                                    # best from previous sweep
    {"name": "baseline_with_prior"},                               # old default for sanity

    # ── Group 1: drop threshold calibration (consistently hurts OOF) ───────
    _cfg("no_threshold", threshold_grid=[0.5]),                    # soft-threshold becomes identity
    _cfg("threshold_narrow", threshold_grid=[0.45, 0.50, 0.55]),   # tighter grid → less overfit
    _cfg("no_threshold_no_residual",
         threshold_grid=[0.5], correction_weight=0.0),

    # ── Group 2: ablate each remaining late stage individually ─────────────
    _cfg("no_residual", correction_weight=0.0),
    _cfg("no_file_conf", file_conf_power=0.0),
    _cfg("no_rank_scaling", rank_power=0.0),
    _cfg("no_smooth", smooth_alpha=0.0),

    # ── Group 3: re-weight proto↔MLP (MLP branch was strongest stage) ──────
    _cfg("mlp_heavy", ensemble_w=0.25),
    _cfg("mlp_only", ensemble_w=0.0),
    _cfg("proto_heavy", ensemble_w=0.75),

    # ── Group 4: stripped / near-prior_mlp (strongest stage alone) ─────────
    _cfg("stripped",
         correction_weight=0.0, file_conf_power=0.0, rank_power=0.0,
         smooth_alpha=0.0, threshold_grid=[0.5]),
    _cfg("prior_mlp_only",                                         # pure-MLP, no proto, no late stages
         ensemble_w=0.0,
         correction_weight=0.0, file_conf_power=0.0, rank_power=0.0,
         smooth_alpha=0.0, threshold_grid=[0.5]),

    # ── Group 5: smoother thresholding + post-scaling combos ───────────────
    _cfg("softer_post",
         file_conf_power=0.2, rank_power=0.2, smooth_alpha=0.1,
         threshold_grid=[0.5]),
    _cfg("keep_smooth_drop_rest",                                  # smoothing was roughly neutral-positive
         correction_weight=0.0, file_conf_power=0.0, rank_power=0.0,
         threshold_grid=[0.5]),

    # ── Group 6: MLP-probe regularization ──────────────────────────────────
    _cfg("mlp_alpha_high", mlp_alpha_blend=0.6),
    _cfg("mlp_alpha_low",  mlp_alpha_blend=0.2),
    _cfg("mlp_more_pca",   mlp_pca_dim=128),
    _cfg("mlp_min_pos_low", mlp_min_pos=3),                        # more active probes

    # ── Group 7: ProtoSSM capacity (less overfit on 46 files) ──────────────
    _cfg("proto_short",      proto_n_epochs=30, proto_patience=10),
    _cfg("proto_very_short", proto_n_epochs=15, proto_patience=6),
]
