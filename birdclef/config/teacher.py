"""Frozen teacher configs for pseudo-labeling.

Each entry pins down the exact SSM pipeline config + random seeds used to
generate a pseudo-label artifact. The teacher name becomes part of the
artifact metadata so you can always trace which config produced which round.

Bumping the teacher: copy the previous entry, give it a new name, edit, and
update TEACHER_LATEST. Don't mutate old entries — they're historical records.
"""
from __future__ import annotations


TEACHER_V0 = dict(
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
    smoothing="gaussian",
    smoothing_alpha=0.20,
    use_boost=True,
    boost_threshold=0.5,
    boost_lift=0.25,
    file_conf_top_k=2,
    file_conf_power=0.40,
    rank_power=0.40,
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
    name="ssm_pca128_teacher",
)

# Multi-seed ensemble — averages out the ±0.02 seed variance we measured on
# V-anchor AUC. Each seed trains its own SSM pipeline on the non-anchor
# labeled rows and runs inference on every soundscape file; the final
# pseudo-label is the mean across seeds.
TEACHER_V0_SEEDS = (42, 7, 13)
TEACHER_V0_NAME = "ssm_pca128_ens3"


TEACHER_LATEST = {
    "name": TEACHER_V0_NAME,
    "config": TEACHER_V0,
    "seeds": list(TEACHER_V0_SEEDS),
}
