"""Config presets for residual_ssm_config_sweep.py.

Each entry is consumed by run_residual_sweep via residual_ssm_config_sweep.py.
Keys: name, d_model, d_state, dropout, meta_dim, n_sites,
      n_epochs, patience, lr, correction_weight.
"""

RESIDUAL_SSM_SWEEP_CONFIGS = [
    # ── Baseline ────────────────────────────────────────────────────────
    {
        "name": "base_d64_s8",
        "d_model": 64, "d_state": 8, "dropout": 0.10, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 30, "patience": 8, "lr": 1e-3, "correction_weight": 0.30,
    },
    # ── Tiny ────────────────────────────────────────────────────────────
    {
        "name": "tiny_d32_s8",
        "d_model": 32, "d_state": 8, "dropout": 0.08, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 30, "patience": 8, "lr": 1.2e-3, "correction_weight": 0.25,
    },
    # ── Medium ──────────────────────────────────────────────────────────
    {
        "name": "med_d96_s16",
        "d_model": 96, "d_state": 16, "dropout": 0.12, "meta_dim": 12, "n_sites": 20,
        "n_epochs": 35, "patience": 10, "lr": 8e-4, "correction_weight": 0.30,
    },
    # ── Large ───────────────────────────────────────────────────────────
    {
        "name": "large_d128_s16",
        "d_model": 128, "d_state": 16, "dropout": 0.15, "meta_dim": 16, "n_sites": 20,
        "n_epochs": 40, "patience": 10, "lr": 7e-4, "correction_weight": 0.30,
    },
    # ── Wide state ──────────────────────────────────────────────────────
    {
        "name": "base_d64_s24",
        "d_model": 64, "d_state": 24, "dropout": 0.10, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 35, "patience": 10, "lr": 1e-3, "correction_weight": 0.30,
    },
    # ── High correction weight ───────────────────────────────────────────
    {
        "name": "base_d64_s8_cw50",
        "d_model": 64, "d_state": 8, "dropout": 0.10, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 30, "patience": 8, "lr": 1e-3, "correction_weight": 0.50,
    },
    # ── Low correction weight ────────────────────────────────────────────
    {
        "name": "base_d64_s8_cw15",
        "d_model": 64, "d_state": 8, "dropout": 0.10, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 30, "patience": 8, "lr": 1e-3, "correction_weight": 0.15,
    },
    # ── Fast LR ─────────────────────────────────────────────────────────
    {
        "name": "base_d64_s8_fastlr",
        "d_model": 64, "d_state": 8, "dropout": 0.10, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 25, "patience": 6, "lr": 2e-3, "correction_weight": 0.30,
    },
    # ── Long training ────────────────────────────────────────────────────
    {
        "name": "base_d64_s8_long",
        "d_model": 64, "d_state": 8, "dropout": 0.10, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 60, "patience": 15, "lr": 7e-4, "correction_weight": 0.30,
    },
    # ── Wide model, low correction ───────────────────────────────────────
    {
        "name": "wide_d192_s16_cw20",
        "d_model": 192, "d_state": 16, "dropout": 0.18, "meta_dim": 24, "n_sites": 20,
        "n_epochs": 45, "patience": 12, "lr": 6e-4, "correction_weight": 0.20,
    },
    # ── High dropout ─────────────────────────────────────────────────────
    {
        "name": "base_d64_s8_drop25",
        "d_model": 64, "d_state": 8, "dropout": 0.25, "meta_dim": 8, "n_sites": 20,
        "n_epochs": 35, "patience": 10, "lr": 1e-3, "correction_weight": 0.30,
    },
    # ── Large + high correction ──────────────────────────────────────────
    {
        "name": "large_d128_s24_cw40",
        "d_model": 128, "d_state": 24, "dropout": 0.15, "meta_dim": 16, "n_sites": 20,
        "n_epochs": 45, "patience": 12, "lr": 6e-4, "correction_weight": 0.40,
    },
    # ── XL ───────────────────────────────────────────────────────────────
    {
        "name": "xl_d256_s32",
        "d_model": 256, "d_state": 32, "dropout": 0.20, "meta_dim": 32, "n_sites": 24,
        "n_epochs": 50, "patience": 12, "lr": 5e-4, "correction_weight": 0.30,
    },
    # ── XL + high correction ─────────────────────────────────────────────
    {
        "name": "xl_d256_s32_cw50",
        "d_model": 256, "d_state": 32, "dropout": 0.22, "meta_dim": 32, "n_sites": 24,
        "n_epochs": 55, "patience": 14, "lr": 4.5e-4, "correction_weight": 0.50,
    },
    # ── XXL ──────────────────────────────────────────────────────────────
    {
        "name": "xxl_d384_s32",
        "d_model": 384, "d_state": 32, "dropout": 0.25, "meta_dim": 48, "n_sites": 24,
        "n_epochs": 55, "patience": 14, "lr": 4e-4, "correction_weight": 0.30,
    },
    # ── XXL wide state ───────────────────────────────────────────────────
    {
        "name": "xxl_d384_s48",
        "d_model": 384, "d_state": 48, "dropout": 0.25, "meta_dim": 48, "n_sites": 24,
        "n_epochs": 60, "patience": 15, "lr": 3.5e-4, "correction_weight": 0.30,
    },
    # ── Mega ─────────────────────────────────────────────────────────────
    {
        "name": "mega_d512_s32",
        "d_model": 512, "d_state": 32, "dropout": 0.28, "meta_dim": 64, "n_sites": 24,
        "n_epochs": 60, "patience": 15, "lr": 3e-4, "correction_weight": 0.30,
    },
    # ── Mega + low correction ────────────────────────────────────────────
    {
        "name": "mega_d512_s32_cw20",
        "d_model": 512, "d_state": 32, "dropout": 0.30, "meta_dim": 64, "n_sites": 24,
        "n_epochs": 60, "patience": 15, "lr": 3e-4, "correction_weight": 0.20,
    },
    # ── Wide state only (small model) ────────────────────────────────────
    {
        "name": "small_d64_s48",
        "d_model": 64, "d_state": 48, "dropout": 0.12, "meta_dim": 16, "n_sites": 20,
        "n_epochs": 40, "patience": 10, "lr": 9e-4, "correction_weight": 0.30,
    },
    # ── Deep-ish state, medium model ─────────────────────────────────────
    {
        "name": "med_d96_s32",
        "d_model": 96, "d_state": 32, "dropout": 0.15, "meta_dim": 16, "n_sites": 20,
        "n_epochs": 45, "patience": 12, "lr": 7e-4, "correction_weight": 0.30,
    },
    # ── Ultra-wide meta embeddings ───────────────────────────────────────
    {
        "name": "xl_d256_meta64",
        "d_model": 256, "d_state": 24, "dropout": 0.20, "meta_dim": 64, "n_sites": 24,
        "n_epochs": 50, "patience": 12, "lr": 5e-4, "correction_weight": 0.30,
    },
    # ── Large + long train + low correction ──────────────────────────────
    {
        "name": "large_d192_s24_long_cw20",
        "d_model": 192, "d_state": 24, "dropout": 0.18, "meta_dim": 24, "n_sites": 24,
        "n_epochs": 80, "patience": 20, "lr": 4e-4, "correction_weight": 0.20,
    },
    # ── XL + aggressive correction ───────────────────────────────────────
    {
        "name": "xl_d256_s24_cw60",
        "d_model": 256, "d_state": 24, "dropout": 0.22, "meta_dim": 32, "n_sites": 24,
        "n_epochs": 50, "patience": 12, "lr": 5e-4, "correction_weight": 0.60,
    },
    # ── Very high dropout, large model ───────────────────────────────────
    {
        "name": "xl_d256_s32_drop35",
        "d_model": 256, "d_state": 32, "dropout": 0.35, "meta_dim": 32, "n_sites": 24,
        "n_epochs": 55, "patience": 14, "lr": 5e-4, "correction_weight": 0.30,
    },
    # ── Fat meta, smaller model ──────────────────────────────────────────
    {
        "name": "med_d128_meta48",
        "d_model": 128, "d_state": 16, "dropout": 0.15, "meta_dim": 48, "n_sites": 24,
        "n_epochs": 40, "patience": 10, "lr": 7e-4, "correction_weight": 0.30,
    },
    # ── Extreme state width ──────────────────────────────────────────────
    {
        "name": "large_d128_s64",
        "d_model": 128, "d_state": 64, "dropout": 0.20, "meta_dim": 16, "n_sites": 24,
        "n_epochs": 50, "patience": 12, "lr": 5e-4, "correction_weight": 0.30,
    },
    # ── Slow LR, large model, long train ─────────────────────────────────
    {
        "name": "xl_d256_s32_slowlr",
        "d_model": 256, "d_state": 32, "dropout": 0.25, "meta_dim": 32, "n_sites": 24,
        "n_epochs": 80, "patience": 20, "lr": 2.5e-4, "correction_weight": 0.30,
    },
    # ── Small but long-trained + patient ─────────────────────────────────
    {
        "name": "tiny_d48_s16_patient",
        "d_model": 48, "d_state": 16, "dropout": 0.10, "meta_dim": 12, "n_sites": 20,
        "n_epochs": 70, "patience": 18, "lr": 8e-4, "correction_weight": 0.25,
    },
]
