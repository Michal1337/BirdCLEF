"""Torch-only probe sweep configurations for standalone OOF experiments.

This module intentionally removes sklearn probe configs.
It generates a broad grid with >100 torch configs, including no-PCA variants.
"""


def _make_name(
    pca_dim,
    hidden_dims,
    lr,
    dropout,
    min_pos,
    epochs,
    max_rows,
):
    pca_tag = "nopca" if int(pca_dim) <= 0 else f"pca{int(pca_dim)}"
    h_tag = "x".join(str(int(x)) for x in hidden_dims)
    return (
        f"torch_{pca_tag}_h{h_tag}_lr{lr:.0e}_do{int(dropout*100):02d}_"
        f"mp{min_pos}_ep{epochs}_mr{max_rows}_abgrid10"
    )


_BASE = {
    "method": "torch",
    "batch_size": 512,
    "weight_decay": 1e-4,
    "patience": 10,
    "val_fraction": 0.15,
    "standardize_features": True,
}


PROBE_SWEEP_CONFIGS = []

PCA_DIMS = [0, 32, 64, 128, 256, 384]
HIDDEN_DIMS = [
    (64,),
    (96, 48),
    (128, 64),
    (192, 96),
    (256, 128),
    (384, 192, 96),
]
LRS = [6e-4, 1e-3]
DROPOUTS = [0.05]
MIN_POS_LIST = [3, 5, 7]
STATIC_ALPHA_BLENDS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

for pca_dim in PCA_DIMS:
    for hidden_dims in HIDDEN_DIMS:
        for lr in LRS:
            for dropout in DROPOUTS:
                for min_pos in MIN_POS_LIST:
                    cfg = dict(_BASE)
                    cfg.update(
                        {
                            "name": _make_name(
                                pca_dim,
                                hidden_dims,
                                lr,
                                dropout,
                                min_pos,
                                epochs=80,
                                max_rows=3000,
                            ),
                            "pca_dim": pca_dim,
                            "alpha_blend": 0.50,
                            "alpha_blends": list(STATIC_ALPHA_BLENDS),
                            "max_rows": 3000,
                            "hidden_dims": hidden_dims,
                            "dropout": dropout,
                            "epochs": 80,
                            "lr": lr,
                            "min_pos": min_pos,
                        }
                    )
                    PROBE_SWEEP_CONFIGS.append(cfg)


_EXTRA = [
    {
        "name": "torch_extreme_nopca_large",
        "method": "torch",
        "min_pos": 3,
        "pca_dim": 0,
        "alpha_blend": 0.50,
        "alpha_blends": list(STATIC_ALPHA_BLENDS),
        "max_rows": 4200,
        "hidden_dims": (384, 192),
        "dropout": 0.15,
        "epochs": 110,
        "batch_size": 512,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "patience": 14,
        "val_fraction": 0.15,
        "standardize_features": True,
    },
    {
        "name": "torch_extreme_pca96_lightreg",
        "method": "torch",
        "min_pos": 5,
        "pca_dim": 96,
        "alpha_blend": 0.50,
        "alpha_blends": list(STATIC_ALPHA_BLENDS),
        "max_rows": 3600,
        "hidden_dims": (256, 128),
        "dropout": 0.05,
        "epochs": 120,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 5e-5,
        "patience": 16,
        "val_fraction": 0.15,
        "standardize_features": True,
    },
]

PROBE_SWEEP_CONFIGS.extend(_EXTRA)
