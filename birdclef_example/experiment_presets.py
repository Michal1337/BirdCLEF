"""Experiment presets for score tuning.

Only architecture, training, threshold, and score-fusion hyperparameters are varied.
No paths, cache locations, or file IO settings are modified here.
"""

KEY_HYPERPARAMETERS = [
	"proto_ssm.d_model",
	"proto_ssm.d_state",
	"proto_ssm.n_ssm_layers",
	"proto_ssm.dropout",
	"proto_ssm.cross_attn_heads",
	"proto_ssm_train.lr",
	"proto_ssm_train.weight_decay",
	"proto_ssm_train.n_epochs",
	"proto_ssm_train.patience",
	"proto_ssm_train.mixup_alpha",
	"proto_ssm_train.focal_gamma",
	"proto_ssm_train.label_smoothing",
	"proto_ssm_train.swa_start_frac",
	"best_fusion.lambda_event",
	"best_fusion.lambda_texture",
	"best_fusion.lambda_proxy_texture",
	"best_fusion.smooth_texture",
	"best_fusion.smooth_event",
	"frozen_best_probe.C",
	"frozen_best_probe.alpha",
	"frozen_best_probe.pca_dim",
	"residual_ssm.correction_weight",
	"temperature.aves",
	"temperature.texture",
	"file_level_top_k",
	"tta_shifts",
	"rank_aware_power",
	"delta_shift_alpha",
	"threshold_grid",
]

EXPERIMENT_PRESETS = {
	# "baseline_v18": {},
	# # V20: Radical architecture space
	# "arch_tiny_fast": {
	# 	"proto_ssm": {"d_model": 112, "d_state": 12, "n_ssm_layers": 2, "dropout": 0.08, "cross_attn_heads": 4},
	# 	"proto_ssm_train": {"n_epochs": 36, "patience": 8},
	# 	"residual_ssm": {"d_model": 56, "d_state": 8, "n_ssm_layers": 1},
	# },
	# "arch_huge_640": {
	# 	"proto_ssm": {"d_model": 640, "d_state": 80, "n_ssm_layers": 5, "dropout": 0.16, "cross_attn_heads": 16},
	# 	"proto_ssm_train": {"n_epochs": 120, "patience": 32},
	# 	"residual_ssm": {"d_model": 256, "d_state": 24, "n_ssm_layers": 3},
	# },
	# "arch_deep_8": {
	# 	"proto_ssm": {"d_model": 288, "d_state": 40, "n_ssm_layers": 8, "dropout": 0.18, "cross_attn_heads": 8},
	# 	"proto_ssm_train": {"n_epochs": 130, "patience": 34},
	# 	"residual_ssm": {"n_ssm_layers": 4, "dropout": 0.14},
	# },
	# "arch_wide_shallow": {
	# 	"proto_ssm": {"d_model": 576, "d_state": 48, "n_ssm_layers": 2, "dropout": 0.10, "cross_attn_heads": 12},
	# 	"proto_ssm_train": {"n_epochs": 88, "patience": 22},
	# 	"residual_ssm": {"d_model": 224},
	# },
	# # V20: Extreme optimization regimes
	# "train_ultra_conservative": {
	# 	"proto_ssm_train": {
	# 		"lr": 2.5e-4,
	# 		"swa_lr": 1.2e-4,
	# 		"weight_decay": 2.0e-4,
	# 		"mixup_alpha": 0.05,
	# 		"focal_gamma": 1.2,
	# 		"label_smoothing": 0.00,
	# 		"n_epochs": 70,
	# 		"patience": 16,
	# 	},
	# 	"residual_ssm": {"lr": 2.5e-4},
	# },
	# "train_ultra_aggressive": {
	# 	"proto_ssm_train": {
	# 		"lr": 1.5e-3,
	# 		"swa_lr": 7.5e-4,
	# 		"weight_decay": 2.5e-3,
	# 		"mixup_alpha": 0.85,
	# 		"focal_gamma": 4.6,
	# 		"label_smoothing": 0.10,
	# 		"pos_weight_cap": 45.0,
	# 		"n_epochs": 110,
	# 		"patience": 24,
	# 	},
	# 	"residual_ssm": {"lr": 1.3e-3, "dropout": 0.14},
	# },
	# "train_distill_max": {
	# 	"proto_ssm_train": {
	# 		"distill_weight": 0.35,
	# 		"mixup_alpha": 0.30,
	# 		"focal_gamma": 2.0,
	# 		"label_smoothing": 0.02,
	# 	},
	# },
	# "train_distill_min": {
	# 	"proto_ssm_train": {
	# 		"distill_weight": 0.02,
	# 		"mixup_alpha": 0.55,
	# 		"focal_gamma": 3.6,
	# 		"label_smoothing": 0.07,
	# 	},
	# },
	# # V20: Radical fusion/temperature/postprocess
	# "fusion_prior_dominant": {
	# 	"best_fusion": {
	# 		"lambda_event": 0.95,
	# 		"lambda_texture": 2.10,
	# 		"lambda_proxy_texture": 1.70,
	# 		"smooth_texture": 0.58,
	# 		"smooth_event": 0.28,
	# 	},
	# 	"temperature": {"aves": 0.95, "texture": 0.78},
	# },
	# "fusion_model_dominant": {
	# 	"best_fusion": {
	# 		"lambda_event": 0.12,
	# 		"lambda_texture": 0.28,
	# 		"lambda_proxy_texture": 0.18,
	# 		"smooth_texture": 0.12,
	# 		"smooth_event": 0.06,
	# 	},
	# 	"temperature": {"aves": 1.25, "texture": 1.20},
	# },
	# "fusion_event_heavy": {
	# 	"best_fusion": {
	# 		"lambda_event": 1.10,
	# 		"lambda_texture": 0.55,
	# 		"lambda_proxy_texture": 0.35,
	# 		"smooth_texture": 0.16,
	# 		"smooth_event": 0.34,
	# 	},
	# },
	# "fusion_texture_heavy": {
	# 	"best_fusion": {
	# 		"lambda_event": 0.28,
	# 		"lambda_texture": 2.35,
	# 		"lambda_proxy_texture": 1.90,
	# 		"smooth_texture": 0.62,
	# 		"smooth_event": 0.09,
	# 	},
	# },
	# "postprocess_extreme_recall": {
	# 	"threshold_grid": [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30],
	# 	"file_level_top_k": 4,
	# 	"rank_aware_power": 0.72,
	# 	"delta_shift_alpha": 0.30,
	# 	"tta_shifts": [0, 1, -1, 2, -2, 3, -3],
	# 	"frozen_best_probe": {"alpha": 0.62, "C": 1.20, "pca_dim": 192, "min_pos": 4},
	# },
	# "postprocess_extreme_precision": {
	# 	"threshold_grid": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
	# 	"file_level_top_k": 1,
	# 	"rank_aware_power": 0.12,
	# 	"delta_shift_alpha": 0.06,
	# 	"tta_shifts": [0],
	# 	"frozen_best_probe": {"alpha": 0.30, "C": 0.35, "pca_dim": 64, "min_pos": 10},
	# },
	"probe_max_capacity": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 3, "C": 1.8, "alpha": 0.65},
		"mlp_params": {"hidden_layer_sizes": (384, 192), "alpha": 0.002, "learning_rate_init": 8.0e-4},
	},
	# "probe_hard_regularized": {
	# 	"frozen_best_probe": {"pca_dim": 48, "min_pos": 12, "C": 0.15, "alpha": 0.22},
	# 	"mlp_params": {"hidden_layer_sizes": (128, 64), "alpha": 0.02, "learning_rate_init": 2.5e-4},
	# },
	# "residual_dominant": {
	# 	"residual_ssm": {
	# 		"correction_weight": 0.65,
	# 		"d_model": 192,
	# 		"d_state": 24,
	# 		"n_ssm_layers": 3,
	# 		"n_epochs": 60,
	# 		"patience": 18,
	# 	},
	# },
	# "residual_minimal": {
	# 	"residual_ssm": {
	# 		"correction_weight": 0.10,
	# 		"d_model": 64,
	# 		"d_state": 8,
	# 		"n_ssm_layers": 1,
	# 		"n_epochs": 24,
	# 		"patience": 8,
	# 	},
	# },
	"probe_wide_deep": {
		"frozen_best_probe": {"pca_dim": 256, "min_pos": 3, "C": 2.2, "alpha": 0.72},
		"mlp_params": {"hidden_layer_sizes": (512, 256), "alpha": 0.0015, "learning_rate_init": 1.0e-3},
	},
	"probe_wide_deep_no_es": {
		"frozen_best_probe": {"pca_dim": 256, "min_pos": 3, "C": 2.5, "alpha": 0.75},
		"mlp_params": {"hidden_layer_sizes": (512, 256), "alpha": 0.001, "learning_rate_init": 1.2e-3, "early_stopping": False},
	},
	"probe_high_bias": {
		"frozen_best_probe": {"pca_dim": 192, "min_pos": 4, "C": 3.0, "alpha": 0.80},
		"mlp_params": {"hidden_layer_sizes": (384, 192), "alpha": 0.0015, "learning_rate_init": 9.0e-4, "activation": "relu"},
	},
	"probe_low_bias": {
		"frozen_best_probe": {"pca_dim": 160, "min_pos": 6, "C": 0.9, "alpha": 0.45},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.006, "learning_rate_init": 4.0e-4},
	},
	"probe_mid_reg": {
		"frozen_best_probe": {"pca_dim": 128, "min_pos": 5, "C": 1.0, "alpha": 0.55},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "learning_rate_init": 5.0e-4},
	},
	"probe_hard_reg": {
		"frozen_best_probe": {"pca_dim": 96, "min_pos": 8, "C": 0.35, "alpha": 0.30},
		"mlp_params": {"hidden_layer_sizes": (128, 64), "alpha": 0.02, "learning_rate_init": 2.5e-4},
	},
	"probe_ultra_reg": {
		"frozen_best_probe": {"pca_dim": 64, "min_pos": 10, "C": 0.15, "alpha": 0.20},
		"mlp_params": {"hidden_layer_sizes": (128, 64), "alpha": 0.05, "learning_rate_init": 1.5e-4, "early_stopping": True},
	},
	"probe_relu_sweep": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 4, "C": 1.5, "alpha": 0.60},
		"mlp_params": {"hidden_layer_sizes": (384, 128), "activation": "relu", "alpha": 0.003, "learning_rate_init": 7.0e-4},
	},
	"probe_tanh_sweep": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 4, "C": 1.5, "alpha": 0.60},
		"mlp_params": {"hidden_layer_sizes": (384, 128), "activation": "tanh", "alpha": 0.003, "learning_rate_init": 7.0e-4},
	},
	"probe_lr_high": {
		"frozen_best_probe": {"pca_dim": 192, "min_pos": 4, "C": 1.4, "alpha": 0.58},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.002, "learning_rate_init": 1.5e-3},
	},
	"probe_lr_low": {
		"frozen_best_probe": {"pca_dim": 192, "min_pos": 4, "C": 1.4, "alpha": 0.58},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.002, "learning_rate_init": 2.0e-4},
	},
	"probe_more_iter": {
		"frozen_best_probe": {"pca_dim": 256, "min_pos": 3, "C": 2.0, "alpha": 0.68},
		"mlp_params": {"hidden_layer_sizes": (512, 256), "alpha": 0.0015, "learning_rate_init": 8.0e-4, "max_iter": 1200},
	},
	"probe_less_iter": {
		"frozen_best_probe": {"pca_dim": 128, "min_pos": 6, "C": 0.7, "alpha": 0.42},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.005, "learning_rate_init": 5.0e-4, "max_iter": 250},
	},
	"probe_big_dropout_bias": {
		"frozen_best_probe": {"pca_dim": 160, "min_pos": 5, "C": 2.0, "alpha": 0.70},
		"mlp_params": {"hidden_layer_sizes": (384, 192), "alpha": 0.001, "learning_rate_init": 9.0e-4, "validation_fraction": 0.25},
	},
	"probe_small_dropout_bias": {
		"frozen_best_probe": {"pca_dim": 160, "min_pos": 5, "C": 1.0, "alpha": 0.50},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "learning_rate_init": 5.0e-4, "validation_fraction": 0.10},
	},
	"probe_class_focus_lowpos": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 2, "C": 1.2, "alpha": 0.64},
		"mlp_params": {"hidden_layer_sizes": (384, 192), "alpha": 0.0025, "learning_rate_init": 8.0e-4},
	},
	"probe_class_focus_highpos": {
		"frozen_best_probe": {"pca_dim": 192, "min_pos": 12, "C": 0.8, "alpha": 0.50},
		"mlp_params": {"hidden_layer_sizes": (384, 192), "alpha": 0.004, "learning_rate_init": 6.0e-4},
	},
	"probe_alpha_high": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 3, "C": 1.8, "alpha": 0.85},
		"mlp_params": {"hidden_layer_sizes": (512, 256), "alpha": 0.001, "learning_rate_init": 1.0e-3},
	},
	"probe_alpha_low": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 3, "C": 1.8, "alpha": 0.25},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "learning_rate_init": 5.0e-4},
	},
	"probe_minimal_pca": {
		"frozen_best_probe": {"pca_dim": 32, "min_pos": 8, "C": 0.5, "alpha": 0.30},
		"mlp_params": {"hidden_layer_sizes": (128, 64), "alpha": 0.02, "learning_rate_init": 3.0e-4},
	},
	"probe_max_pca": {
		"frozen_best_probe": {"pca_dim": 320, "min_pos": 3, "C": 2.5, "alpha": 0.75},
		"mlp_params": {"hidden_layer_sizes": (512, 256), "alpha": 0.001, "learning_rate_init": 1.0e-3},
	},
	"probe_no_early_stop": {
		"frozen_best_probe": {"pca_dim": 224, "min_pos": 3, "C": 1.5, "alpha": 0.62},
		"mlp_params": {"hidden_layer_sizes": (384, 192), "alpha": 0.002, "learning_rate_init": 8.0e-4, "early_stopping": False},
	},
	"probe_strong_early_stop": {
		"frozen_best_probe": {"pca_dim": 128, "min_pos": 6, "C": 0.8, "alpha": 0.48},
		"mlp_params": {"hidden_layer_sizes": (256, 128), "alpha": 0.006, "learning_rate_init": 5.0e-4, "early_stopping": True, "n_iter_no_change": 10},
	},
	"probe_even_more_capacity": {
		"frozen_best_probe": {"pca_dim": 288, "min_pos": 3, "C": 2.8, "alpha": 0.78},
		"mlp_params": {"hidden_layer_sizes": (768, 384), "alpha": 0.001, "learning_rate_init": 1.2e-3, "max_iter": 1500},
	},
}

