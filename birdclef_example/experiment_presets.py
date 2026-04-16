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
	"baseline_v18": {},
	"arch_wide_384": {
		"proto_ssm": {"d_model": 384, "cross_attn_heads": 8},
		"residual_ssm": {"d_model": 160},
	},
	"arch_compact_288": {
		"proto_ssm": {"d_model": 288, "cross_attn_heads": 6},
		"residual_ssm": {"d_model": 112},
	},
	"arch_deep_5layer": {
		"proto_ssm": {"n_ssm_layers": 5, "dropout": 0.14},
		"proto_ssm_train": {"n_epochs": 90, "patience": 24},
	},
	"arch_light_3layer": {
		"proto_ssm": {"n_ssm_layers": 3, "dropout": 0.10},
		"proto_ssm_train": {"n_epochs": 70, "patience": 18},
	},
	"arch_high_state": {
		"proto_ssm": {"d_state": 48, "dropout": 0.13},
	},
	"arch_low_dropout": {
		"proto_ssm": {"dropout": 0.08},
		"residual_ssm": {"dropout": 0.08},
	},
	"arch_high_dropout": {
		"proto_ssm": {"dropout": 0.18},
		"residual_ssm": {"dropout": 0.14},
	},
	"train_lr_high": {
		"proto_ssm_train": {"lr": 1.0e-3, "swa_lr": 5.0e-4},
		"residual_ssm": {"lr": 1.0e-3},
	},
	"train_lr_low": {
		"proto_ssm_train": {"lr": 6.0e-4, "swa_lr": 3.0e-4},
		"residual_ssm": {"lr": 6.0e-4},
	},
	"train_long_100": {
		"proto_ssm_train": {"n_epochs": 100, "patience": 26},
		"residual_ssm": {"n_epochs": 50, "patience": 14},
	},
	"train_mixup_strong": {
		"proto_ssm_train": {"mixup_alpha": 0.6, "distill_weight": 0.12},
	},
	"train_mixup_light": {
		"proto_ssm_train": {"mixup_alpha": 0.2, "distill_weight": 0.18},
	},
	"train_focal_strong": {
		"proto_ssm_train": {"focal_gamma": 3.0, "pos_weight_cap": 30.0},
	},
	"train_focal_light": {
		"proto_ssm_train": {"focal_gamma": 2.0, "pos_weight_cap": 22.0},
	},
	"train_label_smooth_high": {
		"proto_ssm_train": {"label_smoothing": 0.05},
	},
	"fusion_prior_strong": {
		"best_fusion": {
			"lambda_event": 0.55,
			"lambda_texture": 1.30,
			"lambda_proxy_texture": 1.00,
			"smooth_texture": 0.40,
			"smooth_event": 0.18,
		}
	},
	"fusion_prior_light": {
		"best_fusion": {
			"lambda_event": 0.35,
			"lambda_texture": 0.95,
			"lambda_proxy_texture": 0.75,
			"smooth_texture": 0.28,
			"smooth_event": 0.12,
		}
	},
	"fusion_texture_strong": {
		"best_fusion": {
			"lambda_event": 0.42,
			"lambda_texture": 1.45,
			"lambda_proxy_texture": 1.10,
			"smooth_texture": 0.45,
			"smooth_event": 0.14,
		}
	},
	"fusion_event_strong": {
		"best_fusion": {
			"lambda_event": 0.62,
			"lambda_texture": 1.00,
			"lambda_proxy_texture": 0.80,
			"smooth_texture": 0.30,
			"smooth_event": 0.20,
		}
	},
	"threshold_conservative": {
		"threshold_grid": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
		"frozen_best_probe": {"alpha": 0.40},
	},
	"threshold_aggressive": {
		"threshold_grid": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
		"frozen_best_probe": {"alpha": 0.52},
	},
	"post_rank_high": {
		"file_level_top_k": 3,
		"rank_aware_power": 0.55,
		"delta_shift_alpha": 0.24,
	},
	"post_rank_low": {
		"file_level_top_k": 1,
		"rank_aware_power": 0.25,
		"delta_shift_alpha": 0.12,
	},
	"tta_compact": {
		"tta_shifts": [0, 1, -1],
		"delta_shift_alpha": 0.16,
	},
	"tta_wide": {
		"tta_shifts": [0, 1, -1, 2, -2, 3, -3],
		"delta_shift_alpha": 0.22,
	},
	"probe_confident": {
		"frozen_best_probe": {"pca_dim": 160, "min_pos": 6, "C": 1.0, "alpha": 0.50},
	},
	"probe_regularized": {
		"frozen_best_probe": {"pca_dim": 96, "min_pos": 8, "C": 0.50, "alpha": 0.38},
		"mlp_params": {"alpha": 0.008, "learning_rate_init": 4.0e-4},
	},
	"residual_strong": {
		"residual_ssm": {"correction_weight": 0.45, "n_epochs": 50, "patience": 15},
	},
	"temperature_sharp": {
		"temperature": {"aves": 1.00, "texture": 0.88},
	},
}

