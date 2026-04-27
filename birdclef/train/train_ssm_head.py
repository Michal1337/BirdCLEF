"""Perch + SSM head trainer (file-level StratifiedKFold).

Uses the regenerated Perch cache + fold assignment parquet. Exposes
`run_full_evaluation(cfg)` that trains/evaluates one config over CV folds
and returns stitched-OOF macro AUC as the primary metric (V-anchor was
abandoned). Suitable for direct use by the sweep runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from birdclef.config.paths import (
    N_WINDOWS,
    PERCH_LABELS,
    PERCH_META,
    PERCH_NPZ,
)
from birdclef.data.soundscapes import primary_labels
from birdclef.data.splits import load_folds
from birdclef.eval.metrics import compute_stage_metrics, split_rare_frequent
from birdclef.models.losses import build_loss
from birdclef.models.ssm import (
    LightProtoSSM,
    ResidualSSM,
    ResidualSSMConfig,
    SSMHeadConfig,
)
from birdclef.postproc.boost import hard_soundscape_boost, file_confidence_scale, rank_aware_scaling
from birdclef.postproc.calibration import (
    apply_per_class_thresholds,
    build_prior_tables,
    calibrate_and_optimize_thresholds,
    logit_prior_shift,
)
from birdclef.postproc.smoothing import adaptive_delta_smooth, gaussian_smooth
from birdclef.utils.seed import seed_everything


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class PerchCache:
    meta: pd.DataFrame       # one row per 5 s window
    scores: np.ndarray       # (N, C) Perch logits mapped to BirdCLEF classes
    emb: np.ndarray          # (N, 1536) embeddings
    Y: np.ndarray            # (N, C) uint8, zeros for unlabeled rows
    labeled_mask: np.ndarray # (N,) bool


def load_perch_cache() -> PerchCache:
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)
    scores = arr["scores_full_raw"].astype(np.float32)
    emb = arr["emb_full"].astype(np.float32)
    Y = np.load(PERCH_LABELS)
    labeled = meta["is_labeled"].astype(bool).to_numpy()
    return PerchCache(meta=meta, scores=scores, emb=emb, Y=Y, labeled_mask=labeled)


def _site2i(meta: pd.DataFrame, cap: int) -> dict:
    sites = sorted(meta["site"].dropna().astype(str).unique())
    return {s: min(i + 1, cap - 1) for i, s in enumerate(sites)}


def _site_hour_ids(meta_files: pd.DataFrame, site2i: dict, cap: int):
    sids = np.array([min(site2i.get(s, 0), cap - 1) for s in meta_files["site"].tolist()], dtype=np.int64)
    hids = np.array([int(h) % 24 for h in meta_files["hour_utc"].tolist()], dtype=np.int64)
    return sids, hids


def _train_proto_ssm(
    emb: np.ndarray, scores: np.ndarray, Y: np.ndarray, meta_files: pd.DataFrame,
    cfg: dict,
) -> LightProtoSSM:
    n_classes = Y.shape[1]
    cap = int(cfg["n_sites_cap"])
    model = LightProtoSSM(SSMHeadConfig(
        d_input=emb.shape[1], n_classes=n_classes, n_windows=N_WINDOWS,
        n_sites=cap, use_cross_attn=True, cross_attn_heads=2,
    ))
    model.init_prototypes(
        torch.tensor(emb, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    site2i = _site2i(meta_files, cap)
    site_ids, hour_ids = _site_hour_ids(meta_files, site2i, cap)

    n_files = emb.shape[0] // N_WINDOWS
    emb_t = torch.tensor(emb.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    sc_t = torch.tensor(scores.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    lab_t = torch.tensor(Y.reshape(n_files, N_WINDOWS, -1).astype(np.float32))
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    loss_kind = cfg["loss"]
    if loss_kind == "focal_bce":
        loss_fn = build_loss("focal_bce",
                             alpha=float(cfg["focal_alpha"]),
                             gamma=float(cfg["focal_gamma"]),
                             label_smoothing=float(cfg["label_smoothing"]))
    elif loss_kind == "bce_focal_mean":
        loss_fn = build_loss("bce_focal_mean",
                             focal_gamma=float(cfg["focal_gamma"]),
                             focal_alpha=float(cfg["focal_alpha"]),
                             label_smoothing=float(cfg["label_smoothing"]))
    else:
        pos_w = np.minimum(
            (Y.shape[0] - Y.sum(axis=0)) / (Y.sum(axis=0) + 1),
            float(cfg["proto_pos_weight_cap"]),
        ).astype(np.float32)
        loss_fn = build_loss("bce_posw",
                             pos_weight=torch.tensor(pos_w),
                             label_smoothing=float(cfg["label_smoothing"]))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["proto_lr"]), weight_decay=1e-3)
    n_epochs = int(cfg["proto_n_epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(cfg["proto_lr"]), epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos",
    )
    swa = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(n_epochs * float(cfg["proto_swa_start_frac"]))
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=float(cfg["proto_swa_lr"]))
    distill_w = float(cfg["proto_distill_weight"])

    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        out = model(emb_t, sc_t, site_ids=site_t, hours=hour_t)
        loss = loss_fn(out, lab_t)
        if distill_w > 0:
            loss = loss + distill_w * F.mse_loss(out, sc_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep >= swa_start:
            swa.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg["proto_patience"]):
                break
    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa)
        model = swa
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_proto(
    model, emb: np.ndarray, scores: np.ndarray, meta_files: pd.DataFrame,
    cfg: dict,
) -> np.ndarray:
    cap = int(cfg["n_sites_cap"])
    site2i = _site2i(meta_files, cap)
    site_ids, hour_ids = _site_hour_ids(meta_files, site2i, cap)
    n_files = emb.shape[0] // N_WINDOWS
    emb_t = torch.tensor(emb.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    sc_t = torch.tensor(scores.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)
    if cfg["tta"] == "window_roll":
        shifts = tuple(int(x) for x in cfg["tta_shifts"])
        outs = []
        with torch.no_grad():
            for s in shifts:
                if s == 0:
                    e = emb_t; sc = sc_t
                else:
                    e = torch.roll(emb_t, s, dims=1); sc = torch.roll(sc_t, s, dims=1)
                o = model(e, sc, site_ids=site_t, hours=hour_t).numpy()
                if s != 0:
                    o = np.roll(o, -s, axis=1)
                outs.append(o)
        return np.mean(outs, axis=0).reshape(-1, scores.shape[1])
    # waveform_shift fallback — this stage only has frozen Perch embeddings,
    # so we do a single pass. Real waveform-shift TTA lives in the inference
    # template where raw audio is available and Perch runs >1 time.
    with torch.no_grad():
        o = model(emb_t, sc_t, site_ids=site_t, hours=hour_t).numpy()
    return o.reshape(-1, scores.shape[1])


def _train_mlp_probes(
    emb: np.ndarray, scores: np.ndarray, Y: np.ndarray, cfg: dict
):
    seed = int(cfg.get("seed", 42))
    scaler = StandardScaler().fit(emb)
    emb_s = scaler.transform(emb)
    pca_dim = min(int(cfg["mlp_pca_dim"]), emb_s.shape[1] - 1)
    pca = PCA(n_components=pca_dim, random_state=seed).fit(emb_s)
    Z = pca.transform(emb_s).astype(np.float32)
    active = np.where(Y.sum(axis=0) >= int(cfg["mlp_min_pos"]))[0]
    models = {}
    max_rows = 3000
    for ci in active:
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        sc_col = scores[:, ci]
        prev = np.concatenate([sc_col.reshape(-1, N_WINDOWS)[:, :1], sc_col.reshape(-1, N_WINDOWS)[:, :-1]], axis=1).reshape(-1)
        nxt = np.concatenate([sc_col.reshape(-1, N_WINDOWS)[:, 1:], sc_col.reshape(-1, N_WINDOWS)[:, -1:]], axis=1).reshape(-1)
        mean = np.repeat(sc_col.reshape(-1, N_WINDOWS).mean(axis=1), N_WINDOWS)
        mx = np.repeat(sc_col.reshape(-1, N_WINDOWS).max(axis=1), N_WINDOWS)
        std = np.repeat(sc_col.reshape(-1, N_WINDOWS).std(axis=1), N_WINDOWS)
        X = np.hstack([Z, sc_col[:, None], prev[:, None], nxt[:, None], mean[:, None], mx[:, None], std[:, None]])
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        repeat = max(1, int(round(n_neg / max(n_pos, 1))))
        repeat = min(repeat, 8)
        if n_pos * repeat + len(y) > max_rows:
            repeat = max(1, (max_rows - len(y)) // max(n_pos, 1))
        pos_idx = np.where(y == 1)[0]
        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                            max_iter=500, early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=20, learning_rate_init=5e-4, alpha=0.005,
                            random_state=seed)
        clf.fit(X_bal, y_bal)
        models[int(ci)] = clf
    return models, scaler, pca


def _apply_mlp_probes(emb, scores, models, scaler, pca, alpha_blend):
    emb_s = scaler.transform(emb)
    Z = pca.transform(emb_s).astype(np.float32)
    out = scores.copy()
    for ci, clf in models.items():
        sc_col = scores[:, ci]
        prev = np.concatenate([sc_col.reshape(-1, N_WINDOWS)[:, :1], sc_col.reshape(-1, N_WINDOWS)[:, :-1]], axis=1).reshape(-1)
        nxt = np.concatenate([sc_col.reshape(-1, N_WINDOWS)[:, 1:], sc_col.reshape(-1, N_WINDOWS)[:, -1:]], axis=1).reshape(-1)
        mean = np.repeat(sc_col.reshape(-1, N_WINDOWS).mean(axis=1), N_WINDOWS)
        mx = np.repeat(sc_col.reshape(-1, N_WINDOWS).max(axis=1), N_WINDOWS)
        std = np.repeat(sc_col.reshape(-1, N_WINDOWS).std(axis=1), N_WINDOWS)
        X = np.hstack([Z, sc_col[:, None], prev[:, None], nxt[:, None], mean[:, None], mx[:, None], std[:, None]])
        prob = clf.predict_proba(X)[:, 1].astype(np.float32)
        logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        out[:, ci] = (1 - alpha_blend) * scores[:, ci] + alpha_blend * logit
    return out


def _train_residual(
    emb, first_pass_flat, Y, site_ids, hour_ids, cfg
):
    n_classes = Y.shape[1]
    n_files = emb.shape[0] // N_WINDOWS
    emb_f = emb.reshape(n_files, N_WINDOWS, -1)
    fp_f = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f = Y.reshape(n_files, N_WINDOWS, -1).astype(np.float32)
    fp_prob = sigmoid_np(fp_f)
    residuals = lab_f - fp_prob

    # 3-fold internal mini-CV (no leakage) — train/val split for early stopping.
    seed = int(cfg.get("seed", 42))
    rng = torch.Generator(); rng.manual_seed(seed)
    perm = torch.randperm(n_files, generator=rng).numpy()
    n_val = max(1, int(0.15 * n_files))
    tr_i = perm[n_val:]; va_i = perm[:n_val]
    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    fp_t = torch.tensor(fp_f, dtype=torch.float32)
    res_t = torch.tensor(residuals, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    model = ResidualSSM(ResidualSSMConfig(
        d_input=emb.shape[1], d_scores=n_classes, n_classes=n_classes,
        n_windows=N_WINDOWS,
        d_model=int(cfg["residual_d_model"]),
        d_state=int(cfg["residual_d_state"]),
        dropout=float(cfg["residual_dropout"]),
    ))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["residual_lr"]), weight_decay=1e-3)
    n_epochs = int(cfg["residual_n_epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["residual_lr"]),
                                                epochs=n_epochs, steps_per_epoch=1,
                                                pct_start=0.1, anneal_strategy="cos")
    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[tr_i], fp_t[tr_i], site_ids=site_t[tr_i], hours=hour_t[tr_i])
        loss = F.mse_loss(corr, res_t[tr_i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            v_loss = F.mse_loss(model(emb_t[va_i], fp_t[va_i],
                                      site_ids=site_t[va_i], hours=hour_t[va_i]),
                                res_t[va_i]).item()
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg["residual_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _temperature_vector(labels: list, class_map: Dict[str, str]) -> np.ndarray:
    t = np.ones(len(labels), dtype=np.float32)
    for i, lb in enumerate(labels):
        cls = class_map.get(lb, "Aves")
        if cls in {"Amphibia", "Insecta"}:
            t[i] = 0.95
        else:
            t[i] = 1.10
    return t


def _lambda_prior_vector(
    labels: list, class_map: Dict[str, str],
    lambda_birds: float, lambda_texture: float,
) -> np.ndarray:
    """Per-class prior strength: stronger λ for continuous callers (frogs/insects)
    where Perch is weak, weaker λ for birds where Perch is reliable.
    Matches the LB_0931_seed.ipynb cell-26 split that boosted LB by ~0.005.
    """
    lam = np.full(len(labels), float(lambda_birds), dtype=np.float32)
    for i, lb in enumerate(labels):
        if class_map.get(lb, "Aves") in {"Amphibia", "Insecta"}:
            lam[i] = float(lambda_texture)
    return lam


def _postproc(
    first_pass_logits: np.ndarray, temperatures: np.ndarray, cfg: dict,
) -> np.ndarray:
    scores = first_pass_logits / temperatures[None, :]
    probs = sigmoid_np(scores)
    probs = file_confidence_scale(probs, n_windows=N_WINDOWS,
                                  top_k=int(cfg["file_conf_top_k"]),
                                  power=float(cfg["file_conf_power"]))
    probs = rank_aware_scaling(probs, n_windows=N_WINDOWS,
                               power=float(cfg["rank_power"]))
    if cfg["smoothing"] == "gaussian":
        probs = gaussian_smooth(probs, n_windows=N_WINDOWS)
    elif cfg["smoothing"] == "adaptive":
        probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,
                                      base_alpha=float(cfg["smoothing_alpha"]))
    if cfg.get("use_boost", False):
        probs = hard_soundscape_boost(probs, n_windows=N_WINDOWS,
                                      threshold=float(cfg["boost_threshold"]),
                                      lift_weight=float(cfg["boost_lift"]))
    return np.clip(probs, 0.0, 1.0)


def _labeled_file_meta(meta_rows: pd.DataFrame) -> pd.DataFrame:
    return meta_rows.drop_duplicates("filename")[["filename", "site", "hour_utc"]].reset_index(drop=True)


def run_pipeline_for_split(
    cache: PerchCache, train_idx: np.ndarray, val_idx: np.ndarray, cfg: dict,
    temperatures: np.ndarray,
    lambda_prior_vec: np.ndarray | None = None,
) -> dict:
    """Train on train_idx rows, predict on val_idx rows.

    Returns:
        {
            "first_pass": (N_val, C) float32 — sigmoid(ensemble_w*proto + (1-w)*mlp_probe)
                          i.e. the calibrated pre-post-processing teacher signal.
            "final":      (N_val, C) float32 — full pipeline output through post-proc + threshold.
        }
    The caller decides which one to rank on and which one to emit as pseudo-label.
    """
    emb_tr, sc_tr, Y_tr = cache.emb[train_idx], cache.scores[train_idx], cache.Y[train_idx]
    emb_va, sc_va = cache.emb[val_idx], cache.scores[val_idx]
    meta_tr = cache.meta.iloc[train_idx].reset_index(drop=True)
    meta_va = cache.meta.iloc[val_idx].reset_index(drop=True)
    tr_files = _labeled_file_meta(meta_tr)
    va_files = _labeled_file_meta(meta_va)

    proto = _train_proto_ssm(emb_tr, sc_tr, Y_tr, tr_files, cfg)
    proto_va = _predict_proto(proto, emb_va, sc_va, va_files, cfg)

    probe_models, scaler, pca = _train_mlp_probes(emb_tr, sc_tr, Y_tr, cfg)
    prior_tables = build_prior_tables(meta_tr, Y_tr)
    lam = lambda_prior_vec if lambda_prior_vec is not None else float(cfg["lambda_prior"])
    sc_va_prior = logit_prior_shift(sc_va, meta_va, prior_tables, lambda_prior=lam)
    sc_va_mlp = _apply_mlp_probes(emb_va, sc_va_prior, probe_models, scaler, pca,
                                  float(cfg["mlp_alpha_blend"]))
    first_pass_va = float(cfg["ensemble_w"]) * proto_va + (1.0 - float(cfg["ensemble_w"])) * sc_va_mlp

    # Residual trained on train split, applied to val split
    tr_site2i = _site2i(tr_files, int(cfg["n_sites_cap"]))
    tr_site_ids, tr_hour_ids = _site_hour_ids(tr_files, tr_site2i, int(cfg["n_sites_cap"]))
    va_site_ids, va_hour_ids = _site_hour_ids(va_files, tr_site2i, int(cfg["n_sites_cap"]))
    proto_tr = _predict_proto(proto, emb_tr, sc_tr, tr_files, cfg)
    sc_tr_prior = logit_prior_shift(sc_tr, meta_tr, prior_tables, lambda_prior=lam)
    sc_tr_mlp = _apply_mlp_probes(emb_tr, sc_tr_prior, probe_models, scaler, pca,
                                  float(cfg["mlp_alpha_blend"]))
    first_pass_tr = float(cfg["ensemble_w"]) * proto_tr + (1.0 - float(cfg["ensemble_w"])) * sc_tr_mlp

    res = _train_residual(emb_tr, first_pass_tr, Y_tr, tr_site_ids, tr_hour_ids, cfg)
    with torch.no_grad():
        n_va = emb_va.shape[0] // N_WINDOWS
        corr = res(
            torch.tensor(emb_va.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
            torch.tensor(first_pass_va.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
            site_ids=torch.tensor(va_site_ids, dtype=torch.long),
            hours=torch.tensor(va_hour_ids, dtype=torch.long),
        ).numpy().reshape(-1, cache.Y.shape[1])
    final_logits = first_pass_va + float(cfg["correction_weight"]) * corr

    # Calibrated thresholds + post-processing
    first_pass_tr_probs = sigmoid_np(first_pass_tr)
    thresholds = calibrate_and_optimize_thresholds(
        first_pass_tr_probs, Y_tr, threshold_grid=list(cfg["threshold_grid"]),
        n_windows=N_WINDOWS,
    )
    probs = _postproc(final_logits, temperatures, cfg)
    final_probs = apply_per_class_thresholds(probs, thresholds)

    return {
        "first_pass": sigmoid_np(first_pass_va).astype(np.float32),
        "final": final_probs.astype(np.float32),
    }


def _augment_cache_with_pseudo(
    cache: PerchCache, pseudo_round: int, pseudo_tau: float = 0.5,
) -> PerchCache:
    """Concat pseudo-labeled unlabeled rows onto the labeled training pool.

    Pseudo-labels come from `cache/pseudo/round{N}/`:
      - `probs.npz['final']`     : (10658, C) float32 — teacher's final-stage probs
      - `probs.npz['keep_mask']` : (10658, C) uint8   — confidence-filter pass
      - `meta.parquet`           : row alignment to the Perch cache

    A row is included as a training augmentation if it is currently unlabeled
    AND has at least one class with `keep_mask=1` AND `prob >= pseudo_tau`.
    Y for that row = (probs >= pseudo_tau) & keep_mask  → hard pseudo-positives.

    Pseudo rows are concatenated AFTER the labeled rows; they have no fold
    assignment, so the row_fold logic in `run_full_evaluation` puts them in
    every fold's train via the `.fillna(-1)` → `row_fold != f` path.
    """
    from birdclef.config.paths import PSEUDO_DIR
    rd = PSEUDO_DIR / f"round{int(pseudo_round)}"
    if not rd.exists():
        raise SystemExit(f"pseudo round dir missing: {rd}. "
                         "Build via `python -m birdclef.scripts._05_pseudo_label --round N`.")
    arr = np.load(rd / "probs.npz")
    probs = arr["final"].astype(np.float32)        # (N, C)
    keep_mask = arr["keep_mask"].astype(np.uint8)   # (N, C)
    pmeta = pd.read_parquet(rd / "meta.parquet")    # row_id, filename, window, is_labeled
    if len(pmeta) != len(cache.meta):
        raise SystemExit(
            f"pseudo meta rows ({len(pmeta)}) != Perch cache rows ({len(cache.meta)}). "
            "Stale cache or stale pseudo-round; rebuild one.")

    # Hard pseudo-Y: positive only where teacher is confident (keep_mask=1)
    # AND the prob clears the threshold. Other positions stay 0 (negative).
    pseudo_Y = ((probs >= float(pseudo_tau)) & (keep_mask > 0)).astype(np.uint8)

    labeled = cache.labeled_mask
    unlabeled_idx = np.where(~labeled)[0]
    # Skip unlabeled rows that have no positive class — they're pure negatives,
    # add no signal beyond what labeled rows already provide.
    has_pos = pseudo_Y[unlabeled_idx].sum(axis=1) > 0
    keep_unlab = unlabeled_idx[has_pos]

    labeled_idx = np.where(labeled)[0]
    aug_meta = pd.concat([
        cache.meta.iloc[labeled_idx].reset_index(drop=True),
        cache.meta.iloc[keep_unlab].reset_index(drop=True),
    ], ignore_index=True)
    aug_scores = np.concatenate([cache.scores[labeled_idx], cache.scores[keep_unlab]], axis=0)
    aug_emb    = np.concatenate([cache.emb[labeled_idx],    cache.emb[keep_unlab]],    axis=0)
    aug_Y      = np.concatenate([cache.Y[labeled_idx],      pseudo_Y[keep_unlab]],     axis=0)
    aug_labeled_mask = np.ones(len(aug_meta), dtype=bool)

    n_lab, n_pseudo = len(labeled_idx), len(keep_unlab)
    n_pseudo_pos = int(aug_Y[n_lab:].sum())
    print(f"[pseudo] augmenting cache: labeled={n_lab}  "
          f"pseudo-rows-with-positives={n_pseudo}  "
          f"pseudo-positives={n_pseudo_pos}  "
          f"τ={pseudo_tau}  round={pseudo_round}")
    return PerchCache(meta=aug_meta, scores=aug_scores, emb=aug_emb,
                      Y=aug_Y, labeled_mask=aug_labeled_mask)


def run_full_evaluation(cfg: dict) -> Dict:
    """Runs fold-safe stitched OOF for one config. Returns sweep-runner result.

    Optional `cfg["pseudo_round"]` (int) augments the training pool with
    pseudo-labeled unlabeled soundscape rows from `cache/pseudo/round{N}/`.
    Pseudo rows are training-only; the OOF eval still measures on real
    labeled-fold val (no pseudo-label leakage into the metric).
    `cfg["pseudo_tau"]` (float, default 0.5) thresholds the pseudo-probs.
    """
    base_seed = int(cfg.get("seed", 42))
    seed_everything(base_seed)
    cache = load_perch_cache()

    # Optional: extend the training pool with pseudo-labeled unlabeled rows.
    # cfg["pseudo_round"] = None (default) → behavior unchanged.
    if cfg.get("pseudo_round") is not None:
        cache = _augment_cache_with_pseudo(
            cache,
            pseudo_round=int(cfg["pseudo_round"]),
            pseudo_tau=float(cfg.get("pseudo_tau", 0.5)),
        )
        # After augmentation, all rows are training-eligible (real label OR
        # pseudo). The val measurement still respects fold membership: only
        # rows whose FILENAME is in the folds parquet (i.e. labeled files)
        # ever land in val. Pseudo rows have no fold → fold=-1 → train-only.
        labeled_idx = np.arange(len(cache.meta))
    else:
        labeled_idx = np.where(cache.labeled_mask)[0]
    cache_meta = cache.meta.iloc[labeled_idx].reset_index(drop=True)
    cache_scores = cache.scores[labeled_idx]
    cache_emb = cache.emb[labeled_idx]
    cache_Y = cache.Y[labeled_idx]
    cache_sub = PerchCache(meta=cache_meta, scores=cache_scores, emb=cache_emb,
                           Y=cache_Y, labeled_mask=np.ones(len(cache_meta), dtype=bool))

    # Temperature vector
    from birdclef.data.soundscapes import load_taxonomy
    tax = load_taxonomy()
    class_map = tax.set_index("primary_label")["class_name"].to_dict()
    labels = primary_labels()
    temperatures = _temperature_vector(labels, class_map)
    lambda_prior_vec = _lambda_prior_vector(
        labels, class_map,
        lambda_birds=float(cfg["lambda_prior"]),
        lambda_texture=float(cfg.get("lambda_prior_texture", cfg["lambda_prior"])),
    )

    # Rare/frequent split (support from LABELED rows only — that's what we
    # have for this comparative sweep).
    support = cache_Y.sum(axis=0)
    rare_idx, freq_idx = split_rare_frequent(support)

    # OOF by file-level StratifiedKFold (n_splits configurable per cfg)
    n_splits_cfg = int(cfg.get("n_splits", 5))
    folds = load_folds(n_splits=n_splits_cfg)
    fold_of = dict(zip(folds["filename"], folds["fold"].astype(int)))
    # row_fold = -1 means EITHER (a) file isn't in folds parquet (e.g.
    # unlabeled), or (b) file is pinned (fold=-1 in parquet). In both cases
    # the file is excluded from val. The train mask below `row_fold != f`
    # naturally includes pinned files in every fold's train; unlabeled files
    # are filtered out elsewhere by `cache.labeled_mask`.
    row_fold = cache_meta["filename"].map(fold_of).fillna(-1).astype(int).to_numpy()
    n_splits = int(folds.loc[folds["fold"] >= 0, "fold"].max()) + 1 if len(folds) else n_splits_cfg

    oof_final = np.zeros_like(cache_Y, dtype=np.float32)
    oof_fp = np.zeros_like(cache_Y, dtype=np.float32)
    per_fold = {}
    for f in range(n_splits):
        # Train: every row except this fold's val. Pinned (-1) and unlabeled
        # (-1 from fillna) both pass `row_fold != f` for any f >= 0, so they
        # land in train as intended. The labeled-mask filter upstream
        # already removed truly unlabeled rows from cache_meta.
        tr = np.where(row_fold != f)[0]
        va = np.where(row_fold == f)[0]
        if len(va) == 0:
            continue
        seed_everything(base_seed + int(f) + 1)
        out = run_pipeline_for_split(cache_sub, tr, va, cfg, temperatures,
                                     lambda_prior_vec=lambda_prior_vec)
        oof_final[va] = out["final"]
        oof_fp[va] = out["first_pass"]
        va_meta = cache_meta.iloc[va].reset_index(drop=True)
        m_final = compute_stage_metrics(cache_Y[va], out["final"], va_meta,
                                        rare_idx=rare_idx, frequent_idx=freq_idx)
        m_fp = compute_stage_metrics(cache_Y[va], out["first_pass"], va_meta,
                                     rare_idx=rare_idx, frequent_idx=freq_idx)
        m_final["first_pass_auc"] = m_fp.get("macro_auc", float("nan"))
        per_fold[int(f)] = {"final": m_final, "first_pass": m_fp}
        print(f"[ssm] fold {f}  final={m_final['macro_auc']:.4f}  "
              f"first_pass={m_fp['macro_auc']:.4f}  "
              f"site_std={m_final['site_auc_std']:.4f}")

    oof_keep = row_fold >= 0
    keep_meta = cache_meta.iloc[oof_keep].reset_index(drop=True)
    m_global_final = compute_stage_metrics(cache_Y[oof_keep], oof_final[oof_keep], keep_meta,
                                           rare_idx=rare_idx, frequent_idx=freq_idx)
    m_global_fp = compute_stage_metrics(cache_Y[oof_keep], oof_fp[oof_keep], keep_meta,
                                        rare_idx=rare_idx, frequent_idx=freq_idx)
    m_global_final["first_pass_auc"] = m_global_fp.get("macro_auc", float("nan"))
    print(f"[ssm] stitched-OOF  final={m_global_final['macro_auc']:.4f}  "
          f"first_pass={m_global_fp['macro_auc']:.4f}  "
          f"site_std={m_global_final['site_auc_std']:.4f}")

    return {
        "metrics": {
            # Stage selected for sweep ranking = final stitched OOF.
            # `first_pass_auc` is surfaced alongside so you can inspect the
            # post-processing delta. Per-fold dict is informational.
            "global": m_global_final,
            "per_fold": per_fold,
            "global_first_pass": m_global_fp,
        }
    }
