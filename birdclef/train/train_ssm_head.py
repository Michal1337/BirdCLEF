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
from tqdm.auto import tqdm

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
    meta: pd.DataFrame             # one row per 5 s window
    scores: np.ndarray             # (N, C) raw direct-mapped Perch logits;
                                   # zero at unmapped positions
    scores_proxy: np.ndarray       # (N, C) genus-max fill at proxy target
                                   # columns; zero everywhere else.
                                   # All-zero on legacy caches that pre-baked
                                   # the proxy into `scores`.
    emb: np.ndarray                # (N, 1536) embeddings
    Y: np.ndarray                  # (N, C) uint8, zeros for unlabeled rows
    labeled_mask: np.ndarray       # (N,) bool


def load_perch_cache() -> PerchCache:
    """Load the Perch cache from disk. No score merging happens here.

    Returns both arrays separately:
      cache.scores         — direct-mapped Perch logits only.
      cache.scores_proxy   — genus-max proxy fill (zeros if the cache
                             pre-dates the new format, since legacy caches
                             baked the proxy into `scores`).

    Each downstream experiment is responsible for deciding whether to apply
    the proxy. The canonical merge is `apply_proxy_to_scores(scores,
    scores_proxy)` — i.e. element-wise sum, since the two arrays are
    disjoint by construction. The LB notebook (cell 6) and the SSM sweep
    runner do this merge themselves at their own load-equivalent stage.

    Legacy caches without `scores_proxy` get an all-zero placeholder so
    consumers can still call `apply_proxy_to_scores` unconditionally and
    get the same tensor they used to get.
    """
    meta = pd.read_parquet(PERCH_META)
    arr = np.load(PERCH_NPZ)
    scores = arr["scores_full_raw"].astype(np.float32)
    emb = arr["emb_full"].astype(np.float32)
    Y = np.load(PERCH_LABELS)
    labeled = meta["is_labeled"].astype(bool).to_numpy()
    if "scores_proxy" in arr.files:
        scores_proxy = arr["scores_proxy"].astype(np.float32)
    else:
        scores_proxy = np.zeros_like(scores)
    return PerchCache(
        meta=meta, scores=scores, scores_proxy=scores_proxy,
        emb=emb, Y=Y, labeled_mask=labeled,
    )


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
    pbar = tqdm(range(n_epochs), desc=f"proto_ssm[{cfg.get('name','?')}]",
                leave=False, dynamic_ncols=True)
    for ep in pbar:
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
            phase = "swa"
        else:
            sched.step()
            phase = "warm"
        cur = float(loss.item())
        if cur < best_loss:
            best_loss = cur
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        pbar.set_postfix(loss=f"{cur:.4f}", best=f"{best_loss:.4f}",
                         wait=wait, phase=phase, refresh=False)
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


class _GroupedMLP(torch.nn.Module):
    """Per-class MLP heads vectorized via grouped einsum.

    Equivalent to N_active independent (in_dim → 256 → 128 → 1) MLPs but they
    train + infer in a single GPU pass instead of sklearn's sequential CPU
    loop. Each class has its OWN weights — there's no parameter sharing
    across classes. The per-class score features (sc_col, prev, nxt, mean,
    max, std) are still distinct per class; only the PCA(emb) features Z
    are shared.

    Input  : X of shape (C, N, in_dim)
    Output : logits of shape (C, N)
    Params : ~ C × (in_dim·256 + 256·128 + 128·1) ≈ 50k×C; for C≈170 ≈ 8.5M
             — fits anywhere.
    """
    def __init__(self, n_active: int, in_dim: int, h1: int = 256, h2: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.empty(n_active, in_dim, h1))
        self.b1 = torch.nn.Parameter(torch.zeros(n_active, h1))
        self.W2 = torch.nn.Parameter(torch.empty(n_active, h1, h2))
        self.b2 = torch.nn.Parameter(torch.zeros(n_active, h2))
        self.W3 = torch.nn.Parameter(torch.empty(n_active, h2, 1))
        self.b3 = torch.nn.Parameter(torch.zeros(n_active, 1))
        self.dropout = float(dropout)
        for w in (self.W1, self.W2, self.W3):
            torch.nn.init.xavier_uniform_(w)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (C, N, in_dim)
        h = torch.einsum("cnd,cdh->cnh", X, self.W1) + self.b1.unsqueeze(1)
        h = F.relu(h)
        if self.training and self.dropout > 0:
            h = F.dropout(h, self.dropout)
        h = torch.einsum("cnd,cdh->cnh", h, self.W2) + self.b2.unsqueeze(1)
        h = F.relu(h)
        if self.training and self.dropout > 0:
            h = F.dropout(h, self.dropout)
        h = torch.einsum("cnd,cdh->cnh", h, self.W3) + self.b3.unsqueeze(1)
        return h.squeeze(-1)  # (C, N)


def _build_score_features(scores: np.ndarray, ci: int) -> np.ndarray:
    """Returns (N, 6) per-class score features matching the legacy sklearn
    implementation: [sc, prev, next, file_mean, file_max, file_std].
    """
    sc_col = scores[:, ci]
    n_files = sc_col.shape[0] // N_WINDOWS
    sc_f = sc_col.reshape(n_files, N_WINDOWS)
    prev = np.concatenate([sc_f[:, :1], sc_f[:, :-1]], axis=1).reshape(-1)
    nxt  = np.concatenate([sc_f[:, 1:], sc_f[:, -1:]], axis=1).reshape(-1)
    mean = np.repeat(sc_f.mean(axis=1), N_WINDOWS)
    mx   = np.repeat(sc_f.max(axis=1),  N_WINDOWS)
    std  = np.repeat(sc_f.std(axis=1),  N_WINDOWS)
    return np.stack([sc_col, prev, nxt, mean, mx, std], axis=1).astype(np.float32)


def _build_grouped_input(
    Z: np.ndarray, scores: np.ndarray, active: np.ndarray,
) -> np.ndarray:
    """Build the (C, N, in_dim) tensor: shared Z + per-class score features."""
    N = Z.shape[0]
    pca_dim = Z.shape[1]
    in_dim = pca_dim + 6
    out = np.empty((len(active), N, in_dim), dtype=np.float32)
    for k, ci in enumerate(active):
        feats = _build_score_features(scores, int(ci))
        out[k, :, :pca_dim] = Z
        out[k, :, pca_dim:] = feats
    return out


def _train_mlp_probes(
    emb: np.ndarray, scores: np.ndarray, Y: np.ndarray, cfg: dict,
):
    """GPU-batched per-class MLP training.

    Trains all `active` classes' MLPs simultaneously via _GroupedMLP. ~30×
    faster than the sklearn loop on the same data; behavior matches the
    legacy implementation in input features (PCA + 6 score statistics) and
    architecture (256 → 128 → 1). Class imbalance is handled via per-class
    `pos_weight` in BCE rather than sklearn's positive-row replication —
    equivalent loss-side effect, no data tiling.
    """
    seed = int(cfg.get("seed", 42))
    scaler = StandardScaler().fit(emb)
    emb_s = scaler.transform(emb).astype(np.float32)
    pca_dim = min(int(cfg["mlp_pca_dim"]), emb_s.shape[1] - 1)
    pca = PCA(n_components=pca_dim, random_state=seed).fit(emb_s)
    Z = pca.transform(emb_s).astype(np.float32)

    n_pos_per_class = Y.sum(axis=0)
    active_mask = (n_pos_per_class >= int(cfg["mlp_min_pos"])) & (n_pos_per_class < Y.shape[0])
    active = np.where(active_mask)[0].astype(np.int64)
    if len(active) == 0:
        return {"model": None, "active": active, "in_dim": pca_dim + 6}, scaler, pca

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Build inputs once. Shape (C, N, pca_dim+6).
    X = _build_grouped_input(Z, scores, active)
    Yc = Y[:, active].astype(np.float32).T   # (C, N)
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Yc).to(device)

    # 85/15 train/val split — matches sklearn's validation_fraction.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[1])
    n_val = max(1, int(0.15 * X.shape[1]))
    va_i = torch.from_numpy(perm[:n_val]).to(device)
    tr_i = torch.from_numpy(perm[n_val:]).to(device)

    # Per-class pos_weight (capped) — equivalent to legacy positive-row
    # replication (which capped repeat at 8 → effective pos_weight ≤ 8).
    pos = Yc.sum(axis=1)
    neg = Yc.shape[1] - pos
    pos_w_np = np.minimum(neg / np.maximum(pos, 1), 8.0).astype(np.float32)
    pos_w = torch.from_numpy(pos_w_np).to(device)  # (C,)

    model = _GroupedMLP(n_active=len(active), in_dim=X.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)

    n_epochs = 100
    patience = 15
    # Mini-batch over the row axis. Each batch is a subset of (N_tr) rows
    # processed across all C classes simultaneously (the C axis is the
    # parallelism, not the batch axis). This gives ~n_tr/batch_size
    # gradient steps per epoch — comparable to sklearn's default batching
    # — instead of full-batch gradient descent which collapses to one step
    # per epoch and converges much worse.
    batch_size = 512
    n_train = tr_i.shape[0]
    n_batches = max(1, (n_train + batch_size - 1) // batch_size)
    best_val = float("inf"); best_state = None; wait = 0
    pbar = tqdm(range(n_epochs), desc=f"mlp_probes[{cfg.get('name','?')}]",
                leave=False, dynamic_ncols=True)
    for ep in pbar:
        model.train()
        epoch_losses = []
        # Reshuffle row order each epoch.
        perm_ep = torch.randperm(n_train, device=device)
        for b_start in range(0, n_train, batch_size):
            b_local = perm_ep[b_start : b_start + batch_size]
            b_idx = tr_i[b_local]
            logits = model(X_t[:, b_idx, :])              # (C, batch)
            targets = Y_t[:, b_idx]
            # BCE with per-class pos_weight broadcast over the batch axis.
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_w.unsqueeze(-1), reduction="mean",
            )
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(float(loss.item()))
        cur_train = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        model.eval()
        with torch.no_grad():
            v_logits = model(X_t[:, va_i, :])
            v_loss = F.binary_cross_entropy_with_logits(
                v_logits, Y_t[:, va_i], pos_weight=pos_w.unsqueeze(-1),
                reduction="mean",
            ).item()
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        pbar.set_postfix(train=f"{cur_train:.4f}", val=f"{v_loss:.4f}",
                         best=f"{best_val:.4f}", wait=wait,
                         n_cls=len(active), n_steps=n_batches,
                         refresh=False)
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "active": active, "in_dim": X.shape[2]}, scaler, pca


@torch.no_grad()
def _apply_mlp_probes(
    emb: np.ndarray, scores: np.ndarray, models: dict, scaler, pca,
    alpha_blend: float,
) -> np.ndarray:
    """Apply the grouped MLP head to test/val data and blend log-odds with
    the original scores. `models` is the dict returned by `_train_mlp_probes`.
    """
    out = scores.copy()
    model: torch.nn.Module | None = models.get("model")
    active: np.ndarray = models.get("active", np.array([], dtype=np.int64))
    if model is None or len(active) == 0:
        return out

    emb_s = scaler.transform(emb).astype(np.float32)
    Z = pca.transform(emb_s).astype(np.float32)
    X = _build_grouped_input(Z, scores, active)
    device = next(model.parameters()).device
    X_t = torch.from_numpy(X).to(device)
    logits = model(X_t)                                   # (C, N)
    # logit blending matches legacy: out_logit = (1-α)*orig + α*probe_logit.
    probe_logit = logits.cpu().numpy().astype(np.float32) # (C, N)
    for k, ci in enumerate(active):
        out[:, int(ci)] = (1 - alpha_blend) * scores[:, int(ci)] + alpha_blend * probe_logit[k]
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
    pbar = tqdm(range(n_epochs), desc=f"residual_ssm[{cfg.get('name','?')}]",
                leave=False, dynamic_ncols=True)
    for ep in pbar:
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
        cur_train = float(loss.item())
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        pbar.set_postfix(train=f"{cur_train:.4f}", val=f"{v_loss:.4f}",
                         best=f"{best_loss:.4f}", wait=wait, refresh=False)
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
    if bool(cfg.get("apply_thresholds", True)):
        final_probs = apply_per_class_thresholds(probs, thresholds)
    else:
        final_probs = probs

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
    labeled_idx = np.where(labeled)[0]
    unlabeled_idx = np.where(~labeled)[0]

    # The SSM stack reshapes (rows, ...) → (n_files, N_WINDOWS=12, ...) for
    # ProtoSSM. Pseudo augmentation must therefore preserve the 12-windows-
    # per-file invariant: include ALL 12 windows of any file that has at
    # least one high-confidence pseudo-positive in any window. Windows
    # without keep_mask positives become Y=0 (treated as negative) — this
    # introduces some teacher-miss false negatives but is acceptable noise
    # vs. broken reshape.
    rows_per_file = pseudo_Y[unlabeled_idx].sum(axis=1) > 0
    unlab_meta = cache.meta.iloc[unlabeled_idx][["filename"]].reset_index(drop=True)
    unlab_meta["_orig_idx"] = unlabeled_idx
    unlab_meta["_has_pos"]  = rows_per_file
    files_with_pos = set(unlab_meta.loc[unlab_meta["_has_pos"], "filename"].unique())
    keep_unlab = unlab_meta[unlab_meta["filename"].isin(files_with_pos)]["_orig_idx"].to_numpy()

    aug_meta = pd.concat([
        cache.meta.iloc[labeled_idx].reset_index(drop=True),
        cache.meta.iloc[keep_unlab].reset_index(drop=True),
    ], ignore_index=True)
    aug_scores = np.concatenate([cache.scores[labeled_idx], cache.scores[keep_unlab]], axis=0)
    aug_emb    = np.concatenate([cache.emb[labeled_idx],    cache.emb[keep_unlab]],    axis=0)
    aug_Y      = np.concatenate([cache.Y[labeled_idx],      pseudo_Y[keep_unlab]],     axis=0)
    aug_labeled_mask = np.ones(len(aug_meta), dtype=bool)

    n_lab = len(labeled_idx)
    n_pseudo_files = len(files_with_pos)
    n_pseudo_rows  = len(keep_unlab)
    n_pseudo_pos   = int(aug_Y[n_lab:].sum())
    # Sanity: rows must be a clean multiple of 12 for both labeled and pseudo.
    assert n_lab % 12 == 0, f"labeled rows ({n_lab}) not divisible by 12"
    assert n_pseudo_rows % 12 == 0, f"pseudo rows ({n_pseudo_rows}) not divisible by 12 — file grouping bug"
    print(f"[pseudo] augmenting cache: labeled={n_lab} ({n_lab//12} files)  "
          f"pseudo={n_pseudo_rows} ({n_pseudo_files} files)  "
          f"pseudo-positives={n_pseudo_pos}  "
          f"τ={pseudo_tau}  round={pseudo_round}")
    # The augmented cache's scores already include whatever proxy decision
    # the caller made upstream (proxy merged at load stage, or pure direct).
    # Pass an all-zero scores_proxy to satisfy the dataclass invariant — no
    # further merge is needed downstream.
    return PerchCache(
        meta=aug_meta,
        scores=aug_scores,
        scores_proxy=np.zeros_like(aug_scores),
        emb=aug_emb,
        Y=aug_Y,
        labeled_mask=aug_labeled_mask,
    )


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
    # Apply the genus-proxy fill at the same stage the LB notebook does
    # (cell 6 / 8 in cand.ipynb / LB_0931_seed.ipynb — right after reading
    # the cache off disk and before any downstream consumer touches the
    # scores). cfg["use_perch_proxy"] = True is the default so legacy SSM
    # behaviour is preserved; set False to ablate and baseline on pure
    # direct-mapped Perch logits (the public-notebook 0.729 number).
    if bool(cfg.get("use_perch_proxy", True)):
        from birdclef.models.perch import apply_proxy_to_scores
        cache.scores = apply_proxy_to_scores(cache.scores, cache.scores_proxy)

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
    # `cache_scores` already reflects the proxy decision (merged or not) made
    # at the load stage above. Subset's `scores_proxy` is zero so no
    # downstream code re-applies it.
    cache_sub = PerchCache(
        meta=cache_meta,
        scores=cache_scores,
        scores_proxy=np.zeros_like(cache_scores),
        emb=cache_emb,
        Y=cache_Y,
        labeled_mask=np.ones(len(cache_meta), dtype=bool),
    )

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

    # OOF by file-level KFold (n_splits + kind configurable per cfg).
    # kind="strat"    : single-label modal StratifiedKFold on filename (default)
    # kind="site"     : GroupKFold on site (strict but unbalanced)
    # kind="sitedate" : GroupKFold on (site, date) (balanced site-aware)
    n_splits_cfg = int(cfg.get("n_splits", 5))
    fold_kind_cfg = str(cfg.get("fold_kind", "strat"))
    folds = load_folds(n_splits=n_splits_cfg, kind=fold_kind_cfg)
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
    fold_pbar = tqdm(range(n_splits), desc=f"folds[{cfg.get('name','?')}]",
                     dynamic_ncols=True)
    for f in fold_pbar:
        # Train: every row except this fold's val. Pinned (-1) and unlabeled
        # (-1 from fillna) both pass `row_fold != f` for any f >= 0, so they
        # land in train as intended. The labeled-mask filter upstream
        # already removed truly unlabeled rows from cache_meta.
        tr = np.where(row_fold != f)[0]
        va = np.where(row_fold == f)[0]
        if len(va) == 0:
            continue
        fold_pbar.set_postfix(fold=f, n_train=len(tr), n_val=len(va), refresh=False)
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
