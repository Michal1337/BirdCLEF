#!/usr/bin/env python3

from pathlib import Path
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models" / "perch_v2_cpu" / "1"
N_WINDOWS = 12
DEVICE = torch.device("cpu")

# Keep filename parsing identical to other scripts.
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

# Ported ProtoSSM hyperparameters (V18) from predict_ported.py.
CFG = {
    "proto_ssm": {
        "d_model": 320,
        "d_state": 32,
        "n_ssm_layers": 4,
        "dropout": 0.12,
        "n_sites": 20,
        "meta_dim": 24,
        "use_cross_attn": True,
        "cross_attn_heads": 8,
    },
    "proto_ssm_train": {
        "n_epochs": 80,
        "lr": 8e-4,
        "weight_decay": 1e-3,
        "patience": 20,
        "pos_weight_cap": 25.0,
        "distill_weight": 0.15,
        "label_smoothing": 0.03,
        "oof_n_splits": 5,
        "mixup_alpha": 0.4,
        "focal_gamma": 2.5,
        "swa_start_frac": 0.65,
    },
}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]


def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }


def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))


def get_genus_hits(scientific_name, bc_labels):
    genus = str(scientific_name).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
    ].copy()
    return genus, hits


def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


def resolve_full_cache_paths(cache_root: Path):
    meta_path = cache_root / "full_perch_meta.parquet"
    npz_path = cache_root / "full_perch_arrays.npz"
    if meta_path.exists() and npz_path.exists():
        return meta_path, npz_path
    return None, None


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows, f"Expected {n_files * n_windows} rows, got {len(flat_array)}"
    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files


def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    n_sites = len(sites) + 1
    return site_to_idx, n_sites


def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites = meta_df["site"].to_numpy()
    hours = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            sid = site_to_idx.get(sites[row], 0)
            site_ids[fi] = min(sid, n_sites_max - 1)
            hour_ids[fi] = int(hours[row]) % 24
    return site_ids, hour_ids


def mixup_files(emb, logits, labels, site_ids, hours, alpha=0.3):
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    perm = np.random.permutation(n)

    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]
    return emb_mix, logits_mix, labels_mix, site_ids, hours


def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    return (((1 - pt) ** gamma) * bce).mean()


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        bsz, tlen, d = x.shape
        xz = self.in_proj(x)
        x_ssm, _ = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :tlen].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(bsz, d, self.d_state, device=x.device)
        ys = []
        for t in range(tlen):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        r = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = r + attn_out
        r = x
        x = self.norm2(x)
        return r + self.ffn(x)


class ProtoSSMv2(nn.Module):
    def __init__(self, d_input, d_model, d_state, n_ssm_layers, n_classes, n_windows, dropout, n_sites, meta_dim, use_cross_attn, cross_attn_heads):
        super().__init__()
        self.n_classes = n_classes
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.pos_enc.shape[-1], n_families)
        self.register_buffer("class_to_family", torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        bsz, tlen, _ = emb.shape
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :tlen, :]

        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
            r = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + r)

        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal


def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    else:
        group_map = {label: "Unknown" for label in primary_labels}

    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = []
    for label in primary_labels:
        grp = group_map.get(label, "Unknown")
        class_to_group.append(grp_to_idx.get(grp, 0))
    return len(groups), class_to_group, grp_to_idx


def train_proto_ssm_single(
    model,
    emb_train,
    logits_train,
    labels_train,
    site_ids_train,
    hours_train,
    emb_val,
    logits_val,
    labels_val,
    site_ids_val,
    hours_val,
    file_families_train,
    file_families_val,
    cfg,
):
    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * cfg.get("swa_start_frac", 1.0))

    labels_np = labels_train.copy()
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], epochs=n_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos")

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    swa_state = None
    swa_count = 0

    emb_v = torch.tensor(emb_val, dtype=torch.float32)
    logits_v = torch.tensor(logits_val, dtype=torch.float32)
    labels_v = torch.tensor(labels_val, dtype=torch.float32)
    site_v = torch.tensor(site_ids_val, dtype=torch.long)
    hour_v = torch.tensor(hours_val, dtype=torch.long)
    fam_v = torch.tensor(file_families_val, dtype=torch.float32) if file_families_val is not None else None

    for epoch in range(n_epochs):
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix, _, _ = mixup_files(emb_train, logits_train, labels_np, site_ids_train, hours_train, alpha=mixup_alpha)
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr = torch.tensor(site_ids_train, dtype=torch.long)
        hour_tr = torch.tensor(hours_train, dtype=torch.long)
        fam_tr = torch.tensor(file_families_train, dtype=torch.float32) if file_families_train is not None else None

        model.train()
        out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)
        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(out, labels_tr, gamma=focal_gamma, pos_weight=pos_weight[None, None, :])
        else:
            loss_main = F.binary_cross_entropy_with_logits(out, labels_tr, pos_weight=pos_weight[None, None, :])
        loss_distill = F.mse_loss(out, logits_tr)
        loss = loss_main + cfg["distill_weight"] * loss_distill
        if family_out is not None and fam_tr is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
            loss = loss + 0.1 * loss_family

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        model.eval()
        with torch.no_grad():
            val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
            val_loss = F.binary_cross_entropy_with_logits(val_out, labels_v, pos_weight=pos_weight[None, None, :])
            if val_fam is not None and fam_v is not None:
                val_loss = val_loss + 0.1 * F.binary_cross_entropy_with_logits(val_fam, fam_v)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} wait={wait}")

        if wait >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    if swa_state is not None and swa_count >= 3:
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_proto_ssm_oof(
    emb_files,
    logits_files,
    labels_files,
    file_list,
    site_ids_all,
    hours_all,
    file_families,
    n_families,
    class_to_family,
    n_classes,
    cfg_model,
    cfg_train,
):
    n_splits = cfg_train.get("oof_n_splits", 5)
    n_files = len(emb_files)
    oof_preds = np.zeros((n_files, N_WINDOWS, n_classes), dtype=np.float32)

    file_level_labels = labels_files.max(axis=1)
    file_row_sum = file_level_labels.sum(axis=1)
    y_strat = np.where(file_row_sum > 0, np.argmax(file_level_labels, axis=1), -1).astype(np.int32)

    unique_classes, counts = np.unique(y_strat, return_counts=True)
    rare_classes = unique_classes[counts < n_splits]
    y_strat[np.isin(y_strat, rare_classes)] = -1

    groups = np.asarray(file_list)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)
    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(emb_files, y_strat, groups=groups), 1):
        train_files = set(groups[train_idx].tolist())
        val_files = set(groups[val_idx].tolist())
        overlap = train_files.intersection(val_files)
        if overlap:
            raise RuntimeError(
                f"Fold {fold_i}: validation contains seen files ({len(overlap)} overlaps); example={next(iter(overlap))}"
            )
        print(f"\n--- Fold {fold_i}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---")
        model = ProtoSSMv2(
            d_input=emb_files.shape[2],
            d_model=cfg_model["d_model"],
            d_state=cfg_model["d_state"],
            n_ssm_layers=cfg_model["n_ssm_layers"],
            n_classes=n_classes,
            n_windows=N_WINDOWS,
            dropout=cfg_model["dropout"],
            n_sites=cfg_model["n_sites"],
            meta_dim=cfg_model["meta_dim"],
            use_cross_attn=cfg_model.get("use_cross_attn", True),
            cross_attn_heads=cfg_model.get("cross_attn_heads", 4),
        ).to(DEVICE)

        emb_flat_fold = emb_files[train_idx].reshape(-1, emb_files.shape[2])
        labels_flat_fold = labels_files[train_idx].reshape(-1, n_classes)
        model.init_prototypes_from_data(torch.tensor(emb_flat_fold, dtype=torch.float32), torch.tensor(labels_flat_fold, dtype=torch.float32))
        model.init_family_head(n_families, class_to_family)

        model = train_proto_ssm_single(
            model,
            emb_files[train_idx], logits_files[train_idx], labels_files[train_idx].astype(np.float32),
            site_ids_all[train_idx], hours_all[train_idx],
            emb_files[val_idx], logits_files[val_idx], labels_files[val_idx].astype(np.float32),
            site_ids_all[val_idx], hours_all[val_idx],
            file_families[train_idx], file_families[val_idx],
            cfg_train,
        )

        model.eval()
        with torch.no_grad():
            val_emb = torch.tensor(emb_files[val_idx], dtype=torch.float32)
            val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
            val_sites = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
            val_hours = torch.tensor(hours_all[val_idx], dtype=torch.long)
            val_out, _, _ = model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
            oof_preds[val_idx] = val_out.numpy()

    return oof_preds


def main():
    seed_everything(1891)

    cache_root = Path(os.environ.get("PERCH_CACHE_DIR", REPO_ROOT / "data" / "perch_cache"))

    sample_sub = pd.read_csv(BASE / "sample_submission.csv")
    taxonomy = pd.read_csv(BASE / "taxonomy.csv")
    taxonomy = pd.read_csv(BASE / "taxonomy.csv")
    soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)
    meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta], axis=1)

    windows_per_file = sc_clean.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    bc_labels = (
        pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    no_label_index = len(bc_labels)
    manual_scientific_name_map = {}

    taxonomy_lookup = taxonomy.copy()
    taxonomy_lookup["scientific_name_lookup"] = taxonomy_lookup["scientific_name"].replace(manual_scientific_name_map)
    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
    mapping = taxonomy_lookup.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label_index).astype(int)

    class_name_map = taxonomy_lookup.set_index("primary_label")["class_name"].to_dict()
    proxy_taxa = {"Amphibia", "Insecta", "Aves"}

    unmapped_df = mapping[mapping["bc_index"] == no_label_index].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()
    proxy_map = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = row["primary_label"]
        sci = row["scientific_name"]
        genus, hits = get_genus_hits(sci, bc_labels)
        if len(hits) > 0:
            proxy_map[target] = {
                "target_scientific_name": sci,
                "genus": genus,
                "bc_indices": hits["bc_index"].astype(int).tolist(),
            }
    selected_proxy_targets = sorted([
        t for t in proxy_map.keys()
        if class_name_map.get(t) in proxy_taxa
    ])

    print(f"Proxy targets selected: {len(selected_proxy_targets)}")
    if selected_proxy_targets:
        print("Sample proxy targets:", selected_proxy_targets[:20])
    y_sc = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            y_sc[i, idxs] = 1

    full_truth = (
        sc_clean[sc_clean["file_fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )

    cache_meta, cache_npz = resolve_full_cache_paths(cache_root)
    if cache_meta is None or cache_npz is None:
        raise FileNotFoundError(
            f"Could not find full Perch cache files under {cache_root}. Expected full_perch_meta.parquet and full_perch_arrays.npz."
        )

    meta_full = pd.read_parquet(cache_meta)
    arr = np.load(cache_npz)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    emb_full = arr["emb_full"].astype(np.float32)

    full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
    y_full = y_sc[full_truth_aligned["index"].to_numpy()]

    assert np.all(full_truth_aligned["filename"].values == meta_full["filename"].values)
    assert np.all(full_truth_aligned["row_id"].values == meta_full["row_id"].values)

    raw_local_auc = macro_auc_skip_empty(y_full, scores_full_raw)

    emb_files, file_list = reshape_to_files(emb_full, meta_full)
    logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _ = reshape_to_files(y_full, meta_full)

    n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, primary_labels)
    file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
    for fi in range(len(file_list)):
        active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
        for ci in active_classes:
            file_families[fi, class_to_family[ci]] = 1.0

    site_to_idx, _ = build_site_mapping(meta_full)
    site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

    oof_proto_preds = run_proto_ssm_oof(
        emb_files=emb_files,
        logits_files=logits_files,
        labels_files=labels_files,
        file_list=file_list,
        site_ids_all=site_ids_all,
        hours_all=hours_all,
        file_families=file_families,
        n_families=n_families,
        class_to_family=class_to_family,
        n_classes=n_classes,
        cfg_model=CFG["proto_ssm"],
        cfg_train=CFG["proto_ssm_train"],
    )
    oof_proto_flat = oof_proto_preds.reshape(-1, n_classes)
    proto_oof_auc = macro_auc_skip_empty(y_full.astype(np.float32), oof_proto_flat)

    print("=== ProtoSSM-Only Training Summary ===")
    print(f"Rows: {len(y_full)} | Classes: {n_classes}")
    print(f"Raw Perch AUC (in-sample): {raw_local_auc:.6f}")
    print(f"ProtoSSM OOF AUC (grouped by file): {proto_oof_auc:.6f}")
    print(f"Used cache meta: {cache_meta}")
    print(f"Used cache arrays: {cache_npz}")
    print(f"Proto hyperparameters: {CFG['proto_ssm']}")
    print(f"Proto train hyperparameters: {CFG['proto_ssm_train']}")


if __name__ == "__main__":
    main()
