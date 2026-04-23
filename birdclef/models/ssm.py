"""LightProtoSSM + ResidualSSM — clean port from birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py.

No CFG globals; takes explicit dataclass configs. Functionally equivalent to
the original so the regression gate passes within ±0.002 macro-AUC.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSMHeadConfig:
    d_input: int = 1536
    d_model: int = 128
    d_state: int = 16
    n_classes: int = 234
    n_windows: int = 12
    dropout: float = 0.15
    n_sites: int = 20
    meta_dim: int = 16
    use_cross_attn: bool = True
    cross_attn_heads: int = 2


@dataclass
class ResidualSSMConfig:
    d_input: int = 1536
    d_scores: int = 234
    d_model: int = 64
    d_state: int = 8
    n_classes: int = 234
    n_windows: int = 12
    dropout: float = 0.1
    n_sites: int = 20
    meta_dim: int = 8


class SelectiveSSM(nn.Module):
    """Mamba-style selective SSM with O(T*D*N) sequential scan on CPU."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_model, -1)
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, _ = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """Bidirectional SSM + cross-attn + class prototypes (frozen Perch inputs)."""

    def __init__(self, cfg: SSMHeadConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(c.d_input, c.d_model),
            nn.LayerNorm(c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, c.n_windows, c.d_model) * 0.02)
        self.site_emb = nn.Embedding(c.n_sites, c.meta_dim)
        self.hour_emb = nn.Embedding(24, c.meta_dim)
        self.meta_proj = nn.Linear(2 * c.meta_dim, c.d_model)
        self.ssm_fwd = nn.ModuleList([SelectiveSSM(c.d_model, c.d_state) for _ in range(2)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(c.d_model, c.d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * c.d_model, c.d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(c.d_model) for _ in range(2)])
        self.drop = nn.Dropout(c.dropout)
        self.cross_attn = None
        self.cross_norm = None
        if c.use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(c.d_model, num_heads=c.cross_attn_heads,
                                      dropout=c.dropout, batch_first=True)
                for _ in range(2)
            ])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(c.d_model) for _ in range(2)])
        self.prototypes = nn.Parameter(torch.randn(c.n_classes, c.d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(c.n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(c.n_classes))

    def init_prototypes(self, emb_flat: torch.Tensor, labels_flat: torch.Tensor) -> None:
        with torch.no_grad():
            h = self.input_proj(emb_flat)
            for c in range(self.cfg.n_classes):
                m = labels_flat[:, c] > 0.5
                if m.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[m].mean(0), dim=0)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            )
            h = h + meta[:, None, :]
        for i in range(len(self.ssm_fwd)):
            res = h
            h_f = self.ssm_fwd[i](h)
            h_b = self.ssm_bwd[i](h.flip(1)).flip(1)
            h = self.drop(self.ssm_merge[i](torch.cat([h_f, h_b], dim=-1)))
            h = self.ssm_norm[i](h + res)
            if self.cross_attn is not None:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)
        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = (
            torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp)
            + self.class_bias[None, None, :]
        )
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            return alpha * sim + (1 - alpha) * perch_logits
        return sim


class ResidualSSM(nn.Module):
    """Predicts the residual between first-pass sigmoid and labels."""

    def __init__(self, cfg: ResidualSSMConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(c.d_input + c.d_scores, c.d_model),
            nn.LayerNorm(c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.site_emb = nn.Embedding(c.n_sites, c.meta_dim)
        self.hour_emb = nn.Embedding(24, c.meta_dim)
        self.meta_proj = nn.Linear(2 * c.meta_dim, c.d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, c.n_windows, c.d_model) * 0.02)
        self.ssm_fwd = SelectiveSSM(c.d_model, c.d_state)
        self.ssm_bwd = SelectiveSSM(c.d_model, c.d_state)
        self.ssm_merge = nn.Linear(2 * c.d_model, c.d_model)
        self.ssm_norm = nn.LayerNorm(c.d_model)
        self.ssm_drop = nn.Dropout(c.dropout)
        self.output_head = nn.Linear(c.d_model, c.n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        emb: torch.Tensor,
        first_pass: torch.Tensor,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat([
                    self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                    self.hour_emb(hours.clamp(0, 23)),
                ], dim=-1)
            )
            h = h + meta.unsqueeze(1)
        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([h_f, h_b], dim=-1)))
        h = self.ssm_norm(h + res)
        return self.output_head(h)
