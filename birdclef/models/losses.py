"""Loss functions for BirdCLEF 2026.

Design notes:
- Winners 2024/25 used FocalBCE or BCE+Focal mean. Label smoothing ≤ 0.05.
- 2024 3rd: masking secondary-label loss added +0.01 LB. We expose this as an
  explicit mask tensor (1 where the loss should be computed, 0 elsewhere).
- pos_weight is NOT used; focal handles imbalance more cleanly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _apply_label_smoothing(targets: Tensor, eps: float) -> Tensor:
    if eps <= 0.0:
        return targets
    return targets * (1.0 - eps) + (1.0 - targets) * eps


class FocalBCE(nn.Module):
    """Focal Binary Cross-Entropy with optional secondary-label masking.

    alpha: weight of positive class (α=0.25 standard — down-weights easy negs).
    gamma: focusing parameter (γ=2.0 standard — pushes loss toward hard examples).
    label_smoothing: applied before BCE.
    mask_mode: "zero" | "halve" | "off" — how to treat masked positions.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.03,
        mask_mode: str = "zero",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.mask_mode = str(mask_mode)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        t = _apply_label_smoothing(targets, self.label_smoothing)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        probs = torch.sigmoid(logits)
        pt = t * probs + (1.0 - t) * (1.0 - probs)
        w = t * self.alpha + (1.0 - t) * (1.0 - self.alpha)
        loss = w * (1.0 - pt).pow(self.gamma) * bce
        if loss_mask is not None:
            if self.mask_mode == "zero":
                loss = loss * loss_mask
            elif self.mask_mode == "halve":
                loss = loss * (0.5 + 0.5 * loss_mask)
            # "off" -> do nothing
        return loss.mean()


class BCEFocalMean(nn.Module):
    """0.5*BCE + 0.5*Focal mean — 2024 #2 recipe."""

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.03,
    ):
        super().__init__()
        self.focal = FocalBCE(
            alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing
        )
        self.ls = float(label_smoothing)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        t = _apply_label_smoothing(targets, self.ls)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        if loss_mask is not None:
            bce = bce * loss_mask
        return 0.5 * bce.mean() + 0.5 * self.focal(logits, targets, loss_mask)


class BCEPosWeight(nn.Module):
    """Legacy loss (current pipeline's default). Kept for A/B tests."""

    def __init__(self, pos_weight: Optional[Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight, persistent=False)
        self.ls = float(label_smoothing)

    def forward(
        self, logits: Tensor, targets: Tensor, loss_mask: Optional[Tensor] = None
    ) -> Tensor:
        t = _apply_label_smoothing(targets, self.ls)
        bce = F.binary_cross_entropy_with_logits(
            logits, t, reduction="none", pos_weight=self.pos_weight
        )
        if loss_mask is not None:
            bce = bce * loss_mask
        return bce.mean()


LOSS_REGISTRY = {
    "focal_bce": FocalBCE,
    "bce_focal_mean": BCEFocalMean,
    "bce_posw": BCEPosWeight,
}


def build_loss(kind: str, **kwargs) -> nn.Module:
    if kind not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{kind}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[kind](**kwargs)
