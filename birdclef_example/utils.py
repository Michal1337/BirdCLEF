import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def birdclef_roc_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """Competition-style macro ROC AUC that skips degenerate classes."""

    if targets.shape != preds.shape:
        raise ValueError(
            f"Targets and predictions must have the same shape, got {targets.shape} and {preds.shape}."
        )
    if targets.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got targets.ndim={targets.ndim}.")

    class_scores = []
    for class_idx in range(targets.shape[1]):
        class_targets = targets[:, class_idx]
        if np.unique(class_targets).size < 2:
            continue
        class_scores.append(roc_auc_score(class_targets, preds[:, class_idx]))

    if not class_scores:
        return float("nan")
    return float(np.mean(class_scores))


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            all_targets.append(targets.cpu())
            all_preds.append(torch.sigmoid(logits).cpu())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    if not all_targets:
        return avg_loss, float("nan")

    target_array = torch.cat(all_targets, dim=0).numpy()
    pred_array = torch.cat(all_preds, dim=0).numpy()
    auc = birdclef_roc_auc(target_array, pred_array)
    return avg_loss, auc


def is_better_score(current_score: float, best_score: float) -> bool:
    if math.isnan(current_score):
        return False
    if math.isnan(best_score):
        return True
    return current_score > best_score


def save_model(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
