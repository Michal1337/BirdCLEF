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
    """Exact equivalent of Kaggle macro ROC-AUC (skip classes with no positives)."""

    if targets.shape != preds.shape:
        raise ValueError(
            f"Targets and predictions must have the same shape, got {targets.shape} and {preds.shape}."
        )
    if targets.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got targets.ndim={targets.ndim}.")

    # Select classes with at least one positive
    class_sums = targets.sum(axis=0)
    valid_classes = np.where(class_sums > 0)[0]

    if len(valid_classes) == 0:
        raise ValueError("No valid classes with positive samples.")

    # Slice only valid classes
    targets_filtered = targets[:, valid_classes]
    preds_filtered = preds[:, valid_classes]

    # Compute macro ROC-AUC (same as Kaggle)
    return roc_auc_score(targets_filtered, preds_filtered, average="macro")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    use_bf16: bool = False,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    all_targets = []
    all_preds = []
    amp_enabled = use_bf16 and device.type == "cuda"

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(inputs)
                loss = criterion(logits, targets)
            losses.append(loss.item())
            all_targets.append(targets.float().cpu())
            all_preds.append(torch.sigmoid(logits).float().cpu())

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
