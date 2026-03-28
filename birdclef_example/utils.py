import random
from pathlib import Path
from typing import Dict

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


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion, device: torch.device):
    model.eval()
    losses = []
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            all_targets.append(targets.cpu())
            all_preds.append(torch.sigmoid(logits).cpu())
    avg_loss = np.mean(losses) if losses else 0.0
    if all_targets:
        target_tensor = torch.cat(all_targets, dim=0).numpy()
        pred_tensor = torch.cat(all_preds, dim=0).numpy()
        try:
            auc = roc_auc_score(target_tensor, pred_tensor, average="macro")
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")
    return avg_loss, auc


def save_model(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
