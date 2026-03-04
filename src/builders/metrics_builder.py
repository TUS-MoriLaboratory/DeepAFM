import torch
import torch.nn.functional as F
from typing import Dict, Callable

# =========================================================
# Metric Functions
# =========================================================

def calc_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Classification Accuracy (Top-1 Accuracy)"""
    # preds: (B, C), targets: (B)
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)
    correct = (preds == targets).float().sum()
    return correct  

def calc_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (MSE) - Used as both loss and metric"""
    batch_size = targets.size(0)
    num_pixels_per_image = targets.numel() / batch_size

    return F.mse_loss(preds, targets, reduction='sum') / num_pixels_per_image

def calc_mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (MAE)"""
    batch_size = targets.size(0)
    num_pixels_per_image = targets.numel() / batch_size

    return F.l1_loss(preds, targets, reduction='sum') / num_pixels_per_image

# =========================================================
# 2. Builder (Factory)
# =========================================================

def build_metrics(cfg) -> Dict[str, Callable]:
    """
    Returns a dictionary of metric functions based on the Task Mode
    Returns:
        metrics (Dict[str, Callable]): {"acc": calc_accuracy, ...}
    """
    task_mode = cfg.train.task_mode
    metrics = {}

    # -------------------------------------------------
    # Classification
    # -------------------------------------------------
    if task_mode == "classification":
        metrics["acc"] = calc_accuracy

    # -------------------------------------------------
    # Denoise / Reconstruction
    # -------------------------------------------------
    elif task_mode in ["denoise", "reconstruction"]:
        metrics["mse"] = calc_mse
        metrics["mae"] = calc_mae

    # -------------------------------------------------
    # Multi-task
    # -------------------------------------------------
    elif task_mode == "multitask":
        metrics["state_acc"] = lambda p, t: calc_accuracy(p["state"], t["state"])
        metrics["ideal_mse"] = lambda p, t: calc_mse(p["ideal"], t["ideal"])        
        metrics["ideal_mae"] = lambda p, t: calc_mae(p["ideal"], t["ideal"])

        return metrics

    else:
        # Return an empty dictionary if the task is unknown (no evaluation)
        pass

    return metrics