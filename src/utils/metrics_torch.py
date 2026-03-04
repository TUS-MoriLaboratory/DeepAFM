# src/utils/metrics_torch.py

import torch

def mse_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Squared Error."""
    return float(torch.mean((a - b) ** 2))


def rmse_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


def mae_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return float(torch.mean(torch.abs(a - b)))


def correlation_coefficient_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient (PyTorch)."""
    a = a.flatten().float()
    b = b.flatten().float()

    # Avoid division by zero
    if torch.std(a) == 0 or torch.std(b) == 0:
        return 0.0

    cov = torch.mean((a - a.mean()) * (b - b.mean()))
    corr = cov / (a.std() * b.std() + 1e-12)
    return float(corr)


def batch_correlation_coefficient_torch(sim_img: torch.Tensor, ref_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient between a single simulation image 
    and a batch of reference images using broadcasting.

    Args:
        sim_img (torch.Tensor): Simulation image. Shape (1, H, W) or (H, W).
        ref_batch (torch.Tensor): Batch of reference images. Shape (B, H, W).

    Returns:
        torch.Tensor: Correlation coefficients for each reference image. Shape (B).
    """
    # Ensure float
    sim_img = sim_img.float()
    ref_batch = ref_batch.float()

    # Ensure sim_img is (1, H, W) for broadcasting
    if sim_img.dim() == 2:
        sim_img = sim_img.unsqueeze(0)

    # --- Mean subtraction ---
    # dim=(-2, -1) means computing over Height and Width
    mean_sim = sim_img.mean(dim=(-2, -1), keepdim=True) # (1, 1, 1)
    mean_ref = ref_batch.mean(dim=(-2, -1), keepdim=True) # (B, 1, 1)

    sub_sim = sim_img - mean_sim # (1, H, W)
    sub_ref = ref_batch - mean_ref # (B, H, W)

    # --- Numerator (Covariance part) ---
    # Broadcasting happens here: (1, H, W) * (B, H, W) -> (B, H, W)
    numerator = (sub_sim * sub_ref).sum(dim=(-2, -1)) # (B,)

    # --- Denominator (Standard Deviation part) ---
    # Note: The (N-1) term for unbiased estimator cancels out in the division,
    # so we can just use the sum of squared differences.
    sum_sq_sim = (sub_sim ** 2).sum(dim=(-2, -1)) # (1,)
    sum_sq_ref = (sub_ref ** 2).sum(dim=(-2, -1)) # (B,)
    
    denominator = torch.sqrt(sum_sq_sim * sum_sq_ref) # (B,)

    # --- Result ---
    # Add epsilon to prevent division by zero
    return numerator / (denominator + 1e-12)