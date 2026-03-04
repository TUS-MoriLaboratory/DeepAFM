import torch
import math
import numpy as np

def center_crop_or_pad(arr, n):
    """ Center-crop or pad a 2D array to size (n, n)."""
    assert arr.ndim == 2, "Input array must be 2D."

    h, w = arr.shape
    out = np.full((n, n), fill_value=arr.min(), dtype=arr.dtype)

    # crop indices
    rs = max((h-n)//2, 0)
    cs = max((w-n)//2, 0)

    cropped = arr[rs:rs+n, cs:cs+n]

    # paste position
    prs = max((n-h)//2, 0)
    pcs = max((n-w)//2, 0)

    out[prs:prs+cropped.shape[0], pcs:pcs+cropped.shape[1]] = cropped
    return out

def pad_center(
        img: torch.Tensor,
        target_H: int,
        target_W: int,
        value: float = 0.0
    ) -> torch.Tensor:
    """
    Apply centered zero-padding to AFM images (H×W).

    Parameters
    ----------
    img : torch.Tensor
        Input tensor of shape (H, W) or (C, H, W)
    target_H : int
        Target height of output image
    target_W : int
        Target width of output image
    value : float
        Padding value (default: 0.0)

    Returns
    -------
    padded_img : torch.Tensor
        Padded image centered in the output tensor
    """

    if img.ndim == 2:
        H, W = img.shape
        C = None
    elif img.ndim == 3:
        C, H, W = img.shape
    else:
        raise ValueError("img must be 2D or 3D tensor")

    if H > target_H or W > target_W:
        raise ValueError(f"Image ({H},{W}) exceeds target size ({target_H},{target_W}).")

    # padding lengths
    pad_H = target_H - H
    pad_W = target_W - W

    top = pad_H // 2
    bottom = pad_H - top
    left = pad_W // 2
    right = pad_W - left

    if C is None:
        padded = torch.nn.functional.pad(
            img,
            pad=(left, right, top, bottom),  # (left, right, top, bottom)
            mode='constant',
            value=value
        )
    else:
        padded = torch.nn.functional.pad(
            img,
            pad=(left, right, top, bottom),
            mode='constant',
            value=value
        )

    return padded

def translate_image(
        img: torch.Tensor, 
        trans_x: int, 
        trans_y: int
    ) -> torch.Tensor:
    """
    Translate image with Cartesian coordinate behavior (Up is positive Y).
    Uses circular shift (torch.roll).

    Parameters
    ----------
    img : torch.Tensor
        (H, W) or (C, H, W)
    trans_x : int
        > 0: Right, < 0: Left
    trans_y : int
        > 0: Up, < 0: Down  (Cartesian convention)

    Returns
    -------
    torch.Tensor
        Translated image
    """
    # Convert from Cartesian coordinates (positive Y is up) to image coordinates (positive Y is down)
    # Negate the Y translation amount
    shifts = (-trans_y, trans_x)
    
    # By specifying dims=(-2, -1), the operation is applied to the last two dimensions (height and width)
    return torch.roll(img, shifts=shifts, dims=(-2, -1))