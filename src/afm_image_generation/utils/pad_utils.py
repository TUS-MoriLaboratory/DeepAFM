import torch

@torch.jit.script
def pad_center_fast(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Center-pad a 2D tensor (H×W) into (target_h × target_w).
    Zero padding around.
    """
    h, w = img.shape[-2], img.shape[-1]

    out = img.new_zeros((target_h, target_w))  # fast allocation

    top = (target_h - h) // 2
    left = (target_w - w) // 2

    out[top:top+h, left:left+w] = img
    return out

def pad_pair(dist_img, ideal_img, target_h, target_w):
    dist = pad_center_fast(dist_img, target_h, target_w) if dist_img is not None else None
    ideal = pad_center_fast(ideal_img, target_h, target_w) if ideal_img is not None else None
    return dist, ideal