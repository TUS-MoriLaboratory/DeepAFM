import torch

def calculate_center(xyz):
    """
    Calculate the center of the XYZ coordinates.
    Parameters:
    - xyz: 2D tensor of shape (num_atoms, 3) representing the XYZ coordinates of the atoms.

    Returns:
    - center: 1D tensor of shape (3,) representing the center of the XYZ coordinates.
    """

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("Input xyz must be a 2D tensor with shape (num_atoms, 3)")
    center = torch.mean(xyz, dim=0)

    return center

def calculate_center_batch(xyz):
    """
    Calculate the center of the XYZ coordinates.
    Parameters:
    - xyz: 3D tensor of shape (N, num_atoms, 3) representing the XYZ coordinates of the atoms.

    Returns:
    - center: 2D tensor of shape (N, 3) representing the center of the XYZ coordinates.
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("Input xyz must be a 3D tensor with shape (N, num_atoms, 3)")
    center = torch.mean(xyz, dim=1)

    return center