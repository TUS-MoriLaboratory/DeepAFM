import torch

def translate_batch(xyz, x_offset, y_offset, z_offset=None, inplace=False):
    """
    Batch translation of XYZ coordinates.

    Parameters:
        xyz:       (N, num_atoms, 3) tensor
        x_offset:  (N,)      tensor
        y_offset:  (N,)      tensor
        z_offset:  (N,) or None  (optional)
        inplace:   bool  If True, modifies xyz in place.

    Returns:
        translated_xyz: (N, num_atoms, 3)
    """

    # --- Input validation ---
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("xyz must be (N, num_atoms, 3)")

    N = xyz.shape[0]
    device = xyz.device
    dtype = xyz.dtype

    x_offset = x_offset.to(device=device, dtype=dtype)
    y_offset = y_offset.to(device=device, dtype=dtype)
    if z_offset is not None:
        z_offset = z_offset.to(device=device, dtype=dtype)

    # --- expand to (N, 1, 1) ---
    x = x_offset[:, None, None]
    y = y_offset[:, None, None]
    if z_offset is not None:
        z = z_offset[:, None, None]

    if inplace:
        xyz[:, :, 0:1] += x
        xyz[:, :, 1:2] += y
        if z_offset is not None:
            xyz[:, :, 2:3] += z
        return xyz

    # --- out-of-place version ---
    out = xyz.clone()
    out[:, :, 0:1] += x
    out[:, :, 1:2] += y
    if z_offset is not None:
        out[:, :, 2:3] += z

    return out



def translate(xyz, x_offset, y_offset, z_offset=None, inplace=False):
    """
    Single translation of XYZ coordinates.

    Parameters:
        xyz:       (num_atoms, 3) tensor
        x_offset:  float or tensor
        y_offset:  float or tensor
        z_offset:  float or tensor (optional)
        inplace:   bool

    Returns:
        translated_xyz: (num_atoms, 3)
    """

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must be (num_atoms, 3)")
    device = xyz.device
    dtype = xyz.dtype

    x_offset = torch.tensor(x_offset, device=device, dtype=dtype)
    y_offset = torch.tensor(y_offset, device=device, dtype=dtype)
    if z_offset is not None:
        z_offset = torch.tensor(z_offset, device=device, dtype=dtype)

    if inplace:
        xyz[:, 0] += x_offset
        xyz[:, 1] += y_offset
        if z_offset is not None:
            xyz[:, 2] += z_offset
        return xyz

    out = xyz.clone()
    out[:, 0] += x_offset
    out[:, 1] += y_offset
    if z_offset is not None:
        out[:, 2] += z_offset

    return out
