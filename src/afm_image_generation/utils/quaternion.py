import torch
import math

# right-handed coordinate system
# z-x-y order
# clockwise rotation when looking along the axis toward the origin  

# ==============================================================
# Quaternion utilities (clockwise rotation)
# ==============================================================

def quat_from_axis_angle_clockwise(axis, deg):
    """
    Create a quaternion representing a clockwise rotation.
    axis: (3,) tensor
    deg: float or scalar tensor
    return: (4,) tensor
    """
    # Ensure axis is normalized
    axis = axis / axis.norm()

    # Convert degree → rad and put on same device/dtype
    rad = torch.deg2rad(torch.as_tensor(deg, dtype=axis.dtype, device=axis.device))

    half = -rad / 2.0  # clockwise
    w = torch.cos(half)
    s = torch.sin(half)

    xyz = axis * s
    q = torch.cat([w.view(1), xyz], dim=0)

    # Normalize to avoid drift
    return q / q.norm()


def quat_mul(q1, q2):
    """
    Hamilton product of two quaternions.
    q1, q2: (4,)
    return: (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=0)


def quat_zxy_clockwise(z_deg, x_deg, y_deg, xyz):
    """
    Construct Z→X→Y clockwise rotation quaternion.
    Angles are ALL clockwise.
    """
    z_axis = torch.tensor([0., 0., 1.], device=xyz.device, dtype=xyz.dtype)
    x_axis = torch.tensor([1., 0., 0.], device=xyz.device, dtype=xyz.dtype)
    y_axis = torch.tensor([0., 1., 0.], device=xyz.device, dtype=xyz.dtype)

    qz = quat_from_axis_angle_clockwise(z_axis, z_deg)
    qx = quat_from_axis_angle_clockwise(x_axis, x_deg)
    qy = quat_from_axis_angle_clockwise(y_axis, y_deg)

    # Z → X → Y
    return quat_mul(qy, quat_mul(qx, qz))


def quat_to_rotmat(q):
    """
    Convert quaternion → rotation matrix (3x3)
    q: (4,)
    """
    w, x, y, z = q

    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)], dim=0),
        torch.stack([2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=0),
        torch.stack([2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=0)
    ], dim=0)


def rotate_with_quat(xyz, z_deg, x_deg, y_deg):
    """
    xyz: (num_atoms, 3)
    Z→X→Y clockwise rotation
    """
    q = quat_zxy_clockwise(z_deg, x_deg, y_deg, xyz)
    R = quat_to_rotmat(q).to(xyz.device)
    return xyz @ R.T

def rotate_around_center(xyz, z_deg, x_deg, y_deg, center):
    """
    Rotate xyz around a specified center.
    xyz: (num_atoms, 3)
    center: (3,)
    """
    # Translate to origin
    translated = xyz - center
    # Rotate
    rotated = rotate_with_quat(translated, z_deg, x_deg, y_deg)
    # Translate back
    return rotated + center

# ==============================================================
# Batch version
# ==============================================================

def quat_from_axis_angle_clockwise_batch(axis, deg):
    """
    axis: (N, 3)
    deg:  (N,)
    return: (N, 4)
    """
    axis = axis / axis.norm(dim=-1, keepdim=True)

    rad = torch.deg2rad(deg)
    half = -rad / 2.0

    w = torch.cos(half)
    s = torch.sin(half)

    xyz = axis * s.unsqueeze(-1)
    q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)

    return q / q.norm(dim=-1, keepdim=True)


def quat_mul_batch(q1, q2):
    """
    q1, q2: (N, 4)
    return: (N, 4)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


def quat_zxy_clockwise_batch(z_deg, x_deg, y_deg, xyz):
    """
    Batch Z→X→Y clockwise quaternion
    """
    N = z_deg.shape[0]

    z_axis = torch.tensor([0., 0., 1.], device=xyz.device, dtype=xyz.dtype).expand(N, 3)
    x_axis = torch.tensor([1., 0., 0.], device=xyz.device, dtype=xyz.dtype).expand(N, 3)
    y_axis = torch.tensor([0., 1., 0.], device=xyz.device, dtype=xyz.dtype).expand(N, 3)

    qz = quat_from_axis_angle_clockwise_batch(z_axis, z_deg)
    qx = quat_from_axis_angle_clockwise_batch(x_axis, x_deg)
    qy = quat_from_axis_angle_clockwise_batch(y_axis, y_deg)

    # Z → X → Y
    return quat_mul_batch(qy, quat_mul_batch(qx, qz))


def quat_to_rotmat_batch(q):
    """
    q: (N, 4)
    return: (N, 3, 3)
    """
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1)
    ], dim=-2)
    return R


def rotate_batch(xyz, z_deg, x_deg, y_deg):
    """
    xyz: (N, num_atoms, 3)
    """
    q = quat_zxy_clockwise_batch(z_deg, x_deg, y_deg, xyz)  # (N,4)
    R = quat_to_rotmat_batch(q)                                        # (N,3,3)
    return xyz @ R.transpose(1, 2)

def rotate_around_center_batch(xyz, z_deg, x_deg, y_deg, center):
    """
    Rotate xyz around a specified center.
    xyz: (N, num_atoms, 3)
    center: (N, 3)
    """
    # Translate to origin
    translated = xyz - center.unsqueeze(1)
    # Rotate
    rotated = rotate_batch(translated, z_deg, x_deg, y_deg)
    # Translate back
    return rotated + center.unsqueeze(1)