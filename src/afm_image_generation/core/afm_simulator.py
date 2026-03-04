# ref: https://github.com/matsunagalab/ColabBTR/blob/main/colabbtr/morphology.py

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from afm_image_generation.constants.atomic_radii import Atom2Radius

def compute_xc_yc(tip):
    """
    Compute the center position of the tip
        Input: tip (tensor of size (tip_height, tip_width))
        Output: xc, yc (int)
    """
    tip_xsiz, tip_ysiz = tip.size()
    xc = round((tip_xsiz - 1) / 2)
    yc = round((tip_ysiz - 1) / 2)
    return xc, yc

# ref: https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
#@torch.jit.script
def fixed_padding(inputs: torch.Tensor, kH: int, kW: int, dilation: int) -> torch.Tensor:
    #if isinstance(kernel_size, (int, float)):
    #    kH = kW = int(kernel_size)
    #else:
    #kH, kW = kernel_size

    kH_eff = kH + (kH - 1) * (dilation - 1)
    kW_eff = kW + (kW - 1) * (dilation - 1)

    pad_h_total = kH_eff - 1
    pad_h_beg = pad_h_total // 2
    pad_h_end = pad_h_total - pad_h_beg
    
    pad_w_total = kW_eff - 1
    pad_w_beg = pad_w_total // 2
    pad_w_end = pad_w_total - pad_w_beg

    padding_list = [
        int(pad_w_beg), 
        int(pad_w_end), 
        int(pad_h_beg), 
        int(pad_h_end)
    ]

    padded_inputs = F.pad(inputs, padding_list)
    
    return padded_inputs

# ref: https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
#@torch.jit.script
def idilation(image, tip):
    """
    Compute the dilation of surface by tip
        Input: surface (tensor of size (surface_height, surface_width)
               tip (tensor of size (kernel_size, kernel_size)
        Output: r (tensor of size (image_height, image_width)
                where image_heigh is equal to surface_height
                      image_width is equal to surface_width
    """
    in_channels = 1
    out_channels = 1
    H, W = int(image.size(0)), int(image.size(1))
    kH, kW = int(tip.size(0)), int(tip.size(1))
    x = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x = fixed_padding(x, kH, kW, dilation=1)
    x = F.unfold(x, (kH, kW), dilation=1, padding=0, stride=1)  # (B, Cin*kH*kW, L), where L is the numbers of patches
    x = x.unsqueeze(1) # (B, 1, Cin*kH*kW, L)
    #L = x.size(-1)
    #L_sqrt = int(math.sqrt(L))

    weight = tip.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
    weight = weight.reshape(out_channels, -1) # (Cout, Cin*kH*kW)
    weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)
    x = weight + x # (B, Cout, Cin*kH*kW, L)
    x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
    #x = x.view(-1, out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)
    x = x.view(-1, out_channels, H, W)  # (B, Cout, H, W)
    return x.squeeze(0).squeeze(0)

def surfing(xyz, radius, config:dict[str, float]):
    """
    Compute the maximum height (z-value) of molecular surface at grid points on AFM stage (where z=0)
        Input: xyz (tensor of size (*, N, 3))
                radius (tensor of size (N,))
                config (dict)
        Output: z_stage (tensor of size (*, len(y_stage), len(x_stage))
    """
    radius2 = radius**2

    W_pixels = int(round((config["max_x"] - config["min_x"]) / config["resolution_x"]))
    H_pixels = int(round((config["max_y"] - config["min_y"]) / config["resolution_y"]))

    idx_x = torch.arange(W_pixels, device=xyz.device, dtype=xyz.dtype)
    idx_y = torch.arange(H_pixels, device=xyz.device, dtype=xyz.dtype)

    x_stage = config["min_x"] + (idx_x + 0.5) * config["resolution_x"]
    y_stage = config["min_y"] + (idx_y + 0.5) * config["resolution_y"]
    
    dx = xyz[...,0,None] - x_stage #(*,N,W)
    #print(f'{dx.shape=}')
    dx2 = dx**2 #(*,N,W)
    dy = xyz[...,1,None] - y_stage #(*,N,H)
    #print(f'{dy.shape=}') 
    dy2 = dy**2 #(*,N,H)
    r2 = dx2.unsqueeze(-2) + dy2[...,None] #(*,N,H,W)
    #print(f'{r2.shape=}')
    index_within_radius = r2 < radius2[...,None,None] #(*,N,H,W)
    diff = radius2[...,None,None] - r2
    diff = torch.where(index_within_radius, diff, 1) #(*,N,H,W)
    temp = torch.where(index_within_radius, xyz[...,2,None,None] + torch.sqrt(diff), -torch.inf) #(*,N,H,W)
    temp_max = temp.max(dim=-3)[0] #(*,H,W)
    z_stage = torch.where(index_within_radius.any(dim=-3), temp_max, torch.zeros_like(temp_max, dtype=xyz.dtype, device=xyz.device)) #(H,W)
    return z_stage.flip([-2])

def define_tip(tip, resolution_x, resolution_y, probeRadius, probeAngle):
    """
    Define the tip shape by the probe radius and angle
        Input: tip (tensor of size (tip_height, tip_width))
               resolution_x (float)
               resolution_y (float)
               probeRadius (float)
               probeAngle (float) [degree]
        Output: tip (tensor of size (tip_height, tip_width))
    """
    probeAngle = probeAngle * math.pi / 180.0 # convert to radian

    tip_xsiz, tip_ysiz = tip.shape
    xc, yc = compute_xc_yc(tip)
    for ix in range(tip_xsiz):
        for iy in range(tip_ysiz):
            x = resolution_x * abs(ix - xc)
            y = resolution_y * abs(iy - yc)
            d = math.sqrt(x**2 + y**2)
            if d <= probeRadius:
                z = math.sqrt(probeRadius**2 - d**2)
            else:
                theta = (0.5 * math.pi) - probeAngle
                z = -math.tan(theta) * (d - probeRadius)
            tip[ix, iy] = z
    tip -= tip.max()
    return tip

# ref: https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
def define_tip(resolution_x, resolution_y, probeRadius, probeAngle, max_height):
    """
    Define the tip shape by the probe radius and angle using Shoulder-based model
        Input: tip (tensor of size (tip_height, tip_width))
               resolution_x (float)
               resolution_y (float)
               probeRadius (float)
               probeAngle (float) [degree]
        Output: tip (tensor of size (tip_height, tip_width))
    """
    probeAngle = probeAngle * math.pi / 180.0 # convert to radian
    tan_theta = math.tan(probeAngle)
    
    if max_height > probeRadius:
        L_nm = probeRadius + (max_height - probeRadius) * tan_theta
    else:
        L_nm = math.sqrt(probeRadius**2 - (probeRadius - max_height)**2)

    # array size calculation
    nx = int(math.ceil(2 * L_nm / resolution_x))
    ny = int(math.ceil(2 * L_nm / resolution_y))
    nx = nx if nx % 2 != 0 else nx + 1
    ny = ny if ny % 2 != 0 else ny + 1
    
    # Prepare an empty array
    tip = torch.zeros((nx, ny))
    xc, yc = compute_xc_yc(tip)

    # Calculate shape
    for ix in range(nx):
        for iy in range(ny):
            x = resolution_x * abs(ix - xc)
            y = resolution_y * abs(iy - yc)
            d = math.sqrt(x**2 + y**2)
            if d <= probeRadius:
                z = math.sqrt(probeRadius**2 - d**2)
            else:
                theta = (0.5 * math.pi) - probeAngle
                z = -math.tan(theta) * (d - probeRadius)
            tip[ix, iy] = z
    tip -= tip.max()

    return tip

def surfing_vector(
    xyz: torch.Tensor,
    radius: torch.Tensor, 
    attention_mask: torch.Tensor,
    min_x: torch.Tensor,
    max_x: torch.Tensor,
    resolution_x: torch.Tensor,
    min_y: torch.Tensor,
    max_y: torch.Tensor,
    resolution_y: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized AFM surface calculation
    """
    device = xyz.device
    dtype = xyz.dtype
    
    num_x_steps = torch.round((max_x - min_x) / resolution_x).long()
    num_y_steps = torch.round((max_y - min_y) / resolution_y).long()
    
    max_W, max_H = num_x_steps.max().item(), num_y_steps.max().item()
    
    w_indices = torch.arange(max_W, dtype=dtype, device=device)
    h_indices = torch.arange(max_H, dtype=dtype, device=device)
    
    x_stage = min_x.unsqueeze(1) + (w_indices + 0.5) * resolution_x.unsqueeze(1)
    y_stage = min_y.unsqueeze(1) + (h_indices + 0.5) * resolution_y.unsqueeze(1)
    
    x_mask = w_indices < num_x_steps.unsqueeze(1)
    y_mask = h_indices < num_y_steps.unsqueeze(1)
    
    atom_x = xyz[..., 0, None, None]
    atom_y = xyz[..., 1, None, None]
    atom_z = xyz[..., 2, None, None]
    
    grid_x = x_stage.unsqueeze(1).unsqueeze(2)
    grid_y = y_stage.unsqueeze(1).unsqueeze(-1)
    
    r2 = (atom_x - grid_x)**2 + (atom_y - grid_y)**2
    
    radius2 = radius**2
    radius2_expanded = radius2.unsqueeze(-1).unsqueeze(-1)
    
    index_within_radius = r2 < radius2_expanded
    diff = radius2_expanded - r2
    
    diff_safe = torch.where(index_within_radius, diff.clamp(min=0), 0.0)
    surface_heights = torch.where(
        index_within_radius,
        atom_z + torch.sqrt(diff_safe),
        torch.tensor(-torch.inf, dtype=dtype, device=device)
    )
    
    grid_mask = (x_mask.unsqueeze(1).unsqueeze(2) & 
                 y_mask.unsqueeze(1).unsqueeze(3))
    atom_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1)
    final_mask = grid_mask & atom_mask_expanded
    
    surface_heights = torch.where(final_mask, surface_heights, 
                                 torch.tensor(-torch.inf, dtype=dtype, device=device))
    
    max_heights = surface_heights.max(dim=1).values
    
    has_valid_atoms = (final_mask & index_within_radius).any(dim=1)
    z_stage = torch.where(has_valid_atoms, max_heights, 
                         torch.zeros_like(max_heights))
    
    return z_stage.flip([-2])

def idilation_vector(images, tip):
    """
    Vectorized morphological dilation
    """
    in_channels = 1
    out_channels = 1
    
    B = images.size(0)
    H = images.size(1)
    W = images.size(2)
    
    if tip.dim() == 3:
        # (Batch, K, K)
        kernel_size = tip.size(1)
    else:
        # (K, K) -> (Batch, K, K)
        kernel_size = tip.size(0)
        tip = tip.unsqueeze(0).expand(B, -1, -1)

    x = images.unsqueeze(1)
    x = fixed_padding(x, torch.tensor(kernel_size), dilation=torch.tensor(1))
    x = F.unfold(x, kernel_size, dilation=1, padding=0, stride=1)
    x = x.unsqueeze(1)
    L = x.size(-1)

    weight = tip.view(B, 1, -1, 1)
    
    x = weight + x
    x, _ = torch.max(x, dim=2, keepdim=False)
    x = x.view(-1, out_channels, H, W)
    x = x.squeeze(1)

    return x

def scale_to_size_config(
        scale_x, 
        scale_y, 
        width_px,
        height_px, 
        scan_axis, 
        valid_lines=None, 
        selected_line=None
        ):
    """
    scan_axis=0 (Y-scan): line index corresponds to X-axis in nm space.
    scan_axis=1 (X-scan): line index corresponds to Y-axis in nm space.
    """
    W_nm = scale_x * width_px
    H_nm = scale_y * height_px
    half_width = W_nm / 2  
    half_height = H_nm / 2

    # setup scan direction dependent parameters
    if scan_axis == 0:
        nm_per_pixel_slow = scale_x
        half_slow = half_width
        total_px_slow = width_px
    else:
        # scan_axis == 1 (X-scan): line index corresponds to Y-axis in nm space.
        nm_per_pixel_slow = scale_y
        half_slow = half_height
        total_px_slow = height_px

    if valid_lines is not None:
        min_idx = max(min(valid_lines), 0)
        max_idx = min(max(valid_lines), total_px_slow - 1)
        
        range_min = -half_slow + min_idx * nm_per_pixel_slow
        range_max = -half_slow + (max_idx + 1) * nm_per_pixel_slow
    elif selected_line is not None:
        range_min = -half_slow + selected_line * nm_per_pixel_slow
        range_max = -half_slow + (selected_line + 1) * nm_per_pixel_slow
    else:
        range_min = -half_slow
        range_max = half_slow

    if scan_axis == 0: # Y-scan
        size_config = {
            "min_x": float(range_min), 
            "max_x": float(range_max),
            "min_y": float(-half_height),     
            "max_y": float(half_height),
        }
    else: # X-scan
        size_config = {
            "min_x": float(-half_width),     
            "max_x": float(half_width),
            "min_y": float(range_min), 
            "max_y": float(range_max),
        }

    size_config.update({
        "resolution_x": float(scale_x),
        "resolution_y": float(scale_y),
    })

    return size_config

def get_local_size_config(x_range, y_range, scale_x, scale_y, image_size_x, image_size_y):
    """
    Calculate the bounding box in nm space for a local cropped region.

    Parameters:
    - x_range (tuple): (min_x_idx, max_x_idx) in pixels.
    - y_range (tuple): (min_y_idx, max_y_idx) in pixels.
    - scale_x (float): nm per pixel in the X direction.
    - scale_y (float): nm per pixel in the Y direction.
    - image_size_x (int): Total image width in pixels.
    - image_size_y (int): Total image height in pixels.

    Returns:
    - size_config (dict): Bounding box and resolution for the local region.
    """
    # Offset to center the image at (0, 0)
    half_nm_x = (scale_x * image_size_x) / 2
    half_nm_y = (scale_y * image_size_y) / 2

    # Convert pixel indices to nm coordinates
    # Note: max_idx from calculate_buffer_1d is exclusive (slice-style), 
    # which naturally points to the boundary of the next pixel.
    min_x_nm = round(-half_nm_x + x_range[0] * scale_x, 10)
    max_x_nm = round(-half_nm_x + x_range[1] * scale_x, 10)
    
    min_y_nm = round(half_nm_y - y_range[1] * scale_y, 10) # top to bottom
    max_y_nm = round(half_nm_y - y_range[0] * scale_y, 10) # top to bottom
    return {
        "min_x": float(min_x_nm),
        "max_x": float(max_x_nm),
        "min_y": float(min_y_nm),
        "max_y": float(max_y_nm),
        "resolution_x": float(scale_x),
        "resolution_y": float(scale_y),
    }

def calculate_buffer_of_line(tip, line_index, width_px, height_px, scan_axis):
    """
    Calculate the number of buffer pixels needed based on the tip size and image scale.
    Parameters:
    - tip: 2D tensor representing the tip shape.
    - scale: Scale factor for image resolution in nm/pixel.
    - column: Valid column in the image.
    - width_px: Total width of the image in pixels.
    - height_px: Total height of the image in pixels.
    - scan_axis: Axis of scanning (0 for Y-scan, 1 for X-scan).
    Returns:
    - buffer_pixels: Number of buffer pixels needed on each side of the image.
    """

    if scan_axis == 1: # X-scan
        # Tip size along Y-axis is needed
        tip_dim_size = tip.shape[1]
        total_px_slow = height_px

    else: # Y-scan
        tip_dim_size = tip.shape[0] 
        total_px_slow = width_px        

    half_tip = round((tip_dim_size - 1) / 2) + 1 
    
    min_idx = line_index - int(half_tip) - 1
    min_idx = max(0, min_idx) 

    max_idx = line_index + int(half_tip) + 1
    max_idx = min(total_px_slow - 1, max_idx) 

    if line_index - int(half_tip) - 1 <= 0:
        target_line = line_index
    else:
        target_line = int(half_tip) + 1

    return min_idx, max_idx, target_line

def calculate_buffer_of_pixel(tip, px, py, image_wide_pixels, image_height_pixels):
    """
    Calculate the calculation range (cropped region) based on the tip size centered on a specific pixel (px, py).
    
    Parameters:
    - tip: 2D tensor representing the tip shape (tip_h, tip_w)
    - px, py: Coordinates of the currently processed pixel
    - image_wide_pixels: Total width of the image in pixels
    - image_height_pixels: Total height of the image in pixels
    
    Returns:
    - x_range: (min_x, max_x) Cropped X range
    - y_range: (min_y, max_y) Cropped Y range
    - target_pos: (target_px, target_py) Relative position of the center pixel within the cropped region
    """
    tip_h, tip_w = tip.shape
    
    # Calculate half sizes
    half_w = round((tip_w - 1) / 2) + 1 
    half_h = round((tip_h - 1) / 2) + 1 
    
    # 1. X direction range calculation
    min_x = max(0, px - half_w - 1)
    max_x = min(image_wide_pixels - 1, px + half_w + 1)
    
    # 2. Y direction range calculation
    min_y = max(0, py - half_h - 1)
    max_y = min(image_height_pixels - 1, py + half_h + 1)
    
    # 3. Relative position of the original pixel (px, py) within the cropped region
    target_px = px - min_x
    target_py = py - min_y
    
    return min_x, max_x, min_y, max_y, target_px, target_py

def calculate_buffer_1d(tip_size: int, pos: int, limit: int):
    """
    Calculate the 1D calculation range (buffer) for a specific axis.

    This function determines the start and end indices for a cropped region
    centered on a target pixel coordinate. It ensures the tip's footprint
    is fully covered while respecting image boundaries.

    Parameters:
    - tip_size (int): The dimension of the tip shape along the target axis.
    - pos (int): The current pixel coordinate (index) along the target axis.
    - limit (int): The total number of pixels along the target axis.

    Returns:
    - min_val (int): Starting index of the cropped range (inclusive).
    - max_val (int): Ending index of the cropped range (exclusive).
    - target_pos (int): Relative coordinate of the center pixel within the cropped region.
    """
    # Use floor division for consistent radius calculation. 
    half = tip_size // 2
    
    # Add a 1-pixel margin for safety during morphological operations
    margin = 1
    
    # Range calculation
    min_val = max(0, pos - half - margin)
    # Python slices are [start:end), so max_val uses 'limit' as a boundary
    max_val = min(limit, pos + half + margin)
    
    # Calculate the target pixel's position relative to the new cropped window
    target_pos = pos - min_val
    
    return min_val, max_val, target_pos


def fill_zero_line(img, index, axis):
    if axis == 1: # X-scan
        img[index, :] = 0
    else: # Y-scan
        img[:, index] = 0

def fill_zero_pixel(img, slow_px, fast_px, axis):
    if axis == 1: # X-scan
        img[slow_px, fast_px] = 0
    else: # Y-scan
        img[fast_px, slow_px] = 0
