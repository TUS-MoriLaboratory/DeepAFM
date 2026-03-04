"""
afm_image_generation/config/afm_image_parameter_config.py
AFM image generation parameter dataclasses.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

### AFM Image Generation Parameters
# Mapping from scale (nm/pixel) to image size (pixels)
scale_to_image_size = {
    0.4: 60,
    0.5: 60,
    0.6: 44,
    0.7: 40,
    0.8: 36,
}

@dataclass
class AFMImageFixedParams:
    """Parameters for single-image generation (fixed values)."""
    pdb_num: int = 1                             # PDB frame number
    probe_radius: float = 1.0                    # R (nm)    
    half_angle: float = 10.0                     # Θ(degree)
    #tip_shape: tuple = (18, 18)                  # Tip shape array (18x18)
    rotation_x: float = 0.0                      # θ rotation angle around X axis (degree)
    rotation_y: float = 0.0                      # ψ rotation angle around Y axis (degree)
    rotation_z: float = 0.0                      # φ rotation angle around Z axis (degree)
    scale_x: float = 0.8                         # p nm/pixel
    scale_y: float = 0.8                         # p nm/pixel
    width_px: Optional[int] = None               # If None, determine from scale_x
    height_px: Optional[int] = None              # If None, determine from scale_y
    fixed_width_px: Optional[int] = None         # If set, use this fixed image size before saving 
    fixed_height_px: Optional[int] = None        # If set, use this fixed image size before saving 

    def __post_init__(self):
        if self.width_px is None:
            rounded_scale = round(float(self.scale_x), 2)
            self.width_px = scale_to_image_size.get(rounded_scale, 128)
        if self.height_px is None:
            rounded_scale = round(float(self.scale_y), 2)
            self.height_px = scale_to_image_size.get(rounded_scale, 128)

        # Determine fixed image size if not set
        if self.fixed_width_px is None:
            self.fixed_width_px = max(scale_to_image_size.values())
        if self.fixed_height_px is None:
            self.fixed_height_px = max(scale_to_image_size.values())


# Preset scan parameters for random sampling
SCAN_PRESETS = [
    {"scale_x": 0.8, "scale_y": 0.8, "width_px": 36, "height_px": 36}, # Default setting of End-up SecYAEG-NDs 
    #{"scale_x": 0.5, "scale_y": 0.7, "width_px": 48, "height_px": 36}, 
]

@dataclass
class AFMImageRandomRange:
    """Parameter ranges used for random image generation."""
    pdb_num_list: list[int] = None                                          # PDB frame number range
    probe_radius_range: Tuple[float, float, float] = (1.0, 3.0, 0.1)        # R (nm)    
    half_angle_range: Tuple[float, float, float] = (5.0, 30.0, 1.0)         # Θ(degree)
    rotation_x_range: Tuple[float, float, float] = (-20.0, 20.0, 1.0)       # θ rotation angle around X axis (degree)
    rotation_y_range: Tuple[float, float, float] = (-20.0, 20.0, 1.0)       # ψ rotation angle around Y axis (degree)
    rotation_z_range: Tuple[float, float, float] = (-180.0, 179.0, 1.0)     # φ rotation angle around Z axis (degree)

    probe_radius_list: list = field(default_factory=list)
    half_angle_list: list = field(default_factory=list)
    rotation_x_list: list = field(default_factory=list)
    rotation_y_list: list = field(default_factory=list)
    rotation_z_list: list = field(default_factory=list)

    # Scale and image size settings
    scan_preset_list: List[dict] = field(default_factory=lambda: SCAN_PRESETS)

    # -------------------------------
    # After dataclass init, expand ranges automatically
    # -------------------------------
    def __post_init__(self):

        def expand(t):
            lo, hi, step = t
            return [round(float(v), 6) for v in np.arange(lo, hi + 1e-9, step)]

        # Convert all ranges into discrete lists
        self.probe_radius_list = expand(self.probe_radius_range)
        self.half_angle_list = expand(self.half_angle_range)
        self.rotation_x_list = expand(self.rotation_x_range)
        self.rotation_y_list = expand(self.rotation_y_range)
        self.rotation_z_list = expand(self.rotation_z_range)
        #self.scale_list = expand(self.scale_range)

@dataclass
class AFMDataSamplingParams:
    rst_num_range: Tuple[int, int] = (1, 20)   # Range of restart numbers
    data_per_rst: int = 1000       # Number of frames per restart
    skip_ratio: float = 0.3        # Ratio of frames to skip at the start of each restart      

@dataclass
class AFMBrownianMotionParams:
    """
    Parameters for Brownian motion perturbations during image generation.
    Scalable via `range_scale_param` (spatial range) and `variance_scale_param` (diffusion strength).
    """
    range_scale_param: float = 1.0      # range scaling factor (w)
    variance_scale_param: float = 1.0   # variance scaling factor (s)

    # Base parameters (before scaling)
    max_translation: float = 1.2        # base maximum translation distance (nm)
    translation_mu: float = 0.0         # base translation mean
    translation_sigma: float = 0.9      # base translation standard deviation
    
    max_rotation_x: float = 5.0        # base max rotation around X (degrees)
    rotation_x_mu: float = 0.0          # base rotation X mean
    rotation_x_sigma: float = 2.5       # base rotation X standard deviation
    
    max_rotation_y: float = 5.0        # base max rotation around Y (degrees)
    rotation_y_mu: float = 0.0          # base rotation Y mean
    rotation_y_sigma: float = 2.5       # base rotation Y standard deviation
    
    max_rotation_z: float = 10.0        # base max rotation around Z (degrees)
    rotation_z_mu: float = 0.0          # base rotation Z mean
    rotation_z_sigma: float = 5.0       # base rotation Z standard deviation

    def __post_init__(self):
        """Apply scaling factors to range and variance parameters."""
        # Range (geometric) scaling
        self.max_translation *= self.range_scale_param
        self.max_rotation_x *= self.range_scale_param
        self.max_rotation_y *= self.range_scale_param
        self.max_rotation_z *= self.range_scale_param

        # Variance (diffusion) scaling
        sigma_factor = self.variance_scale_param ** 0.5
        self.translation_sigma *= sigma_factor
        self.rotation_x_sigma *= sigma_factor
        self.rotation_y_sigma *= sigma_factor
        self.rotation_z_sigma *= sigma_factor

@dataclass
class AFMScanParams:
    """
    Parameters defining the AFM scanning process.
    """
    scan_Height_nm: float = 80.0   # Scan Height (80nm)
    scan_Wide_nm: float = 100.0    # Scan Width (100nm)
    scan_Height_pixels: int = 100  # Scan Height in pixels
    scan_Wide_pixels: int = 120    # Scan Width in pixels
    fps: float = 2.50              # Frame rate (frames/sec)
    scan_direction: str = "y"      # Scan direction: "x" or "y"

    # Derived properties from calculations
    @property
    def time_per_frame(self) -> float:
        return 1.0 / self.fps # sec/frame

    @property
    def time_per_line(self) -> float:
        """time per scan line"""
        # Divide by the number of pixels in the slow scan (feed) direction
        num_lines = self.scan_Height_pixels if self.scan_direction == 'x' else self.scan_Wide_pixels
        return self.time_per_frame / num_lines # sec/line

    @property
    def time_per_pixel(self) -> float:
        """time per pixel"""
        # Divide by the number of pixels in the fast scan direction
        pixels_in_line = self.scan_Wide_pixels if self.scan_direction == 'x' else self.scan_Height_pixels
        return self.time_per_line / pixels_in_line # sec/pixel