# gen_AFM/core/brown_motion.py
import torch
from afm_image_generation.configs.afm_image_parameter import AFMBrownianMotionParams, AFMImageFixedParams, AFMScanParams
from configs.experiment_config import ExperimentConfig

class AFMBrownianMotion:
    def __init__(
            self, 
            brown_cfg: AFMBrownianMotionParams, 
            params_cfg: AFMImageFixedParams, 
            scan_params: AFMScanParams,
            exp_cfg: ExperimentConfig
            ):
        
        # Store configs
        self.brown_cfg = brown_cfg
        self.scan_params = scan_params
        self.exp_cfg = exp_cfg  
        self.sampling_mode = exp_cfg.system.scan_unit # "line" or "pixel"

        # Image size
        self.width_px = params_cfg.width_px
        self.height_px = params_cfg.height_px

        # Setup mode-dependent parameters
        self._setup_mode_params()

        # translation parameters
        self.max_translation = brown_cfg.max_translation
        self.translation_mu = brown_cfg.translation_mu
        self.translation_sigma = brown_cfg.translation_sigma * self.time_scale_factor

        # rotation parameters
        self.x_max_angle = brown_cfg.max_rotation_x
        self.x_mu = brown_cfg.rotation_x_mu
        self.x_sigma = brown_cfg.rotation_x_sigma * self.time_scale_factor
        
        self.y_max_angle = brown_cfg.max_rotation_y
        self.y_mu = brown_cfg.rotation_y_mu
        self.y_sigma = brown_cfg.rotation_y_sigma * self.time_scale_factor

        self.z_max_angle = brown_cfg.max_rotation_z
        self.z_mu = brown_cfg.rotation_z_mu
        self.z_sigma = brown_cfg.rotation_z_sigma * self.time_scale_factor

        # Note:
        # The original sigma values are assumed to be per-line (column) values.
        # Therefore, if using pixel-unit sampling, apply time_scale_factor to reduce them.

        # device and data type
        self.device = torch.device(exp_cfg.system.device)
        FLOAT = exp_cfg.system.afm_dtype
        if FLOAT == "float16":
            self.dtype = torch.float16
        elif FLOAT == "float32":
            self.dtype = torch.float32
        elif FLOAT == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported FLOAT type: {FLOAT}")
    
        self.seed = exp_cfg.system.seed
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

    def _setup_mode_params(self):
        """Calculate step count and physical scale based on sampling mode"""

        # Determine pixel based on scan direction
        if self.exp_cfg.system.scan_direction == 'x': # x
            self.fast_px = self.width_px
            self.slow_px = self.height_px
            self.full_line_steps = self.scan_params.scan_Wide_pixels  

        else: # y
            self.fast_px = self.height_px
            self.slow_px = self.width_px
            self.full_line_steps = self.scan_params.scan_Height_pixels

        if self.sampling_mode == "pixel":
            
            # Total steps in one frame
            self.full_frame_steps = self.full_line_steps * self.slow_px
            self.total_steps = self.full_frame_steps

            # Time scaling factor for pixel-unit sampling
            self.time_scale_factor = (1.0 / self.full_line_steps) ** 0.5

            # Precompute sampling indices for pixel-unit sampling
            indices = []
            for row in range(self.slow_px):
                start = row * self.full_line_steps
                end = start + self.fast_px
                indices.extend(list(range(start, end)))

            self.sampling_indices = torch.tensor(indices)

        else:
            self.total_steps = self.slow_px
            self.time_scale_factor = 1.0
            self.sampling_indices = None

    def bounded_normal_move(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 2D Brownian-like bounded motion (translation trajectory).
        offset (nm) from origin (0,0) is generated using normal distribution with rejection sampling.
                
        Returns:
            x_vals : (image_size,) tensor
            y_vals : (image_size,) tensor

        """
        if self.max_translation < 1e-9:
            zeros = torch.zeros(self.total_steps, device=self.device, dtype=self.dtype)
            return zeros, zeros

        pos = torch.zeros(self.total_steps, 2, device=self.device, dtype=self.dtype)
        for i in range(1, self.total_steps):
            bfr = pos[i - 1]
            for _ in range(100000):  # retry up to 0.1 million times
                step = torch.randn(2, generator=self.rng, device=self.device, dtype=self.dtype) \
                    * self.translation_sigma + self.translation_mu
                step = torch.round(step, decimals=1)
                candidate = bfr + step
                if torch.linalg.norm(candidate) <= self.max_translation:
                    pos[i] = candidate
                    break
            else:
                pos[i] = bfr  

        # Return sampled positions if in pixel-unit mode
        if self.sampling_mode == "pixel" and self.sampling_indices is not None:
            return pos[self.sampling_indices, 0], pos[self.sampling_indices, 1]
            
        # Otherwise, return full positions
        return pos[:, 0], pos[:, 1]

    def bounded_normal_rotate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 1D bounded angular motion for x/z axes (in degrees).
        
        Returns:
            x_vals : (image_size,) tensor
            z_vals : (image_size,) tensor
        """
        x_vals = torch.zeros(self.total_steps, device=self.device, dtype=self.dtype)
        y_vals = torch.zeros(self.total_steps, device=self.device, dtype=self.dtype)
        z_vals = torch.zeros(self.total_steps, device=self.device, dtype=self.dtype)

        # Check which axes to calculate
        calc_x = self.x_max_angle > 1e-9
        calc_y = self.y_max_angle > 1e-9
        calc_z = self.z_max_angle > 1e-9

        # If none to calculate, return zeros
        if not (calc_x or calc_y or calc_z):
            return x_vals, y_vals, z_vals

        for i in range(1, self.total_steps):
            if calc_x:
                prev_x_val = x_vals[i - 1]
                for _ in range(100000): # retry up to 0.1 million times
                    step = torch.round(
                        torch.randn(1, generator=self.rng, device=self.device, dtype=self.dtype) * self.x_sigma + self.x_mu,
                        decimals=1
                    )
                    candidate = prev_x_val + step
                    if torch.abs(candidate) <= self.x_max_angle:
                        x_vals[i] = candidate
                        break
                else:
                    x_vals[i] = prev_x_val

            if calc_y:
                prev_y_val = y_vals[i - 1]
                for _ in range(100000): # retry up to 0.1 million times
                    step = torch.round(
                    torch.randn(1, generator=self.rng, device=self.device, dtype=self.dtype) * self.y_sigma + self.y_mu,
                    decimals=1
                    )
                    candidate = prev_y_val + step
                    if torch.abs(candidate) <= self.y_max_angle:
                        y_vals[i] = candidate
                        break
                    else:
                        y_vals[i] = prev_y_val

            if calc_z:
                prev_z_val = z_vals[i - 1]
                for _ in range(100000): # retry up to 0.1 million times
                    step = torch.round(
                        torch.randn(1, generator=self.rng, device=self.device, dtype=self.dtype) * self.z_sigma + self.z_mu,
                        decimals=1
                    )
                    candidate = prev_z_val + step
                    if torch.abs(candidate) <= self.z_max_angle:
                        z_vals[i] = candidate
                        break
                else:
                    z_vals[i] = prev_z_val

        # Return sampled positions if in pixel-unit mode
        if self.sampling_mode == "pixel" and self.sampling_indices is not None:
            return x_vals[self.sampling_indices], y_vals[self.sampling_indices], z_vals[self.sampling_indices]
        return x_vals, y_vals, z_vals