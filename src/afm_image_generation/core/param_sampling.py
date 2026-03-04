import torch
from typing import Generator, Dict
from itertools import product

from configs.experiment_config import ExperimentConfig
from afm_image_generation.configs.afm_image_parameter import AFMImageFixedParams, AFMImageRandomRange, AFMDataSamplingParams


class AFMParameterSampler:
    """
    Parameter sampler for AFM image generation with reproducible random seeds.
    Torch-based RNG for both CPU and GPU environments.
    """
    def __init__(
            self, 
            fixed_params: AFMImageFixedParams,
            random_range: AFMImageRandomRange, 
            sampling_params: AFMDataSamplingParams, 
            exp_cfg: ExperimentConfig
        ):
        # Store configs
        self.random_range = random_range
        self.sampling_params = sampling_params
        self.exp_cfg = exp_cfg

        # parameter mode
        # "random": random sampling of parameters and Brownian motion
        # "grid": grid sampling of parameters all combinations except Brownian motion(random)
        self.parameter_mode = exp_cfg.system.parameter_mode

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

        # Preload all ranges as tensors for efficient sampling
        self.rotation_x_list = torch.tensor(random_range.rotation_x_list, device=self.device, dtype=self.dtype)
        self.rotation_y_list = torch.tensor(random_range.rotation_y_list, device=self.device, dtype=self.dtype)
        self.rotation_z_list = torch.tensor(random_range.rotation_z_list, device=self.device, dtype=self.dtype)
        self.probe_radius_list = torch.tensor(random_range.probe_radius_list, device=self.device, dtype=self.dtype)
        self.half_angle_list = torch.tensor(random_range.half_angle_list, device=self.device, dtype=self.dtype)
        self.scan_preset_list = random_range.scan_preset_list

        # if pdb_num_list is set, preload it as tensor
        self.pdb_num_list = None
        if self.random_range.pdb_num_list is not None:
            self.pdb_num_list = torch.tensor(
                self.random_range.pdb_num_list, 
                device=self.device, 
                dtype=self.dtype
                )
            
        if self.random_range.pdb_num_list is None and self.parameter_mode == "grid":
            self.pdb_num_list = torch.tensor(
                [fixed_params.pdb_num],
                device=self.device,
                dtype=self.dtype,
            )

        # Fixed sampling params for output
        self.fixed_params = fixed_params

        # Prepare grid if in grid mode
        if self.parameter_mode == "grid":
            self._prepare_grid_torch()

    # ==============================================================
    # Core random sampling
    # ==============================================================

    def _random_pdb_num(self) -> int:
        """Generate one random pdb frame number."""
        start_rst, end_rst = self.sampling_params.rst_num_range
        data_per_rst = self.sampling_params.data_per_rst
        skip_ratio = self.sampling_params.skip_ratio

        if not (0 <= skip_ratio < 1):
            raise ValueError(f"Invalid skip_ratio={skip_ratio}, must be in [0, 1).")

        start_offset = int(data_per_rst * skip_ratio)

        segment = torch.randint(start_rst, end_rst + 1, (1,), generator=self.rng, device=self.device).item()
        offset = torch.randint(start_offset, data_per_rst, (1,), generator=self.rng, device=self.device).item()
        return (segment - 1) * data_per_rst + offset + 1

    def _sample_once(self) -> Dict[str, float]:
        """Sample one full AFM parameter set."""

        rotation_x_deg = self.rotation_x_list[torch.randint(0, len(self.rotation_x_list), (1,), generator=self.rng, device=self.device)].item()
        rotation_y_deg = self.rotation_y_list[torch.randint(0, len(self.rotation_y_list), (1,), generator=self.rng, device=self.device)].item()
        rotation_z_deg = self.rotation_z_list[torch.randint(0, len(self.rotation_z_list), (1,), generator=self.rng, device=self.device)].item()
        probe_radius = round(self.probe_radius_list[torch.randint(0, len(self.probe_radius_list), (1,), generator=self.rng, device=self.device)].item(), 2)
        half_angle = round(self.half_angle_list[torch.randint(0, len(self.half_angle_list), (1,), generator=self.rng, device=self.device)].item(), 2)
        scan_preset = self.scan_preset_list[torch.randint(0, len(self.scan_preset_list), (1,), generator=self.rng, device=self.device)]

        if self.pdb_num_list is not None:
            pdb_num = int(self.pdb_num_list[torch.randint(0, len(self.pdb_num_list), (1,), generator=self.rng, device=self.device)].item())
        else:
            pdb_num = int(self._random_pdb_num())

        # Set sampled values to fixed_params
        fixed_params = AFMImageFixedParams(
            probe_radius=probe_radius,
            half_angle=half_angle,
            scale_x=scan_preset["scale_x"],
            scale_y=scan_preset["scale_y"],
            width_px=scan_preset["width_px"],
            height_px=scan_preset["height_px"],
            rotation_x=rotation_x_deg,
            rotation_y=rotation_y_deg,
            rotation_z=rotation_z_deg,
            pdb_num=pdb_num,
        )
        return fixed_params
    
    # ==============================================================
    # Core grid sampling
    # ==============================================================

    def _prepare_grid_torch(self):
        param_tensors = [
            self.rotation_x_list, 
            self.rotation_y_list, 
            self.rotation_z_list, 
            self.probe_radius_list, 
            self.half_angle_list, 
            self.scan_preset_list, 
            self.pdb_num_list
        ]

        mesh = torch.meshgrid(*param_tensors, indexing="ij")
        flat = [m.reshape(-1) for m in mesh]
        
        self.grid_torch = torch.stack(flat, dim=1)  # (N, 7)
        self.grid_total = self.grid_torch.shape[0]
        self.grid_idx = 0

    def _sample_grid(self):
        if self.grid_idx >= self.grid_total:
            raise StopIteration     # End of grid

        params = self.grid_torch[self.grid_idx]
        self.grid_idx += 1
        rotation_x, rotation_y, rotation_z, probe_radius, half_angle, \
            scan_preset, pdb_num = params.tolist()

        return AFMImageFixedParams(
            probe_radius=float(round(probe_radius, 2)),
            half_angle=float(round(half_angle, 2)),
            scale_x=scan_preset["scale_x"],
            scale_y=scan_preset["scale_y"],
            width_px=scan_preset["width_px"],
            height_px=scan_preset["height_px"],
            rotation_x=float(rotation_x),
            rotation_y=float(rotation_y),
            rotation_z=float(rotation_z),
            pdb_num=int(pdb_num),
        )

    # ==============================================================
    # Generator API
    # ==============================================================

    def __iter__(self) -> Generator[Dict[str, float], None, None]:
        """
        Infinite generator yielding random AFM parameter dictionaries.
        Use with `itertools.islice()` or manual break to limit count.
        """
        if self.parameter_mode == "random":
            while True:
                yield self._sample_once()

        elif self.parameter_mode == "grid":
            while True:
                try:
                    yield self._sample_grid()
                except StopIteration:
                    return