# gen_AFM/core/afm_input_generator.py
from typing import Iterator
from dataclasses import dataclass
import torch

from configs.experiment_config import ExperimentConfig
from afm_image_generation.configs.afm_image_parameter import AFMImageFixedParams, AFMImageRandomRange, AFMDataSamplingParams, AFMBrownianMotionParams, AFMScanParams
from afm_image_generation.core.brownian_motion import AFMBrownianMotion
from afm_image_generation.core.param_sampling import AFMParameterSampler

@dataclass
class AFMGenerationInput:
    """Unified container for generating one AFM image."""
    fixed_params: AFMImageFixedParams
    brown_translation_x: torch.Tensor
    brown_translation_y: torch.Tensor
    brown_rotation_x: torch.Tensor
    brown_rotation_y: torch.Tensor
    brown_rotation_z: torch.Tensor

class AFMInputGenerator:
    """Generates inputs required for AFM image generation."""
    def __init__(
            self,
            exp_cfg: ExperimentConfig
        ):
        # Store configs
        self.exp_cfg = exp_cfg
        self.fixed_params: AFMImageFixedParams = getattr(exp_cfg.afm, "fixed")
        self.random_range: AFMImageRandomRange = getattr(exp_cfg.afm, "random")
        self.sampling_params: AFMDataSamplingParams = getattr(exp_cfg.afm, "sampling")
        self.brownian_params: AFMBrownianMotionParams = getattr(exp_cfg.afm, "brownian")
        self.scan_params: AFMScanParams = getattr(exp_cfg.afm, "scan")

        # paramater mode
        # "random": random sampling of parameters and Brownian motion
        # "grid": grid sampling of parameters all combinations except Brownian motion(random)
        self.parameter_mode = exp_cfg.system.parameter_mode
        print(f"AFMInputGenerator: parameter_mode = {self.parameter_mode}")

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

        # seed for reproducibility
        seed = getattr(exp_cfg.system, "seed", None)
        if seed is not None:
            self.rng = torch.Generator(device=self.device)
            self.rng.manual_seed(seed)
        else:
            self.rng = torch.Generator(device=self.device)

        # Initialize parameter sampler
        self.param_sampler = AFMParameterSampler(
            self.fixed_params, 
            self.random_range, 
            self.sampling_params, 
            self.exp_cfg,
        )
        # Initialize Brownian motion generator
        self.brownian_motion = AFMBrownianMotion(
            brown_cfg=self.brownian_params, 
            params_cfg=self.fixed_params, 
            scan_params=self.scan_params,
            exp_cfg=self.exp_cfg, 
        )
            
        self.param_iterator = iter(self.param_sampler)

    def generate_input_once(self) -> AFMGenerationInput:
        """
        Generate one AFMGenerationInput instance containing:
          - one set of random fixed parameters
          - one Brownian motion trajectory (x/y translation, x/z rotation)
        """

        # Sample parameters
        sampled_params = next(self.param_iterator)

        # Generate Brownian motion trajectories
        self.brownian_motion = AFMBrownianMotion(
            brown_cfg=self.brownian_params, 
            params_cfg=sampled_params, 
            scan_params=self.scan_params,
            exp_cfg=self.exp_cfg, 
        )

        translation_x, translation_y = self.brownian_motion.bounded_normal_move()
        rotation_x, rotation_y, rotation_z = self.brownian_motion.bounded_normal_rotate()

        return AFMGenerationInput(
            fixed_params=sampled_params,
            brown_translation_x=translation_x,
            brown_translation_y=translation_y,
            brown_rotation_x=rotation_x,
            brown_rotation_y=rotation_y,
            brown_rotation_z=rotation_z
        )
    
    def __iter__(self) -> Iterator[AFMGenerationInput]:
        """
        Infinite generator yielding random AFM parameters and Brownian motion trajectories.
        Use with itertools.islice(...) or manual break to limit count.
        """
        while True:
            try:
                yield self.generate_input_once()

            except StopIteration:
                return 

    def generate_batch(self, n: int) -> list[AFMGenerationInput]:
        return [self.generate_input_once() for _ in range(n)]