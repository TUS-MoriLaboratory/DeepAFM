import os
import math
import numpy as np
import torch

from afm_image_generation.configs.afm_image_parameter import (
    AFMImageFixedParams, AFMImageRandomRange, AFMDataSamplingParams
)
from configs.experiment_config import ExperimentConfig
from afm_image_generation.core.afm_input_generator import AFMInputGenerator, AFMGenerationInput
from afm_image_generation.utils.pdb_utils import PDBUtils

class AFMImageGeneratorBase:
    def __init__(
            self,
            exp_cfg: ExperimentConfig,
            xyz_refs=None, # ref of xyz data on shared memory
            radii_ref=None # ref of radii data on shared memory
        ):
        # Store configs
        self.fixed_params: AFMImageFixedParams = getattr(exp_cfg.afm, "fixed")
        self.random_range: AFMImageRandomRange = getattr(exp_cfg.afm, "random")
        self.sampling_params: AFMDataSamplingParams = getattr(exp_cfg.afm, "sampling")
        self.exp_cfg = exp_cfg  
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

        self.input_generator = AFMInputGenerator(exp_cfg)
        self.input_iterator = iter(self.input_generator)

        # Initialize PDB utils
        self.pdb_utils = PDBUtils(exp_cfg)
        # Load PDB data
        if xyz_refs is not None and radii_ref is not None:
            self.xyz_refs = xyz_refs
            self.radii_ref = radii_ref
        else:
            self.xyz_data, self.atom_radii = self.pdb_utils.load_mdtrj()

        # tip shape array
        #self.tip_shape = torch.tensor(np.zeros(self.fixed_params.tip_shape), dtype=self.dtype, device=self.device)

        # Scan direction
        direction = self.exp_cfg.system.scan_direction.lower()
        if direction == "x":
            self.scan_axis = 1  # Columns (Horizontal)
        elif direction == "y":
            self.scan_axis = 0  # Rows (Vertical)
        else:
            raise ValueError(f"Invalid scan direction: {self.exp_cfg.system.scan_direction}. Use 'x' or 'y'.")
        
        # scan unit
        self.scan_unit = self.exp_cfg.system.scan_unit.lower()

    def generate_image(self, input_data: AFMGenerationInput) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, n_images: int):
        """Main loop for generating multiple images."""
        results = []
        for _ in range(n_images):
            input_data = next(self.input_iterator)
            img = self.generate_image(input_data)
            results.append(img)
        return results
    
