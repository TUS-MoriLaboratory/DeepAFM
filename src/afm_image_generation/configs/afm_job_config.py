"""
afm_image_generation/config/afm_job_config.py
AFM Job Configuration Module
"""
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional
from configs.system_config import SystemConfig

sys_cfg = SystemConfig()
root_dir = sys_cfg.project_root if sys_cfg.project_root else str(
    pathlib.Path(__file__).resolve().parents[3]
)

@dataclass
class AFMGenerationJobConfig:
    """Configuration for AFM image generation jobs."""
    # --- Job Parameters ---
    total_images: int = 1000                                        # Total number of AFM images to generate
    processes: int = 4                                              # Number of main parallel processes to use
    chunk_size: int = 10                                            # Number of images per chunk for each process
    # for hybrid CPU/GPU mode such as multiprocess_batch
    cpu_processes: int = 0                                          # Number of CPU processes (hybrid CPU/GPU mode such as multiprocess_batch)
    batch_size: int = 64                                            # Number of images to generate in a batch within each process    

    # --- Output Directory ---
    save_dir: Optional[str] = None                                  # Directory to save generated AFM images

    # Executing mode
    if sys_cfg.device == "cpu":
        vectorized: bool = False                                  # Whether to use vectorized generation
    else:
        vectorized: bool = True                                   # Whether to use vectorized generation    

    # --- mode ---
    # "distorted": generate distorted images
    # "ideal": generate ideal images
    # "config": generate configuration data
    
    output_mode: List[str] = field(default_factory=lambda: ["distorted", "ideal", "config"])

    def __post_init__(self):
        valid = {"distorted", "ideal", "config"}
        for m in self.output_mode:
            if m not in valid:
                raise ValueError(f"Invalid output_mode: {m}, choose from {valid}")
            
    