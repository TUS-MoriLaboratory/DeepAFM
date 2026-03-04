from dataclasses import dataclass, field
from typing import List, Optional
import torch
import pathlib

@dataclass
class SystemConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = None
    save_interval: int = 5
    log_interval: int = 1
    run_name: Optional[str] = None

    # --data type settings for AFM and NN computations--
    afm_dtype: str = "float32"
    data_dtype: str = "float32"
    nn_dtype: str = "amp"
    precision_mode: str = "amp"  # "fp32", "bf16", "amp"
    allow_tf32: bool = True  # Allow TF32 on Ampere GPUs (RTX 2080 Ti is not Ampere)

    # --afm image generation settings--
    md_load_mode: str = "coarse"    # "all_atom", "coarse"
    scan_direction: str = "y"       # Scan direction: "x" or "y"
    scan_unit: str = "line"         # "line" or "pixel"
    pdb_only: bool = False          #  If True, only load PDB without DCD
    parameter_mode: str = "random"  # "random", "grid"
    save_mode: str = "hdf5"         # "hdf5", "tfrecord", "webdataset"
    use_gpu_for_afm: bool = False   # Whether to use GPU for AFM image generation

    # --data loading settings--
    # "distorted", "ideal", "config"
    data_load_mode: List[str] = field(default_factory=lambda: ["distorted", "config"])  

    # --random seed--
    seed: Optional[int] = 42  # Random seed for reproducibility; if None, random seed is used

    # project root
    project_root: Optional[str] = None 

    # tqdm settings
    tqdm_silent: bool = True  # If True, disable tqdm progress bars

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = str(
        pathlib.Path(__file__).resolve().parents[2]
    )
