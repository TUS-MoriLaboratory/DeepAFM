from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class DataConfig:
    # Data split ratios
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # DataLoader parameters
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers = True
    prefetch_size: int = 2
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0 # for DDP     
    world_size: int = 1 

    # preprocessing options
    # augment options
    add_white_noise: bool = True
    noise_std_range: Tuple[float, float, float] = (0.0, 0.3, 0.01)  # Range of noise stddev for augmentation

    # translation ranges for augmentation
    translate : bool = True
    translation_x_pixel: Tuple[int,int,int] = (-5,5,1)
    translation_y_pixel: Tuple[int,int,int] = (-5,5,1)

    # scaling option
    min_max_scaling: bool = True

    # pdb_num to state option
    pdb_num_to_state: bool = True # enable pdb_num to state mapping
    pdb_num_to_state_mapping_mode : str = "unsupervised"  # unsupervised(./src/unsupervised/PCA_GMM/csv_...) / custom(tutorial mode) / none(no mapping, use original pdb_num as state label)
    
    # for custom mapping mode, specify the mapping from pdb frame number to state label, 
    # e.g. {1:0, 20000:1} means pdb frame number 1 is mapped to state 0 and pdb frame number 20000 is mapped to state 1. 
    # This is only used when pdb_num_to_state_mapping_mode is set to "custom".
    custom_mapping: dict = None 

    def __post_init__(self):

        def expand(t):
            lo, hi, step = t
            return [round(float(v), 6) for v in np.arange(lo, hi + 1e-9, step)]

        # Convert all ranges into discrete lists
        self.noise_std_list = expand(self.noise_std_range)
        self.translation_x_list = expand(self.translation_x_pixel)
        self.translation_y_list = expand(self.translation_y_pixel)
        