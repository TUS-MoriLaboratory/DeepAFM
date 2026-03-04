# src/dataload/preprocess.py
import torch
import numpy as np

class AFMPreprocess:
    """
    Preprocess AFM samples with augmentations:
        - Add random white noise
        - Random pixel translation
        - Min-max scaling
        - PDB number to state conversion

    Args:
        exp_cfg: Experiment configuration object
        add_white_noise (bool): Whether to add white noise
        translate (bool): Whether to apply random pixel translation
        min_max_scaling (bool): Whether to apply min-max scaling
        pdb_num_to_state (bool): Whether to convert pdb_num to state

        if not provided, uses settings from exp_cfg

    """

    def __init__(
        self,
        exp_cfg,
        add_white_noise: bool = None,
        translate: bool = None,
        min_max_scaling: bool = None,
        pdb_num_to_state: bool = None,
    ):
        self.exp_cfg = exp_cfg
        self.data_load_mode = exp_cfg.system.data_load_mode

        # augment flags
        # ---- augment flags (None → use config value) ----
        self.add_white_noise = (
            exp_cfg.data.add_white_noise if add_white_noise is None else add_white_noise
        )
        self.translate = (
            exp_cfg.data.translate if translate is None else translate
        )

        # scaling flag
        self.min_max_scaling = (
            exp_cfg.data.min_max_scaling if min_max_scaling is None else min_max_scaling
        )

        # pdb_num_to_state flag
        self.pdb_num_to_state = (
            exp_cfg.data.pdb_num_to_state if pdb_num_to_state is None else pdb_num_to_state
        )

        # pdb_num to state mapping mode
        # e.g., "unsupervised" or "tutorial"
        self.pdb_num_to_state_mapping_mode = exp_cfg.data.pdb_num_to_state_mapping_mode

        # data type
        FLOAT = exp_cfg.system.data_dtype
        if FLOAT == "float16":
            self.dtype = torch.float16
        elif FLOAT == "float32": 
            self.dtype = torch.float32
        elif FLOAT == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported FLOAT type: {FLOAT}")


        # augmentation ranges
        self.noise_std_list = torch.tensor(exp_cfg.data.noise_std_list, dtype=self.dtype)
        self.translation_x_list = torch.tensor(exp_cfg.data.translation_x_list, dtype=torch.int64)
        self.translation_y_list = torch.tensor(exp_cfg.data.translation_y_list, dtype=torch.int64)

        # load pdb_num to state mapping 
        if self.pdb_num_to_state and self.pdb_num_to_state_mapping_mode == "unsupervised":
            csv_path = exp_cfg.unsupervised.pca_gmm_results_path
            self.pdb_num_to_state_mapping = self._csv_to_mapping(
                path=csv_path,
                key_col="PDB_number",  # assuming CSV has these columns
                value_col="new_label", # cluster label (state)
            )

    def _add_random_white_noise(self, img: torch.Tensor, img2: torch.Tensor = None):
        if img is None: return None

        noise_std = self.noise_std_list[
            torch.randint(
                low=0,
                high=len(self.noise_std_list),
                size=(1,),
                device=img.device,
            ).item()
        ]

        noise = torch.randn_like(img) * noise_std
    
        if img2 is not None:
            return img + noise, img2 + noise

        return img + noise

    def _random_pixel_translation(self, img: torch.Tensor, img2: torch.Tensor = None):
        ix = torch.randint(
            low=0,
            high=len(self.translation_x_list),
            size=(1,),
            device=img.device,
        ).item()
        iy = torch.randint(
            low=0,
            high=len(self.translation_y_list),
            size=(1,),
            device=img.device,
        ).item()

        tx = int(self.translation_x_list[ix].item())
        ty = int(self.translation_y_list[iy].item())

        img = torch.roll(img, shifts=ty, dims=-2)
        img = torch.roll(img, shifts=tx, dims=-1)

        if img2 is not None:
            img2 = torch.roll(img2, shifts=ty, dims=-2)
            img2 = torch.roll(img2, shifts=tx, dims=-1)
            return img, img2

        return img

    def _min_max_scaling(self, img: torch.Tensor):
        min_val = torch.min(img)
        max_val = torch.max(img)
        scaled = (img - min_val) / (max_val - min_val + 1e-9)
        return scaled

    def _csv_to_mapping(self, path, key_col="PDB_number", value_col="new_label"):
        mapping = {}

        with open(path, "r") as f:
            header = next(f).strip().split(",")

            key_idx = header.index(key_col)
            val_idx = header.index(value_col)

            for line in f:
                cols = line.strip().split(",")

                # skip empty lines or corrupt rows
                if len(cols) < max(key_idx, val_idx) + 1:
                    continue

                pdb = int(cols[key_idx])
                label = int(cols[val_idx])

                if value_col == "new_label":
                    label -= 1  # make labels zero-indexed

                mapping[pdb] = label

        return mapping

    def _pdb_num_to_state(self, pdb_num: int):
        # Example mapping, modify as needed
        if self.pdb_num_to_state_mapping_mode == "unsupervised":
            return self.pdb_num_to_state_mapping.get(pdb_num)

        elif self.pdb_num_to_state_mapping_mode == "custom":
            if self.exp_cfg.data.custom_mapping is None:
                raise ValueError("custom_mapping must be provided for 'custom' mapping mode")
            return self.exp_cfg.data.custom_mapping.get(pdb_num)
        
        else:
            raise ValueError(f"Unsupported pdb_num_to_state_mapping_mode: {self.pdb_num_to_state_mapping_mode}")

    def __call__(self, sample):

        # keys
        dist_key = "dist.pt"
        ideal_key = "ideal.pt"
        config_key = "config.pt"
        id_key = "id.pt"

        dist = sample.get(dist_key) if "distorted" in self.data_load_mode else None
        ideal = sample.get(ideal_key) if "ideal" in self.data_load_mode else None

        # convert dtype first
        if dist is not None:
            dist = dist.to(self.dtype)
        if ideal is not None:
            ideal = ideal.to(self.dtype)

        # augment
        # add white noise
        if self.add_white_noise:
            # If denoise task, add white noise to distorted images only
            
            if "distorted" in self.data_load_mode and dist is not None:
                dist = self._add_random_white_noise(dist)

            elif "ideal" in self.data_load_mode and ideal is not None:
                ideal = self._add_random_white_noise(ideal)

        # random pixel translation
        if self.translate:
            if ("distorted" in self.data_load_mode and dist is not None) and ("ideal" in self.data_load_mode and ideal is not None):
                dist, ideal = self._random_pixel_translation(
                    img=dist, img2=ideal
                    )
            elif "distorted" in self.data_load_mode and dist is not None:
                dist = self._random_pixel_translation(dist)

            elif "ideal" in self.data_load_mode and ideal is not None:
                ideal = self._random_pixel_translation(ideal)
            
        # scaling
        # min-max scaling
        if self.min_max_scaling:
            dist = self._min_max_scaling(dist) if "distorted" in self.data_load_mode and dist is not None else None
            ideal = self._min_max_scaling(ideal) if "ideal" in self.data_load_mode and ideal is not None else None

        # pdb_num to state
        if self.pdb_num_to_state and config_key in sample:
            cfg = sample[config_key]
            pdb_num = cfg.pdb_num if hasattr(cfg, "pdb_num") else cfg.get("pdb_num", None)
            if pdb_num is not None:
                state = self._pdb_num_to_state(pdb_num)
                sample["state"] = state

        # update sample and image shapes
        if "distorted" in self.data_load_mode:
            if dist.ndim == 2 and dist.shape[0] != 1:
                dist = dist.unsqueeze(0)  # add channel dim. (1, H, W)
                sample[dist_key] = dist  
            else:
                sample[dist_key] = dist
        else:
            sample[dist_key] = None

        if "ideal" in self.data_load_mode:
            if ideal.ndim == 2 and ideal.shape[0] != 1:
                ideal = ideal.unsqueeze(0)  # add channel dim (1, H, W)
            sample[ideal_key] = ideal

        else:
            sample[ideal_key] = None

        return sample
