# src/webdataset_io/writer.py
import os, io, tarfile
import numpy as np
import torch

from configs.experiment_config import ExperimentConfig
from afm_image_generation.utils.pad_utils import pad_center_fast

class AFM_WebDataset_Writer:
    """Write AFM samples into a WebDataset (.tar) shard."""

    def __init__(self, exp_cfg: ExperimentConfig):
        FLOAT = exp_cfg.system.afm_dtype
 
        self.torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[FLOAT]
        if self.torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {FLOAT}")

        self.H = exp_cfg.afm.fixed.fixed_height_px
        self.W = exp_cfg.afm.fixed.fixed_width_px

        self.seed = exp_cfg.system.seed

    def _add_bytes(self, tar, filename: str, data: bytes):
        info = tarfile.TarInfo(filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


    def add_sample(self, tar, dist, ideal, cfg, img_id):
        """
        Add one AFM sample into the tar shard.
        File naming convention: {SEED}_{ID}.{TYPE}.pt
        Example:
            042_000001234.dist.pt
            042_000001234.ideal.pt
            042_000001234.config.pt
            042_000001234.id.pt
        """

        prefix = f"{int(img_id):09d}"

        # ---- distorted ----
        if dist is not None:
            d_tensor = pad_center_fast(dist, self.H, self.W).cpu().to(self.torch_dtype)
            buf = io.BytesIO()
            torch.save(d_tensor, buf)
            self._add_bytes(tar, f"{self.seed:03d}_{prefix}.dist.pt", buf.getvalue())

        # ---- ideal ----
        if ideal is not None:
            i_tensor = pad_center_fast(ideal, self.H, self.W).cpu().to(self.torch_dtype)
            buf = io.BytesIO()
            torch.save(i_tensor, buf)
            self._add_bytes(tar, f"{self.seed:03d}_{prefix}.ideal.pt", buf.getvalue())

        # ---- config ----
        cfg_bytes = io.BytesIO()
        torch.save(cfg, cfg_bytes)
        self._add_bytes(tar, f"{self.seed:03d}_{prefix}.config.pt", cfg_bytes.getvalue())

        # ---- id ----
        buf = io.BytesIO()
        torch.save(torch.tensor(int(img_id), dtype=torch.int64), buf)
        self._add_bytes(tar, f"{self.seed:03d}_{prefix}.id.pt", buf.getvalue())


    def write_chunk(self, chunk: list, shard_path: str):
        """
        chunk = [(dist, ideal, cfg, img_id), ...]
        """
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        with tarfile.open(shard_path, "w") as tar:
            for dist, ideal, cfg, img_id in chunk:
                self.add_sample(tar, dist, ideal, cfg, img_id)
                
        print(f"[WebDataset] Saved chunk → {os.path.relpath(shard_path)}")