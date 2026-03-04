# src/webdataset_io/reader.py

import os
import tarfile
import io
import json
import numpy as np
import torch
import glob

class AFMWebdatasetReader:
    """
    Add one AFM sample into the tar shard.
    File naming convention: {SEED}_{ID}.{TYPE}.pt
    Example:
        042_000001234.dist.pt
        042_000001234.ideal.pt
        042_000001234.config.pt
        042_000001234.id.pt
    """

    def __init__(self, data_dir: str, exp_cfg, image_size: int = None, dtype: str = None):

        # ---- find all shards ----
        self.filenames = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
        if len(self.filenames) == 0:
            raise FileNotFoundError(f"No .tar files found in {data_dir}")

        self.exp_cfg = exp_cfg

        # ---- image dtype & size ----
        self.dtype_str = dtype or exp_cfg.system.afm_dtype

        # torch dtype
        self.torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[self.dtype_str]

    # =========================================================
    # Helper loaders
    # =========================================================
    def _load_pt(self, tar: tarfile.TarFile, member: tarfile.TarInfo):
        f = tar.extractfile(member)
        return torch.load(io.BytesIO(f.read()), map_location="cpu")

    # =========================================================
    # Main iterator
    # =========================================================
    def __iter__(self):

        for filename in self.filenames:
            with tarfile.open(filename, "r") as tar:

                grouped = {}

                # ---------------------------------------------------------
                # group members by sample prefix
                # ---------------------------------------------------------
                for member in tar.getmembers():
                    basename = os.path.basename(member.name)
                    prefix, suffix = basename.split(".", 1)

                    if prefix not in grouped:
                        grouped[prefix] = {}

                    grouped[prefix][suffix] = member

                # ---------------------------------------------------------
                # iterate sorted (prefix order)
                # ---------------------------------------------------------
                for prefix in sorted(grouped.keys()):
                    files = grouped[prefix]

                    # distorted
                    if "dist.pt" in files:
                        distorted = self._load_pt(tar, files["dist.pt"]).to(self.torch_dtype)
                    else:
                        distorted = None

                    # ideal
                    if "ideal.pt" in files:
                        ideal = self._load_pt(tar, files["ideal.pt"]).to(self.torch_dtype)
                    else:
                        ideal = None

                    # config (dict or dataclass)
                    if "config.pt" in files:
                        cfg = self._load_pt(tar, files["config.pt"])
                    else:
                        cfg = {}

                    # image_id (tensor → int)
                    if "id.pt" in files:
                        image_id = int(self._load_pt(tar, files["id.pt"]))
                    else:
                        image_id = None

                    yield {
                        "distorted": distorted,
                        "ideal": ideal,
                        "config": cfg,
                        "image_id": image_id,
                    }
