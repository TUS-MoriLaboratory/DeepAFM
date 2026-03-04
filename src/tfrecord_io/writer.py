# src/tfrecord/writer.py

import os
import json
import numpy as np
import torch
from tfrecord import TFRecordWriter

from configs.experiment_config import ExperimentConfig
from afm_image_generation.utils.pad_utils import pad_center_fast


class AFM_TFRecord_Writer:
    """Class to write AFM images and configurations into TFRecord format."""
    def __init__(
            self,
            exp_cfg: ExperimentConfig,
        ):

        # data types
        FLOAT = exp_cfg.system.afm_dtype
        if FLOAT == "float16":
            self.dtype = torch.float16
            self.np_dtype = np.float16
        elif FLOAT == "float32":
            self.dtype = torch.float32
            self.np_dtype = np.float32
        elif FLOAT == "float64":
            self.dtype = torch.float64
            self.np_dtype = np.float64
        else:
            raise ValueError(f"Unsupported FLOAT type: {FLOAT}")

        #---- Set fixed image size ----
        self.H = exp_cfg.afm.fixed.fixed_image_size
        self.W = self.H # square image


    def chunk_to_tfrecord(self, filename: str, chunk: list):
        """
        Save a chunk of AFM images into a TFRecord shard.

        chunk = [
            (dist_img or None, ideal_img or None, cfg or None, img_id),
            ...
        ]
        """

        # TFRecord writer (shard)
        writer = TFRecordWriter(filename)

        for (dist_img, ideal_img, cfg, img_id) in chunk:

            # ---- Pad images ----
            if dist_img is not None:
                d_np = pad_center_fast(dist_img, self.H, self.W).cpu().numpy().astype(self.np_dtype)
                d_bytes = d_np.tobytes()
            else:
                d_bytes = b""   # empty byte string = None

            if ideal_img is not None:
                i_np = pad_center_fast(ideal_img, self.H, self.W).cpu().numpy().astype(self.np_dtype)
                i_bytes = i_np.tobytes()
            else:
                i_bytes = b""

            # ---- Config JSON ----
            if cfg is not None:
                cfg_json = json.dumps(
                    cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                    default=str
                ).encode("utf-8")
            else:
                cfg_json = b"{}"

            # ---- Pack record ----
            record = {
                "distorted": (d_bytes, "byte"),
                "ideal":     (i_bytes, "byte"),
                "config":    (cfg_json, "byte"),
                "image_id":  (int(img_id), "int")
            }

            writer.write(record)

        writer.close()

        print(f"[TFRecord] Saved {len(chunk)} samples → {filename}")

