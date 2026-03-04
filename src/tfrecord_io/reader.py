# src/tfrecord_io/reader.py

import json
import numpy as np
import torch
from tfrecord.reader import tfrecord_iterator
from tfrecord import example_pb2
from typing import List, Optional

class AFMTFRecordReader:
    """
    Low-level TFRecord reader compatible with AFM_TFRecord_Writer.
    Reads a single TFRecord file and yields dicts:
        {
            "distorted": Tensor or None,
            "ideal": Tensor or None,
            "config": dict,
            "image_id": int
        }
    """

    def __init__(
        self,
        filename: str,
        exp_cfg,
        image_size: int = None,
        dtype: torch.dtype = None,
    ):
        self.filename = filename
        self.exp_cfg = exp_cfg
        self.image_size = exp_cfg.afm.fixed.fixed_image_size if image_size is None else image_size
        self.dtype = exp_cfg.system.afm_dtype if dtype is None else dtype

        self.H = self.W = self.image_size
        
        self.torch_dtype = {
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
        }[self.dtype]

        self.np_dtype = {
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
        }[self.dtype]

    def _parse_example(self, raw_record: bytes):
        """Parse raw TFRecord bytes into dict using tf.train.Example."""

        example = example_pb2.Example()
        example.ParseFromString(raw_record)

        features = example.features.feature

        def get_bytes(name):
            f = features.get(name)
            if f is None or len(f.bytes_list.value) == 0:
                return None
            return f.bytes_list.value[0]

        def get_int(name):
            f = features.get(name)
            if f is None or len(f.int64_list.value) == 0:
                return 0
            return int(f.int64_list.value[0])

        distorted = get_bytes("distorted")
        ideal = get_bytes("ideal")
        cfg_bytes = get_bytes("config") or b"{}"
        image_id = get_int("image_id")

        return distorted, ideal, cfg_bytes, image_id


    def __iter__(self):

        for raw_record in tfrecord_iterator(self.filename):
            distorted_bytes, ideal_bytes, cfg_bytes, image_id = \
                self._parse_example(raw_record)

            # --- distorted ---
            if distorted_bytes:
                arr = np.frombuffer(distorted_bytes, dtype=self.np_dtype)
                arr = arr.reshape(self.H, self.W)
                distorted = torch.from_numpy(arr).to(self.torch_dtype)
            else:
                distorted = None

            # --- ideal ---
            if ideal_bytes:
                arr = np.frombuffer(ideal_bytes, dtype=self.np_dtype)
                arr = arr.reshape(self.H, self.W)
                ideal = torch.from_numpy(arr).to(self.torch_dtype)
            else:
                ideal = None

            # --- config ---
            cfg = json.loads(cfg_bytes.decode("utf-8"))

            yield {
                "distorted": distorted,
                "ideal": ideal,
                "config": cfg,
                "image_id": image_id,
            }
