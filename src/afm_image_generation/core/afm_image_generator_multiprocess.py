import h5py
import json
import copy
import os
import ray
import torch

from afm_image_generation.utils.pad_utils import pad_center_fast
from configs.experiment_config import ExperimentConfig
from afm_image_generation.core.afm_image_generator_base import AFMImageGeneratorBase
from afm_image_generation.core.afm_image_generator_ray import AFMImageGenerator_Ray

from tfrecord_io.writer import AFM_TFRecord_Writer

class AFMImageGenerator_Multiprocess(AFMImageGeneratorBase):
    """AFM image generator using ray tracing."""
    def __init__(
            self,
            exp_cfg: ExperimentConfig,
        ):
        super().__init__(exp_cfg)
        
        # setup job parameters
        # Validate job config
        if not hasattr(exp_cfg.afm, "job") or exp_cfg.afm.job is None:
            raise ValueError(
                "AFM job configuration (exp_cfg.afm.job) is missing.\n"
                "Required fields:\n"
                "  - processes\n"
                "  - chunk_size\n"
                "  - total_images\n"
                "  - save_dir"
            )

        job = exp_cfg.afm.job
        required_fields = ["processes", "chunk_size", "total_images", "save_dir"]
        missing = [f for f in required_fields if not hasattr(job, f)]

        if missing:
            raise ValueError(
                f"Missing fields in AFM job config: {missing}\n"
                "Please define all job config fields in exp_cfg.afm.job."
            )

        self.processes = job.processes
        self.chunk_size = job.chunk_size
        self.total_images = job.total_images
        self.save_dir = job.save_dir

        print(f"[AFM Multiprocess] Configured for {self.total_images} images using {self.processes} processes with chunk size {self.chunk_size}.")

        # Whether to use GPU for AFM image generation
        self.use_gpu_for_afm = exp_cfg.system.use_gpu_for_afm
        print(f"[AFM Multiprocess] Use GPU for AFM: {self.use_gpu_for_afm}")

        # save mode ("tfrecord" or "hdf5")
        self.save_mode = exp_cfg.system.save_mode
        print(f"[AFM Multiprocess] Save mode set to: {self.save_mode}")

        # AFM mode
        print(f"[AFM Multiprocess] Scan direction: {exp_cfg.system.scan_direction}, Scan unit: {exp_cfg.system.scan_unit}")

        # setup data paths
        self.src_path = os.path.join(exp_cfg.system.project_root, "src") if exp_cfg.system.project_root else None

        self.xyz_refs = None
        self.radii_ref = None
        self.actors = []

    def setup(self):
        """initialize ray and load data to object store"""

        if not ray.is_initialized():
            try:
                ray.init(
                    runtime_env={"working_dir": self.src_path or os.getcwd()},
                    log_to_driver=False
                    )
            except Exception as e:
                print(f"[Warning] Ray already initialized: {e}")

        self.xyz_refs = [ray.put(frame.cpu().numpy()) for frame in self.xyz_data]
        self.radii_ref = ray.put(self.atom_radii.cpu().numpy())

        # wgether to use GPU for AFM image generation
        ActorClass = AFMImageGenerator_Ray.options(
            num_gpus=1 if self.use_gpu_for_afm else 0
        )

        base_seed = self.exp_cfg.system.seed if self.exp_cfg.system.seed is not None else 0

        for i in range(self.processes):
            actor_cfg = copy.deepcopy(self.exp_cfg)
            actor_cfg.system.seed = base_seed + i + 1

            actor = ActorClass.remote(
                exp_cfg=actor_cfg,
                xyz_refs=self.xyz_refs,
                radii_ref=self.radii_ref
            )

            self.actors.append(actor)

    # save chunk as hdf5 file
    @staticmethod
    def _save_chunk_to_hdf5(filename: str, chunk, fixed_image_size: int):

        """
        chunk = [
            (dist_img or None, ideal_img or None, cfg or None),
            ...
        ]
        """

        chunk = sorted(chunk, key=lambda x: x[3]) 

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        H = W = fixed_image_size

        # Pre-allocate padded lists
        dist_list  = []
        ideal_list = []
        cfg_list   = []
        idx_list   = []

        # ===== Center Pad for all items =====
        for (d, i, c, img_id) in chunk:
            d_pad = pad_center_fast(d, H, W) if d is not None else None
            i_pad = pad_center_fast(i, H, W) if i is not None else None
            dist_list.append(d_pad)
            ideal_list.append(i_pad)
            cfg_list.append(c)
            idx_list.append(img_id)

        with h5py.File(filename, "w") as h5f:

            # ===== distorted =====
            if any(x is not None for x in dist_list):
                dset = h5f.create_dataset(
                    "distorted",
                    shape=(len(chunk), H, W),
                    dtype="float32"
                )
                for idx, d in enumerate(dist_list):
                    if d is None:
                        dset[idx] = 0
                    else:
                        dset[idx] = d.cpu().numpy()

            # ===== ideal =====
            if any(x is not None for x in ideal_list):
                dset = h5f.create_dataset(
                    "ideal",
                    shape=(len(chunk), H, W),
                    dtype="float32"
                )
                for idx, i in enumerate(ideal_list):
                    if i is None:
                        dset[idx] = 0
                    else:
                        dset[idx] = i.cpu().numpy()

            # ===== config (JSON) =====
            if any(x is not None for x in cfg_list):
                cfg_json = [
                    json.dumps(c.__dict__, default=str) if c is not None else "{}"
                    for c in cfg_list
                ]
                dt = h5py.string_dtype(encoding="utf-8")
                h5f.create_dataset("config", data=cfg_json, dtype=dt)

            # ===== image IDs =====
            if any(x is not None for x in idx_list):
                h5f.create_dataset("image_id", data=idx_list, dtype="int64")

        print(f"[HDF5] Saved {len(chunk)} samples → {filename}")

    def _save_chunk(self, chunk_count: int, chunk):
        """Save a chunk of AFM images into a file (HDF5 or TFRecord)."""
        # TFRecord writer (shard)

        filename_base =f"{self.exp_cfg.system.seed}_{chunk_count:04d}"
        if self.save_mode == "tfrecord":
            # sort chunk by image_id
            chunk = sorted(chunk, key=lambda x: x[3])
            # make writer
            writer = AFM_TFRecord_Writer(self.exp_cfg)
            filename = os.path.join(self.save_dir, f"{filename_base}.tfrecord")
            writer.chunk_to_tfrecord(filename, chunk)
        
        elif self.save_mode == "webdataset":
            from webdataset_io.writer import AFM_WebDataset_Writer
            # sort chunk by image_id
            chunk = sorted(chunk, key=lambda x: x[3])
            # make writer
            writer = AFM_WebDataset_Writer(self.exp_cfg)
            filename = os.path.join(self.save_dir, f"{filename_base}.tar")
            writer.write_chunk(chunk, filename)
        
        # HDF5 writer
        elif self.save_mode == "hdf5":
            raise NotImplementedError("HDF5 save mode is not supported in multiprocess mode.")
            #filename = os.path.join(self.save_dir, f"{filename_base}.hdf5")
            #self._save_chunk_to_hdf5(filename, chunk, self.fixed_image_size)
        
        else:
            raise ValueError(f"Unsupported save mode: {self.save_mode}")

    def run_parallel_process(self):
        """run parallel process to generate AFM images"""

        if not self.actors:
            raise RuntimeError("Actors not initialized. Run setup() first.")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # request initial tasks
        chunk = [] # current chunk to save 
        chunk_count = 0 # number of chunks saved
        global_index = 0 # global image index
        
        futures = {} # map of future refs to actors
        for actor in self.actors:
            futures[actor.next.remote(global_index)] = actor
            global_index += 1
        
        while futures:
            ready_refs, _ = ray.wait(list(futures.keys()), timeout=10.0)

            for ref in ready_refs:
                # get the result from the completed task
                results = ray.get(ref)
                # get the actor that returned this result
                actor = futures.pop(ref)

                chunk.append(results)

                # check if we need to submit a new task
                if global_index >= self.total_images:
                    print("[INFO] Reached total_images. Stopping submission.")
                else:
                    new_ref = actor.next.remote(global_index)
                    futures[new_ref] = actor
                    global_index += 1

                # save chunk if reached chunk_size
                if len(chunk) >= self.chunk_size:
                    self._save_chunk(chunk_count, chunk[:self.chunk_size])
                    chunk = chunk[self.chunk_size :]
                    chunk_count += 1

            if global_index >= self.total_images and len(futures) == 0:
                break

        # 3. Handle remainder
        if chunk:
            self._save_chunk(chunk_count, chunk)

    def shutdown(self):
        """Safely shutdown Ray environment."""
        if ray.is_initialized():
            try:
                ray.shutdown()
                print("[Shutdown] Ray environment closed successfully.")
            except Exception as e:
                print(f"[Warning] Failed to shutdown Ray: {e}")
