import os
import math
import numpy as np
import ray
import torch

from afm_image_generation.configs.afm_image_parameter import (
    AFMImageFixedParams, AFMImageRandomRange, AFMDataSamplingParams
)
from configs.experiment_config import ExperimentConfig
from afm_image_generation.core.afm_image_generator_base import AFMImageGeneratorBase, AFMGenerationInput
from afm_image_generation.core.afm_simulator import define_tip, scale_to_size_config, get_local_size_config, surfing, idilation, surfing_vector, idilation_vector, calculate_buffer_of_line, calculate_buffer_1d, fill_zero_line, fill_zero_pixel

# PDB Utils
from afm_image_generation.utils.pdb_utils import PDBUtils
from afm_image_generation.utils.quaternion import rotate_around_center, rotate_around_center_batch
from afm_image_generation.utils.translation import translate, translate_batch
from afm_image_generation.utils.center import calculate_center, calculate_center_batch

class AFMImageGenerator_RayBase(AFMImageGeneratorBase):
    """AFM image generator using ray tracing."""
    def __init__(
            self,
            exp_cfg: ExperimentConfig,
            xyz_refs, # ref of xyz data on shared memory
            radii_ref, # ref of radii data on shared memory
        ):
        super().__init__(exp_cfg, xyz_refs=xyz_refs, radii_ref=radii_ref)

        cuda_ids = ray.get_gpu_ids()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # device
        print(f"[AFM Ray] Using device: {self.device}")
        print(cuda_ids)

        # JIT compile functions
        self.compiled_surfing = torch.jit.script(surfing)
        self.compiled_idilation = torch.jit.script(idilation)

        #self.compiled_idilation_vector = torch.jit.script(idilation_vector)
        #self.compiled_surfing_vector = torch.jit.script(surfing_vector)

        # Execution mode
        self.vectorized = exp_cfg.afm.job.vectorized

        # generation mode
        self.mode = exp_cfg.afm.job.output_mode

    def prepare_xyz(
            self, 
            xyz, 
            params, 
            brown_x, 
            brown_y, 
            brown_rx,
            brown_ry, 
            brown_rz,
            ):
        """
        Prepare xyz:
        1. base → rotation → batch repeat → brownian rotation/translation
        """
        slow_px = params.height_px if self.scan_axis == 1 else params.width_px

        # calculate center
        center = calculate_center(xyz)

        # Initial rotation 
        xyz_initial = self.pdb_utils.transform_initial_pose(
            xyz,
            x_deg=params.rotation_x,
            y_deg=params.rotation_y,
            z_deg=params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=center
        )

        # Brownian motion batch
        center = center.unsqueeze(0).repeat(slow_px, 1)  # (slow_px, 3)
        xyz_batch = xyz_initial.unsqueeze(0).repeat(slow_px, 1, 1)

        # rotate and translate for Brownian motion
        xyz_batch = rotate_around_center_batch(
            xyz_batch, 
            z_deg=brown_rz, 
            x_deg=brown_rx, 
            y_deg=brown_ry, 
            center=center 
        )
        xyz_batch = translate_batch(
            xyz_batch, 
            x_offset=brown_x, 
            y_offset=brown_y
        )

        return xyz_initial, xyz_batch

    def compute_distorted_per_line(self, params, xyz_batch, atom_radii):
        
        # Initialize image tensor
        width_px = params.width_px
        height_px = params.height_px

        slow_px = height_px if self.scan_axis == 1 else width_px

        img = torch.zeros((height_px, width_px), dtype=self.dtype, device=self.device)
        
        tip = define_tip(
            resolution_x=params.scale_x,
            resolution_y=params.scale_y,
            probeRadius=params.probe_radius,
            probeAngle=params.half_angle,
            max_height=self.pdb_utils.get_max_height(xyz_batch)
        )

        min_columns, max_columns, target_columns = zip(
            *[calculate_buffer_of_line(
                tip, 
                line_index=i, 
                width_px=params.width_px,
                height_px=params.height_px, 
                scan_axis=self.scan_axis,
                ) 
                for i in range(slow_px)
                ])
        
        size_configs = [scale_to_size_config(
            scale_x=params.scale_x, 
            scale_y=params.scale_y,
            width_px=params.width_px,
            height_px=params.height_px,
            valid_lines=[min_col, max_col],
            scan_axis=self.scan_axis
            ) 
            for min_col, max_col in zip(min_columns, max_columns)
            ]

        if self.scan_axis == 0: # Y-scan
            k_min, k_max, xyz_idx = 'min_x', 'max_x', 0
        else: # X-scan
            k_min, k_max, xyz_idx = 'min_y', 'max_y', 1

        # Tensors for mins and maxs
        mins = torch.tensor([cfg[k_min] for cfg in size_configs], device=self.device, dtype=self.dtype)
        maxs = torch.tensor([cfg[k_max] for cfg in size_configs], device=self.device, dtype=self.dtype)

        # Create masks for each line
        masks = (xyz_batch[:, :, xyz_idx:xyz_idx+1] >= mins.view(-1, 1, 1)) & \
                (xyz_batch[:, :, xyz_idx:xyz_idx+1] <= maxs.view(-1, 1, 1))

        for i in range(slow_px):
            xyz_tmp = xyz_batch[i][masks[i].squeeze(-1)]
            if xyz_tmp.shape[0] == 0:
                fill_zero_line(img, i, self.scan_axis)
                continue

            size_config = size_configs[i]
            surface = surfing(xyz_tmp, atom_radii[masks[i].squeeze(-1)], size_config)

            if surface.numel() == 0: 
                fill_zero_line(img, i, self.scan_axis)
                continue

            image_patch = idilation(surface, tip)
    
            # Insert the column into the images
            if self.scan_axis == 1: # X-scan
                image_patch_size = image_patch.shape[0]
                img[slow_px-1-i, :] = image_patch[image_patch_size-1-target_columns[i], :] 
            else: # Y-scan
                img[:, i] = image_patch[:, target_columns[i]]

        return img
    
    def compute_distorted_per_pixel(self, input_data, xyz_base, center, atom_radii):
        
        # Initialize image tensor
        width_px = input_data.fixed_params.width_px
        height_px = input_data.fixed_params.height_px

        slow_px = height_px if self.scan_axis == 1 else width_px
        fast_px = width_px if self.scan_axis == 1 else height_px

        img = torch.zeros((height_px, width_px), dtype=self.dtype, device=self.device)
        
        # calculate center
        center = center.unsqueeze(0).repeat(fast_px, 1)  # (image_size, 3)

        # scan line by line
        for i in range(slow_px):

            # Determine start and end indices for the current line
            start_idx = i * fast_px
            end_idx = start_idx + fast_px

            # Extract the current line's transformations
            b_tx = input_data.brown_translation_x[start_idx:end_idx] # (image_size,)
            b_ty = input_data.brown_translation_y[start_idx:end_idx]
            b_rz = input_data.brown_rotation_z[start_idx:end_idx]
            b_rx = input_data.brown_rotation_x[start_idx:end_idx]
            b_ry = input_data.brown_rotation_y[start_idx:end_idx]
            
            xyz_batch = xyz_base.unsqueeze(0).repeat(fast_px, 1, 1)  # (image_size, num_atoms, 3)
            xyz_batch = rotate_around_center_batch(
                xyz_batch, 
                z_deg=b_rz,
                x_deg=b_rx, 
                y_deg=b_ry,
                center=center # use calculated center
            )
            xyz_batch = translate_batch(
                xyz_batch,
                x_offset=b_tx,
                y_offset=b_ty
            )

            tip = define_tip(
                resolution_x=input_data.fixed_params.scale_x,
                resolution_y=input_data.fixed_params.scale_y,
                probeRadius=input_data.fixed_params.probe_radius,
                probeAngle=input_data.fixed_params.half_angle,
                max_height=self.pdb_utils.get_max_height(xyz_batch),
            )

            # Calculate buffer for each pixel in the current line
            min_slow, max_slow, tgt_slow = calculate_buffer_1d(
                    tip_size=tip.shape[0 if self.scan_axis == 1 else 1], 
                    pos=slow_px-1-i if self.scan_axis == 1 else i, # slow axis pixel index 
                    limit=slow_px
                )

            fast_bounds = [
                calculate_buffer_1d(
                    tip_size=tip.shape[1 if self.scan_axis == 1 else 0], 
                    pos=fast_px-1-j if self.scan_axis == 0 else j, # fast axis pixel index
                    limit=fast_px
                )
                for j in range(fast_px) # fast axis pixel index
            ]

            size_configs = [get_local_size_config(
                x_range=(min_fast, max_fast) if self.scan_axis == 1 else (min_slow, max_slow),
                y_range=(min_slow, max_slow) if self.scan_axis == 1 else (min_fast, max_fast),
                scale_x=input_data.fixed_params.scale_x,
                scale_y=input_data.fixed_params.scale_y,
                image_size_x=width_px,
                image_size_y=height_px, 
            )
            for min_fast, max_fast, tgt_fast in fast_bounds
            ]

            # get area of interest
            x_mins_t = torch.tensor([cfg["min_x"] for cfg in size_configs], device=self.device, dtype=self.dtype).view(-1, 1)
            x_maxs_t = torch.tensor([cfg["max_x"] for cfg in size_configs], device=self.device, dtype=self.dtype).view(-1, 1)
            y_mins_t = torch.tensor([cfg["min_y"] for cfg in size_configs], device=self.device, dtype=self.dtype).view(-1, 1)
            y_maxs_t = torch.tensor([cfg["max_y"] for cfg in size_configs], device=self.device, dtype=self.dtype).view(-1, 1)
            
            # 2D mask for atoms within the area of interest
            mask_x = (xyz_batch[:, :, 0] >= x_mins_t) & (xyz_batch[:, :, 0] <= x_maxs_t)
            mask_y = (xyz_batch[:, :, 1] >= y_mins_t) & (xyz_batch[:, :, 1] <= y_maxs_t)

            # Select only atoms that satisfy both conditions
            masks = mask_x & mask_y # Shape: (image_size, num_atoms)

            # Process each pixel in the current line
            line_heights = torch.zeros(fast_px, device=self.device, dtype=self.dtype)
            for j in range(fast_px): # fast axis pixel index
                xyz_tmp = xyz_batch[j][masks[j]]
                #print(f"xyz_tmp shape: {xyz_tmp.shape}")
                if xyz_tmp.shape[0] == 0:
                    fill_zero_pixel(
                        img=img, 
                        slow_px=i, 
                        fast_px=j, 
                        axis=self.scan_axis
                        )
                    continue

                size_config = size_configs[j]
                
                surface = surfing(xyz_tmp, atom_radii[masks[j]], size_config)

                if surface.numel() == 0: 
                    fill_zero_pixel(
                        img=img, 
                        slow_px=i, 
                        fast_px=j, 
                        axis=self.scan_axis
                        )
                    continue

                image_patch = idilation(surface, tip)


                tgt_fast = fast_bounds[j][2]

                if self.scan_axis == 1: # X-scan
                    line_heights[j] = image_patch[tgt_slow, tgt_fast]
                else: # Y-scan
                    line_heights[j] = image_patch[tgt_fast, tgt_slow]

            # Insert the line into the image
            if self.scan_axis == 1: # X-scan
                img[slow_px-1-i, :] = line_heights
            else: # Y-scan
                img[:, i] = line_heights.flip(0) # Y-scan needs flipping(bottom to top)

        return img
    
    def compute_ideal(self, params, xyz_initial, atom_radii):

        tip = define_tip(
            resolution_x=params.scale_x,
            resolution_y=params.scale_y,
            probeRadius=params.probe_radius,
            probeAngle=params.half_angle,
            max_height=self.pdb_utils.get_max_height(xyz_initial)
        )

        size_config = scale_to_size_config(
            scale_x=params.scale_x,
            scale_y=params.scale_y,
            width_px=params.width_px,
            height_px=params.height_px,
            scan_axis=self.scan_axis
        )

        surface = self.compiled_surfing(xyz_initial, atom_radii, size_config)
        img = self.compiled_idilation(surface, tip)
        
        return img

    def compute_distorted_per_line_vectorized(self, params, xyz_batch, atom_radii, tip):
        image_size = self.fixed_params.image_size
        img = torch.zeros((image_size, image_size), dtype=self.dtype, device=self.device)

        # Prepare for vectorized processing
        min_columns, max_columns, target_columns = zip(*[
            calculate_buffer_of_line(tip, line_index=i, image_size=image_size, scan_axis=self.scan_axis) 
            for i in range(image_size)
        ])
        
        size_configs = [scale_to_size_config(
            scale=params.scale, 
            image_size=image_size,
            valid_lines=[min_col, max_col],
            scan_axis=self.scan_axis
            ) 
            for min_col, max_col in zip(min_columns, max_columns)
        ]

        # Create masks for valid atoms
        if self.scan_axis == 0: # Y-scan
            k_min, k_max, xyz_idx = 'min_x', 'max_x', 0
            gather_dim = 2 # For Y-scan, we gather along width (columns)
        else: # X-scan
            k_min, k_max, xyz_idx = 'min_y', 'max_y', 1
            gather_dim = 1 # For X-scan, we gather along height (rows)
        
        x_mins = torch.tensor([cfg['min_x'] for cfg in size_configs], device=self.device, dtype=self.dtype)
        x_maxs = torch.tensor([cfg['max_x'] for cfg in size_configs], device=self.device, dtype=self.dtype)
        y_mins = torch.tensor([cfg['min_y'] for cfg in size_configs], device=self.device, dtype=self.dtype)
        y_maxs = torch.tensor([cfg['max_y'] for cfg in size_configs], device=self.device, dtype=self.dtype)
        resolutions_xs = torch.tensor([cfg['resolution_x'] for cfg in size_configs], device=self.device, dtype=self.dtype)
        resolutions_ys = torch.tensor([cfg['resolution_y'] for cfg in size_configs], device=self.device, dtype=self.dtype)

        # Tensors for mins and maxs
        mins = torch.tensor([cfg[k_min] for cfg in size_configs], device=self.device, dtype=self.dtype)
        maxs = torch.tensor([cfg[k_max] for cfg in size_configs], device=self.device, dtype=self.dtype)

        # Create masks for each line
        masks = (xyz_batch[:, :, xyz_idx:xyz_idx+1] >= mins.view(-1, 1, 1)) & \
                (xyz_batch[:, :, xyz_idx:xyz_idx+1] <= maxs.view(-1, 1, 1))
        
        # Batch processing setup
        num_valid_atoms_per_column = masks.sum(dim=1).squeeze(-1)
        max_atoms = num_valid_atoms_per_column.max().item()
        padded_xyz = torch.zeros((image_size, max_atoms, 3), dtype=xyz_batch.dtype, device=self.device)
        padded_radii = torch.zeros((image_size, max_atoms), dtype=atom_radii.dtype, device=self.device)

        attention_mask = torch.arange(max_atoms, device=self.device)[None, :] < num_valid_atoms_per_column[:, None]

        for i in range(image_size):
            valid_mask_i = masks[i].squeeze(-1)
            xyz_tmp = xyz_batch[i][valid_mask_i]
            radii_tmp = atom_radii[valid_mask_i]

            num_atoms = xyz_tmp.shape[0]
            padded_xyz[i, :num_atoms] = xyz_tmp
            padded_radii[i, :num_atoms] = radii_tmp

        # Filter non-empty columns
        non_empty_mask = attention_mask.any(dim=1)
        non_empty_indices = torch.where(non_empty_mask)[0]
        
        if len(non_empty_indices) > 0:
            padded_xyz_filtered = padded_xyz[non_empty_indices]
            padded_radii_filtered = padded_radii[non_empty_indices]
            attention_mask_filtered = attention_mask[non_empty_indices]
            
            # Surface computation
            surface = self.compiled_surfing_vector(
                padded_xyz_filtered, padded_radii_filtered, attention_mask_filtered,
                x_mins[non_empty_indices], x_maxs[non_empty_indices], resolutions_xs[non_empty_indices],
                y_mins[non_empty_indices], y_maxs[non_empty_indices], resolutions_ys[non_empty_indices]
            )
            # Image generation
            image_columns = self.compiled_idilation_vector(surface, tip)

            # --- target line (gather) ---
            target_indices = torch.tensor([target_columns[i] for i in non_empty_indices], device=self.device)
            
            # X-scan (reverse index since scanning from bottom to top)
            if self.scan_axis == 1:
                patch_size = image_columns.size(gather_dim)
                target_indices = patch_size - 1 - target_indices

            # Expand indices for gather operation
            if self.scan_axis == 0: # Y-scan
                target_expanded = target_indices.view(-1, 1, 1).expand(-1, image_columns.size(1), 1)
            else: # X-scan
                target_expanded = target_indices.view(-1, 1, 1).expand(-1, 1, image_columns.size(2))
            
            selected = image_columns.gather(gather_dim, target_expanded).squeeze(gather_dim)

            # --- 3. Assign to final image (with index reversal) ---
            if self.scan_axis == 1: # X-scan
                # Reverse indices to arrange from bottom to top
                flipped_row_indices = image_size - 1 - non_empty_indices
                img[flipped_row_indices, :] = selected
            else: # Y-scan
                img[:, non_empty_indices] = selected.T

        return img
    
    def _prepare_common_assets(self, input_data):
        """ prepare common assets (xyz, tip, atom_radii)"""
        params = input_data.fixed_params
        

        # xyz (load from shared memory)
        idx = params.pdb_num - 1
        xyz = torch.tensor(ray.get(self.xyz_refs[idx]), dtype=self.dtype, device=self.device)

        # Radii
        atom_radii = None
        if self.radii_ref is not None:
            atom_radii = torch.tensor(self.radii_ref[idx], dtype=self.dtype, device=self.device)

        return xyz, atom_radii, params
    
    def next_per_line(self, global_index: int):
        """
        Generate one AFM image (called repeatedly by multiprocess driver).
        
        if generate distorted image, line-wise generation is used.
        """

        input_data = next(self.input_iterator)
        
        xyz, atom_radii, params = self._prepare_common_assets(input_data)

        # 1. only ideal image
        if "ideal" in self.mode and "distorted" not in self.mode:
            xyz_initial = self.pdb_utils.transform_initial_pose(
                xyz,
                x_deg=params.rotation_x,
                y_deg=params.rotation_y,
                z_deg=params.rotation_z,
                x_offset=0.0,
                y_offset=0.0,
                center=None # center will be calculated inside the function
            )
            ideal_img = self.compute_ideal(params, xyz_initial, atom_radii)
            cfg = params if "config" in self.mode else None
            return None, ideal_img, cfg, global_index
        
        # 2. distorted and ideal images
        xyz_initial, xyz_batch = self.prepare_xyz(
            xyz,
            params,
            input_data.brown_translation_x,
            input_data.brown_translation_y,
            input_data.brown_rotation_x,
            input_data.brown_rotation_y,
            input_data.brown_rotation_z,
        )

        dist_img = None
        ideal_img = None

        if "distorted" in self.mode:
            # vectorized or not
            if self.vectorized:
                dist_img = self.compute_distorted_per_line_vectorized(
                    params, xyz_batch, atom_radii
                )
            else:
                dist_img = self.compute_distorted_per_line(
                    params, xyz_batch, atom_radii
                )

        if "ideal" in self.mode:
            ideal_img = self.compute_ideal(
                params, xyz_initial, atom_radii
            )

        cfg = params if "config" in self.mode else None

        return dist_img, ideal_img, cfg, global_index
        
    def next_per_pixel(self, global_index: int):
        """
        Generate one AFM image (called repeatedly by multiprocess driver).
        
        if generate distorted image, pixel-wise generation is used.
        """

        input_data = next(self.input_iterator)
        
        xyz, atom_radii, params = self._prepare_common_assets(input_data)

        # 1. only ideal image
        if "ideal" in self.mode and "distorted" not in self.mode:
            xyz_initial = self.pdb_utils.transform_initial_pose(
                xyz,
                x_deg=params.rotation_x,
                y_deg=params.rotation_y,
                z_deg=params.rotation_z,
                x_offset=0.0,
                y_offset=0.0,
                center=None # center will be calculated inside the function
            )
            ideal_img = self.compute_ideal(params, xyz_initial, atom_radii)
            cfg = params if "config" in self.mode else None
            return None, ideal_img, cfg, global_index
        
        # 2. distorted and ideal images
        center = calculate_center(xyz)
        xyz_initial = self.pdb_utils.transform_initial_pose(
            xyz,
            x_deg=params.rotation_x,
            y_deg=params.rotation_y,
            z_deg=params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=center
        )

        dist_img = None
        ideal_img = None

        if "distorted" in self.mode:
            # vectorized or not
            if self.vectorized:
                raise NotImplementedError(
                    "Vectorized mode is not yet supported for pixel-wise scanning. "
                    "Use 'vectorized=False' or 'scan_unit=\"line\"' instead."
                )
            
            else:
                dist_img = self.compute_distorted_per_pixel(
                    input_data=input_data, 
                    xyz_base=xyz_initial, 
                    center=center, 
                    atom_radii=atom_radii, 
                )

        if "ideal" in self.mode:
            ideal_img = self.compute_ideal(
                params, xyz_initial, atom_radii
            )

        cfg = params if "config" in self.mode else None

        return dist_img, ideal_img, cfg, global_index

    def next(self, global_index: int):
        """
        Generate one AFM image (called repeatedly by multiprocess driver).
        """

        # determine scan unit
        if self.scan_unit == "line":
            return self.next_per_line(global_index)
        else:
            return self.next_per_pixel(global_index)

@ray.remote
class AFMImageGenerator_Ray(AFMImageGenerator_RayBase):
    """AFM image generator using ray tracing."""
    def __init__(
            self,
            exp_cfg: ExperimentConfig,
            xyz_refs, # ref of xyz data on shared memory
            radii_ref, # ref of radii data on shared memory
        ):
        super().__init__(exp_cfg, xyz_refs=xyz_refs, radii_ref=radii_ref)
