import os
import math
import numpy as np
import torch

from afm_image_generation.configs.afm_image_parameter import (
    AFMImageFixedParams, AFMImageRandomRange, AFMDataSamplingParams
)
from configs.experiment_config import ExperimentConfig
from afm_image_generation.core.afm_image_generator_base import AFMImageGeneratorBase, AFMGenerationInput
from afm_image_generation.core.afm_simulator import define_tip, scale_to_size_config, get_local_size_config, surfing, idilation, calculate_buffer_of_line, calculate_buffer_1d, fill_zero_line, fill_zero_pixel

# PDB Utils
from afm_image_generation.utils.pdb_utils import PDBUtils
from afm_image_generation.utils.quaternion import rotate_around_center, rotate_around_center_batch
from afm_image_generation.utils.translation import translate, translate_batch
from afm_image_generation.utils.center import calculate_center, calculate_center_batch


class AFMImageGeneratorCPU(AFMImageGeneratorBase):
    """AFM image generator using CPU computations."""
    def __init__(
            self,
            exp_cfg: ExperimentConfig
        ):
        super().__init__(exp_cfg)

    def generate_ideal_image(
            self, 
            input_data: AFMGenerationInput, 
            xyz: torch.Tensor = None, 
            atom_radii: torch.Tensor = None
            ) -> torch.Tensor:
        
        # generate AFM image using tip and molecular structure
        idx = int(input_data.fixed_params.pdb_num - 1)
        
        # Use provided xyz if available, else load from data
        if xyz is None:
            xyz = self.xyz_data[idx, :, :]
        else:
            xyz = xyz

        # Use provided atom_radii if available, else load from data
        if atom_radii is None:
            atom_radii = self.atom_radii[idx, :]
        else:
            atom_radii = atom_radii

        # Rotate molecular structure and set 0-z position
        xyz = self.pdb_utils.transform_initial_pose(
            xyz, 
            x_deg=input_data.fixed_params.rotation_x, 
            y_deg=input_data.fixed_params.rotation_y,
            z_deg=input_data.fixed_params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=None # center will be calculated inside the function
        )

        # Placeholder implementation for CPU-based AFM image generation
        # Actual AFM image generation logic would go here

        tip = define_tip(
            resolution_x=input_data.fixed_params.scale_x,
            resolution_y=input_data.fixed_params.scale_y,
            probeRadius=input_data.fixed_params.probe_radius,
            probeAngle=input_data.fixed_params.half_angle,
            max_height=self.pdb_utils.get_max_height(xyz)
            )

        size_config = scale_to_size_config(
            scale_x=input_data.fixed_params.scale_x,
            scale_y=input_data.fixed_params.scale_y,
            width_px=input_data.fixed_params.width_px,
            height_px=input_data.fixed_params.height_px,
            scan_axis=self.scan_axis
        )

        surface = surfing(xyz, atom_radii, size_config)
        img = idilation(surface, tip)

        return img
    
    def generate_distorted_image(
            self, 
            input_data: AFMGenerationInput, 
            xyz: torch.Tensor = None, 
            atom_radii: torch.Tensor = None
            ) -> torch.Tensor:
        
        #Assert scan unit
        assert self.scan_unit == "line", f"Please set scan unit: {self.scan_unit}. "

        # Placeholder implementation for CPU-based distorted AFM image generation
        # Actual distorted AFM image generation logic would go here
        
        # Initialize image tensor
        width_px = input_data.fixed_params.width_px
        height_px = input_data.fixed_params.height_px

        slow_px = height_px if self.scan_axis == 1 else width_px

        img = torch.zeros((height_px, width_px), dtype=self.dtype, device=self.device)
        
        brown_translation_x = input_data.brown_translation_x
        brown_translation_y = input_data.brown_translation_y
        brown_rotation_x = input_data.brown_rotation_x
        brown_rotation_y = input_data.brown_rotation_y
        brown_rotation_z = input_data.brown_rotation_z
        
        # generate AFM image using tip and molecular structure
        idx = int(input_data.fixed_params.pdb_num - 1)

        # Use provided xyz if available, else load from data
        if xyz is None:
            xyz = self.xyz_data[idx, :, :]
        else:
            xyz = xyz

        # Use provided atom_radii if available, else load from data
        if atom_radii is None:
            atom_radii = self.atom_radii[idx, :]
        else:
            atom_radii = atom_radii

        # calculate center
        center = calculate_center(xyz)

        # Rotate molecular structure for initial orientation and set 0-z position
        xyz = self.pdb_utils.transform_initial_pose(
            xyz, 
            x_deg=input_data.fixed_params.rotation_x, 
            y_deg=input_data.fixed_params.rotation_y,
            z_deg=input_data.fixed_params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=center
        )

        # calculate center 
        center = center.unsqueeze(0).repeat(slow_px, 1)  # (slow_px, 3)

        xyz = xyz.unsqueeze(0).repeat(slow_px, 1, 1)  # (slow_px, num_atoms, 3)
        xyz = rotate_around_center_batch(
            xyz, 
            z_deg=brown_rotation_z,
            x_deg=brown_rotation_x, 
            y_deg=brown_rotation_y,
            center=center # use calculated center
        )

        xyz = translate_batch(
            xyz,
            x_offset=brown_translation_x,
            y_offset=brown_translation_y
        )

        tip = define_tip(
            resolution_x=input_data.fixed_params.scale_x,
            resolution_y=input_data.fixed_params.scale_y,
            probeRadius=input_data.fixed_params.probe_radius,
            probeAngle=input_data.fixed_params.half_angle,
            max_height=self.pdb_utils.get_max_height(xyz)
            )

        min_columns, max_columns, target_columns = zip(
            *[calculate_buffer_of_line(
                tip, 
                line_index=i, 
                width_px=input_data.fixed_params.width_px,
                height_px=input_data.fixed_params.height_px, 
                scan_axis=self.scan_axis,
                ) 
                for i in range(slow_px)
                ])
        
        size_configs = [scale_to_size_config(
            scale_x=input_data.fixed_params.scale_x, 
            scale_y=input_data.fixed_params.scale_y,
            width_px=input_data.fixed_params.width_px,
            height_px=input_data.fixed_params.height_px,
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
        masks = (xyz[:, :, xyz_idx:xyz_idx+1] >= mins.view(-1, 1, 1)) & \
                (xyz[:, :, xyz_idx:xyz_idx+1] <= maxs.view(-1, 1, 1))

        for i in range(slow_px):
            xyz_tmp = xyz[i][masks[i].squeeze(-1)]
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

    def generate_distorted_image_simple(
            self, 
            input_data: AFMGenerationInput, 
            xyz: torch.Tensor = None,
            atom_radii: torch.Tensor = None
            ) -> torch.Tensor:
        
        # Assert scan unit
        assert self.scan_unit == "line", f"Please set scan unit: {self.scan_unit}."

        # Placeholder implementation for CPU-based distorted AFM image generation
        # Actual distorted AFM image generation logic would go here
        
        # Initialize image tensor
        width_px = input_data.fixed_params.width_px
        height_px = input_data.fixed_params.height_px

        slow_px = height_px if self.scan_axis == 1 else width_px

        img = torch.zeros((height_px, width_px), dtype=self.dtype, device=self.device)
        
        brown_translation_x = input_data.brown_translation_x
        brown_translation_y = input_data.brown_translation_y

        brown_rotation_x = input_data.brown_rotation_x
        brown_rotation_y = input_data.brown_rotation_y
        brown_rotation_z = input_data.brown_rotation_z
        
        # generate AFM image using tip and molecular structure
        idx = int(input_data.fixed_params.pdb_num - 1)
           
        # Use provided xyz if available, else load from data
        if xyz is None:
            xyz = self.xyz_data[idx, :, :]
        else:
            xyz = xyz

        # Use provided atom_radii if available, else load from data
        if atom_radii is None:
            atom_radii = self.atom_radii[idx, :]
        else:
            atom_radii = atom_radii

        # calculate center
        center = calculate_center(xyz)
        
        # Rotate molecular structure for initial orientation and set 0-z position
        xyz = self.pdb_utils.transform_initial_pose(
            xyz, 
            x_deg=input_data.fixed_params.rotation_x, 
            y_deg=input_data.fixed_params.rotation_y,
            z_deg=input_data.fixed_params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=center
        )

        # calculate center
        center = center.unsqueeze(0).repeat(slow_px, 1)  # (slow_px, 3)
        xyz = xyz.unsqueeze(0).repeat(slow_px, 1, 1)  # (slow_px, num_atoms, 3)
        xyz = rotate_around_center_batch(
            xyz, 
            z_deg=brown_rotation_z,
            x_deg=brown_rotation_x, 
            y_deg=brown_rotation_y,
            center=center # use calculated center
        )
        xyz = translate_batch(
            xyz,
            x_offset=brown_translation_x,
            y_offset=brown_translation_y
        )

        tip = define_tip(
            resolution_x=input_data.fixed_params.scale_x,
            resolution_y=input_data.fixed_params.scale_y,
            probeRadius=input_data.fixed_params.probe_radius,
            probeAngle=input_data.fixed_params.half_angle,
            max_height=self.pdb_utils.get_max_height(xyz)
            )

        size_config = scale_to_size_config(
            scale_x=input_data.fixed_params.scale_x,
            scale_y=input_data.fixed_params.scale_y,
            width_px=input_data.fixed_params.width_px,
            height_px=input_data.fixed_params.height_px,
            scan_axis=self.scan_axis
        )

        for i in range(slow_px):
            surface = surfing(xyz[i, :, :], atom_radii, size_config)
            image_column = idilation(surface, tip)

            if self.scan_axis == 1: # X-scan from bottom to top
                img[slow_px-1-i, :] = image_column[slow_px-1-i, :] 
            else: # Y-scan
                img[:, i] = image_column[:, i] 

        return img
    
    def generate_distorted_image_pixel_by_pixel(
            self, 
            input_data: AFMGenerationInput, 
            xyz: torch.Tensor = None,
            atom_radii: torch.Tensor = None
            ) -> torch.Tensor:
        
        # Assert scan unit
        assert self.scan_unit == "pixel", f"Please set scan unit: {self.scan_unit}."

        # Placeholder implementation for CPU-based distorted AFM image generation
        # Actual distorted AFM image generation logic would go here
        
        # Initialize image tensor
        width_px = input_data.fixed_params.width_px
        height_px = input_data.fixed_params.height_px

        slow_px = height_px if self.scan_axis == 1 else width_px
        fast_px = width_px if self.scan_axis == 1 else height_px

        img = torch.zeros((height_px, width_px), dtype=self.dtype, device=self.device)
        
        # generate AFM image using tip and molecular structure
        idx = int(input_data.fixed_params.pdb_num - 1)
           
        # Use provided xyz if available, else load from data
        if xyz is None:
            xyz = self.xyz_data[idx, :, :]
        else:
            xyz = xyz

        # Use provided atom_radii if available, else load from data
        if atom_radii is None:
            atom_radii = self.atom_radii
        else:
            atom_radii = atom_radii

        # calculate center
        center = calculate_center(xyz)
        
        # Rotate molecular structure for initial orientation and set 0-z position
        xyz_base = self.pdb_utils.transform_initial_pose(
            xyz, 
            x_deg=input_data.fixed_params.rotation_x, 
            y_deg=input_data.fixed_params.rotation_y,
            z_deg=input_data.fixed_params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=center
        )

        # calculate center
        center = center.unsqueeze(0).repeat(fast_px, 1)  # (fast_px, 3)

        # scan line by line
        for i in range(slow_px):

            # Determine start and end indices for the current line
            start_idx = i * fast_px
            end_idx = start_idx + fast_px

            # Extract the current line's transformations
            b_tx = input_data.brown_translation_x[start_idx:end_idx] # (fast_px,)
            b_ty = input_data.brown_translation_y[start_idx:end_idx]
            b_rz = input_data.brown_rotation_z[start_idx:end_idx]
            b_rx = input_data.brown_rotation_x[start_idx:end_idx]
            b_ry = input_data.brown_rotation_y[start_idx:end_idx]
            
            xyz_batch = xyz_base.unsqueeze(0).repeat(fast_px, 1, 1)  # (fast_px, num_atoms, 3)
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
            masks = mask_x & mask_y # Shape: (fast_px, num_atoms)

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
                #print(f"size_config: {size_config}")
                
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
    
    def generate_distorted_image_pixel_by_pixel_simple(
            self, 
            input_data: AFMGenerationInput, 
            xyz: torch.Tensor = None,
            atom_radii: torch.Tensor = None
            ) -> torch.Tensor:
        
        # Assert scan unit
        assert self.scan_unit == "pixel", f"Please set scan unit: {self.scan_unit}."

        # Placeholder implementation for CPU-based distorted AFM image generation
        # Actual distorted AFM image generation logic would go here
        
        # Initialize image tensor
        width_px = input_data.fixed_params.width_px
        height_px = input_data.fixed_params.height_px

        slow_px = height_px if self.scan_axis == 1 else width_px
        fast_px = width_px if self.scan_axis == 1 else height_px

        img = torch.zeros((height_px, width_px), dtype=self.dtype, device=self.device)

        # generate AFM image using tip and molecular structure
        idx = int(input_data.fixed_params.pdb_num - 1)
           
        # Use provided xyz if available, else load from data
        if xyz is None:
            xyz = self.xyz_data[idx, :, :]
        else:
            xyz = xyz

        # Use provided atom_radii if available, else load from data
        if atom_radii is None:
            atom_radii = self.atom_radii[idx, :]
        else:
            atom_radii = atom_radii

        # calculate center
        center = calculate_center(xyz)
        
        # Rotate molecular structure for initial orientation and set 0-z position
        xyz_base = self.pdb_utils.transform_initial_pose(
            xyz, 
            x_deg=input_data.fixed_params.rotation_x, 
            y_deg=input_data.fixed_params.rotation_y,
            z_deg=input_data.fixed_params.rotation_z,
            x_offset=0.0,
            y_offset=0.0,
            center=center
        )

        # calculate center
        center = center.unsqueeze(0).repeat(fast_px, 1)  # (fast_px, 3)

        # scan line by line
        for i in range(slow_px):

            # Determine start and end indices for the current line
            start_idx = i * fast_px
            end_idx = start_idx + fast_px

            # Extract the current line's transformations
            b_tx = input_data.brown_translation_x[start_idx:end_idx] # (fast_px,)
            b_ty = input_data.brown_translation_y[start_idx:end_idx]
            b_rz = input_data.brown_rotation_z[start_idx:end_idx]
            b_rx = input_data.brown_rotation_x[start_idx:end_idx]
            b_ry = input_data.brown_rotation_y[start_idx:end_idx]

            xyz_batch = xyz_base.unsqueeze(0).repeat(fast_px, 1, 1)  # (fast_px, num_atoms, 3)

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
                max_height=self.pdb_utils.get_max_height(xyz_batch)
            )

            size_config = scale_to_size_config(
                scale_x=input_data.fixed_params.scale_x,
                scale_y=input_data.fixed_params.scale_y,
                width_px=input_data.fixed_params.width_px,
                height_px=input_data.fixed_params.height_px,
                scan_axis=self.scan_axis
            )

            line_heights = torch.zeros(fast_px, device=self.device, dtype=self.dtype)
            # Process each pixel in the current line
            for j in range(fast_px):
                surface = surfing(xyz_batch[j, :, :], atom_radii, size_config)
                full_column = idilation(surface, tip)

                #print(f"full_column shape: {full_column.shape}")

                if self.scan_axis == 1: # X-scan
                    line_heights[j] = full_column[slow_px-1-i, j]
                else: # Y-scan
                    line_heights[j] = full_column[fast_px-1-j, i]

            # Insert the line into the image
            if self.scan_axis == 1: # X-scan
                img[slow_px-1-i, :] = line_heights
            else:
                img[:, i] = line_heights.flip(0) # Y-scan needs flipping(bottom to top)

        return img

