import os
import math

import warnings
import MDAnalysis as mda

import numpy as np
import torch
from typing import Union, List, Optional

from configs.experiment_config import ExperimentConfig
from afm_image_generation.configs.afm_image_parameter import AFMImageFixedParams
from afm_image_generation.constants.atomic_radii import Atom2Radius, Residue2Radius

from afm_image_generation.utils.quaternion import rotate_around_center, rotate_around_center_batch
from afm_image_generation.utils.translation import translate, translate_batch
from afm_image_generation.utils.center import calculate_center, calculate_center_batch

# ignore warnings from MDAnalysis
warnings.filterwarnings("ignore", message="Element information is missing")
warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")

# Device
class PDBUtils:
    def __init__(self, cfg: ExperimentConfig):
        self.device = cfg.system.device
        FLOAT = cfg.system.afm_dtype

        if FLOAT == "float16":
            self.dtype = torch.float16
        elif FLOAT == "float32":
            self.dtype = torch.float32
        elif FLOAT == "float64":
            self.dtype = torch.float64
        
        self.pdb_path = cfg.data_dir.pdb_path
        self.dcd_path = cfg.data_dir.dcd_path

        self.md_load_mode = cfg.system.md_load_mode
        self.pdb_only = cfg.system.pdb_only

    def load_mdtrj(
            self, 
            pdb_path: Optional[Union[str, List[str]]] = None,
            dcd_path: Optional[str] = None,
            md_load_mode: Optional[str] = None,
            pdb_only: Optional[bool] = None,
            ):
        """
        Load trajectory using MDAnalysis.

        Args:
            pdb_path: path to PDB file if None, use default
            dcd_path: path to DCD file if None, use default
            md_load_mode: "all_atom" or "coarse" if None, use default
            frame: if specified, only load this frame if not None used frame of all trajectory

        Converts Å → nm and returns:
            xyz: (n_frames, n_atoms, 3) torch tensor
            radii: (n_atoms,) torch tensor
        """

        # Determine numpy dtype
        if self.dtype == torch.float16:
            np_dtype = np.float16

        elif self.dtype == torch.float32:
            np_dtype = np.float32

        elif self.dtype == torch.float64:
            np_dtype = np.float64

        # Use provided pdb_only or default
        pdb_only = self.pdb_only if pdb_only is None else pdb_only

        # Use provided paths or default
        target_path = pdb_path or self.pdb_path
        pdb_list = [target_path] if isinstance(target_path, str) else target_path
        
        dcd_path = dcd_path or self.dcd_path
        md_load_mode = md_load_mode or self.md_load_mode

        all_xyz_list = []
        all_radii_list = []
        for p in pdb_list:
            # --------------------------
            # Path validation
            # --------------------------
            if not os.path.exists(p):
                raise FileNotFoundError(f"PDB file not found: {p}")

            if not pdb_only and not os.path.exists(dcd_path):
                print(f"[Warning] DCD not found at {dcd_path}, falling back to PDB-only mode.")
                pdb_only = True

            # === Load trajectory ===
            if pdb_only:
                u = mda.Universe(p)
            else:
                u = mda.Universe(p, dcd_path)

            n_frames = len(u.trajectory)
            n_atoms = u.atoms.n_atoms

            # === Pre-allocate numpy array (float32) ===
            current_xyz = np.zeros((n_frames, n_atoms, 3), dtype=np_dtype)

            # === Read coordinates from each frame ===
            for i, ts in enumerate(u.trajectory):
                current_xyz[i] = ts.positions.astype(np_dtype) / 10.0   # Å → nm 

            # === Atom radius lookup ===
            current_radii = []
            # all-atom mode
            if md_load_mode == "all_atom":
                for atom in u.atoms:
                    name = atom.name.strip().upper()   # " CA " → "CA"
                    r = Atom2Radius.get(name, Atom2Radius.get(name[0], 0.0))
                    current_radii.append(r)
                    if r == 0.0:
                        print(f"[Warning] Radius not found for atom {name} in residue {atom.resname}, using 0.0")

            # coarse-grained mode
            elif md_load_mode == "coarse":
                for atom in u.atoms:
                    r = Residue2Radius.get(atom.resname, 0.0)
                    current_radii.append(r)  
                    if r == 0.0:
                        print(f"[Warning] Radius not found for residue {atom.resname}")

            all_xyz_list.append(current_xyz)
            all_radii_list.append(np.array(current_radii, dtype=np_dtype))

        max_atoms = max(xyz.shape[1] for xyz in all_xyz_list)
        total_frames = sum(xyz.shape[0] for xyz in all_xyz_list)
        
        padded_xyz = np.zeros((total_frames, max_atoms, 3), dtype=np_dtype)
        padded_radii = np.zeros((total_frames, max_atoms), dtype=np_dtype) # radii of padding atoms is set to 0.0
        
        current_frame_idx = 0
        for xyz, radii in zip(all_xyz_list, all_radii_list):
            n_f, n_a, _ = xyz.shape
            next_frame_idx = current_frame_idx + n_f
            
            padded_xyz[current_frame_idx:next_frame_idx, :n_a, :] = xyz
            padded_radii[current_frame_idx:next_frame_idx, :n_a] = radii
            
            if n_a < max_atoms:
                #  (shape: n_f,)
                atom0_coords = xyz[:, 0, :][:, np.newaxis, :]
                padded_xyz[current_frame_idx:next_frame_idx, n_a:, :] = atom0_coords
                
            current_frame_idx = next_frame_idx

        xyz_t = torch.tensor(padded_xyz, dtype=self.dtype, device=self.device)
        radii_t = torch.tensor(padded_radii, dtype=self.dtype, device=self.device)

        return xyz_t, radii_t

    def save_xyz_to_pdb(
        self,
        xyz,
        out_path: str,
        coords_in_angstrom: bool = False
    ):
        """Save coordinates to a PDB file using MDAnalysis.

        Args:
            xyz: (n_atoms, 3) or (n_frames, n_atoms, 3)
            out_path: save path
            coords_in_angstrom: if True → Å → nm に変換
        """

        # --- translate to numpy ---
        if isinstance(xyz, torch.Tensor):
            coords = xyz.detach().cpu().numpy()
        else:
            coords = np.asarray(xyz)

        # --- Convert Å to nm (MDAnalysis uses Å, so unify to nm→Å) ---
        if coords_in_angstrom:
            coords_nm = coords / 10.0
        else:
            coords_nm = coords

        # MDAnalysis uses Angstrom, convert nm → Å
        coords_angstrom = coords_nm * 10.0

        # --- shape (n_frames, n_atoms, 3) ---
        if coords_angstrom.ndim == 2:
            coords_angstrom = coords_angstrom[np.newaxis, ...]

        n_frames, n_atoms, _ = coords_angstrom.shape

        # --- 4) Load topology (pdb_path) ---
        u = mda.Universe(self.pdb_path)

        if u.atoms.n_atoms != n_atoms:
            raise ValueError(
                f"Atom count mismatch: topology={u.atoms.n_atoms}, coords={n_atoms}"
            )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with mda.coordinates.PDB.PDBWriter(out_path, multiframe=True) as w:
            for f in range(n_frames):
                u.atoms.positions = coords_angstrom[f]
                w.write(u)

        print(f"[PDB saved] {out_path} ({n_frames} frames)")

    def transform_initial_pose(
            self, 
            xyz, 
            x_deg,      # Rotation angle around the x-axis in degrees.
            y_deg,      # Rotation angle around the y-axis in degrees.
            z_deg,      # Rotation angle around the z-axis in degrees.
            x_offset,   # x_offset: Translation offset in x direction (nm).
            y_offset,   # y_offset: Translation offset in y direction (nm). 
            center=None # center of rotation (if None, calculate from xyz)
            ):
        """
        Apply initial rotation, setting stage position and translation to the molecular structure.
        xyz: (num_atoms, 3)
        Parameters:
        - xyz: 2D tensor of shape (num_atoms, 3) representing the XYZ coordinates of the atoms.
        - x_deg: Rotation angle around the x-axis in degrees.
        - y_deg: Rotation angle around the y-axis in degrees.
        - z_deg: Rotation angle around the z-axis in degrees.
        - x_offset: Translation offset in x direction (nm).
        - y_offset: Translation offset in y direction (nm).

        Returns:
            - xyz: 2D tensor of shape (num_atoms, 3) representing the transformed XYZ coordinates.
        """

        assert xyz.ndim == 2 and xyz.shape[1] == 3, "Input xyz must be a 2D tensor with shape (num_atoms, 3)"

        # Rotate around center (calculate center inside the function)
        xyz = rotate_around_center(
            xyz=xyz, 
            z_deg=z_deg, 
            x_deg=x_deg, 
            y_deg=y_deg, 
            center=calculate_center(xyz) if center is None else center
            )
        xyz[:, 2] -= xyz[:, 2].min()  # set stage position (z=0)
        xyz = translate(xyz, x_offset, y_offset)

        return xyz

    def get_max_height(self, xyz: torch.Tensor) -> float:
        """
        Get the maximum z-coordinate (height) from the given xyz coordinates.

        Args:
            xyz: (n_atoms, 3) or (n_frames, n_atoms, 3) tensor 
        Returns:
            max_height: maximum z value (float)
        """
        if xyz.ndim == 2:
            return float(torch.max(xyz[:, 2]).item())
        elif xyz.ndim == 3:
            return float(torch.max(xyz[:, :, 2]).item())
        else:
            raise ValueError("xyz must be a 2D or 3D tensor.")
        


    
