import MDAnalysis as mda
import freesasa
import numpy as np
import csv
from tqdm import tqdm
from typing import Union, Optional

# Import Atomic Radii Dictionary
from afm_image_generation.constants.atomic_radii import Atom2Radius

class SurfaceDepthAnalyzer:
    """
    MD trajectory analyzer to extract atoms within a specific depth from the surface.
    Independent from PyTorch or specific GPU utils.
    """

    def __init__(
        self, 
        input_data: Union[mda.Universe, str], 
        dcd_path: Optional[str] = None,
        selection: str = "protein and not (name H* or type H)"
    ):
        """
        Initialize with either an existing Universe or file paths.

        Args:
            input_data: mda.Universe object OR path to PDB file (str).
            dcd_path: Path to DCD file (only used if input_data is a path).
            selection: Atom selection string (Default: Protein excluding Hydrogens).
        """
        # --- Flexible Loading Logic ---
        if isinstance(input_data, mda.Universe):
            # Case 1: Use existing Universe
            self.u = input_data
            if dcd_path is not None:
                print("Warning: 'dcd_path' is ignored because a Universe object was passed.")
        elif isinstance(input_data, str):
            # Case 2: Load from files
            if dcd_path:
                print(f"Loading Trajectory: {input_data} + {dcd_path}")
                self.u = mda.Universe(input_data, dcd_path)
            else:
                print(f"Loading PDB only: {input_data}")
                self.u = mda.Universe(input_data)
        else:
            raise TypeError("input_data must be an mda.Universe or a file path string.")

        # --- Setup AtomGroup ---
        self.selection_str = selection
        self.atom_group = self.u.select_atoms(selection)
        
        # --- Prepare Atomic Radii ---
        self._setup_radii()

        print(f"--- Initialization Complete ---")
        print(f"Target Atoms: {len(self.atom_group)}")
        print(f"Frames: {len(self.u.trajectory)}")

        self.current_surface_atoms = None
    
    def _setup_radii(self):
        """
        Setup atomic radii with a priority mechanism.
        
        Priority 1: Radii from Topology (PSF/TPR) if available.
        Priority 2: Predefined dictionary (Atom2Radius) if PDB only.
        """
        # --- Import Dictionary (Fallbacks) ---
        try:
            from afm_image_generation.constants.atomic_radii import Atom2Radius
        except ImportError:
            print("Warning: Could not import Atom2Radius. Using internal fallback.")
            Atom2Radius = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}

        # --- Check Logic ---
        use_topology_radii = False
        try:
            if hasattr(self.atom_group, 'radii'):
                current_radii = self.atom_group.radii
                if np.any(current_radii > 0.0):
                    use_topology_radii = True
        except AttributeError:
            pass

        # --- Branching ---
        if use_topology_radii:
            # Case A: PSF/Topology available
            print("Info: Valid radii found in input topology (PSF/TPR). Using them.")
            self.radii = np.round(self.atom_group.radii, 3)
            
        else:
            # Case B: PDB only (Dictionary Lookup)
            print("Info: No valid topology radii found. Using Atom2Radius dictionary.")
            radii_list = []
            
            for atom in self.atom_group:
                name = atom.name.strip().upper()
                
                # 1. Exact Match (e.g., "CA")
                if name in Atom2Radius:
                    radii_list.append(Atom2Radius[name])
                    continue
                
                # 2. Element Match (e.g., "C")
                elem = name[0]
                if elem in Atom2Radius:
                    radii_list.append(Atom2Radius[elem])
                    continue
                
                # 3. Fallback
                radii_list.append(1.70) # Default to Carbon
                print(f"[Warning] Radius not found for atom '{name}' (elem='{elem}'). Using default 1.70 Å.")
            
            self.radii = np.array(radii_list) * 10.0  # Convert nm to Å

            # Assign back to AtomGroup
            if not hasattr(self.u.atoms, 'radii'):
                self.u.add_TopologyAttr('radii')

            self.atom_group.radii = self.radii

    def compute_surface_atoms(self, sasa_threshold=1.0, probe_radius=1.4):
        """
        Identify surface atoms using FreeSASA.
        
        Parameters:
        - sasa_threshold: Minimum SASA value to consider an atom as 'surface' (Å^2).
        - probe_radius: Radius of the probe (default 1.4Å for water).
        """
        # Get coordinates as flat array
        coords = self.atom_group.positions.flatten()
        
        # FreeSASA parameters
        params = freesasa.Parameters({'probe-radius': probe_radius})
        
        # Execute calculation
        res = freesasa.calcCoord(coords, self.radii, params)
        areas = np.array([res.atomArea(i) for i in range(len(self.atom_group))])

        print(f"Computed SASA for atoms. Sample areas: {areas[:5]} ...")

        # Filter atoms
        is_surface = areas > sasa_threshold
        self.current_surface_atoms = self.atom_group[is_surface]
        
        return self.current_surface_atoms

    def get_atoms_near_surface(self, depth_x):
        """ Retrieve atoms within depth_x (radius-aware offset)."""
        if self.current_surface_atoms is None:
            raise ValueError("Error: Run compute_surface_atoms() first.")
        
        if len(self.current_surface_atoms) == 0:
            return self.atom_group[[]]

        final_selection = self.atom_group[[]]
        surface_radii = self.current_surface_atoms.radii
        unique_radii = np.unique(surface_radii)
        
        for r in unique_radii:
            sub_surface = self.current_surface_atoms[surface_radii == r]
            if len(sub_surface) == 0: continue
            cutoff = depth_x + r
            
            nearby = self.u.select_atoms(
                f"around {cutoff} group sub_grp", 
                sub_grp=sub_surface, 
                updating=True
                )
            
            final_selection = final_selection | nearby
        
        return (final_selection | self.current_surface_atoms) & self.atom_group

    def write_union_surface_dcd(self, depth_x, output_pdb, output_dcd, step=1):
        """Output PDB/DCD containing UNION of all relevant atoms."""
        
        union_indices = set()
        for ts in tqdm(self.u.trajectory[::step], desc="Scanning"):
            self.compute_surface_atoms()
            targets = self.get_atoms_near_surface(depth_x)
            union_indices.update(targets.indices)
            
        final_indices = sorted(list(union_indices))
        final_atom_group = self.u.atoms[final_indices]
        
        n_selected = len(final_atom_group)
        
        final_atom_group.write(output_pdb)
        with mda.Writer(output_dcd, n_selected) as W:
            for ts in tqdm(self.u.trajectory[::step], desc="Writing"):
                W.write(final_atom_group)
        
        print("Done.")