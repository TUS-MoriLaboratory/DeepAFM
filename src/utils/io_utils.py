import h5py
import json
import numpy as np
import torch
from typing import List, Optional, Tuple


def load_afm_hdf5(
    path: str,
    include: Optional[List[str]] = None,
):
    """
    Flexible HDF5 loader for AFM multiprocess output.

    Parameters
    ----------
    path : str
        Path to HDF5 file.
    include : list[str]
        Any subset of ["distorted", "ideal", "config"].
        Controls what is returned. Missing datasets → None.

    Returns
    -------
    distorted : np.ndarray or None
    ideal     : np.ndarray or None
    configs   : list[dict] or None
    """

    include = include or ["distorted", "ideal", "config", "image_id"]

    want_distorted = "distorted" in include
    want_ideal     = "ideal" in include
    want_config    = "config" in include
    want_image_id  = "image_id" in include
    distorted = None
    ideal = None
    configs = None
    ids = None

    with h5py.File(path, "r") as f:

        # ---- distorted ----
        if want_distorted:
            if "distorted" in f:
                distorted = f["distorted"][:]
            else:
                distorted = None

        # ---- ideal ----
        if want_ideal:
            if "ideal" in f:
                ideal = f["ideal"][:]
            else:
                ideal = None

        # ---- configs ----
        if want_config:
            if "config" in f:
                raw = f["config"][:]
                configs = [json.loads(x) for x in raw]
            else:
                configs = None

        #---- idx ----
        if want_image_id:
            if "image_id" in f:
                ids = f["image_id"][:]
            else:
                ids = None

    return distorted, ideal, configs, ids


def load_file_to_tensor(
        file_path, 
        dtype=torch.float32, 
        skiprows=0, 
        delimiter=None
        ):
    """
    Load data from a file and return it as a PyTorch Tensor of the specified type.

    Args:
        file_path (str): File path
        dtype (torch.dtype): Output Tensor data type
        skiprows (int): Number of rows to skip from the top (set to 1 if there is a header)
        delimiter (str): Delimiter character (if None, whitespace/tab is automatically detected)
        
    Returns:
        torch.Tensor: Data tensor
    """
    
    if delimiter is None:
        if file_path.endswith('.csv'):
            delimiter = ','
        else:
            delimiter = None 
            
    try:
        np_data = np.loadtxt(
            file_path, 
            delimiter=delimiter, 
            skiprows=skiprows, 
            dtype=np.float32  
        )
        # Convert to PyTorch Tensor
        tensor = torch.from_numpy(np_data)
        
        # Ensure correct dtype
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
            
        return tensor

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return torch.empty(0)
