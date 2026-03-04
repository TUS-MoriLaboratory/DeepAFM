import os
from dataclasses import dataclass, field
import pathlib

@dataclass
class DataDirConfig:
    root_dir: str
    exp_id: str
    run_name: str
    protein_of_project: str
    scan_direction: str = "y"  # "x" or "y"

    # will be built dynamically
    md_dir: str = field(init=False)
    pdb_path: str = field(init=False)
    dcd_path: str = field(init=False)
    afm_data_dir: str = field(init=False)
    train_data_dir: str = field(init=False)
    val_data_dir: str = field(init=False)
    test_data_dir: str = field(init=False)
    exp_data_dir: str = field(init=False)
    exp_raw_dir: str = field(init=False)
    exp_processed_dir: str = field(init=False)
    runs_dir: str = field(init=False)

    def __post_init__(self):
        root = pathlib.Path(self.root_dir)

        # ----- Base directories -----
        root = self.root_dir

        self.md_dir         = f"{root}/data/md_raw/{self.protein_of_project}"
        self.pdb_path       = f"{self.md_dir}/initial.pdb"
        self.dcd_path       = f"{self.md_dir}/rst_total.dcd"
        self.exp_data_dir   = f"{root}/data/Experiment/{self.protein_of_project}"
        self.exp_processed_dir = f"{self.exp_data_dir}/processed_imgs_{self.scan_direction}"
        self.exp_raw_dir    = f"{self.exp_data_dir}/raw_imgs"
        base                = f"{root}/data/{self.exp_id}"
        self.afm_data_dir   = f"{base}/afm_generated"
        self.train_data_dir = f"{base}/train"
        self.val_data_dir   = f"{base}/val"
        self.test_data_dir  = f"{base}/test"
        self.runs_dir       = f"{root}/runs/{self.run_name}"

    def set_morphing_dirs(self, morph_start: str, morph_end: str):
        morphing_name = f"{morph_start}_to_{morph_end}"
        self.morphing_dir = os.path.join(
            self.md_dir, 
            "Morphing",
            morphing_name
            )
        
        return self.morphing_dir
