# src/utils/setup_utils.py
import os

def make_experiment_dirs(exp_cfg):
    """
    Create necessary directories for the experiment based on the provided configuration.

    Args:
        exp_cfg: An ExperimentConfig object containing directory paths.
    """
    dirs_to_create = [
        exp_cfg.data_dir.md_dir,
        exp_cfg.data_dir.exp_data_dir,
        exp_cfg.data_dir.afm_data_dir,
        exp_cfg.data_dir.train_data_dir,
        exp_cfg.data_dir.val_data_dir,
        exp_cfg.data_dir.test_data_dir,
        exp_cfg.data_dir.runs_dir,
        os.path.join(exp_cfg.data_dir.runs_dir, "checkpoints"),
        os.path.join(exp_cfg.data_dir.runs_dir, "logs"),
        os.path.join(exp_cfg.data_dir.runs_dir, "evaluation_results"),
    ]

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)