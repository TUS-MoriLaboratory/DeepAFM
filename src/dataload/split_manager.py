# src/dataload/split_and_move_tfrecords.py

import shutil
from pathlib import Path
from configs.experiment_config import ExperimentConfig

def split_and_move_data(
        exp_cfg: ExperimentConfig,
        extension: str = "tfrecord"
    ):
    """
    move tfrecord files into train/val/test directories using given ratio and experiment configuration.

    Args:
        exp_cfg: ExperimentConfig
            Experiment configuration containing data directory info.
    
    """
    # set
    # get split ratios
    train_ratio = exp_cfg.data.train_split
    val_ratio   = exp_cfg.data.val_split
    test_ratio  = exp_cfg.data.test_split
    
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Train/Val/Test split ratios must sum to 1.0"

    # get dirs 
    src_dir = Path(exp_cfg.afm.job.save_dir)

    out_train = Path(exp_cfg.data_dir.train_data_dir)
    out_val   = Path(exp_cfg.data_dir.val_data_dir)
    out_test  = Path(exp_cfg.data_dir.test_data_dir)

    # create dirs
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    # collect shards
    shards = sorted(src_dir.glob(f"*.{extension}"))
    if not shards:
        raise RuntimeError(f"No {extension} found in {src_dir}")

    n = len(shards)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = shards[:n_train]
    val_files   = shards[n_train:n_train + n_val]
    test_files  = shards[n_train + n_val:]

    def move(files, dst):
        for f in files:
            dst_path = dst / f.name
            shutil.move(str(f), str(dst_path))
    
    # move files
    move(train_files, out_train)
    move(val_files, out_val)
    move(test_files, out_test)

