# src/dataload/webdataset_loader.py

from dataload.preprocess import AFMPreprocess

import webdataset as wds
import torch
import io
import glob
import os


#=========================================================
# decoders (byte -> tensor)
#=========================================================

def torch_decoder(data):
    return torch.load(io.BytesIO(data), map_location="cpu")

def id_decoder(data):
    tensor = torch.load(io.BytesIO(data), map_location="cpu")
    return int(tensor.item())

def identity(x):
    return x

#=========================================================
# Filters
#=========================================================
class RequiredKeysFilter:
    def __init__(self, required_keys):
        self.required_keys = required_keys

    def __call__(self, sample):
        return all(k in sample for k in self.required_keys)

class KeyRenamer:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, sample):
        new_sample = {}
        for suffix, logical_name in self.mapping.items():
            if suffix in sample:
                new_sample[logical_name] = sample[suffix]
  
        if "state" in sample:
            new_sample["state"] = sample["state"]
            
        return new_sample

# =========================================================
# Collation Function (for batching)
# =========================================================
def dict_collation_fn(samples):
    """
    take dict list ([{'a': 1}, {'a': 2}]) and convert to batch ({'a': [1, 2]}) に変換する。
    if value is Tensor, stack them into a batch Tensor.
    """
    batch = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        
        elif isinstance(values[0], (int, float)):
            batch[key] = torch.tensor(values)
            
        else:
            batch[key] = values
            
    return batch

#=========================================================
# Configuration Mapping
#=========================================================
# logical_name -> (file_suffix, decoder_func)
MODE_MAP = {
    "distorted": ("dist.pt",   torch_decoder),
    "ideal":     ("ideal.pt",  torch_decoder),
    "config":    ("config.pt", torch_decoder),
    "id":        ("id.pt",     id_decoder),
}

SUFFIX_TO_MODE = {v[0]: k for k, v in MODE_MAP.items()}

#=========================================================
# WebDataset loader
#=========================================================

def create_afm_dataloader(
    url_pattern: str,
    exp_cfg,
    preprocessor: AFMPreprocess = None,
    batch_size: int = None,
    shuffle: bool = None,
    num_workers: int = None,
    pin_memory: bool = None,
    persistent_workers: bool = None, 
    prefetch_factor: int = None,     
    is_distributed: bool = None,
    rank: int = None,
    world_size: int = None,
    max_shards: int = None, # limit the number of shards to load
):

    # ---- use defaults from exp_cfg if not provided ----
    batch_size = batch_size or exp_cfg.train.batch_size
    shuffle = shuffle if shuffle is not None else exp_cfg.data.shuffle
    num_workers = num_workers if num_workers is not None else exp_cfg.data.num_workers
    pin_memory = pin_memory if pin_memory is not None else exp_cfg.data.pin_memory

    if persistent_workers is None:
        persistent_workers = getattr(exp_cfg.data, 'persistent_workers', True)
    
    if prefetch_factor is None:
        prefetch_factor = getattr(exp_cfg.data, 'prefetch_size', 2)

    is_distributed = is_distributed if is_distributed is not None else exp_cfg.data.is_distributed
    rank = rank if rank is not None else exp_cfg.data.rank
    world_size = world_size if world_size is not None else exp_cfg.data.world_size

    load_modes = exp_cfg.system.data_load_mode

    required_keys = []
    decoder_dict = {}
    output_keys = []

    for mode in load_modes:
        if mode not in MODE_MAP:
            raise ValueError(f"Unknown data_load_mode: {mode}. Available: {list(MODE_MAP.keys())}")
        
        suffix, decoder = MODE_MAP[mode]
        
        required_keys.append(suffix)      # filtering ["dist.pt", "ideal.pt"])
        decoder_dict[suffix] = decoder    # decode
        output_keys.append(suffix)        # for output sort

    # ---- find all shard files ----
    urls = sorted(glob.glob(url_pattern))
    if not urls:
        raise FileNotFoundError(f"No files found matching: {url_pattern}")

    if max_shards is not None:
        if max_shards > len(urls):
            print(f"[Warning] Requested {max_shards} shards, but only {len(urls)} found. Using all.")
        else:
            original_len = len(urls)
            urls = urls[:max_shards]
            print(f"[DataLoader] Using {len(urls)}/{original_len} shards (Limited by max_shards).")
    
    # change node splitter based on distributed setting
    nodesplitter = wds.split_by_node if is_distributed else identity

    active_mapping = {}
    for mode in load_modes:
        suffix = MODE_MAP[mode][0]
        active_mapping[suffix] = mode

    # preprocess instance
    if preprocessor is None:
        preprocessor = AFMPreprocess(exp_cfg=exp_cfg)

    # --- define pipeline ---
    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        nodesplitter,        # split by node or worker
        wds.split_by_worker, # split by worker within node
        wds.tarfile_to_samples(handler=wds.warn_and_continue), # tarfile_to_samples: .tar -> samples
        wds.map_dict(**decoder_dict),                          # decode files
        wds.select(RequiredKeysFilter(required_keys)),         # filter samples that have all required files
        wds.shuffle(1000) if shuffle else identity,            # shuffle samples within worker
        wds.map(preprocessor),
        wds.map(KeyRenamer(active_mapping)),                   # rename keys to logical names
        wds.batched(batch_size, collation_fn=dict_collation_fn)
    )

    # ---Setting WebLoader (DataLoader Wrapper) ---
    loader = wds.WebLoader(
        dataset,
        batch_size=None, 
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,             
        persistent_workers=True,       
        pin_memory=pin_memory          
    )

    return loader