# src/utils/distributed.py
import os
import torch
import torch.distributed as dist

def setup_distributed():
    """
    Initialize the DDP environment.
    If started with torchrun, initialize using rank information from environment variables.
    
    Returns:
        dict: {
            "is_distributed": bool,
            "rank": int,
            "world_size": int,
            "local_rank": int,
            "device": torch.device
        }
    """
    # Check if started with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist_backend = "nccl" # Recommended for NVIDIA GPUs
        
        # Initialize process group
        dist.init_process_group(backend=dist_backend)
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Set the GPU device for this process (important)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        is_distributed = True
        
    else:
        # singele process / non-distributed setup
        rank = 0
        world_size = 1
        local_rank = 0
        
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            
        is_distributed = False

    # Return distributed info dictionary if initialized
    if dist.is_initialized():
        return {
            "is_distributed": is_distributed,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": device
        }

def cleanup_distributed():
    """Cleanup DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Helper to check if current process is rank 0 (main process)."""
    return not dist.is_initialized() or dist.get_rank() == 0