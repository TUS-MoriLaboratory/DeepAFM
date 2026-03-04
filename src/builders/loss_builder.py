import torch.nn as nn

def build_loss(cfg):
    """
    build loss functions and weights based on the task mode.
    
    Returns:
        loss_fns (Dict[str, nn.Module]): Loss functions dictionary
        loss_weights (Dict[str, float]): Loss weights dictionary
    """

    task_mode = cfg.train.task_mode
    loss_fns = {}
    loss_weights = {}

    # common reconstruction loss type
    recon_loss_type = getattr(cfg.train, "recons_loss_type", "mse").lower()
    
    def get_recon_loss(loss_type):
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported reconstruction loss: {loss_type}")

    # -------------------------------------------------
    # 1. Classification
    # -------------------------------------------------
    if task_mode == "classification":
        loss_fns["state"] = nn.CrossEntropyLoss()
        loss_weights["state"] = 1.0

    # -------------------------------------------------
    # 2. Denoise / Reconstruction
    # -------------------------------------------------
    elif task_mode == "denoise" or task_mode == "reconstruction":
        # It's convenient to choose between MSE, L1, or Huber for reconstruction losses
        loss_fns["ideal"] = get_recon_loss(recon_loss_type)
        loss_weights["ideal"] = 1.0

    # -------------------------------------------------
    # 3. Multi-task (Classification + Denoise/Reconstruction)
    # -------------------------------------------------
    elif task_mode == "multitask":
        # Loss A: Classification
        loss_fns["state"] = nn.CrossEntropyLoss()
        loss_weights["state"] = getattr(cfg.train, "weight_cls", 0.5) # Configurable weight
        
        # Loss B: Reconstruction
        loss_fns["ideal"] = get_recon_loss(recon_loss_type)
        loss_weights["ideal"] = getattr(cfg.train, "weight_recon", 0.5)

    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")

    return loss_fns, loss_weights