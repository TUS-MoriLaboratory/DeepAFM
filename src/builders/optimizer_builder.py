# builders/optimizer_builder.py
import torch

def build_optimizer(cfg, model, silent=False):
    opt_name = cfg.train.optimizer.lower()
    params = model.parameters()

    # learning rate
    if hasattr(cfg.data, "world_size") and cfg.data.world_size > 1:
        lr = cfg.train.learning_rate * cfg.data.world_size
        if not silent:
            print(f"[Optimizer Builder] Adjusted learning rate for distributed training: {lr}")
    else:
        lr = cfg.train.learning_rate

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    