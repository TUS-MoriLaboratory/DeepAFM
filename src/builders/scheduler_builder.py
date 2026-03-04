import torch.optim.lr_scheduler as lr_sched

def build_scheduler(cfg, optimizer, silent=False):
    if cfg.train.scheduler is None:
        return None

    name = cfg.train.scheduler.lower()
    epochs = cfg.train.epochs

    if name == "step":
        main_scheduler = lr_sched.StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == "cosine":
        main_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    # Warmup scheduler if DDP is used
    if hasattr(cfg.data, "world_size") and cfg.data.world_size > 1:
        warmup_epochs = cfg.train.warmup_epochs if hasattr(cfg.train, "warmup_epochs") else 5
       
        warmup_scheduler = lr_sched.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    
        scheduler = lr_sched.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
        if not silent:
            print(f"[Scheduler Builder] Using warmup scheduler for distributed training: {warmup_epochs}")
        return scheduler
    
    else:
        return main_scheduler