import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
import os
from tqdm import tqdm
from glob import glob
import re

from configs.experiment_config import ExperimentConfig
from dataload.batch_adapter import prepare_batch_for_model
from .logger import CSVLogger

class ModelTrainer:
    def __init__(
        self,
        exp_cfg: ExperimentConfig,
        model: nn.Module,
        loss_fn: Dict[str, nn.Module],
        loss_weights: Dict[str, float],
        metrics: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        # initialize trainer
        self.exp_cfg = exp_cfg

        # distribution settings
        self.rank = exp_cfg.data.rank
        self.is_distributed = exp_cfg.data.is_distributed

        if self.is_distributed:
            if hasattr(exp_cfg.data, "local_rank"):
                self.local_rank = exp_cfg.data.local_rank
            else:
                # simplified assumption (setting exp_cfg.data.local_rank is recommended)
                self.local_rank = self.rank % torch.cuda.device_count()
            
            # In DDP, fix device to local_rank
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device(exp_cfg.system.device)

        # model
        self.model = model.to(self.device)
        
        # loss
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        # metrics
        self.metrics = metrics

        # optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.use_amp = exp_cfg.train.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # setting logging and saving directory
        self.run_dir = os.path.join('runs', exp_cfg.system.run_name)
        if exp_cfg.system.checkpoint_dir is not None:
            self.save_checkpoint_dir = exp_cfg.system.checkpoint_dir
        else:
            self.save_checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        
        if self.rank == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.save_checkpoint_dir, exist_ok=True)
            self.logger = CSVLogger(self.run_dir)
        else:
            self.logger = None
    
        # early stopping settings
        self.patience = exp_cfg.train.patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # defeine task mode
        self.task_mode = exp_cfg.train.task_mode

        # distributed data parallel setup
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        # ray tune availability check
        # ray tune is not available always, so we check its availability here
        try:
            from ray import train
            self._ray_available = False
        except ImportError:
            self._ray_available = False

    def load_latest_checkpoint(self):
        
        pattern = os.path.join(self.save_checkpoint_dir, "model_epoch*.pt")
        ckpts = glob(pattern)

        if len(ckpts) == 0:
            if self.rank == 0:
                print("[Trainer] No checkpoints found, start from scratch.")
            return 1  # 1-based epoch start

        # extract epoch numbers from filenames
        def extract_epoch(path):
            m = re.search(r"model_epoch(\d+)\.pt", path)
            return int(m.group(1)) if m else -1

        # sort by epoch number (not alphabetically!)
        ckpts = sorted(ckpts, key=extract_epoch)
        latest = ckpts[-1]

        if self.rank == 0:
            print(f"[Trainer] Loading latest checkpoint: {latest}")

        state = torch.load(latest, map_location=self.device)

        # load model
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(state["model_state"])

        # load optimizer if present
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])

        # checkpoint uses 1-based epoch
        epoch_1based = state.get("epoch", 1)

        # resume from next epoch (1-based)
        start_epoch_1based = epoch_1based + 1
        if self.rank == 0:
            print(f"[Trainer] Resumed from epoch {start_epoch_1based}")

        return start_epoch_1based

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.exp_cfg.system.device)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        return state.get("epoch", 0) # 1-based epoch

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        # initialize losses and metrics
        epoch_loss = 0.0
        epoch_loss_components = {k: 0.0 for k in self.loss_fn.keys()}
        epoch_metrics = {k: 0.0 for k in self.metrics.keys()}

        total_samples = 0

        if self.rank == 0:
            iterator = tqdm(
                self.train_loader, 
                desc=f"[Train] Epoch {epoch+1}", 
                leave=False, 
                disable=self.exp_cfg.system.tqdm_silent
            )
        else:
            iterator = self.train_loader

        for batch in iterator:
            # batch adaptation for model input
            imgs, labels = prepare_batch_for_model(batch, self.task_mode)
            imgs = imgs.to(self.device, non_blocking=True)

            if isinstance(labels, dict):
                labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
            else:
                labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(imgs)
                
                # loss calculation
                batch_loss = 0.0
                for key, loss_fn in self.loss_fn.items():
                    # handle both single and multi-output cases
                    if isinstance(outputs, dict) and isinstance(labels, dict):
                        pred = outputs[key]
                        target = labels[key]
                    else:
                        pred = outputs
                        target = labels

                    loss_val = loss_fn(pred, target)
                    weight = self.loss_weights.get(key, 1.0)

                    batch_loss += weight * loss_val

                    epoch_loss_components[key] += loss_val.item() * imgs.size(0)

            # Backward
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Aggregate metrics
            batch_size = imgs.size(0)
            total_samples += batch_size
            epoch_loss += batch_loss.item() * batch_size

            # -- Metrics Calculation --
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    val = metric_fn(outputs, labels)
                    epoch_metrics[name] += val.item()

            if self.rank == 0 and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": epoch_loss / total_samples if total_samples > 0 else 0})

        if self.rank == 0 and isinstance(iterator, tqdm):
            iterator.close()
        if self.scheduler is not None:
            self.scheduler.step()

        loss_comp_keys = sorted(list(epoch_loss_components.keys()))
        metric_keys = sorted(list(epoch_metrics.keys()))
        
        stats_list = [epoch_loss, total_samples]
        stats_list += [epoch_loss_components[k] for k in loss_comp_keys]
        stats_list += [epoch_metrics[k] for k in metric_keys]
        
        stats_tensor = torch.tensor(stats_list, device=self.device)

        if self.is_distributed:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        global_total = stats_tensor[1].item()
        results = {
            "train_loss": stats_tensor[0].item() / global_total
        }

        idx = 2
        for k in loss_comp_keys:
            results[f"train_loss_{k}"] = stats_tensor[idx].item() / global_total
            idx += 1
            
        for k in metric_keys:
            results[f"train_{k}"] = stats_tensor[idx].item() / global_total
            idx += 1

        return results
    
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()

        # initialize losses and metrics
        epoch_loss = 0.0
        epoch_loss_components = {k: 0.0 for k in self.loss_fn.keys()}
        epoch_metrics = {k: 0.0 for k in self.metrics.keys()}

        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)
                imgs = imgs.to(self.device, non_blocking=True)

                if isinstance(labels, dict):
                    labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
                else:
                    labels = labels.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(imgs)
                    # loss calculation
                    batch_loss = 0.0
                    for key, loss_fn in self.loss_fn.items():
                        # handle both single and multi-output cases
                        if isinstance(outputs, dict) and isinstance(labels, dict):
                            pred = outputs[key]
                            target = labels[key]
                        else:
                            pred = outputs
                            target = labels

                        loss_val = loss_fn(pred, target)
                        weight = self.loss_weights.get(key, 1.0)

                        batch_loss += weight * loss_val
                        epoch_loss_components[key] += loss_val.item() * imgs.size(0)

                # Aggregate 
                batch_size = imgs.size(0)
                total_samples += batch_size
                epoch_loss += batch_loss.item() * batch_size

                for name, metric_fn in self.metrics.items():
                    val = metric_fn(outputs, labels)
                    epoch_metrics[name] += val.item()

        # --- DDP All-Reduce ---
        loss_comp_keys = sorted(list(epoch_loss_components.keys()))
        metric_keys = sorted(list(epoch_metrics.keys()))
        
        stats_list = [epoch_loss, total_samples]
        stats_list += [epoch_loss_components[k] for k in loss_comp_keys]
        stats_list += [epoch_metrics[k] for k in metric_keys]
        
        stats_tensor = torch.tensor(stats_list, device=self.device)

        if self.is_distributed:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        global_total = stats_tensor[1].item()
        if global_total == 0: return {} 

        results = {
            "val_loss": stats_tensor[0].item() / global_total
        }

        idx = 2
        for k in loss_comp_keys:
            results[f"val_loss_{k}"] = stats_tensor[idx].item() / global_total
            idx += 1
            
        for k in metric_keys:
            results[f"val_{k}"] = stats_tensor[idx].item() / global_total
            idx += 1

        return results

    def fit(self):
        start_epoch = self.load_latest_checkpoint()  # 1-based epoch

        if self.rank == 0:
            print(f"[Trainer] Starting training for {self.exp_cfg.train.epochs} epochs")
            print(f"[Trainer] Starting training from epoch {start_epoch}")

        # 0-based epoch loop
        for epoch in range(start_epoch - 1, self.exp_cfg.train.epochs):

            if self.is_distributed: dist.barrier()

            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()

            # display_epoch: 1-based
            display_epoch = epoch + 1
            current_lr = self.optimizer.param_groups[0]['lr']

            # Ray Tune reporting
            if self._ray_available:
                report_dict = val_metrics.copy()
                report_dict.update({
                    "loss": val_metrics["val_loss"], 
                    "accuracy": val_metrics["val_acc"], 
                    "epoch": display_epoch,
                    "lr": current_lr
                })
                try:
                    train.report(report_dict)
                except RuntimeError:
                    pass 

            # early stopping and best model saving
            val_loss = val_metrics.get("val_loss", float("inf"))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if self.rank == 0:
                    print(f" >> New Best Model found! Val Loss: {val_loss:.6f}")
                    self.save_checkpoint(epoch, is_best=True)
            
            else:
                self.patience_counter += 1
                if self.patience is not None and self.patience_counter >= self.patience:
                    if self.rank == 0:
                        print(f"Early stopping triggered at epoch {display_epoch}")            
                    break

            # Logging
            if self.rank == 0 and self.logger is not None:

                log_dict = {
                    "epoch": display_epoch,
                    "lr": current_lr
                }
                log_dict.update(train_metrics) # train_loss, train_acc...
                log_dict.update(val_metrics)   # val_loss, val_acc...

                self.logger.log(log_dict)

                if display_epoch % self.exp_cfg.system.log_interval == 0:

                    log_items = [f"Epoch {display_epoch}", f"LR: {current_lr:.2e}"]

                    for key, value in sorted(log_dict.items()):
                        if key in ["epoch", "lr"]:
                            continue
                            
                        if isinstance(value, float):
                            val_str = f"{value:.4f}"
                        else:
                            val_str = str(value)
                            
                        log_items.append(f"{key}: {val_str}")
                    
                    print(" | ".join(log_items))

                if display_epoch % self.exp_cfg.system.save_interval == 0:
                    self.save_checkpoint(epoch)

        # plot logs
        if self.logger is not None:
            self.logger.plot()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if self.rank != 0:
            return

        display_epoch = epoch + 1  # 1-based
        if is_best:
            ckpt_path = os.path.join(self.save_checkpoint_dir, "best_model.pt")
        else:
            ckpt_path = os.path.join(self.save_checkpoint_dir, f"model_epoch{display_epoch}.pt")
        
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        torch.save(
            {
                "epoch": display_epoch, # 0-based -> 1-based
                "model_state": model_to_save.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.exp_cfg.to_dict(),
            },
            ckpt_path,
        )
        if self.rank == 0:
            if is_best:
                print(f"[Checkpoint] Saved best model: {ckpt_path}")
            else:
                print(f"[Checkpoint] Saved: {ckpt_path}")