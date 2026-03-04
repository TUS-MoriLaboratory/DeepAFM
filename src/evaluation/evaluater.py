# evaluation/evaluator.py

import os 
from tqdm import tqdm
import csv
import re
from glob import glob
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from configs.experiment_config import ExperimentConfig
from dataload.batch_adapter import prepare_batch_for_model

from utils.scaler import AFMScaler
from utils.metrics_torch import mse_torch, mae_torch
from evaluation.visualizer import visualize_confusion_matrix, visualize_denoising, visualize_image_with_attention, visualize_multitask_comprehensive
from visualization.plot import plot_heightmap, plot_three_heightmap_with_lineprofile

class Evaluator:
    def __init__(
            self, 
            exp_cfg: ExperimentConfig, 
            model: torch.nn.Module,  
            test_data_loader: DataLoader,
            save_dir=None,
            checkpoint_path=None,
            ):
        # set attributes
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
        self.model = model

        # data loader
        self.test_data_loader = test_data_loader

        # AFM Scaler
        self.afm_scaler = AFMScaler(exp_cfg)

        # AMP settings
        self.use_amp = exp_cfg.train.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # set device
        self.device = torch.device(exp_cfg.system.device)
        self.model.to(self.device)

        # setting save directory
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            root_dir = exp_cfg.system.project_root
            self.save_dir = os.path.join(root_dir, 'runs', exp_cfg.system.run_name, 'evaluation_results')

        os.makedirs(self.save_dir, exist_ok=True)

        # setting checkpoint directory
        self.run_dir = os.path.join('runs', exp_cfg.system.run_name)
        if exp_cfg.system.checkpoint_dir is not None:
            self.checkpoint_dir = exp_cfg.system.checkpoint_dir
        else:
            self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')

        # define task mode (e.g., ideal_classification)
        self.task_mode = exp_cfg.train.task_mode
    
        # load latest checkpoint if available
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        else:
            try:
                self.load_latest_checkpoint()
            except (ValueError, FileNotFoundError):
                if self.rank == 0:
                    print("[Evaluator] Warning: No checkpoint found. Using random weights.")

        # distributed data parallel setup
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

    def load_latest_checkpoint(self):
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not set for Evaluator.")

        ckpts = glob(os.path.join(self.checkpoint_dir, "model_epoch*.pt"))
        if len(ckpts) == 0:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        # extract epoch numbers
        def extract_epoch(path):
            m = re.search(r"model_epoch(\d+)\.pt", path)
            return int(m.group(1)) if m else -1

        # sort by epoch number
        ckpts = sorted(ckpts, key=extract_epoch)

        latest = ckpts[-1]
        print(f"[Evaluator] Loading latest checkpoint: {latest}")
        return self._load_checkpoint_file(latest)

    def load_best_checkpoint(self):
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not set for Evaluator.")

        best_chk = os.path.join(self.checkpoint_dir, "best_model.pt")

        print(f"[Evaluator] Loading best checkpoint: {best_chk}")
        return self._load_checkpoint_file(best_chk)

    # Load specific checkpoint
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"[Evaluator] Loading checkpoint: {path}")
        return self._load_checkpoint_file(path)

    # Common loader
    def _load_checkpoint_file(self, path):
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state"])

        epoch = state.get("epoch", None)
        print(f"[Evaluator] Loaded model at epoch {epoch}")

        return epoch

    @torch.no_grad()
    def compute_loss_and_metrics(self, loss_fn, loss_weights, metrics, dataloader: DataLoader=None):
        """ 
        evaluate loss / accuracy 
        
        Args:
            metrics: list of metric functions to compute
            dataloader: DataLoader for evaluation (if None, use self.test_data_loader)
         Returns:
            average loss and accuracy over the dataset
        """
        # dataloader
        # use test data loader if not provided
        if dataloader is None:
            dataloader = self.test_data_loader
        if self.rank == 0:
            iterator = tqdm(
                dataloader, 
                desc=f"[TEST]", 
                leave=False, 
                disable=self.exp_cfg.system.tqdm_silent
            )
        else:
            iterator = dataloader


        self.model.eval()
        # initialize losses and metrics
        epoch_loss = 0.0
        epoch_loss_components = {k: 0.0 for k in loss_fn.keys()}
        epoch_metrics = {k: 0.0 for k in metrics.keys()}

        total_samples = 0

        with torch.no_grad():
            for batch in iterator:
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
                    for key, criterion in loss_fn.items():
                        # handle both single and multi-output cases
                        if isinstance(outputs, dict) and isinstance(labels, dict):
                            pred = outputs[key]
                            target = labels[key]
                        else:
                            pred = outputs
                            target = labels

                        loss_val = criterion(pred, target)
                        weight = loss_weights.get(key, 1.0)

                        batch_loss += weight * loss_val
                        epoch_loss_components[key] += loss_val.item() * imgs.size(0)

                # Aggregate 
                batch_size = imgs.size(0)
                total_samples += batch_size
                epoch_loss += batch_loss.item() * batch_size

                for name, metric_fn in metrics.items():
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
            "test_loss": stats_tensor[0].item() / global_total
        }

        idx = 2
        for k in loss_comp_keys:
            results[f"test_loss_{k}"] = stats_tensor[idx].item() / global_total
            idx += 1
            
        for k in metric_keys:
            results[f"test_{k}"] = stats_tensor[idx].item() / global_total
            idx += 1

        # save results
        if self.rank == 0:
            results_path = os.path.join(self.save_dir, "evaluation_results.txt")
            with open(results_path, "w") as f:
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
            print(f"[Evaluator] Saved evaluation results to {os.path.relpath(results_path)}")

        return results
    
    def compute_confusion_matrix(
            self, 
            dataloader: DataLoader=None, 
            save_dir: str=None
            ):
        
        # dataloader
        # use test data loader if not provided
        if dataloader is None:
            dataloader = self.test_data_loader
        
        if self.rank == 0:
            iterator = tqdm(
                dataloader, 
                desc="[TEST] Confusion Matrix", 
                leave=False, 
                disable=self.exp_cfg.system.tqdm_silent
            )
        else:
            iterator = dataloader

        # number of classes
        try:
            num_classes = self.exp_cfg.model.num_classes
        except AttributeError:
            raise ValueError("num_classes must be specified in exp_cfg.model for confusion matrix computation.")
        
        self.model.eval()
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=self.device)
        with torch.no_grad():            
            for batch in iterator:
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)
                imgs = imgs.to(self.device, non_blocking=True)

                # prepare labels
                if self.task_mode == "ideal_classification" or self.task_mode == "distorted_classification":
                    labels = labels.to(self.device, non_blocking=True)
                    labels = labels.view(-1)

                elif self.task_mode == "multitask" or self.task_mode == "multitask_ideal":
                    if isinstance(labels, dict):
                        labels = labels["state"].to(self.device, non_blocking=True).view(-1)
                    else:
                        labels = labels.to(self.device, non_blocking=True).view(-1)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(imgs)
                    if isinstance(outputs, dict):
                        preds = outputs["state"]
                    else:
                        preds = outputs

                    if preds.ndim == 2:
                        preds = preds.argmax(dim=1)

                if labels.numel() == preds.numel():
                    indices = torch.stack((labels, preds), dim=0)
                    ones = torch.ones(labels.size(0), dtype=torch.long, device=self.device)            
                    confusion_matrix.index_put_(tuple(indices), ones, accumulate=True)

        if self.is_distributed:
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)

        # move to CPU for further processing
        confusion_matrix = confusion_matrix.cpu()
        if self.rank == 0:
            cm_numpy = confusion_matrix.numpy()

            if save_dir is None:
                save_dir = self.save_dir
                
            cm_path = os.path.join(save_dir, "confusion_matrix.npy")
            np.save(cm_path, cm_numpy)
            print(f"[Evaluator] Saved confusion matrix to {os.path.relpath(cm_path)}")

            cm_pdf_path = os.path.join(save_dir, "confusion_matrix.pdf")
            visualize_confusion_matrix(cm_numpy, pdf_path=cm_pdf_path)
            print(f"[Evaluator] Saved confusion matrix to {os.path.relpath(cm_pdf_path)}")

    def compute_physical_metrics(self, dataloader: DataLoader = None):
        """ 
        Evaluate physical metrics (MSE, MAE) for reconstruction task 
        
        Args:
            dataloader: DataLoader for evaluation (if None, use self.test_data_loader)

        """
        self.model.eval()

        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        if dataloader is None:
            dataloader = self.test_data_loader

        with torch.no_grad():
            for batch in dataloader:
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)

                assert imgs.max() > 1.0, "Input imgs must be raw data (not scaled)"

                imgs = imgs.to(self.device, non_blocking=True)

                imgs_scaled, inp_min, inp_max = self.afm_scaler.scale(imgs)

                if isinstance(labels, dict):
                    labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
                else:
                    labels = labels.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(imgs_scaled)

                recon_scaled = outputs["ideal"] if isinstance(outputs, dict) else outputs
                recon_phys = self.afm_scaler.descale(recon_scaled, inp_min, inp_max)

                if isinstance(labels, dict):
                    target_phys = labels["ideal"].to(self.device, non_blocking=True)
                else:
                    target_phys = labels.to(self.device, non_blocking=True)
                
                assert target_phys.max() > 1.0, "Target imgs must be raw data (not scaled)"

                mse = torch.sum((recon_phys - target_phys) ** 2)
                mae = torch.sum(torch.abs(recon_phys - target_phys))

                total_mse += mse.item() / (target_phys.shape[1] * target_phys.shape[2] * target_phys.shape[3])
                total_mae += mae.item() / (target_phys.shape[1] * target_phys.shape[2] * target_phys.shape[3])
                
                total_samples += imgs.size(0)

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples

        results = {
            "avg_physical_mse": avg_mse,
            "avg_physical_mae": avg_mae,
            "total_samples": total_samples
        }

        if self.rank == 0:
            results_path = os.path.join(self.save_dir, "evaluation_results.txt")
            
            with open(results_path, "a") as f:
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
            
            print(f"--- Physical Evaluation Results (nm) ---")
            for k, v in results.items():
                print(f"{k}: {v:.4f}")
            print(f"[Evaluator] Saved physical results to {os.path.relpath(results_path)}")

        return results

    def run_predictions(self, imgs):
        """ 
        Run model predictions on a batch of images 
        
        Args:
            imgs: input batch of images (B, C, H, W)
        """

        assert imgs.ndim == 4, "Input imgs must be a 4D tensor (B, C, H, W)"
        assert imgs.min() >= 0.0 and imgs.max() <= 1.0, "Input imgs must be scaled to [0, 1] range"

        self.model.eval()
        with torch.no_grad():
            imgs = imgs.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(imgs)

        return outputs

    def run_denoising(self, imgs):
        """ 
        Run model denoising on a batch of images 

        Args:
            imgs: input batch of images (B, C, H, W)
         Returns:
            denoised images (B, C, H, W)
        """
        assert imgs.ndim == 4, "Input imgs must be a 4D tensor (B, C, H, W)"
        assert imgs.max() > 1.0, "Input imgs must be raw data (not scaled)"

        # scaling images
        imgs, inp_min_val, inp_max_val = self.afm_scaler.scale(imgs)
        outputs = self.run_predictions(imgs)
        if isinstance(outputs, dict):
            denoised_imgs = outputs["ideal"]
        else:
            denoised_imgs = outputs

        # descale images
        batch_size = denoised_imgs.size(0)
        descaled_list = []
        for i in range(batch_size):
            denoised_imgs_i = self.afm_scaler.descale(
                x=denoised_imgs[i].cpu(),
                min_val=inp_min_val[i].cpu(),
                max_val=inp_max_val[i].cpu()
                )
            descaled_list.append(denoised_imgs_i)
       
        return torch.stack(descaled_list)

    def run_state_prediction(self, imgs):
        """ 
        Run model state prediction on a batch of images 

        Args:
            imgs: input batch of images (B, C, H, W)
         Returns:
            predicted states (B,)
        """
        assert imgs.ndim == 4, "Input imgs must be a 4D tensor (B, C, H, W)"
        assert imgs.max() > 1.0, "Input imgs must be raw data (not scaled)"

        # scaling images
        imgs, _, _ = self.afm_scaler.scale(imgs)
        outputs = self.run_predictions(imgs)
        if isinstance(outputs, dict):
            state_logits = outputs["state"]
        else:
            state_logits = outputs

        predicted_states = state_logits.argmax(dim=1)

        return predicted_states

    def run_img_visualization(
            self, 
            dataloader: DataLoader=None,
            num_samples=5, 
            save_dir=None,
            img_specifics="distorted", # "distorted", "ideal" or "denoised" 
            extension=".pdf",
            colorbar=True
            ):
        """
        Visualize specified imgs 
        Args:
            dataloader: DataLoader for evaluation (if None, use self.test_data_loader)
        """

        if self.rank != 0: return  # only main process
        if dataloader is None: dataloader = self.test_data_loader
        if save_dir is None: save_dir = self.save_dir

        assert extension in [".pdf", ".png"], "extension must be .pdf or .png"

        output_dir = os.path.join(save_dir, "denoise_visualizations")
        os.makedirs(output_dir, exist_ok=True)

        scaler = AFMScaler(self.exp_cfg)

        self.model.eval()
        count = 0

        iterator = iter(dataloader)
        try:
            batch = next(iterator)
        except StopIteration:
            return

        with torch.no_grad():
            for batch in iterator:
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)
                imgs = imgs.to(self.device, non_blocking=True)

                assert imgs.max() > 1.0, \
                "Input images must be raw data (not scaled) for visualization."

                # scaling images
                imgs, inp_min_val, inp_max_val = scaler.scale(imgs)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(imgs)
                
                #--- Get reconstructed images ---
                if isinstance(outputs, dict):
                    recon_batch = scaler.descale(outputs["ideal"], inp_min_val, inp_max_val)
                else:
                    recon_batch = scaler.descale(outputs, inp_min_val, inp_max_val)
                
                #--- Get states ---
                predicted_states = None
                if self.task_mode == "multitask":
                    if isinstance(outputs, dict) and "state" in outputs:
                        predicted_states = outputs["state"].argmax(dim=1)
                    else:
                        raise ValueError("For multitask mode, model output must be a dict containing 'state' key.")

                # get input and target images
                input_batch = scaler.descale(imgs, inp_min_val, inp_max_val)
                target_batch = labels["ideal"] if isinstance(labels, dict) else labels
                
                # get target states if available
                target_states = labels["state"] if isinstance(labels, dict) and "state" in labels else None

                # check if raw images
                assert target_batch.max() > 1.0, \
                "Input and target images must be raw data (not scaled) for visualization."

                # --- Loop for Visualization ---
                batch_size = input_batch.size(0)
                
                for i in range(min(batch_size, num_samples)):
                    # Tensor -> Numpy
                    inp = input_batch[i]
                    rec = recon_batch[i]
                    tgt = target_batch[i]

                    if img_specifics == "distorted":
                        vis_img = inp
                    elif img_specifics == "ideal":
                        vis_img = tgt
                    elif img_specifics == "denoised":
                        vis_img = rec
                    else:
                        raise ValueError("img_specifics must be 'distorted', 'ideal' or 'denoised'.")
                    
                    vis_img = vis_img.squeeze(0)  # (H, W)

                    plot_heightmap(
                        arr=vis_img.cpu().numpy(),
                        save_path=os.path.join(output_dir, f"{img_specifics}_sample_{count}{extension}"),
                        colorbar=colorbar
                        )
                    print(f"[Evaluator] Saved {img_specifics} sample: {os.path.relpath(os.path.join(output_dir, f'{img_specifics}_sample_{count}{extension}'))}")

                    count += 1
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break

    def run_denoise_visualization(
            self, 
            dataloader: DataLoader=None,
            num_samples=5, 
            start=None,
            angle_deg=None,
            num_points=None,
            save_dir=None,
            save=True,
            extension=".pdf"
            ):
        """ 
        Visualize attention maps on denoising task 
        
        Args:
            dataloader: DataLoader for evaluation (if None, use self.test_data_loader)
            distorted_img, ideal_img in dataloader must be not scaled and raw images
         """
        if self.rank != 0: return  # only main process
        if dataloader is None: dataloader = self.test_data_loader
        if save_dir is None: save_dir = self.save_dir
        
        output_dir = os.path.join(save_dir, "denoise_visualizations")
        os.makedirs(output_dir, exist_ok=True)

        scaler = AFMScaler(self.exp_cfg)

        self.model.eval()
        count = 0

        iterator = iter(dataloader)
        try:
            batch = next(iterator)
        except StopIteration:
            return

        with torch.no_grad():
            for batch in iterator:
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)
                imgs = imgs.to(self.device, non_blocking=True)

                assert imgs.max() > 1.0, \
                "Input images must be raw data (not scaled) for visualization."

                # scaling images
                imgs, inp_min_val, inp_max_val = scaler.scale(imgs)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(imgs)
                
                #--- Get reconstructed images ---
                if isinstance(outputs, dict):
                    recon_batch = scaler.descale(outputs["ideal"], inp_min_val, inp_max_val)
                else:
                    recon_batch = scaler.descale(outputs, inp_min_val, inp_max_val)
                
                #--- Get states ---
                predicted_states = None
                if self.task_mode == "multitask":
                    if isinstance(outputs, dict) and "state" in outputs:
                        predicted_states = outputs["state"].argmax(dim=1)
                    else:
                        raise ValueError("For multitask mode, model output must be a dict containing 'state' key.")

                # get input and target images
                input_batch = scaler.descale(imgs, inp_min_val, inp_max_val)
                target_batch = labels["ideal"] if isinstance(labels, dict) else labels
                
                # get target states if available
                target_states = labels["state"] if isinstance(labels, dict) and "state" in labels else None

                # check if raw images
                assert target_batch.max() > 1.0, \
                "Input and target images must be raw data (not scaled) for visualization."

                # --- Loop for Visualization ---
                batch_size = input_batch.size(0)
                
                for i in range(min(batch_size, num_samples)):
                    # Tensor -> Numpy
                    inp = input_batch[i]
                    rec = recon_batch[i]
                    tgt = target_batch[i]

                    tgt_state = target_states[i] if target_states is not None else None
                    pred_state = predicted_states[i] if predicted_states is not None else None
                        
                    # states (0-based -> 1-based)
                    texts = [
                        None, 
                        f'state:{int(pred_state.item()+1)}' if pred_state is not None else None, 
                        f'state:{int(tgt_state.item()+1)}' if tgt_state is not None else None
                        ]
                    
                    # visualize and save
                    if save:
                        save_path = os.path.join(output_dir, f"denoise_sample_{count}{extension}")
                    else:
                        save_path = None
                    
                    if start is not None and angle_deg is not None and num_points is not None:
                        plot_three_heightmap_with_lineprofile(
                            arr_list=[inp.cpu().numpy(), rec.cpu().numpy(), tgt.cpu().numpy()],
                            start=start,
                            angle_deg=angle_deg,
                            num_points=num_points,
                            texts=texts,
                            save_path=save_path
                        )
                    else:
                        visualize_denoising(
                            input_img=inp, 
                            denoised_img=rec, 
                            target_img=tgt, 
                            texts=texts, 
                            save_path=save_path
                            )
                    
                    if save_path:
                        print(f"[Evaluator] Saved reconstruction sample: {os.path.relpath(save_path)}")
                    
                    count += 1
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break

    def compute_attention_maps(self, img_scaled_batch):
        """ 
        Extract attention maps from ViT model 
        
        Args:
            img_scaled_batch: input batch of scaled images
        """
        self.model.eval()
        img_scaled_batch = img_scaled_batch.to(self.device)

        with torch.no_grad():
            _, attn_maps = self.model(img_scaled_batch, return_attn=True)
    
        return attn_maps # (num_layers, batch_size, N, N)

    def compute_attention_rollout(
            self, 
            x_scaled_batch, 
            add_residual=True,
            analyze_task="classification" 
            ):
        """
        Attention Rollout calculation
        attn_maps: Tensor (num_layers, batch_size, N, N)

        Args:
            batch: input batch from dataloader
            add_residual: whether to add residual connection in attention
            analyze_task: "classification" or "denoising" to select which head to analyze
        """
        result = None
        if self.task_mode == "multitask" or self.task_mode == "experiment_image_inference":
            attn_maps = self.compute_attention_maps(x_scaled_batch)  # (num_layers, batch_size, N, N)

            if analyze_task == "classification":

                attn_maps = torch.cat(
                    [attn_maps['encoder_attn'], attn_maps['classifier_attn']],
                    dim=0
                ) # (num_layers, batch_size, N, N)

            elif analyze_task == "classification_only":
                attn_maps = attn_maps['classifier_attn']  # (num_layers, batch_size, N, N)

            elif analyze_task == "denoising":
                attn_maps = torch.cat(
                    [attn_maps['encoder_attn'], attn_maps['decoder_attn']],
                    dim=0
                ) # (num_layers, batch_size, N, N)

        else:
            attn_maps = self.compute_attention_maps(x_scaled_batch)  # (num_layers, batch_size, N, N)

        for A in attn_maps:
            if add_residual:
                I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)  # (1, N, N)
                A = A + I
                A = A / A.sum(dim=-1, keepdim=True)  # 正規化

            result = A if result is None else torch.bmm(A, result)

        return result  # (B, N, N)

    def visualize_attention(self, rollout, grid_size, save_name="attention_rollout.png"):
        """
        Visualize flow from CLS token to each patch
        rollout: (num_layers, batch_size, N, N)
        """
        batch_size, N, _ = rollout.shape

        for i in range(batch_size):
            cls_to_patch = rollout[i, 0, 1:]  # CLS → patches
            attn_map = cls_to_patch.reshape(grid_size, grid_size)
            attn_map = attn_map / attn_map.max()

            plt.figure(figsize=(4, 4))
            #sns.heatmap(attn_map.cpu(), cmap="viridis")
            plt.title(f"Attention Rollout (sample {i})")
            plt.axis("off")

            if self.save_dir:
                path = f"{self.save_dir}/{save_name.replace('.png', f'_{i}.png')}"
                plt.savefig(path)
                print(f"[Evaluator] Saved: {os.path.relpath(path)}")

            plt.close()

    def visualize_attention_with_image(
            self, 
            dataloader: DataLoader,
            visualize_num=5,
            add_residual=True,
            analyze_task="classification", # "classification" or "classification_only" 
            save_name="attention_rollout.png"
            ):
        """
        Visualize flow from CLS token to each patch
        rollout: (num_layers, batch_size, N, N)
        """

        output_dir = os.path.join(self.save_dir, "attention_visualizations")
        os.makedirs(output_dir, exist_ok=True)

        count=0
        for batch in dataloader:
            x, _ = prepare_batch_for_model(batch, self.task_mode)

            # scale
            x_scaled, min_val, max_val = self.afm_scaler.scale(x)

            rollout = self.compute_attention_rollout(
                x_scaled,
                add_residual=add_residual,
                analyze_task=analyze_task
            )
            batch_size, N, _ = rollout.shape

            grid_size = int((self.exp_cfg.model.image_size / getattr(self.exp_cfg.model, "patch_size", 3)))
            for i in range(batch_size):
                cls_to_patch = rollout[i, 0, 1:]  # CLS → patches
                attn_map = cls_to_patch.reshape(grid_size, grid_size)
                attn_map = attn_map / attn_map.max()

                # descale image for visualization
                x_descaled = self.afm_scaler.descale(
                    x=x_scaled[i].cpu(),
                    min_val=min_val[i].cpu(),
                    max_val=max_val[i].cpu()
                    )

                visualize_image_with_attention(
                    img_tensor=x_descaled,
                    attn_map=attn_map,
                    save_path=f"{output_dir}/{save_name.replace('.png', f'_{count}.png')}"
                )

                full_path = f"{output_dir}/{save_name.replace('.png', f'_{count}.png')}"
                print(f"[Evaluator] Saved: {os.path.relpath(full_path)}")
                count += 1
                if count >= visualize_num:
                    break
            if count >= visualize_num:
                break

    def visualize_multitask_comprehensive(
            self, 
            start,
            angle_deg,
            dataloader: DataLoader=None,
            num_points=None,
            num_samples=5, 
            add_residual=True,
            analyze_task="classification", # "classification" or "classification_only" 
            save_dir=None,
            save=True,
            extension=".pdf",
            state_diff_threshold=None, 
            overlay_line=True,
            save_datapack=False
        ):
        """ 
        Visualize attention maps on denoising task 
        
        Args:
            dataloader: DataLoader for evaluation (if None, use self.test_data_loader)
            distorted_img, ideal_img in dataloader must be not scaled and raw images

            num_points: number of points for line profile (if None, line profile is not plotted)
            num_samples: number of samples to visualize
            add_residual: whether to add residual connection in attention
            analyze_task: "classification" or "denoising" to select which head to analyze
            save_dir: directory to save visualizations   
            save: whether to save the visualizations
            extension: file extension for saved images (".png" or ".pdf")
            state_diff_threshold: if specified, only visualize samples where |predicted_state - target_state| >= threshold
            overlay_line: whether to overlay line profile on heightmap
            save_datapack: whether to save the data used for visualization

         """
        if self.rank != 0: return  # only main process
        if dataloader is None: dataloader = self.test_data_loader
        if save_dir is None: save_dir = self.save_dir
        
        assert extension in [".png", ".pdf"], "extension must be .png or .pdf"

        output_dir = os.path.join(save_dir, "multitask_comprehensive")
        os.makedirs(output_dir, exist_ok=True)

        scaler = AFMScaler(self.exp_cfg)

        self.model.eval()
        count = 0

        iterator = iter(dataloader)
        with torch.no_grad():
            exceed_diff_list = []
            metrics_log = []  
            for batch in iterator:
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)
                imgs = imgs.to(self.device, non_blocking=True)
                labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}

                assert imgs.max() > 1.0, \
                "Input images must be raw data (not scaled) for visualization."

                # scaling images
                imgs, inp_min_val, inp_max_val = scaler.scale(imgs)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(imgs)
                
                #--- Get reconstructed images ---
                if isinstance(outputs, dict):
                    recon_batch = scaler.descale(outputs["ideal"], inp_min_val, inp_max_val)
                else:
                    recon_batch = scaler.descale(outputs, inp_min_val, inp_max_val)
                
                #--- Get target states if available ---
                target_states = labels["state"] if isinstance(labels, dict) and "state" in labels else None

                #--- Get states ---
                predicted_states = None
                if self.task_mode == "multitask":
                    if isinstance(outputs, dict) and "state" in outputs:
                        predicted_states = outputs["state"].argmax(dim=1)
                        
                    else:
                        raise ValueError("For multitask mode, model output must be a dict containing 'state' key.")

                #--- Filter by state difference threshold if specified ---
                if state_diff_threshold is not None and target_states is not None and predicted_states is not None:
                    abs_diffs = torch.abs(predicted_states - target_states)
                    
                    # check which samples exceed the threshold
                    mask = abs_diffs >= state_diff_threshold
                    selected_indices = torch.nonzero(mask).squeeze()
                    
                    if selected_indices.ndim == 0:
                        selected_indices = selected_indices.unsqueeze(0)
                    # if no samples exceed the threshold, skip this batch
                    if selected_indices.numel() == 0:
                        continue 

                    # Filtering data
                    imgs = imgs[selected_indices]
                    inp_min_val = inp_min_val[selected_indices]
                    inp_max_val = inp_max_val[selected_indices]
                    
                    outputs = {k: v[selected_indices] for k, v in outputs.items()} if isinstance(outputs, dict) else outputs[selected_indices]
                    recon_batch = recon_batch[selected_indices]

                    if isinstance(labels, dict):
                        for k in labels.keys():
                            labels[k] = labels[k][selected_indices]
                    else:
                        labels = labels[selected_indices]

                    target_states = target_states[selected_indices]
                    predicted_states = predicted_states[selected_indices]

                    # store samples exceeding threshold
                    batch_append = {}
                    for key in batch.keys():
                        if key not in batch:
                            continue
                        if isinstance(batch[key], torch.Tensor):
                            batch_append[key] = batch[key][selected_indices]

                    exceed_diff_list.append(batch_append)

                #--- Attention Rollout ---
                rollout = self.compute_attention_rollout(
                    imgs,
                    add_residual=add_residual,
                    analyze_task=analyze_task
                )
                batch_size, N, _ = rollout.shape
                grid_size = int((self.exp_cfg.model.image_size / getattr(self.exp_cfg.model, "patch_size", 3)))


                # get input and target images
                input_batch = scaler.descale(imgs, inp_min_val, inp_max_val)
                target_batch = labels["ideal"] if isinstance(labels, dict) else labels
                
                # check if raw images
                assert target_batch.max() > 1.0, \
                "Input and target images must be raw data (not scaled) for visualization."


                # --- Calculate Metrics ---
                target_batch = labels["ideal"] if isinstance(labels, dict) else labels
                mse_batch = torch.mean((recon_batch - target_batch) ** 2, dim=[1,2,3])
                mae_batch = torch.mean(torch.abs(recon_batch - target_batch), dim=[1,2,3])

                # --- Loop for Visualization ---
                batch_size = imgs.size(0)
                
                for i in range(min(batch_size, num_samples)):
                    # Tensor -> Numpy
                    inp = input_batch[i]
                    rec = recon_batch[i]
                    tgt = target_batch[i]

                    tgt_state = target_states[i] if target_states is not None else None
                    pred_state = predicted_states[i] if predicted_states is not None else None
                        
                    # states (0-based -> 1-based)
                    texts = [
                        None, 
                        f'state:{int(pred_state.item()+1)}' if pred_state is not None else None, 
                        f'state:{int(tgt_state.item()+1)}' if tgt_state is not None else None
                        ]
                    
                    # attention map for this sample
                    cls_to_patch = rollout[i, 0, 1:]  # CLS → patches
                    attn_map = cls_to_patch.reshape(grid_size, grid_size)
                    attn_map = attn_map / attn_map.max()
                    
                    # visualize and save
                    if save:
                        save_path = os.path.join(output_dir, f"sample_{count}{extension}")

                        # append state difference info if applicable
                        if state_diff_threshold is not None:
                            save_path = save_path.replace(extension, f"_statediff{state_diff_threshold}{extension}")
                    else:
                        save_path = None
                
                    visualize_multitask_comprehensive(
                        input_img=inp,
                        denoised_img=rec,
                        target_img=tgt,
                        attn_map=attn_map.cpu().numpy(),
                        class_logits=outputs["state"][i].cpu().numpy(),
                        true_label=tgt_state.item() if tgt_state is not None else None,
                        start=start,
                        angle_deg=angle_deg,
                        texts=texts,
                        save_path=save_path,
                        num_points=num_points,
                        overlay_line=overlay_line,
                    )
                    
                    if save:
                        print(f"[Evaluator] Saved reconstruction sample: {os.path.relpath(save_path)}")
    
                    #--- Log Metrics ---
                    metrics_dict = {
                        "sample_id": count,
                        "filename": os.path.basename(save_path) if save_path is not None else None,
                        "mse": mse_batch[i].item(),
                        "mae": mae_batch[i].item(),
                        # 0-based -> 1-based
                        "pred_state": int(predicted_states[i].item() + 1) if predicted_states is not None else None, 
                        # 0-based -> 1-based
                        "true_state": int(target_states[i].item() + 1) if target_states is not None else None, 
                        }
                    metrics_log.append(metrics_dict)

                    #-- Save Data Pack if specified ---
                    if save_datapack and save:
                        datapack_path = os.path.join(output_dir, f"datapack_statediff{state_diff_threshold}_{count}.pt")
                        
                        datapack = {
                            "input": inp.cpu().numpy() if torch.is_tensor(inp) else inp,
                            "denoised": rec.cpu().numpy() if torch.is_tensor(rec) else rec,
                            "ideal": tgt.cpu().numpy() if torch.is_tensor(tgt) else tgt,
                            "attn": attn_map.cpu().numpy() if torch.is_tensor(attn_map) else attn_map,
                            "logits": outputs["state"][i].cpu().numpy(),
                            "meta": metrics_dict 
                        }
                        
                        torch.save(datapack, datapack_path)
                        print(f"[Evaluator] Saved raw datapack: {os.path.relpath(datapack_path)}")

                    count += 1
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break

            #--- Save Metrics Log to CSV ---
            if len(metrics_log) > 0:                
                csv_filename = "evaluation_metrics.csv"
                if state_diff_threshold is not None:
                    csv_filename = f"evaluation_metrics_statediff{state_diff_threshold}.csv"
                
                csv_path = os.path.join(output_dir, csv_filename)

                # 2. Write to CSV
                fieldnames = list(metrics_log[0].keys())
                
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    writer.writeheader()       # Write header (column names)
                    writer.writerows(metrics_log) # Write all data rows
                
                print(f"[Evaluator] Saved metrics log to {os.path.relpath(csv_path)} (Total {len(metrics_log)} samples)")        #f len(exceed_diff_list) > 0:        #   

    #-- Export Latent Embeddings --#
    def export_latent_embeddings(
            self, 
            dataloader: DataLoader=None,
            mode="encoder" 
            ):
        """ 
        Extract latent embeddings from the model 
        
        Args:
            dataloader: DataLoader for evaluation (if None, use self.test_data_loader)
            mode: "encoder" or "classifier" to select which embeddings to extract
            Returns:
                embeddings: Tensor of shape (num_samples, latent_dim)
         """
        
        output_dir = os.path.join(self.save_dir, "latent_embeddings")
        os.makedirs(output_dir, exist_ok=True)

        # Dataloader
        if dataloader is None:
            dataloader = self.test_data_loader

        self.model.eval()

        save_path = os.path.join(output_dir, f"latent_embeddings_{mode}.npz")

        feature_list = []
        gt_class_list = []
        pred_class_list = []
        config_list = []
        print(f"Start exporting features via mode='{mode}'...")
        with torch.no_grad():
            for batch in tqdm(
                dataloader, 
                desc="Extracting Features", 
                disable=self.exp_cfg.system.tqdm_silent
                ):
                
                if "config" in batch:
                    config_list.extend(batch["config"])
                    
                imgs, labels = prepare_batch_for_model(batch, self.task_mode)
                imgs = imgs.to(self.device, non_blocking=True)
                
                # Encoder forward
                enc_features, _ = self.model.encoder(imgs)
                
                # pass through classifier head
                class_features = enc_features
                for blk in self.model.classifier.blocks:
                    class_features, _ = blk(class_features)
                class_features = self.model.classifier.norm(class_features)

                # predict classes
                preds = self.model.classifier.head(class_features[:, 0])
                pred_class_list.append(preds.argmax(dim=1).cpu().numpy())

                if isinstance(labels, dict):
                    if "state" in labels:
                        lbl_data = labels["state"]
                        
                        # If True label is None (When evaluate Struct Dependency or Experiment Img)
                        if lbl_data is None:
                            lbl_data = batch_size = imgs.shape[0]
                            lbl_data = np.full((batch_size,), -1, dtype=np.int32)
                            gt_class_list.append(lbl_data)

                        elif isinstance(lbl_data, torch.Tensor):
                            gt_class_list.append(lbl_data.cpu().numpy())

                        else:
                            gt_class_list.append(np.array(lbl_data))
                    else:
                        pass 

                elif isinstance(labels, torch.Tensor):
                    gt_class_list.append(labels.cpu().numpy())
                
                elif isinstance(labels, (list, np.ndarray)):
                    gt_class_list.append(np.array(labels))

                # select features based on mode
                if mode == "encoder":
                    enc_features = enc_features[:, 0, :]
                    feature_list.append(enc_features.cpu().numpy())

                elif mode == "classifier":
                    class_features = class_features[:, 0, :]
                    feature_list.append(class_features.cpu().numpy())
        
        all_features = np.concatenate(feature_list, axis=0) # shape: (TotalSamples, 512)
        all_labels = np.concatenate(gt_class_list, axis=0)     # (N, ) or (N, ClassNum)
        all_preds = np.concatenate(pred_class_list, axis=0)     # (N, ) or (N, ClassNum)

        print(f"Saving to {os.path.relpath(save_path)}...")
        np.savez_compressed(
            save_path, 
            embeddings=all_features, 
            labels=all_labels,
            preds=all_preds,
            configs=config_list
            )
        
        # memory cleanup
        del feature_list, gt_class_list, pred_class_list, all_features, all_labels

        return save_path

    # -- structure dependency analysis -- #
    @torch.no_grad()
    def _run_analysis_single(self, save_name, structure_id=None, dataloader: DataLoader=None):
        """
        Run analysis for a single structure and save results to CSV
        
        Args:
            save_name: base name for saving CSV
            structure_id: identifier for the structure (used in saving)
            dataloader: optional DataLoader to use for analysis (defaults to self.dataloader)
        Returns:
            csv_path: path to the saved CSV file
            results_list: list of dicts containing results
        """

        assert self.task_mode in ["ideal_classification", "distorted_classification", "multitask"], "Unsupported task mode for analysis."

        if dataloader is None:
            dataloader = self.test_data_loader

        results_list = []
        
        # --- prediction loop ---
        for batch in tqdm(
            dataloader, 
            desc=f"[Eval] Analysis ({structure_id})", 
            disable=self.exp_cfg.system.tqdm_silent
        ):
            imgs, labels = prepare_batch_for_model(batch, self.task_mode)
            imgs = imgs.to(self.device, non_blocking=True)

            outputs = self.model(imgs)

            #--- Get probabilities ---
            if isinstance(outputs, dict):
                class_logits = outputs["state"]
            else:
                class_logits = outputs

            probs = torch.softmax(class_logits.float(), dim=1).cpu().numpy()  # (B, num_classes)

            batch_size = probs.shape[0]
            for i in range(batch_size): 
                row = {
                    "structure_id": structure_id if structure_id else "unknown"
                }
                for cls_idx, p in enumerate(probs[i]):
                    row[f"prob_{cls_idx}"] = float(p)
                
                results_list.append(row)

        # --- CSV saving (using standard library) ---
        csv_path = os.path.join(self.save_dir, f"{save_name}.csv")
        
        if results_list:
            # Get header (column names)
            fieldnames = list(results_list[0].keys())
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_list)
                
        print(f"[Evaluator] Saved results to {os.path.relpath(csv_path)}")
        
        # Return path and data for aggregation by caller
        return csv_path, results_list
