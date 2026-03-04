import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from configs.experiment_config import ExperimentConfig
from models.ViT_AutoEncoder import ViTAutoEncoder
from models.ViT_MultiTask_AutoEncoder import ViTMultiTaskAutoEncoder

def load_pretrained_weights(model, checkpoint_path, device='cpu'):
    rel_path = os.path.relpath(checkpoint_path) if os.path.exists(checkpoint_path) else checkpoint_path
    print(f"[Model] Loading pretrained weights with adaptation from: {rel_path}")
    
    state = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = state["model_state"] if "model_state" in state else state
    
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            # 1. Patch Embedding resize (if image size or patch size changed)
            if "patch_embed.proj.weight" in k and v.shape != model_dict[k].shape:
                print(f"[Model] Resizing {k}: {v.shape} -> {model_dict[k].shape}")
                new_v = F.interpolate(
                    v, size=model_dict[k].shape[2:], 
                    mode='bilinear', align_corners=False
                )
                new_state_dict[k] = new_v
            
            # 2. Decoder final projection layer 
            elif "decoder_pred" in k and v.shape != model_dict[k].shape:
                print(f"[Model] Skipping {k} due to shape mismatch (will be randomly initialized)")
                continue
                
            elif v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"[Model] Skipping {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print(f"[Model] {len(missing_keys)} keys missing (expected for new layers or mismatches).")
    
    return model

def change_patch_embedding(model, exp_cfg):
    """
    Dynamically adapts the model's patch embedding to a new patch size.
    """
    
    old_proj = model.encoder.patch_embed.proj
    in_ch = old_proj.in_channels
    embed_dim = old_proj.out_channels
    
    model.encoder.patch_embed.proj = nn.Conv2d(
        in_channels=in_ch,
        out_channels=embed_dim,
        kernel_size=(exp_cfg.model.patch_size, exp_cfg.model.patch_size),
        stride=(exp_cfg.model.patch_size, exp_cfg.model.patch_size)
    ).to(next(model.parameters()).device) 
    
    return model

def create_model_from_pretrained(exp_cfg, checkpoint_path, config_path):
    """
    Factory function to build either AE or MultiTask model based on task_mode,
    initialized with pretrained weights.
    """
    # Load the model structure configuration
    model_cfg = ExperimentConfig.load_yaml(config_path)
    
    # Select the class based on the current experiment's task mode
    if exp_cfg.train.task_mode == "multitask":
        model_class = ViTMultiTaskAutoEncoder
    elif exp_cfg.train.task_mode == "denoise":
        model_class = ViTAutoEncoder
    else:
        raise ValueError(f"Unsupported task_mode: {exp_cfg.train.task_mode}")
    
    # 1. Instantiate the model
    model = model_class.from_config(model_cfg)

    # If patch size or image size has changed, adapt the patch embedding layer before loading weights
    if exp_cfg.model.patch_size != model_cfg.model.patch_size:
        print(f"[Model Factory] Adapting patch embedding from patch size {model_cfg.model.patch_size} to {exp_cfg.model.patch_size}")
        model = change_patch_embedding(model, exp_cfg)  # Adapt patch embedding to new patch size if needed

    # 2. Load weights using the utility function
    load_pretrained_weights(
        model, 
        checkpoint_path, 
        device=exp_cfg.system.device
    )

    # 3. If multitask, replace the classifier head to match the number of classes in the new config
    if exp_cfg.train.task_mode == "multitask":
        num_classes = exp_cfg.model.num_classes
    
        # input features of the classifier head
        in_features = model.classifier.head.in_features
        
        # Replace the classifier head with a new one that has the correct number of output classes
        model.classifier.head = nn.Linear(in_features, num_classes)
        
        # initialize the new classifier head's weights and biases
        nn.init.trunc_normal_(model.classifier.head.weight, std=0.02)
        nn.init.constant_(model.classifier.head.bias, 0)
    
    if exp_cfg.model.patch_size != model_cfg.model.patch_size or exp_cfg.model.image_size != model_cfg.model.image_size:

        # 4. Decoder final projection layer adaptation (if image size or patch size changed)
        dec_features = model_cfg.model.dec_embed_dim
        out = exp_cfg.model.patch_size ** 2 * exp_cfg.model.in_channels
        model.decoder.decoder_pred = nn.Linear(dec_features, out)

        # initialize the new decoder_pred layer's weights and biases
        nn.init.trunc_normal_(model.decoder.decoder_pred.weight, std=0.02)
        nn.init.constant_(model.decoder.decoder_pred.bias, 0)

        # renew patch size
        model.patch_size = exp_cfg.model.patch_size

    return model