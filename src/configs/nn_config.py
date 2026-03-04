from dataclasses import dataclass
from typing import Optional

# -------------------------
# Training Configuration
# -------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    warmup_epochs: int = 5      # warmup epochs for DDP
    patience: int = 10          # early stopping patience
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    optimizer: str = "adamw"
    scheduler: Optional[str] = None    # e.g., "cosine", "step", etc.
    task_mode: str = "classification"  # "classification" or "denoise" or "multitask"

    # loss 
    recons_loss_type: str = "mse"     # for denoise/reconstruction task: "mse", "l1", "huber"
    weight_cls: float = 0.5           # for multitask
    weight_recon: float = 0.5         # for multitask

# -------------------------
# Neural Network Configuration
# -------------------------

@dataclass
class BaseModelConfig:
    name: str = "base" 

# -------------------------
# Classifier Configuration
# -------------------------

# ViT
@dataclass
class ViTClassifierConfig(BaseModelConfig):
    name: str = "vit_classifier"
    
    # Input / Output
    in_channels: int = 1
    image_size: int = 40
    num_classes: int = 2
    
    # Architecture
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1

# -------------------------
# Denoiser Configuration
# -------------------------
@dataclass
class ViTAutoEncoderConfig(BaseModelConfig):
    name: str = "vit_autoencoder"

    # Input 
    in_channels: int = 1
    image_size: int = 40
    patch_size: int = 4

    # --- Encoder Settings ---
    enc_embed_dim: int = 256
    enc_depth: int = 4
    enc_num_heads: int = 8
    enc_mlp_ratio: float = 4.0

    # --- Decoder Settings ---
    dec_embed_dim: int = 128    
    dec_depth: int = 2          
    dec_num_heads: int = 4
    dec_mlp_ratio: float = 4.0

    dropout_rate: float = 0.1

# -------------------------
# MultiTask AE Configuration
# -------------------------
@dataclass
class ViTMultiTaskAutoEncoderConfig(BaseModelConfig):
    name: str = "vit_multitask_ae"

    # Input 
    in_channels: int = 1
    image_size: int = 40
    patch_size: int = 4

    # --- Encoder Settings ---
    enc_embed_dim: int = 256
    enc_depth: int = 4
    enc_num_heads: int = 8
    enc_mlp_ratio: float = 4.0

    # --- Decoder Settings ---
    dec_embed_dim: int = 128    
    dec_depth: int = 2          
    dec_num_heads: int = 4
    dec_mlp_ratio: float = 4.0

    # --- Classifier Settings ---
    class_depth: int = 2
    num_classes: int = 2 

    dropout_rate: float = 0.1