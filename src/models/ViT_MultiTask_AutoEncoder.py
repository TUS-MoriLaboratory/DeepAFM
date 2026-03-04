# models/ViT_MultiTask_AutoEncoder.py
"""
Vision Transformer MultiTask AutoEncoder
-----------------------------------
Implements a Vision Transformer-based AutoEncoder model.
Includes encoder and decoder with attention map extraction capability.
-----------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.ViT_component import ViTEncoder, ViTDecoder, ViTClassifierEncoder, unpatchify

# ------------------------------
# Vision Transformer AutoEncoder
# ------------------------------
class ViTMultiTaskAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_chans=1,
        enc_dim=128,
        enc_depth=4,
        enc_num_heads=4,
        enc_mlp_ratio=4.0,
        dec_dim=128,
        dec_depth=4,
        dec_num_heads=4,
        dec_mlp_ratio=4.0,
        class_depth=2,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        # Encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=enc_dim,
            depth=enc_depth,
            num_heads=enc_num_heads,
            mlp_ratio=enc_mlp_ratio,
            dropout=dropout,
        )

        # Decoder
        self.decoder = ViTDecoder(
            img_size=img_size,
            patch_size=patch_size,
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            depth=dec_depth,
            num_heads=dec_num_heads,
            mlp_ratio=dec_mlp_ratio,
            dropout=dropout,
            in_chans=in_chans,
        )

        # Classifier Encoder
        self.classifier = ViTClassifierEncoder(
            embed_dim=enc_dim,
            depth=class_depth,
            num_heads=enc_num_heads,
            mlp_ratio=enc_mlp_ratio,
            dropout=dropout,
            num_classes=num_classes
        )

        # Store parameters for patchify/unpatchify
        self.patch_size = patch_size
        self.in_chans = in_chans

    def forward(self, x, return_attn=False):
        shared_feats, enc_attn = self.encoder(x, return_attn=return_attn) # (B, N+1, D)
        
        # decoder
        patch_tokens = shared_feats[:, 1:, :]  # remove CLS token
        recon_patches, dec_attn = self.decoder(patch_tokens, return_attn=return_attn) # (B, N, P*P*C)
        recon_img = unpatchify(recon_patches, self.patch_size, in_chans=self.in_chans)  # (B, C, H, W)

        # classifier head
        class_logits, class_attn = self.classifier(shared_feats, return_attn=return_attn)  # (B, num_classes)

        attn_list = {
            "encoder_attn": enc_attn,
            "decoder_attn": dec_attn,
            "classifier_attn": class_attn
        }

        outputs = {
            "ideal": recon_img,
            "state": class_logits
        }

        if return_attn:
            return outputs, attn_list
        else:
            return outputs

    @staticmethod
    def from_config(cfg) -> "ViTMultiTaskAutoEncoder":
        """Factory from ExperimentConfig"""
        return ViTMultiTaskAutoEncoder(
            img_size=cfg.model.image_size,
            patch_size=getattr(cfg.model, "patch_size", 4),
            in_chans=cfg.model.in_channels,
            enc_dim=getattr(cfg.model, "enc_embed_dim", 128),
            enc_depth=getattr(cfg.model, "enc_depth", 4),
            enc_num_heads=getattr(cfg.model, "enc_num_heads", 4),
            enc_mlp_ratio=getattr(cfg.model, "enc_mlp_ratio", 4.0),
            dec_dim=getattr(cfg.model, "dec_embed_dim", 128),
            dec_depth=getattr(cfg.model, "dec_depth", 4),
            dec_num_heads=getattr(cfg.model, "dec_num_heads", 4),
            dec_mlp_ratio=getattr(cfg.model, "dec_mlp_ratio", 4.0),
            class_depth=getattr(cfg.model, "class_depth", 2),
            dropout=getattr(cfg.model, "dropout_rate", 0.1),
            num_classes=getattr(cfg.model, "num_classes", 2),
        )
    
