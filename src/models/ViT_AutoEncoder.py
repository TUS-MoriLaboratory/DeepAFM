# models/ViT_AutoEncoder.py
"""
Vision Transformer AutoEncoder
-----------------------------------
Implements a Vision Transformer-based AutoEncoder model.
Includes encoder and decoder with attention map extraction capability.
-----------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.ViT_component import ViTEncoder, ViTDecoder, unpatchify

# ------------------------------
# Vision Transformer AutoEncoder
# ------------------------------
class ViTAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_chans=1,
        enc_dim=128,
        dec_dim=128,
        enc_depth=4,
        dec_depth=4,
        enc_num_heads=4,
        dec_num_heads=4,
        enc_mlp_ratio=4.0,
        dec_mlp_ratio=4.0,
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

        # Store parameters for patchify/unpatchify
        self.patch_size = patch_size
        self.in_chans = in_chans

    def forward(self, x, return_attn=False):
        enc_out, enc_attn = self.encoder(x, return_attn=return_attn) # (B, N+1, D)
        enc_out = enc_out[:, 1:, :]  # remove CLS token
        dec_out, dec_attn = self.decoder(enc_out, return_attn=return_attn) # (B, N, P*P*C)
        
        x = unpatchify(dec_out, self.patch_size, in_chans=self.in_chans)  # (B, C, H, W)

        if return_attn:
            return x, {"enc_attn": enc_attn, "dec_attn": dec_attn}
        else:
            return x

    @staticmethod
    def from_config(cfg) -> "ViTAutoEncoder":
        """Factory from ExperimentConfig"""
        return ViTAutoEncoder(
            img_size=cfg.model.image_size,
            patch_size=getattr(cfg.model, "patch_size", 4),
            in_chans=cfg.model.in_channels,
            enc_dim=getattr(cfg.model, "enc_embed_dim", 128),
            dec_dim=getattr(cfg.model, "dec_embed_dim", 128),
            enc_depth=getattr(cfg.model, "enc_depth", 4),
            dec_depth=getattr(cfg.model, "dec_depth", 4),
            enc_num_heads=getattr(cfg.model, "enc_num_heads", 4),
            dec_num_heads=getattr(cfg.model, "dec_num_heads", 4),
            enc_mlp_ratio=getattr(cfg.model, "enc_mlp_ratio", 4.0),
            dec_mlp_ratio=getattr(cfg.model, "dec_mlp_ratio", 4.0),
            dropout=getattr(cfg.model, "dropout_rate", 0.1),
        )