# models/ViT_model.py
"""
Custom Vision Transformer (from scratch)
-----------------------------------
Minimal ViT implementation compatible with Trainer.
Easily extendable for Attention visualization (Flow/GradCAM).
-----------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.ViT_component import PatchEmbedding, TransformerBlock

# ------------------------------
# Vision Transformer (custom)
# ------------------------------
class CustomViT(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_chans=1,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_attn=False):
        B = x.size(0)
        x = self.patch_embed(x)                       # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, N+1, D)
        x = x + self.pos_embed

        attn_list = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_list.append(attn)

        x = self.norm(x)
        cls_out = x[:, 0]                              # CLS token
        logits = self.head(cls_out)

        if return_attn:
            attn_torch = torch.stack(attn_list, dim=0)  # (num_layers, B, N+1, N+1)
            return logits, attn_torch
        else:
            return logits

    @staticmethod
    def from_config(cfg) -> "CustomViT":
        """Factory from ExperimentConfig"""
        return CustomViT(
            img_size=cfg.model.image_size,
            patch_size=getattr(cfg.model, "patch_size", 4),
            in_chans=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            embed_dim=getattr(cfg.model, "embed_dim", 128),
            depth=getattr(cfg.model, "depth", 4),
            num_heads=getattr(cfg.model, "num_heads", 4),
            mlp_ratio=getattr(cfg.model, "mlp_ratio", 4.0),
            dropout=getattr(cfg.model, "dropout", 0.1),
        )


if __name__ == "__main__":
    model = CustomViT(img_size=28, patch_size=4, in_chans=1, num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    y, attn = model(x, return_attn=True)
    print("Output:", y.shape)
    print("Attention:", [a.shape for a in attn])
