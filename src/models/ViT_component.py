# models/VIT_component.py

"""
Vision Transformer Components
-----------------------------------
Modular components for building Vision Transformer models.
Includes Patch Embedding and Transformer Encoder Block.
-----------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Patch embedding
# ------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)                      # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x


# ------------------------------
# Transformer Block
# ------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        attn_out, attn_weights = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), 
            need_weights=return_attn
            )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

# ------------------------------
# Encoder Module
# ------------------------------
class ViTEncoder(nn.Module):
    def __init__(
            self, 
            img_size, 
            patch_size, 
            in_chans, 
            embed_dim, 
            depth, 
            num_heads, 
            mlp_ratio, 
            dropout
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

    def forward(self, x, return_attn=False):
        B = x.size(0)
        x = self.patch_embed(x)                       # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, N+1, D)
        x = x + self.pos_embed

        attn_list = []
        for blk in self.blocks:
            x, attn_weights = blk(x, return_attn=return_attn) 
            attn_list.append(attn_weights)

        x = self.norm(x)

        attn_torch = None
        if return_attn:
            attn_torch = torch.stack(attn_list, dim=0)  # (num_layers, B, N+1, N+1)

        return x, attn_torch

# ------------------------------
# Decoder Module 
# ------------------------------
class ViTDecoder(nn.Module):
    def __init__(
            self, 
            img_size, 
            patch_size,
            enc_dim, 
            dec_dim, 
            depth, 
            num_heads, 
            mlp_ratio, 
            dropout, 
            in_chans
            ):
        
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        self.decoder_embed = nn.Linear(enc_dim, dec_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Transformer Blocks 
        self.blocks = nn.ModuleList([
            TransformerBlock(dec_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dec_dim)

        self.decoder_pred = nn.Linear(dec_dim, (patch_size ** 2) * in_chans, bias=True)

    def forward(self, x, return_attn=False):
        # x: (B, N, enc_dim) -> (B, N, dec_dim)
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        attn_list = []
        for blk in self.blocks:
            x, attn_weights = blk(x, return_attn=return_attn)
            attn_list.append(attn_weights)

        x = self.norm(x)
        x = self.decoder_pred(x) # (B, N, P*P*C)
        
        attn_torch = None
        if return_attn:
            attn_torch = torch.stack(attn_list, dim=0)  # (num_layers, B, N, N)
            
        return x, attn_torch


# ------------------------------
# Classifier Encoder Module
# ------------------------------
class ViTClassifierEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            depth, 
            num_heads, 
            mlp_ratio, 
            dropout,
            num_classes
            ): 
        super().__init__()
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # Example: binary classification

    def forward(self, x, return_attn=False):

        attn_list = []
        for blk in self.blocks:
            x, attn_weights = blk(x, return_attn=return_attn) 
            attn_list.append(attn_weights)

        x = self.norm(x)
        x = self.head(x[:, 0])  # CLS token

        attn_torch = None
        if return_attn:
            attn_torch = torch.stack(attn_list, dim=0)  # (num_layers, B, N+1, N+1)

        return x, attn_torch



def patchify(imgs, patch_size):
    """
    imgs: (B, C, H, W)
    x: (B, N, patch_size**2 * C)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    in_chans = imgs.shape[1]

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * in_chans))
    return x

def unpatchify(x, patch_size, in_chans=1):
    """
    x: (B, N, patch_size**2 * in_chans)
    imgs: (B, in_chans, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, h * p))
    return imgs