"""
ViT and SpectralFormer Inference Pipeline Utilities for Electrolyzers-HSI Dataset (built upon the original implementation of SpectralFormer)

This module provides the inference utilities of Vision Transformer (ViT) and SpectralFormer
architectures for hyperspectral image (HSI) classification on Electrolyzers-HSI dataset.
The code builds upon methods introduced in the SpectralFormer paper.

### Components:

1. **Transformer Building Blocks**:
   - `Residual`, `PreNorm`, `FeedForward`: Core transformer layer components.
   - `Attention`: Implements multi-head self-attention with optional masking.
   - `Transformer`: Composable transformer layer stack supporting both standard ViT and spectral-feature attention fusion (CAF mode).

2. **Vision Transformer (ViT) Model**:
   - `ViT`: End-to-end transformer-based model for patch-wise HSI classification using neighborhood spectral embedding.
   - Supports both standard ViT mode and CAF (Cross-Attention Fusion) mode for integrating spectral-spatial features.

3. **Dataset Wrapper**:
   - `HSIDataset`: Custom PyTorch Dataset for patch extraction from HSI cubes with optional class ignoring and neighborhood-aware band expansion.
   - `gain_neighborhood_band`: Expands the spectral band dimension using contextual neighborhood information, critical for SpectralFormer/VIT architectures.

### Notable Features:
- Position embeddings for patch-based transformer input.
- Learnable CLS token for classification.
- Neighborhood-aware spectral context augmentation via `gain_neighborhood_band`.
- Flexible transformer depth and attention configuration.
- Fully compatible with PyTorch data pipelines and GPU acceleration.

### Usage:
- Instantiate the `ViT` model with desired parameters (`image_size`, `num_patches`, etc.).
- Load `HSIDataset` with the appropriate patch size and band patch configuration.
- Use `gain_neighborhood_band` to prepare transformer-friendly patch inputs.
- Feed processed data into the model and use the CLS token output for classification.

### Dependencies:
- PyTorch
- NumPy
- `einops` (for tensor reshaping and repetition)

This module enables training and inference of transformer-based models for HSI classification, with support for spectral context modeling and performance 
evaluation on the Electrolyzers-HSI dataset.

Author: HiF-Explo: Elias Arbash
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x

class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask = None):
       
        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x) #[b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x = torch.cat((cls_tokens, x), dim = 1) #[b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:,0])

        # MLP classification layer
        return self.mlp_head(x)
    


class HSIDataset(Dataset):
    def __init__(self, hsi, gt, patch_size, times=1, fill_value_X=0, ignore_classes=[0], band_patch=1):
        if isinstance(hsi, list):
            self.hsi_list = hsi
            self.gt_list = gt
        else:
            self.hsi_list = [hsi]
            self.gt_list = [gt]

        self.patch_size = patch_size
        self.times = times
        self.fill_value_X = fill_value_X
        self.ignore_classes = ignore_classes
        self.band_patch = band_patch

        self.patches, self.labels = self.create_patches()
        self.labels = self.labels - 1
        self.original_patch_count = len(self.patches)  # Store original patch count



    def create_patches(self):
        hsi_patches = []
        labels = []

        # Calculate the center offset
        center_offset = self.patch_size // 2

        for hsi, mask in zip(self.hsi_list, self.gt_list):
            h, w, channels = hsi.shape
            pad_hsi = np.pad(hsi, ((center_offset, center_offset), (center_offset, center_offset), (0, 0)), mode='constant', constant_values=self.fill_value_X)
            pad_mask = np.pad(mask, ((center_offset, center_offset), (center_offset, center_offset)), mode='constant', constant_values=self.fill_value_X)

            for i in range(h):
                for j in range(w):
                    patch_hsi = pad_hsi[i:i+self.patch_size, j:j+self.patch_size, :]
                    label = mask[i, j]

                    if label not in self.ignore_classes:
                        hsi_patches.append(patch_hsi)
                        labels.append(label)

        hsi_patches = np.array(hsi_patches)
        labels = np.array(labels)

        return hsi_patches, labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Apply gain_neighborhood_band : this is for SpectralFormer (VIT, SF patch wise, SF Pixel wise) models, had to 
        # deploy the function inside the class for the sake of memory efficiency when creating big numbers of patches
        band = patch.shape[-1]  # The number of bands in the patch
        patch = gain_neighborhood_band(np.expand_dims(patch, axis=0), band, self.band_patch, self.patch_size).squeeze(0)
        # Transpose to get the desired shape [18, 243]
        patch = np.transpose(patch, ( 1, 0))

        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_original_patch_count(self):
        return self.original_patch_count


# Define the gain_neighborhood_band function
def gain_neighborhood_band(x_train, band, band_patch, patch):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    
    # Center region
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    
    # Left mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    
    # Right mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    
    return x_train_band


# Define the gain_neighborhood_band function
def gain_neighborhood_band(x_train, band, band_patch, patch):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    
    # Center region
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    
    # Left mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    
    # Right mirror
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    
    return x_train_band