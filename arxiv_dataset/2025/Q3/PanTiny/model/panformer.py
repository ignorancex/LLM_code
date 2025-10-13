#!/usr/bin/env python
# coding=utf-8
"""
PanFormer for Pan-sharpening
Adapted from original PanFormer implementation with configurable parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.base_net import *


class SwinModule(nn.Module):
    """Memory-efficient Swin Transformer Module for PanFormer"""
    def __init__(self, in_channels, hidden_dimension, layers=2, downscaling_factor=1, 
                 num_heads=4, head_dim=16, window_size=4, relative_pos_embedding=True, 
                 cross_attn=False):
        super(SwinModule, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dimension = hidden_dimension
        self.layers = layers
        self.downscaling_factor = downscaling_factor
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.cross_attn = cross_attn
        
        # Input projection - memory efficient
        if downscaling_factor > 1:
            self.patch_partition = nn.Conv2d(in_channels, hidden_dimension, 
                                           kernel_size=downscaling_factor, 
                                           stride=downscaling_factor, 
                                           padding=0)
        else:
            self.patch_partition = nn.Conv2d(in_channels, hidden_dimension, 
                                           kernel_size=1, stride=1, padding=0)
        
        # Simplified transformer blocks - reduce layers for memory efficiency
        effective_layers = max(1, layers // 2) if cross_attn else layers
        self.transformer_blocks = nn.ModuleList()
        for _ in range(effective_layers):
            if cross_attn:
                self.transformer_blocks.append(MemoryEfficientCrossAttentionBlock(hidden_dimension, num_heads))
            else:
                self.transformer_blocks.append(MemoryEfficientSelfAttentionBlock(hidden_dimension, num_heads))
        
        # Output projection
        self.norm = nn.LayerNorm(hidden_dimension)
    
    def forward(self, x, cross_feat=None):
        # Input projection
        x = self.patch_partition(x)
        B, C, H, W = x.shape
        
        # For cross-attention, ensure feature size compatibility
        if self.cross_attn and cross_feat is not None:
            if cross_feat.shape[2:] != x.shape[2:]:
                cross_feat = F.interpolate(cross_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Reshape for transformer - memory efficient
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            if self.cross_attn and cross_feat is not None:
                cross_feat_flat = cross_feat.flatten(2).transpose(1, 2)
                x = block(x, cross_feat_flat)
            else:
                x = block(x)
        
        x = self.norm(x)
        
        # Reshape back
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


class MemoryEfficientSelfAttentionBlock(nn.Module):
    """Memory-efficient self-attention block"""
    def __init__(self, dim, num_heads):
        super(MemoryEfficientSelfAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = min(num_heads, dim // 8)  # Ensure reasonable head dimension
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, self.num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        # Reduced FFN expansion for memory efficiency
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Reduced from 4x to 2x
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)  # Don't return attention weights
        x = shortcut + x
        
        # FFN with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x
        
        return x


class MemoryEfficientCrossAttentionBlock(nn.Module):
    """Memory-efficient cross-attention block"""
    def __init__(self, dim, num_heads):
        super(MemoryEfficientCrossAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = min(num_heads, dim // 8)  # Ensure reasonable head dimension
        
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, self.num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        # Reduced FFN expansion for memory efficiency
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Reduced from 4x to 2x
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x, cross_feat):
        # Cross-attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x, _ = self.cross_attn(x, cross_feat, cross_feat, need_weights=False)  # Don't return attention weights
        x = shortcut + x
        
        # FFN with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x
        
        return x


class CrossSwinTransformer(nn.Module):
    """Memory-efficient Cross Swin Transformer for PanFormer"""
    def __init__(self, num_channels=4, n_feats=64, n_heads=4, head_dim=16, 
                 win_size=4, n_blocks=3, cross_module=['pan', 'ms'], 
                 cat_feat=['pan', 'ms'], sa_fusion=False):
        super(CrossSwinTransformer, self).__init__()
        
        self.num_channels = num_channels
        self.n_blocks = min(n_blocks, 2)  # Limit blocks for memory efficiency
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion
        
        # Simplified PAN encoder - no downscaling to match MS size
        pan_encoder = [
            SwinModule(in_channels=1, hidden_dimension=n_feats, layers=1,
                      downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                      window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        
        # Simplified MS encoder
        ms_encoder = [
            SwinModule(in_channels=num_channels, hidden_dimension=n_feats, layers=1,
                      downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                      window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        
        # Cross-attention modules - simplified
        if 'ms' in self.cross_module:
            self.ms_cross_pan = nn.ModuleList()
            for _ in range(self.n_blocks):
                self.ms_cross_pan.append(
                    SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=1,
                              downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                              window_size=win_size, relative_pos_embedding=True, cross_attn=True)
                )
        
        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(self.n_blocks):
                self.pan_cross_ms.append(
                    SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=1,
                              downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                              window_size=win_size, relative_pos_embedding=True, cross_attn=True)
                )
        
        # Memory-efficient HR reconstruction tail
        concat_feats = n_feats * len(cat_feat)
        self.HR_tail = nn.Sequential(
            # First upsampling stage
            ConvBlock(concat_feats, n_feats * 4, 3, 1, 1, activation='relu'),
            nn.PixelShuffle(2),  # n_feats
            # Second upsampling stage  
            ConvBlock(n_feats, n_feats * 4, 3, 1, 1, activation='relu'),
            nn.PixelShuffle(2),  # n_feats
            # Final refinement
            ConvBlock(n_feats, n_feats, 3, 1, 1, activation='relu'),
            ConvBlock(n_feats, num_channels, 3, 1, 1, activation=None)
        )
        
        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)
        
        # Upsampling layer to match PAN and MS resolution
        self.ms_upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
    
    def forward(self, pan, ms):
        # Upsample MS to match PAN resolution
        ms_upsampled = self.ms_upsample(ms)
        
        # Encode features
        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms_upsampled)
        
        # Ensure feature map sizes match
        if pan_feat.shape[2:] != ms_feat.shape[2:]:
            ms_feat = F.interpolate(ms_feat, size=pan_feat.shape[2:], mode='bilinear', align_corners=False)
        
        last_pan_feat = pan_feat
        last_ms_feat = ms_feat
        
        # Cross-attention blocks
        for i in range(self.n_blocks):
            if 'pan' in self.cross_module:
                pan_cross_ms_feat = self.pan_cross_ms[i](last_pan_feat, last_ms_feat)
                last_pan_feat = pan_cross_ms_feat
            if 'ms' in self.cross_module:
                ms_cross_pan_feat = self.ms_cross_pan[i](last_ms_feat, last_pan_feat)
                last_ms_feat = ms_cross_pan_feat
        
        # Concatenate features
        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(last_pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(last_ms_feat)
        
        # HR reconstruction
        output = self.HR_tail(torch.cat(cat_list, dim=1))
        
        return output


class Net(nn.Module):
    """PanFormer Network with configurable parameters"""
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()
        
        self.args = args
        self.num_channels = num_channels
        
        # Get model configuration
        model_config = args.get('model', {})
        
        # Support different model sizes through base_filter or size parameter
        if 'base_filter' in model_config:
            base_filter = model_config['base_filter']
            # Map base_filter to n_feats for panformer
            if base_filter <= 32:
                n_feats = 32
                n_heads = 2
                n_blocks = 2
            elif base_filter <= 48:
                n_feats = 48
                n_heads = 3
                n_blocks = 3
            elif base_filter <= 64:
                n_feats = 64
                n_heads = 4
                n_blocks = 3
            elif base_filter <= 96:
                n_feats = 96
                n_heads = 6
                n_blocks = 4
            else:
                n_feats = 128
                n_heads = 8
                n_blocks = 4
        else:
            model_size = model_config.get('size', 'original')
            
            if model_size == 'tiny':
                # ~60K parameters equivalent
                n_feats = 32
                n_heads = 2
                n_blocks = 2
            elif model_size == 'small':
                # ~180K parameters equivalent
                n_feats = 48
                n_heads = 3
                n_blocks = 3
            elif model_size == 'large':
                # ~300K parameters equivalent
                n_feats = 64
                n_heads = 4
                n_blocks = 3
            else:
                # Original size
                n_feats = 64
                n_heads = 4
                n_blocks = 3
        
        # Override with specific parameters if provided
        n_feats = model_config.get('n_feats', n_feats)
        n_heads = model_config.get('n_heads', n_heads)
        n_blocks = model_config.get('n_blocks', n_blocks)
        head_dim = model_config.get('head_dim', 16)
        win_size = model_config.get('win_size', 4)
        
        # Cross attention configuration
        cross_module = model_config.get('cross_module', ['pan', 'ms'])
        cat_feat = model_config.get('cat_feat', ['pan', 'ms'])
        sa_fusion = model_config.get('sa_fusion', False)
        
        # Initialize the core transformer
        self.panformer = CrossSwinTransformer(
            num_channels=num_channels,
            n_feats=n_feats,
            n_heads=n_heads,
            head_dim=head_dim,
            win_size=win_size,
            n_blocks=n_blocks,
            cross_module=cross_module,
            cat_feat=cat_feat,
            sa_fusion=sa_fusion
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, l_ms, b_ms, x_pan):
        """
        Forward pass for PanFormer
        Args:
            l_ms: Low-resolution MS image
            b_ms: Bicubic upsampled MS image (not used directly in PanFormer)
            x_pan: PAN image
        """
        # PanFormer uses l_ms and x_pan directly
        output = self.panformer(x_pan, l_ms)
        
        return output
    
    def get_parameter_count(self):
        """Get the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


if __name__ == "__main__":
    # Test the model
    args = {
        'model': {
            'size': 'large',
            'n_feats': 64,
            'n_heads': 4,
            'n_blocks': 3
        },
        'data': {'upsacle': 4}
    }
    
    model = Net(num_channels=4, base_filter=64, args=args)
    print(f"Parameters: {model.get_parameter_count():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    # Test forward pass
    l_ms = torch.randn(1, 4, 32, 32)
    b_ms = torch.randn(1, 4, 128, 128)
    x_pan = torch.randn(1, 1, 128, 128)
    
    output = model(l_ms, b_ms, x_pan)
    print(f"Output shape: {output.shape}")

