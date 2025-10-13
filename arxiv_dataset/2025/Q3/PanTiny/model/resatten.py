#!/usr/bin/env python
# coding=utf-8
"""
PanFlowV2: Flexible version of PanFlow with configurable parameters
@Description: Enhanced PanFlow implementation with adjustable model size
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from math import exp


def initialize_weights(net_l, scale=1):
    """Initialize network weights"""
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def mean_channels(F):
    """Calculate mean across spatial dimensions"""
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    """Calculate standard deviation across spatial dimensions"""
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class HinResBlock(nn.Module):
    """Configurable HinResBlock with adjustable feature size"""
    def __init__(self, channel_in, channel_out, feature_dim=64):
        super(HinResBlock, self).__init__()
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv2d(channel_in, feature_dim, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature_dim + channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature_dim // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


class CALayer(nn.Module):
    """Channel Attention Layer with configurable reduction"""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block with configurable features"""
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def conv(self, in_channels, out_channels, kernel_size, bias=True):
        """Simple convolution layer"""
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    """Residual Group with configurable number of RCAB blocks"""
    def __init__(self, n_feat, kernel_size, reduction, act, res_scale, n_resblocks=20):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) 
                       for _ in range(n_resblocks)]
        modules_body.append(self.conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def conv(self, in_channels, out_channels, kernel_size, bias=True):
        """Simple convolution layer"""
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Net(nn.Module):
    """PanFlowV2: Configurable PanFlow network"""
    
    def __init__(self, args=None, num_channels=4, base_filter=None):
        super(Net, self).__init__()
        
        # Parse arguments with defaults
        if args is None:
            args = {}
        args = args.get('model', {}) # special
        
        # Handle different argument formats for compatibility  
        if hasattr(args, 'base_filter'):
            base_filter = args.base_filter
        elif hasattr(args, 'get'):
            base_filter = args.get('base_filter', 64)
        elif isinstance(args, dict) and 'base_filter' in args:
            base_filter = args['base_filter']
            
        if base_filter is None:
            base_filter = 64
        
        # Configuration parameters - more flexible than original
        self.size = getattr(args, 'size', 'medium') if hasattr(args, 'size') else args.get('size', 'medium')
        self.num_channels = getattr(args, 'num_channels', num_channels) if hasattr(args, 'num_channels') else args.get('num_channels', num_channels)
        self.base_filter = base_filter  # Store base_filter
        
        # Size configurations - much more conservative
        size_configs = {
            'small': {
                'n_feats': 16,         # Much reduced from 32
                'kernel_size': 3,
                'reduction': 8,        # Reduced from 16
                'n_resgroups': 1,      # Much reduced from 2
                'n_resblocks': 2,      # Much reduced from 5
                'feature_dim': 16,     # Much reduced from 32
                'middle_feats': 8      # Much reduced from 16
            },
            'medium': {  # Still original-ish but smaller
                'n_feats': 24,         # Much reduced from 64
                'kernel_size': 3,
                'reduction': 12,       # Reduced from 16
                'n_resgroups': 1,      # Much reduced from 3
                'n_resblocks': 3,      # Much reduced from 6
                'feature_dim': 24,     # Much reduced from 64
                'middle_feats': 12     # Much reduced from 32
            },
            'large': {
                'n_feats': 32,         # Much reduced from 96
                'kernel_size': 3,
                'reduction': 16,
                'n_resgroups': 2,      # Much reduced from 4
                'n_resblocks': 4,      # Much reduced from 8
                'feature_dim': 32,     # Much reduced from 96
                'middle_feats': 16     # Much reduced from 48
            },
            'custom': {
                # Allow custom configuration via args
                'n_feats': getattr(args, 'n_feats', 64) if hasattr(args, 'n_feats') else args.get('n_feats', 64),
                'kernel_size': getattr(args, 'kernel_size', 3) if hasattr(args, 'kernel_size') else args.get('kernel_size', 3),
                'reduction': getattr(args, 'reduction', 16) if hasattr(args, 'reduction') else args.get('reduction', 16),
                'n_resgroups': getattr(args, 'n_resgroups', 3) if hasattr(args, 'n_resgroups') else args.get('n_resgroups', 3),  # Reduced from 10
                'n_resblocks': getattr(args, 'n_resblocks', 6) if hasattr(args, 'n_resblocks') else args.get('n_resblocks', 6),  # Reduced from 20
                'feature_dim': getattr(args, 'feature_dim', 64) if hasattr(args, 'feature_dim') else args.get('feature_dim', 64),
                'middle_feats': getattr(args, 'middle_feats', 32) if hasattr(args, 'middle_feats') else args.get('middle_feats', 32),
            }
        }
        
        config = size_configs[self.size]
        
        # Extract configuration
        n_feats = config['n_feats']
        kernel_size = config['kernel_size']
        reduction = config['reduction']
        n_resgroups = config['n_resgroups']
        n_resblocks = config['n_resblocks']
        feature_dim = config['feature_dim']
        middle_feats = config['middle_feats']
        
        act = nn.ReLU(True)
        
        # Initialize layers
        # Head - shallow feature extraction
        self.head = nn.Conv2d(self.num_channels + 1, n_feats, kernel_size, padding=(kernel_size//2))
        
        # Body - deep feature extraction using residual groups
        modules_body = [ResidualGroup(n_feats, kernel_size, reduction, act, res_scale=1, n_resblocks=n_resblocks) 
                       for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))
        self.body = nn.Sequential(*modules_body)
        
        # Tail - reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, middle_feats, kernel_size, padding=(kernel_size//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_feats, self.num_channels, kernel_size, padding=(kernel_size//2))
        )
        
        # Alternative architecture using HinResBlocks for fusion
        self.fusion_blocks = nn.Sequential(
            HinResBlock(self.num_channels + 1, n_feats, feature_dim),
            HinResBlock(n_feats, n_feats, feature_dim),
            HinResBlock(n_feats, self.num_channels, feature_dim)
        )
        
        # Use fusion approach selector
        self.use_residual_groups = getattr(args, 'use_residual_groups', True) if hasattr(args, 'use_residual_groups') else args.get('use_residual_groups', True)
        
        # Initialize weights
        initialize_weights([self.head, self.body, self.tail, self.fusion_blocks], 0.1)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass
        Args:
            ms: multispectral image (low resolution) 
            bms: bicubic upsampled multispectral image
            pan: panchromatic image (high resolution)
        """
        # Concatenate bms and pan
        x = torch.cat([bms, pan], dim=1)
        
        if self.use_residual_groups:
            # Use residual group architecture
            x = self.head(x)
            res = self.body(x)
            res += x
            x = self.tail(res)
        else:
            # Use HinResBlock fusion architecture
            x = self.fusion_blocks(x)
        
        # Add residual connection with bms
        out = x + bms
        
        return out

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'PanFlowV2_{self.size}',
            'size': self.size,
            'num_channels': self.num_channels,
            'total_params': total_params,
            'use_residual_groups': self.use_residual_groups
        }


if __name__ == "__main__":
    # Test different configurations
    import argparse
    
    # Test small configuration
    args_small = {'size': 'small', 'num_channels': 4}
    model_small = Net(args_small)
    print(f"Small model info: {model_small.get_model_info()}")
    
    # Test medium configuration  
    args_medium = {'size': 'medium', 'num_channels': 4}
    model_medium = Net(args_medium)
    print(f"Medium model info: {model_medium.get_model_info()}")
    
    # Test large configuration
    args_large = {'size': 'large', 'num_channels': 4}
    model_large = Net(args_large)
    print(f"Large model info: {model_large.get_model_info()}")
    
    # Test with dummy data
    ms = torch.randn(1, 4, 64, 64)
    bms = torch.randn(1, 4, 256, 256)  
    pan = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        out = model_medium(ms, bms, pan)
        print(f"Output shape: {out.shape}")
