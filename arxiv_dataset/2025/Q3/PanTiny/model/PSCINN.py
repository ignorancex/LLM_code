#!/usr/bin/env python
# coding=utf-8
"""
PSCINN module compatibility wrapper
@Description: Import wrapper for PSCINN.py to handle case sensitivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the actual PSCINN implementation
from .PSCINN import Net as PSCINNNet

# Create compatible Net class with base_filter parameter
class Net(PSCINNNet):
    """Wrapper for PSCINN with base_filter parameter support"""
    def __init__(self, args=None, num_channels=4, base_filter=None):
        # Handle different argument formats for compatibility
        if args is None:
            args = type('args', (), {})()
        elif isinstance(args, dict):
            # Convert dict to namespace
            args_dict = args
            args = type('args', (), {})()
            for key, value in args_dict.items():
                setattr(args, key, value)
            
        # Add base_filter to args if provided
        if base_filter is not None:
            args.base_filter = base_filter
        elif not hasattr(args, 'base_filter'):
            args.base_filter = 64  # Default value
            
        if num_channels is not None:
            args.num_channels = num_channels
        elif not hasattr(args, 'num_channels'):
            args.num_channels = 4  # Default value
            
        # Store for model info
        self.base_filter = getattr(args, 'base_filter', 64)
        self.num_channels = getattr(args, 'num_channels', 4)
        
        # Initialize parent class
        super(Net, self).__init__(args)
        
        # Add output projection layer for correct channel output
        self.output_proj = nn.Conv2d(64, 4, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
    def forward(self, ms, bms, pan, **kwargs):
        """
        Wrapper forward method to match expected interface
        Args:
            ms: low resolution multispectral image [B, 4, H//4, W//4]
            bms: bicubic upsampled multispectral image [B, 4, H, W]
            pan: panchromatic image [B, 1, H, W]
        Returns:
            Enhanced multispectral image [B, 4, H, W]
        """
        # Generate h_ms by upsampling ms
        h_ms = F.interpolate(ms, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Call parent forward with required arguments
        out = super(Net, self).forward(ms, bms, pan, h_ms, rev=False)
        
        # Project to correct number of channels (4)
        out = self.output_proj(out)
        
        # Ensure output has correct spatial dimensions to match bms
        if out.shape[2:] != bms.shape[2:]:
            out = F.interpolate(out, size=bms.shape[2:], mode='bilinear', align_corners=False)
            
        return out
        
    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'PSCINN_{self.base_filter}',
            'base_filter': self.base_filter,
            'num_channels': self.num_channels,
            'total_params': total_params
        }
