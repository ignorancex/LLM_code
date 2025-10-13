#!/usr/bin/env python
# coding=utf-8
'''
Enhanced PNN (PNNv2) for research experiments
Description: Configurable PNN model with parameter control from config
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()
        
        self.args = args
        
        # Get model parameters from config
        model_config = args.get('model', {})
        
        # Support base_filter parameter (unified interface)
        if 'base_filter' in model_config:
            base_filter = model_config['base_filter']
            # Map base_filter to model size for backward compatibility
            if base_filter <= 32:
                model_size = 'small'
            elif base_filter <= 48:
                model_size = 'original'
            elif base_filter <= 64:
                model_size = 'large'
            elif base_filter <= 96:
                model_size = 'huge'
            else:
                model_size = 'huge'
        else:
            model_size = model_config.get('size', 'original')
        
        # Support different model sizes
        if model_size == 'small':
            # ~30K parameters
            self.hidden_dim1 = model_config.get('hidden_dim1', 24)
            self.hidden_dim2 = model_config.get('hidden_dim2', 16)
            self.kernel_size1 = model_config.get('kernel_size1', 7)
            self.kernel_size2 = model_config.get('kernel_size2', 3)
            self.kernel_size3 = model_config.get('kernel_size3', 3)
        elif model_size == 'bigger':
            # ~150K parameters
            self.hidden_dim1 = model_config.get('hidden_dim1', 64)
            self.hidden_dim2 = model_config.get('hidden_dim2', 48)
            self.kernel_size1 = model_config.get('kernel_size1', 9)
            self.kernel_size2 = model_config.get('kernel_size2', 7)
            self.kernel_size3 = model_config.get('kernel_size3', 5)
        elif model_size == 'large':
            # ~300K parameters
            self.hidden_dim1 = model_config.get('hidden_dim1', 96)
            self.hidden_dim2 = model_config.get('hidden_dim2', 64)
            self.kernel_size1 = model_config.get('kernel_size1', 11)
            self.kernel_size2 = model_config.get('kernel_size2', 7)
            self.kernel_size3 = model_config.get('kernel_size3', 5)
        elif model_size == 'huge':
            # ~1M+ parameters
            self.hidden_dim1 = model_config.get('hidden_dim1', 128)
            self.hidden_dim2 = model_config.get('hidden_dim2', 96)
            self.kernel_size1 = model_config.get('kernel_size1', 13)
            self.kernel_size2 = model_config.get('kernel_size2', 9)
            self.kernel_size3 = model_config.get('kernel_size3', 7)
        else:  # original or any other value
            # ~60K+ parameters (original)
            self.hidden_dim1 = model_config.get('hidden_dim1', 48)
            self.hidden_dim2 = model_config.get('hidden_dim2', 32)
            self.kernel_size1 = model_config.get('kernel_size1', 9)
            self.kernel_size2 = model_config.get('kernel_size2', 5)
            self.kernel_size3 = model_config.get('kernel_size3', 5)
        
        # Input channels: 4 (MS) + 1 (PAN) + 2 (NDVI, NDWI) = 7
        input_channels = 7
        output_channels = num_channels  # Should be 4 for MS
        
        # Network architecture
        self.head = ConvBlock(
            input_channels, 
            self.hidden_dim1, 
            self.kernel_size1, 
            1, 
            self.kernel_size1//2, 
            activation='relu', 
            norm=None, 
            bias=True
        )
        
        self.body = ConvBlock(
            self.hidden_dim1, 
            self.hidden_dim2, 
            self.kernel_size2, 
            1, 
            self.kernel_size2//2, 
            activation='relu', 
            norm=None, 
            bias=True
        )
        
        self.output_conv = ConvBlock(
            self.hidden_dim2, 
            output_channels, 
            self.kernel_size3, 
            1, 
            self.kernel_size3//2, 
            activation=None,  # No activation for output layer
            norm=None, 
            bias=True
        )
        
        # Initialize weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, l_ms, b_ms, x_pan):
        """
        Args:
            l_ms: Low-resolution MS image
            b_ms: Bicubic upsampled MS image 
            x_pan: PAN image
        """
        # Calculate spectral indices
        eps = 1e-8
        
        # Get upscale factor - handle both 'upscale' and 'upsacle' (typo in config files)
        upscale_factor = self.args['data'].get('upscale', self.args['data'].get('upsacle', 4))
        
        # NDWI = (Green - NIR) / (Green + NIR)
        # Assuming band order: Blue(0), Green(1), Red(2), NIR(3)
        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / 
                (l_ms[:, 1, :, :] + l_ms[:, 3, :, :] + eps)).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=upscale_factor, 
                           mode='bicubic', align_corners=False)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / 
                (l_ms[:, 3, :, :] + l_ms[:, 2, :, :] + eps)).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=upscale_factor, 
                           mode='bicubic', align_corners=False)
        
        # Concatenate all inputs
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        
        # Forward pass
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)
        
        return x_f
    
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
