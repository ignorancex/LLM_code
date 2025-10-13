#!/usr/bin/env python
# coding=utf-8
'''
Deep PNN (DeepPNN) for research experiments
Description: Deep PNN model with residual connections and configurable depth
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, use_skip=True):
        super(ResidualBlock, self).__init__()
        self.use_skip = use_skip and (in_channels == out_channels)
        
        padding = kernel_size // 2
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size, 1, padding,
            activation='relu', norm=None, bias=True
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size, 1, padding,
            activation=None, norm=None, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Channel adjustment for skip connection if needed
        if not self.use_skip and in_channels != out_channels:
            self.channel_adjust = ConvBlock(
                in_channels, out_channels, 1, 1, 0,
                activation=None, norm=None, bias=True
            )
        else:
            self.channel_adjust = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_skip:
            out += identity
        elif self.channel_adjust is not None:
            out += self.channel_adjust(identity)
        
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()
        
        self.args = args
        
        # Get model parameters from config
        model_config = args.get('model', {})
        
        # Support different model configurations
        model_size = model_config.get('size', 'deep_original')
        
        if model_size == 'deep_small':
            # ~80K parameters with depth
            self.hidden_dim = model_config.get('hidden_dim', 20)
            self.num_blocks = model_config.get('num_blocks', 10)
            self.kernel_size = model_config.get('kernel_size', 3)
        elif model_size == 'deep_medium':
            # ~180K parameters with depth
            self.hidden_dim = model_config.get('hidden_dim', 28)
            self.num_blocks = model_config.get('num_blocks', 12)
            self.kernel_size = model_config.get('kernel_size', 3)
        elif model_size == 'deep_large':
            # ~270K parameters with depth
            self.hidden_dim = model_config.get('hidden_dim', 32)
            self.num_blocks = model_config.get('num_blocks', 14)
            self.kernel_size = model_config.get('kernel_size', 3)
        elif model_size == 'deep_huge':
            # ~650K parameters with depth
            self.hidden_dim = model_config.get('hidden_dim', 44)
            self.num_blocks = model_config.get('num_blocks', 18)
            self.kernel_size = model_config.get('kernel_size', 3)
        elif model_size == 'deep_large_plus':
            # ~500K parameters with depth
            self.hidden_dim = model_config.get('hidden_dim', 80)
            self.num_blocks = model_config.get('num_blocks', 24)
            self.kernel_size = model_config.get('kernel_size', 3)
        elif model_size == 'deep_extreme':
            # ~1000K parameters with extreme depth
            self.hidden_dim = model_config.get('hidden_dim', 96)
            self.num_blocks = model_config.get('num_blocks', 32)
            self.kernel_size = model_config.get('kernel_size', 3)
        else:  # deep_original or any other value
            # ~80K+ parameters (original depth)
            self.hidden_dim = model_config.get('hidden_dim', 20)
            self.num_blocks = model_config.get('num_blocks', 10)
            self.kernel_size = model_config.get('kernel_size', 3)
        
        # Input channels: 4 (MS) + 1 (PAN) + 2 (NDVI, NDWI) = 7
        input_channels = 7
        output_channels = num_channels  # Should be 4 for MS
        
        # Network architecture
        # Initial feature extraction
        self.head = ConvBlock(
            input_channels, 
            self.hidden_dim, 
            7,  # Larger kernel for initial feature extraction
            1, 
            3,  # padding
            activation='relu', 
            norm=None, 
            bias=True
        )
        
        # Deep residual body
        self.body = nn.ModuleList()
        for i in range(self.num_blocks):
            # First block might change channels, rest maintain same channels
            if i == 0 and self.hidden_dim != self.hidden_dim:
                block = ResidualBlock(self.hidden_dim, self.hidden_dim, self.kernel_size, use_skip=False)
            else:
                block = ResidualBlock(self.hidden_dim, self.hidden_dim, self.kernel_size, use_skip=True)
            self.body.append(block)
        
        # Feature compression before output
        self.neck = ConvBlock(
            self.hidden_dim, 
            self.hidden_dim // 2, 
            1,  # 1x1 conv for channel reduction
            1, 
            0, 
            activation='relu', 
            norm=None, 
            bias=True
        )
        
        # Output layer
        self.output_conv = ConvBlock(
            self.hidden_dim // 2, 
            output_channels, 
            self.kernel_size, 
            1, 
            self.kernel_size // 2, 
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
        
        # NDWI = (Green - NIR) / (Green + NIR)
        # Assuming band order: Blue(0), Green(1), Red(2), NIR(3)
        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / 
                (l_ms[:, 1, :, :] + l_ms[:, 3, :, :] + eps)).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=self.args['data']['upsacle'], 
                           mode='bicubic', align_corners=False)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / 
                (l_ms[:, 3, :, :] + l_ms[:, 2, :, :] + eps)).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=self.args['data']['upsacle'], 
                           mode='bicubic', align_corners=False)
        
        # Concatenate all inputs
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        
        # Forward pass
        x_f = self.head(x_f)
        
        # Deep residual processing
        for block in self.body:
            x_f = block(x_f)
        
        # Final processing
        x_f = self.neck(x_f)
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
