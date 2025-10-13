#!/usr/bin/env python
# coding=utf-8
"""
PanNet V2 - Scalable version of PanNet with configurable model sizes
@Description: Enhanced PanNet implementation supporting different parameter counts (60K, 180K, 300K)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
import numpy as np


class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        fft_x = torch.fft.fftshift(fft_x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        Mag = torch.nn.functional.pad(Mag, (W // 2, W // 2, H // 2, H // 2))
        Pha = torch.nn.functional.pad(Pha, (W // 2, W // 2, H // 2, H // 2))

        real = Mag * torch.cos(Pha)
        imag = Mag * torch.sin(Pha)
        out = torch.complex(real, imag)

        out = torch.fft.ifftshift(out)
        output = torch.fft.ifft2(out)
        output = self.post(torch.abs(output))

        return output


def get_D_map_optimized(feature):
    B, C, H, W = feature.shape
    d_map = torch.zeros((1, 1, H, W), dtype=torch.float32).to(feature.device)
    
    #Create a grid to store the indices of all (i, j) pairs
    i_indices = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1).repeat(1, 1, 1, W).cuda()
    j_indices = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W).repeat(1, 1, H, 1).cuda()
    
    # Compute d_map using vectorization operations
    d_map[:, :, :, :] = calculate_d(i_indices, j_indices, H, W)
    
    return d_map


def calculate_d(x, y, M, N):
    term1 = 1
    term2 = torch.exp(1j * torch.tensor(np.pi) * x / M).cuda()
    term3 = torch.exp(1j * torch.tensor(np.pi) * y / N).cuda()
    term4 = torch.exp(1j * torch.tensor(np.pi) * (x/M + y/N)).cuda()

    result = term1 + term2 + term3 + term4
    return torch.abs(result) / 4


class freup_inter(nn.Module):
    def __init__(self, channels):
        super(freup_inter, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        
        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        
        output = torch.fft.ifft2(out)
        output = torch.abs(output)
        
        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H/2), 0:int(W/2)] = output[:, :, 0:int(H/2), 0:int(W/2)]
        crop[:, :, int(H/2):H, 0:int(W/2)] = output[:, :, int(H*1.5):2*H, 0:int(W/2)]
        crop[:, :, 0:int(H/2), int(W/2):W] = output[:, :, 0:int(H/2), int(W*1.5):2*W]
        crop[:, :, int(H/2):H, int(W/2):W] = output[:, :, int(H*1.5):2*H, int(W*1.5):2*W]

        return self.post(crop)


class freup_pad(nn.Module):
    def __init__(self, channels):
        super(freup_pad, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        Mag = torch.nn.functional.pad(Mag, (W//2, W//2, H//2, H//2))
        Pha = torch.nn.functional.pad(Pha, (W//2, W//2, H//2, H//2))

        real = Mag * torch.cos(Pha)
        imag = Mag * torch.sin(Pha)
        out = torch.complex(real, imag)
        
        output = torch.fft.ifft2(out)
        output = torch.abs(output)
        
        return self.post(output)


class fresadd(nn.Module):
    def __init__(self, channels=32, intermediate_channels=48):
        super(fresadd, self).__init__()

        self.opspa = ConvBlock(intermediate_channels, channels, 5, 1, 2, activation=None, norm=None, bias=False)
        self.opfre = freup_pad(channels)

        self.fuse1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.fuse2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.fuse = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')

        x1 = self.opspa(x1)
        x2 = self.opspa(x2)
        x3 = self.opspa(x3)

        x3f = self.opfre(x3)
        x3s = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]), mode='bilinear')
        x32 = self.fuse1(x3f + x3s)

        x2m = x2 + x32

        x2f = self.opfre(x2m)
        x2s = F.interpolate(x2m, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x21 = self.fuse2(x2f + x2s)

        x1m = x1 + x21
        x = self.fuse(x1m)

        return x


class frescat(nn.Module):
    def __init__(self, channels=32, intermediate_channels=48):
        super(frescat, self).__init__()

        self.opspa = ConvBlock(intermediate_channels, channels, 5, 1, 2, activation=None, norm=None, bias=False)
        self.opfre = freup_Cornerdinterpolation(channels)

        self.fuse1 = nn.Conv2d(2*channels, channels, 1, 1, 0)
        self.fuse2 = nn.Conv2d(2*channels, channels, 1, 1, 0)
        self.fuse = nn.Conv2d(2*channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')

        x1 = self.opspa(x1)
        x2 = self.opspa(x2)
        x3 = self.opspa(x3)

        x3f = self.opfre(x3)
        x3s = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]), mode='bilinear')
        x32 = self.fuse1(torch.cat([x3f, x3s], dim=1))

        x2m = x2 + x32

        x2f = self.opfre(x2m)
        x2s = F.interpolate(x2m, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x21 = self.fuse2(torch.cat([x2f, x2s], dim=1))

        x = self.fuse(torch.cat([x1, x21], dim=1))

        return x


# Model size configurations
MODEL_CONFIGS = {
    'tiny': {
        'head_channels': 24,
        'body_channels': 16,
        'head_kernel': 7,
        'body_kernel': 3,
        'target_params': '60K'
    },
    'small': {
        'head_channels': 32,
        'body_channels': 24,
        'head_kernel': 7,
        'body_kernel': 5,
        'target_params': '180K'
    },
    'large': {
        'head_channels': 48,
        'body_channels': 32,
        'head_kernel': 9,
        'body_kernel': 5,
        'target_params': '300K'
    },
    'huge': {
        'head_channels': 64,
        'body_channels': 48,
        'head_kernel': 11,
        'body_kernel': 7,
        'target_params': '500K'
    }
}


class Net(nn.Module):
    def __init__(self, num_channels=4, base_filter=64, args=None):
        super(Net, self).__init__()
        
        self.args = args if args is not None else {}
        
        # Get model configuration
        if isinstance(args, dict) and 'model' in args:
            model_config = args['model']
            size = model_config.get('size', 'large')
            
            # Override with specific parameters if provided
            head_channels = model_config.get('head_channels', MODEL_CONFIGS[size]['head_channels'])
            body_channels = model_config.get('body_channels', MODEL_CONFIGS[size]['body_channels'])
            head_kernel = model_config.get('head_kernel', MODEL_CONFIGS[size]['head_kernel'])
            body_kernel = model_config.get('body_kernel', MODEL_CONFIGS[size]['body_kernel'])
        else:
            # Default to large configuration
            size = 'large'
            head_channels = MODEL_CONFIGS[size]['head_channels']
            body_channels = MODEL_CONFIGS[size]['body_channels']
            head_kernel = MODEL_CONFIGS[size]['head_kernel']
            body_kernel = MODEL_CONFIGS[size]['body_kernel']
        
        # Calculate input channels (MS + PAN + NDVI + NDWI)
        input_channels = num_channels + 1 + 2  # 4 MS + 1 PAN + 2 indices = 7
        out_channels = num_channels  # 4 MS channels
        
        # Network layers
        head_padding = head_kernel // 2
        self.head = ConvBlock(
            input_channels, head_channels, head_kernel, 1, head_padding, 
            activation='relu', norm=None, bias=False
        )
        
        # Use frescat for better performance (concatenation version)
        self.body = frescat(channels=body_channels, intermediate_channels=head_channels)
        
        body_padding = body_kernel // 2
        self.output_conv = ConvBlock(
            body_channels, out_channels, body_kernel, 1, body_padding,
            activation=None, norm=None, bias=False
        )
        
        # Store configuration for analysis
        self.config = {
            'size': size,
            'head_channels': head_channels,
            'body_channels': body_channels,
            'head_kernel': head_kernel,
            'body_kernel': body_kernel,
            'input_channels': input_channels,
            'output_channels': out_channels,
            'target_params': MODEL_CONFIGS[size]['target_params']
        }
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
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
        Forward pass
        Args:
            l_ms: Low resolution multispectral image
            b_ms: Bicubic upsampled multispectral image (not used directly)
            x_pan: Panchromatic image
        """
        # Upscale low resolution MS image
        upsacle_factor = self.args.get('data', {}).get('upsacle', 4)
        b_ms = F.interpolate(l_ms, scale_factor=upsacle_factor, mode='bicubic')
        
        # Calculate spectral indices
        # NDWI = (Green - NIR) / (Green + NIR) for water detection
        # Assuming channel order: [Blue, Green, Red, NIR] (typical for WV2/WV3)
        if l_ms.size(1) >= 4:
            # Standard 4-band case
            green_idx, nir_idx = 1, 3
            red_idx = 2
        else:
            # Fallback for different channel counts
            green_idx, red_idx, nir_idx = 0, 1, min(2, l_ms.size(1)-1)
        
        # NDWI calculation with numerical stability
        green = l_ms[:, green_idx, :, :]
        nir = l_ms[:, nir_idx, :, :]
        eps = 1e-8
        NDWI = ((green - nir) / (green + nir + eps)).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=upsacle_factor, mode='bicubic')
        
        # NDVI calculation with numerical stability
        red = l_ms[:, red_idx, :, :]
        NDVI = ((nir - red) / (nir + red + eps)).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=upsacle_factor, mode='bicubic')
        
        # Concatenate all inputs: [MS_upsampled, PAN, NDVI, NDWI]
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        
        # Forward through network
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)
        
        # Residual connection with upsampled MS
        x_f = torch.add(x_f, b_ms)
        
        return x_f
    
    def get_model_info(self):
        """Get model configuration and parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'PanNetV2',
            'configuration': self.config,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        return info
    
    def print_model_info(self):
        """Print detailed model information"""
        info = self.get_model_info()
        print("\n" + "="*60)
        print(f"Model: {info['model_name']}")
        print("="*60)
        print(f"Size Configuration: {info['configuration']['size']}")
        print(f"Target Parameters: {info['configuration']['target_params']}")
        print(f"Actual Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print("\nArchitecture Details:")
        print(f"  Head Channels: {info['configuration']['head_channels']}")
        print(f"  Body Channels: {info['configuration']['body_channels']}")
        print(f"  Head Kernel: {info['configuration']['head_kernel']}")
        print(f"  Body Kernel: {info['configuration']['body_kernel']}")
        print(f"  Input Channels: {info['configuration']['input_channels']}")
        print(f"  Output Channels: {info['configuration']['output_channels']}")
        print("="*60)


def test_model_sizes():
    """Test different model configurations"""
    print("Testing PanNetV2 configurations...")
    
    for size in ['tiny', 'small', 'large']:
        args = {'model': {'size': size}, 'data': {'upsacle': 4}}
        model = Net(num_channels=4, args=args)
        model.print_model_info()
        
        # Test forward pass
        with torch.no_grad():
            l_ms = torch.randn(1, 4, 32, 32)
            x_pan = torch.randn(1, 1, 128, 128)
            b_ms = torch.randn(1, 4, 128, 128)
            
            output = model(l_ms, b_ms, x_pan)
            print(f"Output shape: {output.shape}")
            print()


if __name__ == "__main__":
    test_model_sizes()
