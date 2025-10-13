import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class EqualizedWeight(nn.Module):
    """Equalized learning rate weight scaling"""
    def __init__(self, shape, gain=math.sqrt(2)):
        super().__init__()
        fan_in = np.prod(shape[:-1]) if len(shape) > 1 else shape[0]
        self.scale = gain / math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.randn(shape))
    
    def forward(self):
        return self.weight * self.scale

class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 gain=math.sqrt(2), use_wscale=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.use_wscale = use_wscale
        
        if use_wscale:
            self.weight = EqualizedWeight([out_channels, in_channels, kernel_size, kernel_size], gain)
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            fan_in = in_channels * kernel_size * kernel_size
            std = gain / math.sqrt(fan_in)
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
            self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x):
        if self.use_wscale:
            weight = self.weight()
        else:
            weight = self.weight
            
        # Handle reflection padding for kernel > 1
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode='reflect')
            
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=0)

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    def __init__(self, in_features, out_features, gain=math.sqrt(2), use_wscale=True):
        super().__init__()
        self.use_wscale = use_wscale
        
        if use_wscale:
            self.weight = EqualizedWeight([out_features, in_features], gain)
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            std = gain / math.sqrt(in_features)
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
            self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        if self.use_wscale:
            weight = self.weight()
        else:
            weight = self.weight
            
        return F.linear(x, weight, self.bias)

class MinibatchStddev(nn.Module):
    """Minibatch standard deviation layer"""
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        group_size = min(self.group_size, batch_size)
        
        if group_size > 1:
            # [NCHW] -> [GMCHW]
            y = x.view(group_size, -1, channels, height, width)
            # Subtract mean over group
            y = y - y.mean(dim=0, keepdim=True)
            # Calculate variance over group
            y = y.pow(2).mean(dim=0)
            # Calculate stddev
            y = (y + 1e-8).sqrt()
            # Average over feature maps and pixels
            y = y.mean(dim=[1, 2, 3], keepdim=True)
            # Replicate over group and pixels
            y = y.repeat(group_size, 1, height, width)
            # Concatenate with input
            x = torch.cat([x, y], dim=1)
        
        return x

class GaussianBlur(nn.Module):
    """Gaussian blur for anti-aliasing"""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32) / 256.0
        
        # Create separate kernels for each channel
        self.register_buffer('kernel', kernel.view(1, 1, 5, 5))
    
    def forward(self, x):
        # Apply gaussian blur to each channel separately
        channels = []
        for i in range(x.size(1)):
            channel = x[:, i:i+1, :, :]
            # Reflection padding
            channel = F.pad(channel, [2, 2, 2, 2], mode='reflect')
            # Apply convolution
            blurred = F.conv2d(channel, self.kernel, padding=0)
            channels.append(blurred)
        
        return torch.cat(channels, dim=1)

class Yu2019AttributionClassifier(nn.Module):
    """Progressive Patch-based Classifier (PyTorch version of C_patch)"""
    
    def __init__(self,
                 num_channels=3,
                 resolution=128,
                 label_size=1000,
                 fmap_base=1024,
                 fmap_decay=1.0,
                 fmap_max=512,
                 latent_res=-1,
                 mode='postpool',
                 switching_res=4,
                 use_wscale=True,
                 mbstd_group_size=0,
                 fused_scale=False):
        super().__init__()
        
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.latent_res = latent_res
        self.mode = mode
        self.switching_res = switching_res
        self.use_wscale = use_wscale
        self.mbstd_group_size = mbstd_group_size
        self.fused_scale = fused_scale
        
        self.resolution_log2 = int(np.log2(resolution))
        self.latent_res_log2 = 2 if latent_res == -1 else int(np.log2(latent_res))
        self.switching_res_log2 = int(np.log2(switching_res))
        
        # Initialize gaussian blur for anti-aliasing
        self.gaussian_blur = GaussianBlur()
        
        # Initialize minibatch stddev if needed
        if mbstd_group_size > 1:
            self.minibatch_stddev = MinibatchStddev(mbstd_group_size)
        
        # Build the network blocks
        self.blocks = nn.ModuleDict()
        self._build_blocks()
    
    def nf(self, stage):
        """Calculate number of feature maps for a given stage"""
        # For progressive networks, higher resolution stages should have fewer channels
        # Stage 8 (256x256) should have base channels, stage 2 (4x4) should have max channels
        channels = min(int(self.fmap_base * (2.0 ** ((self.resolution_log2 - stage) * self.fmap_decay))), self.fmap_max)
        return max(channels, 1)  # Ensure at least 1 channel
    
    def _build_blocks(self):
        """Build all the network blocks"""
        for res in range(self.resolution_log2, self.latent_res_log2-1, -1):
            block_name = f'{2**res}x{2**res}'
            
            if res > self.latent_res_log2:
                # Intermediate blocks - create all of them
                # First layer: input channels are either num_channels (for first block) or previous block's output channels
                if res == self.resolution_log2:
                    # First block: RGB input
                    in_channels = self.num_channels
                else:
                    # Subsequent blocks: input comes from previous resolution's output
                    in_channels = self.nf(res + 1)  # Previous resolution's channels
                
                out_channels = self.nf(res)  # Current resolution's channels
                
                # Convolution blocks
                conv0 = EqualizedConv2d(
                    in_channels,
                    out_channels, 
                    kernel_size=3, 
                    padding=1,
                    use_wscale=self.use_wscale
                )
                
                if self.fused_scale:
                    # Fused conv + downscale (simplified as conv with stride 2)
                    conv1 = EqualizedConv2d(
                        out_channels,  # Input is output from conv0
                        out_channels,  # Keep same number of channels
                        kernel_size=3, 
                        stride=2,
                        padding=1,
                        use_wscale=self.use_wscale
                    )
                    self.blocks[f'{block_name}_conv0'] = conv0
                    self.blocks[f'{block_name}_conv1_down'] = conv1
                else:
                    conv1 = EqualizedConv2d(
                        out_channels,  # Input is output from conv0
                        out_channels,  # Keep same number of channels
                        kernel_size=3, 
                        padding=1,
                        use_wscale=self.use_wscale
                    )
                    self.blocks[f'{block_name}_conv0'] = conv0
                    self.blocks[f'{block_name}_conv1'] = conv1
                
            else:
                # Final classification block
                in_channels = self.nf(res)
                if self.mbstd_group_size > 1:
                    in_channels += 1  # +1 for minibatch stddev channel
                
                conv0 = EqualizedConv2d(
                    in_channels,
                    self.nf(res),  # Keep same number of channels
                    kernel_size=3, 
                    padding=1,
                    use_wscale=self.use_wscale
                )
                self.blocks[f'{block_name}_conv0'] = conv0
                
                if self.latent_res == -1:
                    # Fully connected layers
                    dense1 = EqualizedLinear(
                        self.nf(res) * (2**res) * (2**res),
                        self.nf(res),  # Keep same number of channels
                        use_wscale=self.use_wscale
                    )
                    dense2 = EqualizedLinear(
                        self.nf(res),
                        self.label_size,
                        gain=1,
                        use_wscale=self.use_wscale
                    )
                    self.blocks[f'{block_name}_dense1'] = dense1
                    self.blocks[f'{block_name}_dense2'] = dense2
                else:
                    # Fully convolutional layers
                    conv1 = EqualizedConv2d(
                        self.nf(res),
                        self.nf(res),  # Keep same number of channels
                        kernel_size=1,
                        use_wscale=self.use_wscale
                    )
                    conv2 = EqualizedConv2d(
                        self.nf(res),
                        self.label_size,
                        kernel_size=1,
                        gain=1,
                        use_wscale=self.use_wscale
                    )
                    self.blocks[f'{block_name}_conv1'] = conv1
                    self.blocks[f'{block_name}_conv2'] = conv2
    
    def forward(self, x):
        """Forward pass through the network"""
        # Process through all resolution blocks
        for res in range(self.resolution_log2, self.latent_res_log2-1, -1):
            x = self._forward_block(x, res)
        
        # Ensure output is properly shaped to match NaiveClassifier [B, 1]
        if x.dim() > 2:
            x = x.squeeze(-1).squeeze(-1)  # Remove spatial dimensions if present
        
        # Make sure output is [B, 1] not [B]
        if x.dim() == 1:
            x = x.unsqueeze(1)  # Add channel dimension to make [B, 1]
        
        return x
    
    def _forward_block(self, x, res):
        """Forward pass through a single resolution block"""
        block_name = f'{2**res}x{2**res}'
        
        if res > self.latent_res_log2:
            # Intermediate blocks - process all of them
            # Convolution first
            conv0 = self.blocks[f'{block_name}_conv0']
            x = F.leaky_relu(conv0(x), negative_slope=0.2)
            
            if self.fused_scale:
                # Fused convolution + downscale
                conv1_down = self.blocks[f'{block_name}_conv1_down']
                x = F.leaky_relu(conv1_down(x), negative_slope=0.2)
            else:
                # Separate convolution and downscale
                conv1 = self.blocks[f'{block_name}_conv1']
                x = F.leaky_relu(conv1(x), negative_slope=0.2)
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                    
        else:
            # Final classification block
            if self.mbstd_group_size > 1:
                x = self.minibatch_stddev(x)
            
            conv0 = self.blocks[f'{block_name}_conv0']
            x = F.leaky_relu(conv0(x), negative_slope=0.2)
            
            if self.latent_res == -1:
                # Fully connected path
                dense1 = self.blocks[f'{block_name}_dense1']
                dense2 = self.blocks[f'{block_name}_dense2']
                x = F.leaky_relu(dense1(x), negative_slope=0.2)
                x = dense2(x)
            else:
                # Fully convolutional path
                conv1 = self.blocks[f'{block_name}_conv1']
                conv2 = self.blocks[f'{block_name}_conv2']
                x = F.leaky_relu(conv1(x), negative_slope=0.2)
                x = conv2(x)
                # Global average pooling to get final classification
                x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        
        return x

# Example usage and testing
if __name__ == "__main__":
    # Create the classifier with the same config as the original
    classifier = Yu2019AttributionClassifier(
        num_channels=3,
        resolution=128,
        label_size=1000,  # ImageNet classes
        fmap_base=1024,
        fmap_max=512,
        latent_res=-1,  # Use fully connected layers
        mode='postpool',
        switching_res=4,
        use_wscale=True,
        mbstd_group_size=0,  # Disable for simplicity
        fused_scale=False
    )
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 128, 128)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = classifier(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Show network structure
    print(f"\nNetwork structure:")
    for name, module in classifier.named_modules():
        if isinstance(module, (EqualizedConv2d, EqualizedLinear)):
            print(f"  {name}: {module}")