import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pcdet.ops.basicblock.modules.modulated_deform_conv import ModulatedDeformConv 




class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, downsample=False,deformable_groups=1, ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.downsample = downsample
        if self.downsample:
            offset_mask_channels = 3 * 3 * (2+1)
            self.conv_offset_mask1 = nn.Conv2d(
            dim, 
            deformable_groups * offset_mask_channels, 
            kernel_size=3,
            stride=2, 
            padding=1,
            bias=True)
            self.down_layer = ModulatedDeformConv(
            dim,
            dim,
            stride=2,
            kernel_size=3,
            padding=1,
            deformable_groups=deformable_groups,
            bias=False)

    def forward(self, x):
        if self.downsample:
            offset_mask_1 = self.conv_offset_mask1(x)
            dcn1_o1, dcn1_o2, dcn1_mask = torch.chunk(offset_mask_1, 3, dim=1)
            dcn1_offset = torch.cat((dcn1_o1, dcn1_o2), dim=1)
            dcn1_mask = torch.sigmoid(dcn1_mask)
            x = self.down_layer(x, dcn1_offset, dcn1_mask)
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)
        x += identity

        return x
    
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
