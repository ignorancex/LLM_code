import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pcdet.ops.basicblock.modules.modulated_deform_conv import ModulatedDeformConv 


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=False,kernel_size=3, deformable_groups=1):
        super(BasicBlock, self).__init__()
        offset_mask_channels = kernel_size * kernel_size * (2+1)
        
        assert norm_fn is not None
        bias = norm_fn is not None
        self.downsample = downsample
        if self.downsample:
            self.down_layer = nn.Conv2d(inplanes, planes, 3, stride=2, padding=1, bias=bias)
        self.conv_offset_mask1 = nn.Conv2d(
            inplanes, 
            deformable_groups * offset_mask_channels, 
            kernel_size=kernel_size,
            stride=1, 
            padding=(kernel_size-1) // 2,
            bias=True)
        self.dcn1 = ModulatedDeformConv(
            inplanes,
            planes,
            stride=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups,
            bias=False)
        self.bn1 = norm_fn(planes)
        self.gelu = nn.GELU()
        
        self.conv_offset_mask2 = nn.Conv2d(
            inplanes, 
            deformable_groups * offset_mask_channels, 
            kernel_size=kernel_size,
            stride=1, 
            padding=(kernel_size-1) // 2,
            bias=True)
        self.dcn2 = ModulatedDeformConv(
            planes,
            planes,
            stride=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups,
            bias=False)
        self.bn2 = norm_fn(planes)
        self.stride = stride
        self.init_offset()

        
    def init_offset(self):
        self.conv_offset_mask1.weight.data.zero_()
        self.conv_offset_mask1.bias.data.zero_()
        
        self.conv_offset_mask2.weight.data.zero_()
        self.conv_offset_mask2.bias.data.zero_()
        

    def forward(self, x):
        if self.downsample:
            x = self.down_layer(x)
        identity = x
        offset_mask_1 = self.conv_offset_mask1(x)
        dcn1_o1, dcn1_o2, dcn1_mask = torch.chunk(offset_mask_1, 3, dim=1)
        dcn1_offset = torch.cat((dcn1_o1, dcn1_o2), dim=1)
        dcn1_mask = torch.sigmoid(dcn1_mask)
        out = self.dcn1(x, dcn1_offset, dcn1_mask)
        out = self.bn1(out)
        out = self.gelu(out)

        offset_mask_2 = self.conv_offset_mask2(out)
        dcn2_o1, dcn2_o2, dcn2_mask = torch.chunk(offset_mask_2, 3, dim=1)
        dcn2_offset = torch.cat((dcn2_o1, dcn2_o2), dim=1)
        dcn2_mask = torch.sigmoid(dcn2_mask)
        
        out = self.dcn2(out, dcn2_offset, dcn2_mask)
        out = self.bn2(out)
        out += identity
        out = self.gelu(out)

        return out