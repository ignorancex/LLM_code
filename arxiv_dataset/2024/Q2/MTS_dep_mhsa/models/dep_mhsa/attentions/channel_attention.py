import torch
import torch.nn as nn
from typing import Tuple

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding='same', bias=bias, stride = stride)

class Conv2DPlus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding:int = 1) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )
    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride
    
## Channel Attention Layer
class CALayer3D(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer3D, self).__init__()
        midplanes = (channel * channel * 3 * 3 * 3) // (channel * 3 * 3 + 3 * channel)
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                Conv2DPlus1D(in_planes=channel, out_planes=channel // reduction, midplanes=midplanes,stride= 1, padding=1),
                nn.ReLU(inplace=True),
                Conv2DPlus1D(channel // reduction, channel, midplanes=midplanes,stride= 1, padding=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        # batch , channel = 256, frame = 3, h=w=16
        y = self.conv_du(x)
        y = self.avg_pool(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)

class CAB3D(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(CAB3D, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        # act -> ReLU, GeLU, PReLU
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer3D(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) # torch.Size([64, 256, 3, 16, 16])
        res = self.CA(res)
        res += x
        return res