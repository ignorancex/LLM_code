import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

# test attention mechanism
from models.dep_mhsa.attentions.mhsa_collection import MHSA3D, DepMHSA
from models.dep_mhsa.attentions.single_module_attn import SingleModalAtten
from models.dep_mhsa.attentions.channel_attention import CAB3D
from einops import rearrange

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        attn_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        n_frames_last_layer=2,
        resolution_last_layer=(7, 7),
        **kwargs

    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        super().__init__()
        self.attn_builder = attn_builder
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride), 
            nn.BatchNorm3d(planes), 
            nn.ReLU(inplace=True)
        )

        # Attention
        if attn_builder is None:
            self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))

        elif attn_builder.__name__ in [MHSA3D.__name__]:
            self.conv2 = nn.Sequential(attn_builder(planes, 
                                                    n_frames=n_frames_last_layer,
                                                    width=resolution_last_layer[1],
                                                    height=resolution_last_layer[0]),
                                        nn.BatchNorm3d(planes))
            
        elif attn_builder.__name__ == DepMHSA.__name__:
            if 'q_kernel' not in kwargs.keys():
                kwargs['q_kernel'] = '133'
                kwargs['k_kernel'] = '311'
                kwargs['v_kernel'] = '311133'
            self.conv2 = nn.Sequential(attn_builder(planes, 
                                                    n_frames=n_frames_last_layer,
                                                    width=resolution_last_layer[1],
                                                    height=resolution_last_layer[0],
                                                    q_kernel = kwargs['q_kernel'],
                                                    k_kernel = kwargs['k_kernel'],
                                                    v_kernel = kwargs['v_kernel'],
                                                    ),
                                        nn.BatchNorm3d(planes))
            
        elif attn_builder.__name__ == SingleModalAtten.__name__:
            self.spatial_attention_block = SingleModalAtten(
                channels=planes, num_heads=4, 
                num_head_channels=-1, 
                use_checkpoint=False)

            self.temporal_attention_block = SingleModalAtten(
                channels=planes, num_heads=4, 
                num_head_channels=-1, 
                use_checkpoint=False)
            
        elif attn_builder.__name__ == CAB3D.__name__:
            self.conv2 = CAB3D(n_feat=planes)
            
        else:
            raise TypeError("Unknown Type for attention block")
            
        # self.relu2 = nn.LeakyReLU()
        self.relu2 = nn.PReLU()
        
        self.downsample = downsample
        self.stride = stride
        
                # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        b,c,f,h,w = out.shape

        # Attention
        if (self.attn_builder) is None or (self.attn_builder.__name__ in ["MHSA3D", "CAB3D","DepMHSA"]):
            out = self.conv2(out)

        elif self.attn_builder.__name__== "SingleModalAtten":
            out = rearrange(out, "b c f h w -> (b f) c (h w)")
            out = self.spatial_attention_block(out)
            out = rearrange(out, "(b f) c (h w) -> (b h w) c f", f=f, h=h, c=c)
            out = self.temporal_attention_block(out)
            out = rearrange(out, "(b h w) c f -> b c f h w", h=h, w=w)
        else:
            raise TypeError("Unknown Type for attention block")
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class BasicBlockWithAttention(nn.Module):
    expansion = 1
    """        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        attn_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        n_frames_last_layer=2,
        resolution_last_layer=(7, 7)
        """
    def __init__(self, inplanes, planes, conv_builder, attn_builder, stride=1, downsample=None, **kwargs):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlockWithAttention, self).__init__()

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(planes, planes // 16, kernel_size=(1, 3, 3),padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes // 16, 1, kernel_size=(1, 3, 3), padding='same'),
            nn.Sigmoid()
        )

        # Temporal Attention
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(planes, planes // 16, kernel_size=(3, 1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes // 16, 1, kernel_size=(1, 1, 1), padding='same'),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x) # size same to x
        # Apply spatial attention
        spatial_weights = self.spatial_attention(out) # Batch, channel = 1(from 64), frames = 11, height=width=60
        out = out * spatial_weights

        out = self.conv2(out)
        # Apply temporal attention
        temporal_weights = self.temporal_attention(out)
        out = out * temporal_weights

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        attn_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:

        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out