import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class BN_Conv2d_Leaky(nn.Module):
    """
    BN_CONV_LeakyRELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.leaky_relu(self.seq(x))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BN_Conv_Mish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv_Mish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return Mish()(out)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Identity(nn.Module):
    def forward(self, x):
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class bilateral_prompt(nn.Module):
    def __init__(self, vis_chans, lan_chans, m_chans=None) -> None:
        super().__init__()
        if m_chans is None:
            m_chans = vis_chans 
        self.v_proj1 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True)
        )
        self.v_proj2 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True)
        )
        self.v_proj3 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True)
        )

        self.t_proj1 = nn.Sequential(
            nn.Linear(lan_chans, m_chans),
            nn.ReLU(inplace=True)
        )
        self.t_proj2 = nn.Sequential(
            nn.Linear(lan_chans, m_chans),
            nn.ReLU(inplace=True)
        )
        self.t_proj3 = nn.Sequential(
            nn.Linear(lan_chans, m_chans),
            nn.ReLU(inplace=True)
        )

        self.v_output = nn.Sequential(
            nn.Conv2d(m_chans, vis_chans, 1),
            nn.InstanceNorm2d(vis_chans, affine=True)
        )
        
        self.t_output = nn.Sequential( 
            nn.Linear(m_chans, lan_chans)
        )
    
    def forward(self, vis, lan):
        B, C, H, W = vis.shape
        lan = lan.transpose(1, 2)
        B, N ,C = lan.shape 

        Ci = lan.shape[-1]

        Qv, Kv, Vv = self.v_proj1(vis), self.v_proj2(vis), self.v_proj3(vis)
        Qt, Kt, Vt = self.t_proj1(lan), self.t_proj2(lan), self.t_proj3(lan)

        Qv = Qv.reshape(B, C, -1).transpose(1,2)
        Av = F.softmax(Qv.matmul(Kt.transpose(1, 2)) / math.sqrt(Ci), dim=2)  

        Kv = Kv.reshape(B, C, -1)
        At = F.softmax(Qt.matmul(Kv) / math.sqrt(Ci), dim=2)  

        new_vis = Av.matmul(Vt)  
        
        Vv = Vv.reshape(B, C, -1).transpose(1, 2)
        new_lan = At.matmul(Vv)  

        new_vis = new_vis.permute(0, 2, 1).reshape(B, C, H, W)

        new_vis = self.v_output(new_vis)
        new_lan = self.t_output(new_lan)
        return new_vis, new_lan 

