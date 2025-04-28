# -*- coding: utf-8 -*-
import sys

import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans import ChannelTransformer


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DownBlock1(nn.Module):
    """Downscaling with maxpool convolution and SE attention"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', reduction=16):
        super(DownBlock1, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.se_block = SEBlock(out_channels, reduction)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.nConvs(out)
        out = self.se_block(out)
        return out

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        #self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        print(out.size())
        print(skip_x.size())
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        print(x.size())
        x = self.nConvs(x)
        print(x.size())
        print("-------------------------------")
        return x



class UpBlock_low(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock_low, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_channels,in_channels,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        #out = self.up(x)
        #x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class UpSampling_high(nn.Module):

    def __init__(self, C):
        super(UpSampling_high, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


class UpSampling_low(nn.Module):

    def __init__(self):
        super(UpSampling_low, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        #self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)



## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src


class DownBlock2(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock2, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        #print("down",out.size())
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class DPCABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(DPCABlock, self).__init__()

        # 设计自适应卷积核
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.conv1(avg_pool_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.conv2(avg_pool_g.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention_high(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt_1 = DPCABlock(channels=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt_1(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class UpBlock_attention_low(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.coatt_1 = DPCABlock(channels=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        #skip_x_att = self.coatt(g=up, x=skip_x)
        skip_x_att = self.coatt_1(g=x, x=skip_x)
        x = torch.cat([skip_x_att, x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class bridge(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        #self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.coatt_1 = DPCABlock(channels=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        #x = self.up(x)
        #print(up.size())
        x = _upsample_like(x, skip_x)
        skip_x_att = self.coatt_1(g=x, x=skip_x)
        x = torch.cat([skip_x_att, x], dim=1)
        return self.nConvs(x)


class MCFNet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size_s1=256, img_size_s2=224, vis=False):
        super().__init__()
        self.vis = vis
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel

        # # decoder initialization
        # if (self.decoder_aggregation == 'additive'):
        #     self.decoder = CASCADE_Add(channels=self.channels)
        # elif (self.decoder_aggregation == 'concatenation'):
        #     self.decoder = CASCADE_Cat(channels=self.channels)
        # else:
        #     sys.exit(
        #         "'" + self.decoder_aggregation + "' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 3, kernel_size=1),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True)
        # )
        self.ub_stem = ConvBatchNorm(n_channels, in_channels)
        self.down1_1 = DownBlock1(in_channels, in_channels * 2, nb_Conv=2)
        self.down1_2 = DownBlock1(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down1_3 = DownBlock1(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down1_4 = DownBlock1(in_channels * 8, in_channels * 8, nb_Conv=2)
        # self.ub_bridge = UpBlock_low(in_channels * 16, in_channels * 8, nb_Conv=2)
        # self.up1_4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        # self.up1_3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        # self.up1_2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        # self.up1_1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        #self.with_cab = nn.Conv2d(1024, 1024 // 2, 1, 1)
        self.up1_4 = UpSampling_low()
        self.conv1_4 = Conv(in_channels * 16, in_channels * 4)
        self.up1_3 = UpSampling_low()
        self.conv1_3 = Conv(in_channels * 8, in_channels * 2)
        self.up1_2 = UpSampling_low()
        self.conv1_2 = Conv(in_channels * 4, in_channels * 1)
        self.up1_1 = UpSampling_low()
        self.conv1_1 = Conv(in_channels * 2, in_channels * 1)
        self.oc_s = nn.Conv2d(in_channels, 1, 1)
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(256, self.n_classes, 1)
        self.out_head2 = nn.Conv2d(128, self.n_classes, 1)
        self.out_head3 = nn.Conv2d(64, self.n_classes, 1)
        self.out_head4 = nn.Conv2d(64, self.n_classes, 1)
        #self.outc1 = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        self.fb_stem = ConvBatchNorm(n_channels, in_channels)
        self.down2_1 = DownBlock2(in_channels, in_channels*2, nb_Conv=2)
        self.down2_2 = DownBlock2(in_channels*2, in_channels*4, nb_Conv=2)
        self.down2_3 = DownBlock2(in_channels*4, in_channels*8, nb_Conv=2)
        self.cab_down = DownBlock2(in_channels*8, in_channels*8, nb_Conv=2)
        self.mtc = ChannelTransformer(config, vis, img_size_s2,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.bridge = bridge(in_channels*16,in_channels*8, nb_Conv=2)
        self.up2_4 = UpBlock_attention_low(in_channels*16, in_channels*4, nb_Conv=2)
        self.up2_3 = UpBlock_attention_high(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2_2 = UpBlock_attention_high(in_channels*4, in_channels, nb_Conv=2)
        self.up2_1 = UpBlock_attention_high(in_channels*2, in_channels, nb_Conv=2)
        self.outc2 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))

        self.last_activation = nn.Sigmoid() # if using BCELoss
        #self.w = nn.Parameter(torch.ones(4))
        self.w = nn.Parameter(torch.Tensor([1,1,1,1]),requires_grad=False)
    def forward(self, x):
        # if (x.shape[2] % 14 != 0):
        # if x.size()[1] == 1:
        #     x = self.conv(x)
        # print(x.size())
        a_1 = x
        s1 = F.interpolate(x, size=(self.img_size_s1, self.img_size_s1), mode='bilinear')

        s1 = s1.float()
        s1 = self.ub_stem(s1)
        # print(s1.size())
        x1_1 = self.down1_1(s1)
        # print(x1_1.size())
        x1_2 = self.down1_2(x1_1)
        # print(x1_2.size())
        x1_3 = self.down1_3(x1_2)
        # print(x1_3.size())
        x1_4 = self.down1_4(x1_3)

        # x_new = _upsample_like(x1_4, x1_3)
        # print(x_new.size())
        # print(x1_1.size())
        # print(x1_2.size())
        # print(x1_3.size())
        # print(x1_4.size())
        # x1_4_with_cab = self.with_cab(x1_4)
        # print(x1_4_with_cab.size())
        #print("------------------------------")
        # x1_4, x1_3, x1_2, x1_1 = self.decoder(x1_4, [x1_3, x1_2, x1_1])
        #x1_5_o = self.ub_bridge(x1_4)

        x1_4_1 = self.up1_4(x1_4, x1_3)
        x1_4_2 = self.conv1_4(x1_4_1)
        #print("x1_4_2:", x1_4_2.size())
        x1_3_1 = self.up1_3(x1_4_2, x1_2)
        #print("x1_3_1:", x1_3_1.size())
        x1_3_2 = self.conv1_3(x1_3_1)
        x1_2_1 = self.up1_2(x1_3_2, x1_1)
        x1_2_2 = self.conv1_2(x1_2_1)
        x1_1_1 = self.up1_1(x1_2_2, s1)
        x1_1_2 = self.conv1_1(x1_1_1)
        # print(x1_4_2.size())
        # print(x1_3_2.size())
        # print(x1_2_2.size())
        # print(x1_1_2.size())
        p14 = self.out_head1(x1_4_2)
        p13 = self.out_head2(x1_3_2)
        p12 = self.out_head3(x1_2_2)
        p11 = self.out_head4(x1_1_2)
        # print([p11.shape, p12.shape, p13.shape, p14.shape])
        # OC-S模块
        x1_1_in = self.oc_s(x1_1_2)
        # print("x1_1_in:", x1_1_in.size())
        x1_1_in = nn.Sigmoid()(x1_1_in)

        p14 = F.interpolate(p14, scale_factor=8, mode='bilinear')
        p13 = F.interpolate(p13, scale_factor=4, mode='bilinear')
        p12 = F.interpolate(p12, scale_factor=2, mode='bilinear')
        p11 = F.interpolate(p11, scale_factor=1, mode='bilinear')

        #print([p11.shape, p12.shape, p13.shape, p14.shape])

        #x1_1_in = F.interpolate(x1_1_in, scale_factor=4, mode='bilinear')
        x_in = a_1 * x1_1_in
        # print("x_in:", x_in.size())
        s2 = F.interpolate(x_in, size=(self.img_size_s2, self.img_size_s2), mode='bilinear')
        # print("s2:", s2.size())

        s2 = s2.float()
        s2 = self.fb_stem(s2)
        x2_1 = self.down2_1(s2)
        x2_2 = self.down2_2(x2_1)
        x2_3 = self.down2_3(x2_2)
        x2_4 = self.cab_down(x2_3)
        x2_3_1 = x2_3
        x1, x2, x3, x4 = self.mtc(s2, x2_1, x2_2, x2_3)
        x = self.bridge(x2_4, x2_3_1)
        # print(x.size())

        skip1_1 = F.interpolate(s1, size=(x1.shape[2:]), mode='bilinear')
        skip1_2 = F.interpolate(x1_1, size=(x2.shape[2:]), mode='bilinear')
        skip1_3 = F.interpolate(x1_2, size=(x3.shape[2:]), mode='bilinear')
        skip1_4 = F.interpolate(x1_3, size=(x4.shape[2:]), mode='bilinear')

        s1_4 = F.interpolate(x1_4, size=(x.shape[2:]), mode='bilinear')
        skip2_1 = skip1_1 + x1
        skip2_2 = skip1_2 + x2
        skip2_3 = skip1_3 + x3
        skip2_4 = skip1_4 + x4
        # print([skip2_1.size(), skip2_2.size(), skip2_3.size(), skip2_4.size()])
        x = x + s1_4
        # print(x.size())
        p24 = self.up2_4(x, skip2_4)
        p23 = self.up2_3(p24, skip2_3)
        p22 = self.up2_2(p23, skip2_2)
        p21 = self.up2_1(p22, skip2_1)

        #print([p21.shape, p22.shape, p23.shape, p24.shape])
        p24 = self.out_head1(p24)
        p23 = self.out_head2(p23)
        p22 = self.out_head3(p22)
        p21 = self.out_head4(p21)
        #print([p21.shape, p22.shape, p23.shape, p24.shape])

        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode='bilinear')
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode='bilinear')
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode='bilinear')
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode='bilinear')
        #print([p21.shape, p22.shape, p23.shape, p24.shape])

        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        # print([p1.size(), p2.size(), p3.size(), p4.size()])
        # else:
        #     a_1 = x
        #     s1 = F.interpolate(x, size=self.img_size_s1, mode='bilinear')
        #     s1 = s1.float()
        #     s1 = self.inc2(s1)
        #     x2_1 = self.down2_1(s1)
        #     x2_2 = self.down2_2(x2_1)
        #     x2_3 = self.down2_3(x2_2)
        #     x2_4 = self.cab_down(x2_3)
        #     x2_3_1 = x2_3
        #     x1, x2, x3, x4 = self.mtc(s1, x2_1, x2_2, x2_3)
        #     x = self.bridge(x2_4, x2_3_1)
        #     x2_4_o = self.up2_3(x, x4)
        #     x2_3_o = self.up2_2(x, x3)
        #     x2_2_o = self.up2_1(x, x2)
        #     x2_1_o = self.up2_0(x, x1)
        #
        #     p24 = self.out_head2(x2_4_o)
        #     p23 = self.out_head3(x2_3_o)
        #     p22 = self.out_head4(x2_2_o)
        #
        #     # OC-S模块
        #     x2_1_in = self.oc_s(x2_1_o)
        #     x2_1_in = nn.Sigmoid()(x2_1_in)
        #
        #     p24 = F.interpolate(p24, scale_factor=32, mode=self.interpolation)
        #     p23 = F.interpolate(p23, scale_factor=16, mode=self.interpolation)
        #     p22 = F.interpolate(p22, scale_factor=8, mode=self.interpolation)
        #     p21 = F.interpolate(x2_1_o, scale_factor=4, mode=self.interpolation)
        #
        #     x2_1_in = F.interpolate(x2_1_in, scale_factor=4, mode=self.interpolation)
        #
        #     x_in = a_1 * x2_1_in
        #
        #     s2 = F.interpolate(x_in, size=self.img_size_s2, mode='bilinear')
        #
        #     s2 = s2.float()
        #     s2 = self.inc1(s2)
        #     x1_1 = self.down1_1(s2)
        #     x1_2 = self.down1_2(x1_1)
        #     x1_3 = self.down1_3(x1_2)
        #     x1_4 = self.down1_4(x1_3)
        #
        #     # x1_4, x1_3, x1_2, x1_1 = self.decoder(x1_4, [x1_3, x1_2, x1_1])
        #     skip2_1 = F.interpolate(x2_1, size=(x1_1.shape[2:]), mode='bilinear')
        #     skip2_2 = F.interpolate(x2_2, size=(x1_2.shape[2:]), mode='bilinear')
        #     skip2_3 = F.interpolate(x2_3, size=(x1_3.shape[2:]), mode='bilinear')
        #     s2_4 = F.interpolate(x2_4, size=(x1_4.shape[2:]), mode='bilinear')
        #     skip1_1 = skip2_1 + x1_1
        #     skip1_2 = skip2_2 + x1_2
        #     skip1_3 = skip2_3 + x1_3
        #     x = x + s2_4
        #     p14 = self.up1_4(x)
        #     p13 = self.up1_3(p14, skip1_3)
        #     p12 = self.up1_2(p13, skip1_2)
        #     p11 = self.up1_1(p12, skip1_1)
        #
        #     p14 = self.out_head1(p14)
        #     p13 = self.out_head2(p13)
        #     p12 = self.out_head3(p12)
        #     p11 = self.out_head4(p11)
        #
        #     p1 = p11 + p21
        #     p2 = p12 + p22
        #     p3 = p13 + p23
        #     p4 = p14
        return p1, p2, p3, p4




