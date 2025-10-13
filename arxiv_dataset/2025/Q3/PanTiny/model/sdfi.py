# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from .modules import InvertibleConv1x1
from .refine import Refine
import math
import numpy as np


class cdc_vg(nn.Module):
    def __init__(self, mid_ch, theta=0.7):

        super(cdc_vg, self).__init__()

        self.cdc = Conv2d_cd(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta= theta)
        self.cdc_bn = nn.BatchNorm2d(mid_ch)
        self.cdc_act = nn.PReLU()

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.vg_bn = nn.BatchNorm2d(mid_ch)
        self.vg_act = nn.PReLU()

        # self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        out_0 = self.cdc_act(self.cdc_bn(self.cdc(x)))

        out1 = self.h_conv(out_0)
        out2 = self.d_conv(out_0)
        out = self.vg_act(self.vg_bn(0.5 * out1 + 0.5 * out2))
        # out = out1 + out2
        return out + x


class ResBlock_cdc(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1, theta=0.8):

        super(ResBlock_cdc, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=n_feats, out_channels=n_feats, kernel_size=3,
                                             stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1,
                                        padding=1, bias=False, theta=theta)
        # self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        out1 = self.h_conv(x)
        out2 = self.d_conv(x)
        # out = torch.sigmoid(self.HP_branch) * out1 + (1 - torch.sigmoid(self.HP_branch)) * out2
        out = out1 + out2

        res += x + out

        return res


class cdcconv(nn.Module):
    def __init__(self, in_channels, out_channels, theta=0.8):

        super(cdcconv, self).__init__()

        self.h_conv = Conv2d_Hori_Veri_Cross(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)
        self.d_conv = Conv2d_Diag_Cross(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=theta)

        self.HP_branch = Parameter(torch.FloatTensor(1))

    def forward(self, x):
        out1 = self.h_conv(x)
        out2 = self.d_conv(x)
        out = torch.sigmoid(self.HP_branch) * out1 + (1 - torch.sigmoid(self.HP_branch)) * out2 + x
        # out = out1 + out2
        return out


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff



class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                 self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1], tensor_zeros,
                                 self.conv.weight[:, :, :, 2], tensor_zeros, self.conv.weight[:, :, :, 3], tensor_zeros,
                                 self.conv.weight[:, :, :, 4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride,
                              padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3





class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


class SpaFre(nn.Module):
    def __init__(self, channels):
        super(SpaFre, self).__init__()
        self.panprocess = nn.Conv2d(channels,channels,3,1,1)
        self.panpre = nn.Conv2d(channels,channels,1,1,0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.fre_process = Freprocess(channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, msf, pan):
        panpre = self.panprocess(pan)
        panf = self.panpre(panpre)
        spafuse = self.spa_process(torch.cat([msf,panf],1))
        frefuse = self.fre_process(msf,panf)
        spa_map = self.spa_att(spafuse-frefuse)
        spa_res = frefuse*spa_map+spafuse
        cat_f = torch.cat([spa_res,frefuse],1)
        cha_res =  self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f))*cat_f)
        out = cha_res+msf


        return out, panpre


class FeatureProcess(nn.Module):
    def __init__(self, channels):
        super(FeatureProcess, self).__init__()

        self.conv_p = nn.Conv2d(4, channels, 3, 1, 1)
        self.conv_p1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.block = SpaFre(channels)
        self.block1 = SpaFre(channels)
        self.block2 = SpaFre(channels)
        self.block3 = SpaFre(channels)
        self.block4 = SpaFre(channels)
        self.fuse = nn.Conv2d(5*channels,channels,1,1,0)


    def forward(self, ms, pan):
        msf = self.conv_p(ms)
        panf = self.conv_p1(pan)
        msf0, panf0 = self.block(msf, panf)
        msf1, panf1 = self.block1(msf0,panf0)
        msf2, panf2 = self.block2(msf1, panf1)
        msf3, panf3 = self.block3(msf2, panf2)
        msf4, panf4 = self.block4(msf3, panf3)
        msout = self.fuse(torch.cat([msf0,msf1,msf2,msf3,msf4],1))

        return msout


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class Net(nn.Module):
    def __init__(self, num_channels, base_filter=None, args=None):
        super(Net, self).__init__()
        self.args = args.get('model', {}) if args else {}
        channels = num_channels
        self.process = FeatureProcess(channels)
        # self.cdc = nn.Sequential(nn.Conv2d(1, 4, 1, 1, 0), cdcconv(4, 4), cdcconv(4, 4))
        self.refine = Refine(channels, 4)

    def forward(self, l_ms, b_ms, x_pan):
        # l_ms - low-resolution multi-spectral image [N,C,h,w]
        # b_ms - bicubic upsampled multi-spectral image [N,C,H,W]  
        # x_pan - high-resolution panchromatic image [N,1,H,W]
        if type(x_pan) == torch.Tensor:
            pass
        elif x_pan == None:
            raise Exception('User does not provide pan image!')

        HRf = self.process(b_ms, x_pan)
        HR = self.refine(HRf) + b_ms

        return HR



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

if __name__ == '__main__':
    # Example usage
    model = Net(num_channels=4)
    ms = torch.randn(1, 4, 64, 64)  # Low resolution multispectral image
    bms = torch.randn(1, 4, 256, 256)  # Bicubic upsampled multispectral image
    pan = torch.randn(1, 1, 256, 256)  # Panchromatic image

    output = model(ms, bms, pan)
    print(output.shape)  # Should be [1, 4, 256, 256]