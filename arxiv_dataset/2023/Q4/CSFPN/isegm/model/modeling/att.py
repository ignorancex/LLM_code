import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import pickle as pkl
import torch.nn as nn
from torch.nn.functional import upsample, normalize
from .layernorm import LayerNorm

torch_ver = torch.__version__[:3]


class Efficient_PAM_Module(Module):
    """ Position attention module"""

    def __init__(self, in_dim, ratio=2, reduction=9):
        super(Efficient_PAM_Module, self).__init__()
        self.channel_in = in_dim
        self.ratio = ratio
        self.reduction = reduction
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x, points):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        points[(points[:, :, 0] >= height*self.ratio) | (points[:, :, 1] >= width*self.ratio)] = -1
        select_points = (points[:, :, 0] // self.ratio) * width + points[:, :, 1] // self.ratio
        proj_select_value = torch.zeros((m_batchsize, C, 48), device=proj_value.device)
        proj_select_key = torch.zeros((m_batchsize, C//self.reduction, 48), device=proj_value.device)
        Ns = []
        for b in range(m_batchsize):
            valid_point = self.reduce_point(select_points[b])
            n = valid_point.shape[0]
            Ns.append(n)
            if n == 0:
                continue
            proj_select_key[b, :, :n] = proj_key[b, :, valid_point].clone()
            proj_select_value[b, :, :n] = proj_value[b, :, valid_point].clone()
        energy = torch.bmm(proj_query, proj_select_key) * np.sqrt(self.reduction / C)
        attention = torch.zeros_like(energy)

        for b, n in enumerate(Ns):
            attention[b, :, :n] = self.softmax(energy[b, :, :n])

        out = torch.bmm(proj_select_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

    def reduce_point(self, points):
        valid_point = torch.nonzero(points > 0, as_tuple=True)
        return points[valid_point].long()


class Att(Module):
    """ Channel attention module"""

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, ratio=4, reduction=3):
        super(Att, self).__init__()
        inter_channels = in_channels
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   LayerNorm(inter_channels), nn.ReLU())
        self.sa = Efficient_PAM_Module(inter_channels, ratio=ratio, reduction=reduction)

    def forward(self, x, points):
        feat1 = self.conva(x)
        sa_feat = self.sa(feat1, points)
        return sa_feat


class Multihead(Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, reduction=3, heads=4, ratio=4):
        super(Multihead, self).__init__()
        self.atts = nn.ModuleList()
        self.heads = heads
        for h in range(heads):
            self.atts.append(Efficient_PAM_Module(in_channels, ratio=ratio, reduction=reduction))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   norm_layer(in_channels), nn.ReLU())

    def forward(self, x, points):
        feats = 0
        for h in range(self.heads):
            _feats = self.atts[h](x, points)
            feats += _feats / self.heads

        out = self.conv1(feats)

        return out


class MultiScaleAtt(Module):
    def __init__(self, high_dim=128, low_dim=64, out_dim=128, reduction=4, ratio=2, mode='normal'):
        super(MultiScaleAtt, self).__init__()
        self.channel_in = high_dim
        self.ratio = ratio
        self.reduction = reduction
        self.query_conv = Conv2d(in_channels=high_dim, out_channels=high_dim // reduction, kernel_size=1)
        self.key_conv = Conv2d(in_channels=high_dim, out_channels=high_dim // reduction, kernel_size=1)
        self.value_conv = Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(high_dim + low_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        # self.softmax = Softmax(dim=1)
        self.mode = mode

    def forward(self, high_feature, low_feature, points, mode='normal'):
        if self.mode == 'normal':
            return self.normal_forward(high_feature, low_feature, points)
        else:
            raise NotImplementedError

    def normal_forward(self, high_feature, low_feature, points):
        x = F.interpolate(high_feature, low_feature.size()[2:], mode='bilinear', align_corners=True)
        value = self.fusion(torch.cat([x, low_feature], dim=1))
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(value).view(m_batchsize, -1, width * height)

        # filter the point outside the current zoom-in
        points[(points[:, :, 0] >= height*self.ratio) | (points[:, :, 1] >= width*self.ratio)] = -1
        select_points = (points[:, :, 0] // self.ratio) * width + points[:, :, 1] // self.ratio
        proj_select_value = torch.zeros((m_batchsize, C, 48), device=proj_value.device)
        proj_select_key = torch.zeros((m_batchsize, C//self.reduction, 48), device=proj_value.device)
        Ns = []
        for b in range(m_batchsize):
            valid_point = self.reduce_point(select_points[b])
            n = valid_point.shape[0]
            Ns.append(n)
            if n == 0:
                continue
            proj_select_key[b, :, :n] = proj_key[b, :, valid_point].clone()
            proj_select_value[b, :, :n] = proj_value[b, :, valid_point].clone()

        # Similarity
        energy = torch.bmm(proj_query, proj_select_key) * np.sqrt(self.reduction / C)
        attention = torch.zeros_like(energy)

        # calculate softmax, exclude the zero points.
        for b, n in enumerate(Ns):
            attention[b, :, :n] = self.softmax(energy[b, :, :n])

        # Information propagation
        out = torch.bmm(proj_select_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + value
        return out

    def reduce_point(self, points):
        # Exclude the points outside the image because of the zoom-in.
        valid_point = torch.nonzero(points > 0, as_tuple=True)
        return points[valid_point].long()
