import math
import torch.serialization
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import copy
import torchvision
import numpy as np
import sys
# import matplotlib.pyplot as plt
import config

# from utils import utils
# from pwc.flow_pwc import Flow_PWC
# from toolbox.arch import Feature_CARB
# from operations import ResBlock, ResBlock_Dil, ResBlock_GTASM_SE, ResBlock_GTASM_SE_GCN,  SELayer, ResBlock_GTASM, DeformShiftTemAtt_SpatialAtt
# from Fusion import Unet_Fusion_C_7OPS
import logging
logger = logging.getLogger('base')
from spynet import SpyNet, flow_warp
# from toolbox.tsam import GatedTemporalShift_Gated_NoDCN
from lib.N_modules import SELayer


class TOFlow(torch.nn.Module):
    def __init__(self, h=0, w=0, task="", cuda_flag=True):
        super(TOFlow, self).__init__()
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag
        print('Toflow based on Deformable Conv !!! \n')

        nf = config.C

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.spynet = SpyNet()

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_encoder_carb = nn.Sequential(
            self.make_carb_layer(block=CARB, ch_num=nf, num_blocks=4),
        )
        print('params of FeaEncode: ', count_parameters_in_MB(self.conv_first)+count_parameters_in_MB(self.feature_encoder_carb))

        self.L1_Cell_DeformAlign_mxy = DeformableConv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deform_groups=8, bias=True)
        self.tsm = GatedTemporalShift_v2(C=nf, n_segment=3, n_div=8, inplace=False)
        print('params of Alignment: ', count_parameters_in_MB(self.spynet) + count_parameters_in_MB(self.L1_Cell_DeformAlign_mxy)+count_parameters_in_MB(self.tsm))

        # reconstruction module
        self.conv9 = nn.Conv2d(nf * 3, nf, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Sequential(
            self.make_carb_layer(block=CARB, ch_num=nf, num_blocks=12),
        )

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 1, bias=True)
        print('params of Reconstruction: ', count_parameters_in_MB(self.conv)+count_parameters_in_MB(self.HRconv)+count_parameters_in_MB(self.conv_last))
        # activation function
        self.relu = nn.ReLU(inplace=True)

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def make_carb_layer(self, block, ch_num, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_num))
        return nn.Sequential(*layers)

    # frames should be TensorFloat
    def forward(self, frames, init=True, x_h=None, x_o=None, iters=12, test_mode=False):
        """
        :param frames: [batch_size=1, img_num=3, n_channels=3, h, w]
        :return: img_tensor:
        """
        x = frames
        B, N, C, H, W = x.size()  # N input video frames: N = 3 != 2

        x = self.normalize(x.view(-1, 3, H, W))
        x = x.view(B, N, 3, H, W)

        flow01 = self.spynet(x[:, 1, :, :, :], x[:, 0, :, :, :])
        flow21 = self.spynet(x[:, 1, :, :, :], x[:, 2, :, :, :])

        # extract LR features
        # L1
        L1_fea = self.feature_encoder_carb(self.relu(self.conv_first(x.view(-1, C, H, W))))
        L1_fea = L1_fea.view(B, N, -1, H, W)

        L1_fea_0 = L1_fea[:, 0, :, :, :].clone()
        L1_fea_1 = L1_fea[:, 1, :, :, :].clone()
        L1_fea_2 = L1_fea[:, 2, :, :, :].clone()
        L1_fea_01 = self.L1_Cell_DeformAlign_mxy(L1_fea_1, L1_fea_0, flow01)  #  torch.cat((L1_fea_0, L1_fea_1), dim=1)
        L1_fea_21 = self.L1_Cell_DeformAlign_mxy(L1_fea_1, L1_fea_2, flow21)  #  torch.cat((L1_fea_2, L1_fea_1), dim=1)

        fea = torch.stack((L1_fea_01, L1_fea_1, L1_fea_21), dim=1)
        # print(x.view(-1, C, H, W).shape)
        fea = self.tsm(fea)
        B, N, C, H, W = fea.size()
        # hr = self.relu(self.conv_1(x))
        # hr = self.relu(self.conv_2(hr))
        # hr = self.relu(self.conv_3(hr))
        # hr = self.relu(self.conv_4(hr))
        fea = self.conv9(fea.view(B, -1, H, W))   # 直接将 ch=64x3 降到 ch=64
        hr = self.conv(fea)
        x_o = hr    # .view(B, N, -1, H, W).view(B, -1, H, W)
        # x_o = torch.cat((hr[0, :, :, :], hr[1, :, :, :], hr[2, :, :, :]), dim=1)
        res = self.conv_last(self.relu(self.HRconv(x_o)))

        lr_ref = x[:, 1, :, :, :]
        predict = lr_ref + res

        return self.denormalize(predict)




# from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
# from mmcv.cnn import ConvModule, constant_init, kaiming_init


# class DCNv2Pack_mxy(ModulatedDeformConv2d):
#     """Modulated Deformable Convolutional Pack.
#     Different from the official DCN, which generates offsets and masks from
#     the preceding features, this ModulatedDCNPack takes another different
#     feature to generate masks and offsets.
#     Args:
#         in_channels (int): Same as nn.Conv2d.
#         out_channels (int): Same as nn.Conv2d.
#         kernel_size (int or tuple[int]): Same as nn.Conv2d.
#         stride (int or tuple[int]): Same as nn.Conv2d.
#         padding (int or tuple[int]): Same as nn.Conv2d.
#         dilation (int or tuple[int]): Same as nn.Conv2d.
#         groups (int): Same as nn.Conv2d.
#         bias (bool or str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
#             False.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.max_residue_magnitude = 25
#         self.flowwarp = flow_warp   # function: x (Tensor): Tensor with size (n, c, h, w). flow (Tensor): Tensor with size (n, h, w, 2), normal value

#         self.firstConv = torch.nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels, kernel_size=3,
#                                          stride=1, padding=1)
#         self.dilatedConv = ResBlock(Channels=self.in_channels, kSize=3)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.conv_offset = nn.Conv2d(
#             self.in_channels,
#             self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
#             kernel_size=self.kernel_size,
#             stride=_pair(self.stride),
#             padding=_pair(self.padding),
#             bias=True)
#         self.init_offset()

#     def init_offset(self):
#         constant_init(self.conv_offset, val=0, bias=0)

#     def forward(self, ref, supp, flow=None):
#         # # ref: 当前帧 supp: 相邻帧 supp
#         # print('flow: ', flow.shape)
#         supp_warped = self.flowwarp(supp, flow.permute(0, 2, 3, 1))
#         # print('warped: ', supp_warped.shape)
#         feat = torch.cat((supp_warped, ref), dim=1)
#         feat = self.lrelu(self.firstConv(feat))

#         feat = self.dilatedConv(feat)

#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)

#         # offset = torch.cat((o1, o2), dim=1)
#         # offset
#         offset = self.max_residue_magnitude * torch.tanh(
#             torch.cat((o1, o2), dim=1))
#         offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
#         offset_1 = offset_1 + flow[:, 0:1, :, :].flip(1).repeat(1,
#                                                                 offset_1.size(1), 1,
#                                                                 1)
#         offset_2 = offset_2 + flow[:, 1:2, :, :].flip(1).repeat(1,
#                                                                 offset_2.size(1), 1,
#                                                                 1)
#         offset = torch.cat([offset_1, offset_2], dim=1)

#         mask = torch.sigmoid(mask)
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             print('Offset abs mean > 50, offset is ', offset_absmean.detach().data())
        
#         if torchvision.__version__ >= '0.9.0':
#             return torchvision.ops.deform_conv2d(supp, offset, self.weight, self.bias, self.stride, self.padding,
#                                                  self.dilation, mask)
#         # else:
#         #     return modulated_deform_conv2d(supp, offset, mask, self.weight, self.bias,
#         #                                 self.stride, self.padding,
#         #                                 self.dilation, self.groups,
#         #                                 self.deform_groups)


class DeformableConv2d(nn.Module):
    # https://zhuanlan.zhihu.com/p/373228012
    # https://zhuanlan.zhihu.com/p/519793194
    # https://github.com/XPixelGroup/BasicSR/blob/243b85fd8cb56cab9178a79c41cd6a0c439b61be/basicsr/archs/arch_util.py#L6
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=8,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        kernel_size = _pair(kernel_size)

        # -----------------------
        self.max_residue_magnitude = 25 # 10 超参数max_residue_magnitude (int): The maximum magnitude of the offset
        self.flowwarp = flow_warp   # function: x (Tensor): Tensor with size (n, c, h, w). flow (Tensor): Tensor with size (n, h, w, 2), normal value

        self.firstConv = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3,
                                         stride=1, padding=1)
        self.dilatedConv = ResBlock(Channels=in_channels, kSize=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # -----------------------
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0],
                         kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.offset_conv = nn.Conv2d(in_channels,
                                     3 * deform_groups * kernel_size[0] *
                                     kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)


    # def forward(self, x):
    def forward(self, ref, supp, flow=None):
        # # ref: 当前帧 supp: 相邻帧 supp
        # print('flow: ', flow.shape)
        supp_warped = self.flowwarp(supp, flow.permute(0, 2, 3, 1))
        # print('warped: ', supp_warped.shape)
        feat = torch.cat((supp_warped, ref), dim=1)
        feat = self.lrelu(self.firstConv(feat))

        feat = self.dilatedConv(feat)

        out = self.offset_conv(feat)
        # https://github.com/XPixelGroup/BasicSR/blob/243b85fd8cb56cab9178a79c41cd6a0c439b61be/basicsr/archs/basicvsrpp_arch.py
        # 估计offset的时候，也可以将flow加到输入中
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset = torch.cat((o1, o2), dim=1)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow[:, 0:1, :, :].flip(1).repeat(1,
                                                                offset_1.size(1), 1,
                                                                1)
        offset_2 = offset_2 + flow[:, 1:2, :, :].flip(1).repeat(1,
                                                                offset_2.size(1), 1,
                                                                1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 100:
            print('Offset abs mean > 100, offset is ', offset_absmean.detach().cpu().data)
        
        if torchvision.__version__ >= '0.9.0':
            return torchvision.ops.deform_conv2d(supp, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            print('-------torchvision 版本需要大于等于0.9.0-------')
            assert False


def count_parameters_in_MB(model):
    return np.sum(np.fromiter((np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name),
                              np.float64)) / 1e6


class CARB(torch.nn.Module):

    def __init__(self, ch=128):
        super(CARB, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.se_attention = SELayer(channel=ch)
        self.conv_256_128_1x1 = torch.nn.Conv2d(in_channels=ch+ch, out_channels=ch, kernel_size=1)

    def forward(self, feature):
        """
        Args:
            feature: 1x128xHxW

        Returns: 1x128xHxW

        """
        x = self.conv1(feature)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        # f = torch.nn.functional.relu(x)
        x_se = self.se_attention(x)
        f = x_se
        # f = torch.cat([x, x_se], dim=1)
        # f = self.conv_256_128_1x1(f)
        return feature + f


class ResBlock(nn.Module):
    def __init__(self, Channels, kSize=3):
        super(ResBlock, self).__init__()
        Ch = Channels
        self.relu  = nn.ReLU()

        self.conv1 = nn.Conv2d(Ch, Ch, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(Ch, Ch, 3, padding=1, stride=1)

        self.conv3 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)
        self.conv4 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)

        self.conv5 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)
        self.conv6 = nn.Conv2d(Ch, Ch, 3, dilation=4, padding=4, stride=1)

    def forward(self, x, prev_x=None, is_the_second=0):
        if is_the_second==1:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + 0.1*self.relu(self.conv4(self.relu(self.conv3(x)))) + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1 + prev_x
        else:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + self.relu(self.conv4(self.relu(self.conv3(x))))*0.1 + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1
        return x


class GatedTemporalShift_Gated_NoDCN(nn.Module):
    def __init__(self, C, n_segment=3, n_div=8, inplace=False):
        super(GatedTemporalShift_Gated_NoDCN, self).__init__()
        nf = C

        self.net = torch.nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=1, padding=1)
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        # define gated convolution layer
        self.gating_conv = copy.deepcopy(self.net)
        self.sigmoid = nn.Sigmoid()

        # if inplace:
        #     print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        B, L, C, H, W = x.size()
        # x = self.sigmoid(self.gating_conv(x)) * self.net(x)     # Should the value of the gate be  computed after shift operator?
        x = self.sigmoid(self.gating_conv(x.view(-1, C, H, W))) * self.net(x.view(-1, C, H, W))
        x = x.view(B, L, -1, H, W)

        return x

    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False):
        B, L, C, H, W = x.size()
        # x = x.transpose(1, 2)  # when x.shape is B C L H W

        fold = C // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out  # out.transpose(1, 2)

class GatedTemporalShift_v2(GatedTemporalShift_Gated_NoDCN):

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        B, L, C, H, W = x.size()
        # x = self.sigmoid(self.gating_conv(x)) * self.net(x)     # Should the value of the gate be  computed after shift operator?
        x = self.sigmoid(self.gating_conv(x.view(-1, C, H, W))) * self.net(x.view(-1, C, H, W))
        x = x.view(B, L, -1, H, W)

        return x


if __name__ == '__main__':
    toflow = TOFlow()
    toflow = toflow.cuda()
    rainy = torch.randn(size=(1, 3, 3, 256, 256), device='cuda')
    out = toflow(rainy)