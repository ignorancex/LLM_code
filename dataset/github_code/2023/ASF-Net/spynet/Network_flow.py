import math
# import torch.utils.serialization
# import torchfile
import torch.serialization
import torch.nn.functional as F
import torch
import torch.nn as nn
# from torch.autograd import Variable
import sys
# import matplotlib.pyplot as plt
# import numpy as np
import config

sys.path.append('core')     # 注意一下
# from raft import RAFT       # 没有问题
# from toolbox.arch import Feature_CARB_mutil
# from EDVR.basicsr.models.archs.edvr_arch import EDVR
# from models.modules.Sakuya_arch import LunaTokis
# from utils import utils
# from pwc.flow_pwc import Flow_PWC
# from toolbox.arch import Feature_CARB
# from toolbox.unet_ywh_multi import U_Net
# from lib.FJDN import FJDN
# from pietorch.DuRN_S import cleaner
# from pietorch.DuRN_U import cleaner
# from deform_conv_v2 import DeformConv2d
# from operations import ResBlock, ResBlock_Dil, ResBlock_GTASM_SE, ResBlock_GTASM_SE_GCN,  SELayer, ResBlock_GTASM, DeformShiftTemAtt_SpatialAtt
# from Fusion import Unet_Fusion_C_7OPS
# try:
#     from models.modules.DCNv2.dcn_v2 import dcn_v2_conv, DCNv2
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')
import logging
logger = logging.getLogger('base')
# from toolbox.cbam import CBAM
# # from toolbox.nlatt import NonLocalBlock2D_fxy
from spynet import SpyNet, Warp, flow_warp


# def normalize(tensorInput):
#     tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
#     tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
#     tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
#     return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)
#
#
# def denormalize(tensorInput):
#     tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
#     tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
#     tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
#     return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


class TOFlow(torch.nn.Module):
    def __init__(self, h=0, w=0, task="", cuda_flag=True):
        super(TOFlow, self).__init__()
        print('Toflow based on SpynetFlow !!! \n')
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag

        nf = C = config.C

        self.ref_idx = config.N // 2    # 中间帧

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.spynet = SpyNet(cuda_flag=self.cuda_flag)  # SpyNet层
        # for param in self.SpyNet.parameters():  # fix
        #     param.requires_grad = False

        self.warp = Warp(self.height, self.width, cuda_flag=self.cuda_flag)

        # reconstruction module
        self.conv_1 = nn.Conv2d(3 * 3, 64, 9, 1, 4)
        self.conv_2 = nn.Conv2d(64, 64, 9, 1, 4)
        self.conv_3 = nn.Conv2d(64, 64, 1)
        self.conv_4 = nn.Conv2d(64, 3, 1)

        # activation function
        self.relu = nn.ReLU(inplace=True)

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    # frames should be TensorFloat
    def forward(self, lrs, init=True, x_h=None, x_o=None, iters=12, test_mode=False):
        """
        :param frames: [batch_size=1, img_num=3, n_channels=3, h, w]
        :return: img_tensor:
        """
        # for i in range(frames.size(1)):
        #     frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])
        #
        # x = frames
        # B, N, C, H, W = x.size()  # N input video frames: N = 3 != 2
        # #### extract LR features
        # if self.cuda_flag:
        #     opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4)).cuda()
        #     warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4)).cuda()
        # else:
        #     opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4))
        #     warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4))

        # process_index = []
        # for i in range(config.N):
        #     if i != (config.N // 2):
        #         process_index.append(i)
        # for i in process_index:
        #     opticalflows[:, i, :, :, :] = self.SpyNet(frames[:, config.N // 2, :, :, :], frames[:, i, :, :, :])
        # warpframes[:, config.N // 2, :, :, :] = frames[:, config.N // 2, :, :, :]
        # for i in process_index:
        #     warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])

        num_batches, num_lrs, _, h, w = lrs.size()      # b, t, c, h, w

        lrs = self.normalize(lrs.view(-1, 3, h, w))
        lrs = lrs.view(num_batches, num_lrs, 3, h, w)

        lr_ref = lrs[:, self.ref_idx, :, :, :]
        lr_aligned = []
        for i in range(config.N):  # 3 frames
            if i == self.ref_idx:
                lr_aligned.append(lr_ref)
            else:
                lr_supp = lrs[:, i, :, :, :]

                flow = self.spynet(lr_ref, lr_supp)
                # print('flow shape: ', flow.shape)   # torch.Size([1, 2, 256, 256])
                lr_aligned.append(flow_warp(lr_supp, flow.permute(0, 2, 3, 1)))


        # reconstruction
        hr = torch.stack(lr_aligned, dim=1)
        hr = hr.view(num_batches, -1, h, w)
        hr = self.relu(self.conv_1(hr))
        hr = self.relu(self.conv_2(hr))
        hr = self.relu(self.conv_3(hr))
        hr = lr_ref - self.conv_4(hr)

        return self.denormalize(hr)



# from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
# from basicsr.utils import get_root_logger
#
#
# class DCNv2Pack_mxy(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#     def __init__(self, *args, **kwargs):
#         super(DCNv2Pack_mxy, self).__init__(*args, **kwargs)
#         self.firstConv = torch.nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)
#         # self.cbam = CBAM(gate_channels=ch, no_spatial=True)
#         # self.gcn = NonLocalBlock2D_fxy(in_channels=ch, inter_channels=ch)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, x, feat):
#         # x: 相邻帧 feat: [相邻帧, 当前帧]
#         feat = self.lrelu(self.firstConv(feat))
#
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)


import numpy as np


def count_parameters_in_MB(model):
    return np.sum(np.fromiter((np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name),
                              np.float)) / 1e6


# class Cell_Fusion_C_7OPS(nn.Module):
#
#     def __init__(self, steps=7, C=64):
#         super(Cell_Fusion_C_7OPS, self).__init__()
#
#         self._steps = steps
#         # self.first_conv = nn.Conv2d(C * 3, C, 3, 1, 1)
#
#         self._ops_1 = DeformShiftTemAtt_SpatialAtt(C=C)
#         self._ops_2 = DeformShiftTemAtt_SpatialAtt(C=C)
#         self._ops_3 = DeformShiftTemAtt_SpatialAtt(C=C)
#         self._ops_4 = DeformShiftTemAtt_SpatialAtt(C=C)
#         self._ops_5 = DeformShiftTemAtt_SpatialAtt(C=C)
#         self._ops_6 = DeformShiftTemAtt_SpatialAtt(C=C)
#         self._ops_7 = DeformShiftTemAtt_SpatialAtt(C=C)
#
#         # self.conv_7C_7C = nn.Conv2d(C*7, C*7, 3, 1, 1)
#         self.att_se = SELayer(channel=C*3)
#         self.conv = nn.Conv2d(C*3, C*3, 1)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         # self.conv_h = nn.Conv2d(C, C, 3, 1, 1)
#         # self.conv_o = nn.Conv2d(C, C, 3, 1, 1)
#
#     def forward(self, inp_features):
#         B, L, C, H, W = inp_features.size()
#         # inp_features = F.relu(self.first_conv(inp_features))
#
#         feat1 = self._ops_1(inp_features)
#
#         feat2 = self._ops_2(feat1)
#
#         feat3 = self._ops_3(feat2)
#
#         feat4 = self._ops_4(feat3)
#
#         feat5 = self._ops_5(feat4)
#         feat6 = self._ops_6(feat5)
#         feat7 = self._ops_7(feat6)
#
#         return self.lrelu(self.conv(self.att_se(feat7.view(B, -1, H, W))))
#         # res = self.att(feat7.view(-1, C, H, W))
#         # return inp_features.view(-1, C, H, W) - res
#
#         # feat = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6, feat7], dim=2).view(-1, C*7, H, W)
#         # feat = self.conv_7C_C(self.att(feat))
#         # return feat
#         # return inp_features.view(-1, C, H, W) - feat
#         # res = inp_features.view(-1, C, H, W) - self.conv_7C_C(self.att(self.conv_7C_7C(feat)))
#         #
#         # x_o = self.conv_o(res)
#         # return x_o  # x_h, x_o