"""
Author: Yifan Lu<yifan_lu@sjtu.edu.cn>

Intermediate fusion for camera based collaboration
"""

from numpy import record
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from icecream import ic
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, BevEncodeSSFusion, Up, CamEncode, BevEncode
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt


class LiftSplatShootIntermediate(LiftSplatShoot):
    def __init__(self, args): 
        super(LiftSplatShootIntermediate, self).__init__(args)

        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        self.kd = False
        if 'kd_flag' in args:
            self.kd = args['kd_flag']
            print('LiftSplatShootIntermediate->kd_flag status: {}'.format(self.kd))
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.bevencode = BevEncodeSSFusion(fusion_args)
        self.supervise_single = args['supervise_single']
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        for p in self.camencode.parameters():
            p.requires_grad_(False)

        if self.supervise_single:
            self.cls_head_before_fusion = nn.Conv2d(self.bevout_feature, args['anchor_number'], kernel_size=1)                 
            self.reg_head_before_fusion = nn.Conv2d(self.bevout_feature, 7 * args['anchor_number'], kernel_size=1)
            if self.use_dir:
                self.dir_head_before_fusion = nn.Conv2d(self.bevout_feature, args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) 

    
    def forward(self, data_dict):
        return self._forward(data_dict)

    def _forward(self, data_dict):
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans) 
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x_single, x_fuse = self.bevencode(x, record_len, pairwise_t_matrix)
        if self.shrink_flag:
            x_fuse = self.shrink_conv(x_fuse)
        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'depth_items': depth_items}
        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dir_preds": dm})

        if self.supervise_single:
            psm_single = self.cls_head_before_fusion(x_single)
            rm_single = self.reg_head_before_fusion(x_single)
            output_dict.update({'cls_preds_single': psm_single,
                                'reg_preds_single': rm_single})
            if self.use_dir:
                dm_single = self.dir_head_before_fusion(x_single)
                output_dict.update({"dir_preds_single": dm_single})

        if self.kd:
            output_dict.update({'feature': x_fuse,
                                'cls_preds': psm,
                                'reg_preds': rm,
                                })

        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShootIntermediate(grid_conf, data_aug_conf, outC)