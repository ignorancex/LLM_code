# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.data_utils.post_processor import UncertaintyVoxelPostprocessor
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.utils.transformation_utils import normalize_pairwise_tfm, regroup
from opencood.models.fuse_modules.fusion_in_one import DiscoFusion, SumFusion, SumFusion2, AttFusion, DiscoFusion2

class PointPillarDiscoNet(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNet, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.cyc_distill = True if 'cyc_distill' in args else False
        print('Model.cyc_distill:', self.cyc_distill)
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.voxel_size = args['voxel_size']
        
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        if 'fusion_net' in args['fusion_args']:
            if args['fusion_args']['fusion_net'] == 'disconet':
                self.fusion_net = DiscoFusion2(self.out_channel, args['fusion_args'])
            else:
                self.fusion_net = SumFusion2(args['fusion_args']) 
        else:
            self.fusion_net = SumFusion2(args['fusion_args']) 
        print('using {} for student'.format(self.fusion_net))

        if 'fusion_args' in args.keys():
            self.multi_scale = args['fusion_args']['multi_scale']
        else:
            self.multi_scale = False
        print('multi_scale status:', self.multi_scale)

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        # teacher_voxel_features = data_dict['teacher_processed_lidar']['voxel_features']
        # teacher_voxel_coords = data_dict['teacher_processed_lidar']['voxel_coords']
        # teacher_voxel_num_points = data_dict['teacher_processed_lidar']['voxel_num_points']

        record_len = data_dict['record_len']
        lidar_pose = data_dict['lidar_pose']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix}


        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)

        _, _, H0, W0 = batch_dict['spatial_features'].shape
        t_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        batch_dict = self.backbone(batch_dict)


        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        output_dict = {}
        if self.multi_scale:
            spatial_features_2d, multiscale_feats = self.fusion_net(batch_dict['spatial_features'], record_len, t_matrix, self.backbone)
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
                if self.cyc_distill:
                    #spatial_features_2d has global view
                    gloabl_H, global_W = spatial_features_2d.shape[-2:]
                    partial_H, partial_W = int(H0/2), int(W0/2)
                    pad_x, pad_y = int((gloabl_H - partial_H)/2), int((global_W - partial_W)/2)
                    #crop the partial view for head prediction
                    spatial_features_partial = spatial_features_2d[:,:,pad_x:pad_x+partial_H, pad_y:pad_y+partial_W]
                    psm = self.cls_head(spatial_features_partial)
                    rm = self.reg_head(spatial_features_partial)

                else:
                    psm = self.cls_head(spatial_features_2d)
                    rm = self.reg_head(spatial_features_2d)
            output_dict.update({'multiscale_feats': multiscale_feats})
        else:
            spatial_features_2d = self.fusion_net(spatial_features_2d, record_len, t_matrix)
            if self.cyc_distill:
                #spatial_features_2d has global view
                gloabl_H, global_W = spatial_features_2d.shape[-2:]
                partial_H, partial_W = int(H0/2), int(W0/2)
                pad_x, pad_y = int((gloabl_H - partial_H)/2), int((global_W - partial_W)/2)
                #crop the partial view for head prediction
                spatial_features_partial = spatial_features_2d[:,:,pad_x:pad_x+partial_H, pad_y:pad_y+partial_W]
                psm = self.cls_head(spatial_features_partial)
                rm = self.reg_head(spatial_features_partial)
            else:
                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)

        output_dict.update({'feature': spatial_features_2d,
                       'cls_preds': psm,
                       'reg_preds': rm})
        if self.use_dir:
            if self.cyc_distill:
                output_dict.update({'dir_preds': self.dir_head(spatial_features_partial)})
            else:
                output_dict.update({'dir_preds': self.dir_head(spatial_features_2d)})

        return output_dict
