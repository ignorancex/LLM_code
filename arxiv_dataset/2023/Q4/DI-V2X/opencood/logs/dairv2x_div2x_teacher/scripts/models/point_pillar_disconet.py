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
from opencood.models.fuse_modules.fusion_in_one import DiscoFusion, SumFusion, MaxFusion, SumFusion_multiscale, SumFusion_multiscale2, \
    AttFusion, AttFusion2, AttFusion4, AttFusion5, AttFusion6, AttFusion6_1, AttFusion6_2, AttFusion6_3, AttFusion7, AttFusion7_1, AttFusion7_1_2, AttFusion8, \
        AttFusion6_2_2, AttFusion6_4_2,  AttFusion6_1_2, AttFusion7_2_2, AttFusion7_3_2, DiscoFusion2, MaxFusion_2, SumFusion_2, TransformerFusion

class PointPillarDiscoNet(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNet, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
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
                self.fusion_net = DiscoFusion(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'attention':
                self.fusion_net = AttFusion(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'max':
                self.fusion_net = MaxFusion()
            elif args['fusion_args']['fusion_net'] == 'max_2':
                self.fusion_net = MaxFusion_2()
            elif args['fusion_args']['fusion_net'] == 'sum_2':
                self.fusion_net = SumFusion_2()
            elif args['fusion_args']['fusion_net'] == 'attention_multiscale':
                self.fusion_net = AttFusion2(self.out_channel, args['fusion_args'])
            elif args['fusion_args']['fusion_net'] == 'att4':
                self.fusion_net = AttFusion4(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att5':
                self.fusion_net = AttFusion5(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att6':
                self.fusion_net = AttFusion6(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att61':
                self.fusion_net = AttFusion6_1(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att61_2':
                self.fusion_net = AttFusion6_1_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att62':
                self.fusion_net = AttFusion6_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att62_2':
                self.fusion_net = AttFusion6_2_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att63':
                self.fusion_net = AttFusion6_3(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att64_2':
                self.fusion_net = AttFusion6_4_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att7':
                self.fusion_net = AttFusion7(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att71':
                self.fusion_net = AttFusion7_1(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att71_2':
                self.fusion_net = AttFusion7_1_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att72_2':
                self.fusion_net = AttFusion7_2_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att73_2':
                self.fusion_net = AttFusion7_3_2(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'att8':
                self.fusion_net = AttFusion8(self.out_channel)
            elif args['fusion_args']['fusion_net'] == 'sumfusion_multiscale':
                self.fusion_net = SumFusion_multiscale(args['fusion_args']) 
            elif args['fusion_args']['fusion_net'] == 'transformer':
                self.fusion_net = TransformerFusion(channels=args['fusion_args']['in_channels'], n_head=args['fusion_args']['n_head'], dropout=args['fusion_args']['dropout_rate']) 
            else:
                self.fusion_net = SumFusion_multiscale2(args['fusion_args']) 
        else:
            self.fusion_net = SumFusion_multiscale(args['fusion_args']) 
        print('using {} for student'.format(self.fusion_net))

        if 'fusion_args' in args.keys():
            self.multi_scale = args['fusion_args']['multi_scale']
        else:
            self.multi_scale = False
        print('multi_scale status:', self.multi_scale)
        
        if 'early_distill' in args['fusion_args']:
            self.early_distill = args['fusion_args']['early_distill']
            print('early_distill status:', self.early_distill)
        else:
            self.early_distill = False

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
                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)
            output_dict.update({'multiscale_feats': multiscale_feats})
        else:
            fusion_features_2d = self.fusion_net(spatial_features_2d, record_len, t_matrix)
            psm = self.cls_head(fusion_features_2d)
            rm = self.reg_head(fusion_features_2d)

        output_dict.update({'feature': fusion_features_2d,
                       'cls_preds': psm,
                       'reg_preds': rm})
        if self.early_distill:
            output_dict.update({'single_features': spatial_features_2d, 'record_len': record_len, 't_matrix': t_matrix})
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fusion_features_2d)})

        return output_dict
