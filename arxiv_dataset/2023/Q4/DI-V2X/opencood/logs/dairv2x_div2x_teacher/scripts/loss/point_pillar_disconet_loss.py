"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.models.fuse_modules.fusion_in_one import regroup

class PointPillarDiscoNetLoss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarDiscoNetLoss, self).__init__(args)
        self.kd = args['kd']
        self.multiscale_kd = self.kd.get('multiscale_kd', False)
        print('self.multiscale_kd: ', self.multiscale_kd)

        self.early_distill = self.kd.get('early_distill', False)
        print('self.early_distill: ', self.early_distill)

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = super().forward(output_dict, target_dict)

        ########## KL loss ############
        # rm = output_dict['reg_preds']  # [B, 14, 50, 176]
        # psm = output_dict['cls_preds'] # [B, 2, 50, 176]
        feature = output_dict['feature']

        # teacher_rm = output_dict['teacher_reg_preds']
        # teacher_psm = output_dict['teacher_cls_preds']
        teacher_feature = output_dict['teacher_feature']
        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

        N, C, H, W = teacher_feature.shape
        teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
        student_feature = feature.permute(0,2,3,1).reshape(N*H*W, C)
        kd_loss_feature = kl_loss_mean(
                F.log_softmax(student_feature, dim=1), F.softmax(teacher_feature, dim=1)
            )
        kd_loss = kd_loss_feature * self.kd['weight']

        if self.multiscale_kd:
            s_feat_multiscale = output_dict['multiscale_feats']
            t_feat_multiscale = output_dict['teacher_multiscale_feats']
            for i in range(len(s_feat_multiscale)):
                N, C, H, W = s_feat_multiscale[i].shape
                teacher_feature = t_feat_multiscale[i].permute(0,2,3,1).reshape(N*H*W, C)
                student_feature = s_feat_multiscale[i].permute(0,2,3,1).reshape(N*H*W, C)
                kd_loss += kl_loss_mean(F.log_softmax(student_feature, dim=1), F.softmax(teacher_feature, dim=1)) * self.kd['multiscale_kd_weight']
        
        # if self.cyc_distill:
        #     #spatial_features_2d has global view
        #     gloabl_H, global_W = output_dict['teacher_feature'].shape[-2:]
        #     partial_H, partial_W = output_dict['cls_preds'].shape[-2:]
        #     pad_x, pad_y = int((gloabl_H - partial_H)/2), int((global_W - partial_W)/2)
        #     #crop the partial view for head prediction
        #     teacher_features_partial = output_dict['teacher_feature'][:,:,pad_x:pad_x+partial_H, pad_y:pad_y+partial_W]
        #     student_features_partial = output_dict['feature'][:,:,pad_x:pad_x+partial_H, pad_y:pad_y+partial_W]
        #     N, C, H, W = teacher_features_partial.shape
        #     teacher_features_partial = teacher_features_partial.permute(0,2,3,1).reshape(N*H*W, C)
        #     student_features_partial = student_features_partial.permute(0,2,3,1).reshape(N*H*W, C)
        #     kd_loss += kl_loss_mean(F.log_softmax(student_features_partial, dim=1), F.softmax(teacher_features_partial, dim=1)) * self.kd['cyc_distill_kd_weight']
        
        if self.early_distill:
            single_features, record_len, t_matrix = output_dict['single_features'], output_dict['record_len'], output_dict['t_matrix']
            split_x = regroup(single_features, record_len)
            B, L = t_matrix.shape[:2]
            ego_feature, inf_feature, inf_idx = [], [], []


            for b in range(B):
                cav_num = split_x[b].shape[0]
                ego_feature.append(split_x[b][0])
                if cav_num > 1:
                    inf_feature.append(split_x[b][1])
                    inf_idx.append(b)
            ego_feature = torch.stack(ego_feature) #[B, C, H, W]
            if len(inf_feature) > 0:
                inf_feature = torch.stack(inf_feature) ##[B2, C, H, W]

            #inf_idx = torch.stack(inf_idx)

            teacher_feature = output_dict['teacher_feature']
            N, C, H, W = ego_feature.shape
            teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
            ego_feature = ego_feature.permute(0,2,3,1).reshape(N*H*W, C)
            kd_loss_feature = kl_loss_mean(F.log_softmax(ego_feature, dim=1), F.softmax(teacher_feature, dim=1))
            kd_ego_loss = kd_loss_feature * self.early_distill
            kd_loss += kd_ego_loss

            #generate overlap mask
            overlap_mask = torch.ones(B, 1, H, W).to(ego_feature)
            t_matrix_inf = t_matrix[:, 0, 1, :, :]
            grid = F.affine_grid(t_matrix_inf, [B, 1, H, W], align_corners=True).to(ego_feature)
            overlap_mask = F.grid_sample(overlap_mask, grid, align_corners=True)  #[B, 1, H, W]

            if len(inf_feature)> 0 :
                teacher_feature = output_dict['teacher_feature']
                N, C, H, W = inf_feature.shape
                teacher_feature *= overlap_mask #mask
                teacher_feature = teacher_feature[inf_idx]
                teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
                inf_feature = inf_feature.permute(0,2,3,1).reshape(N*H*W, C)
                kd_loss_feature = kl_loss_mean(F.log_softmax(inf_feature, dim=1), F.softmax(teacher_feature, dim=1))
                kd_inf_loss = kd_loss_feature * self.early_distill
                kd_loss += kd_inf_loss

        total_loss += kd_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                              'kd_loss': kd_loss.item()})


        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=''):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        kd_loss = self.loss_dict.get('kd_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || KD Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, kd_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Kd_loss'+suffix, kd_loss,
                            epoch*batch_len + batch_id)
            

class PointPillarUniDistilLoss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarUniDistilLoss, self).__init__(args)
        self.kd = args['kd']

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = super().forward(output_dict, target_dict)

        ########## KL loss ############
        rm = output_dict['reg_preds']  # [B, 14, 50, 176]
        psm = output_dict['cls_preds'] # [B, 2, 50, 176]
        feature = output_dict['feature']

        teacher_rm = output_dict['teacher_reg_preds']
        teacher_psm = output_dict['teacher_cls_preds']

        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)
        # feature = output_dict['feature']
        # teacher_feature = output_dict['teacher_feature']
        # N, C, H, W = teacher_feature.shape
        # teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
        # student_feature = feature.permute(0,2,3,1).reshape(N*H*W, C)
        # kd_loss_feature = kl_loss_mean(
        #         F.log_softmax(student_feature, dim=1), F.softmax(teacher_feature, dim=1)
        #     )

        if self.kd.get('feature_kd', False):
            feature = output_dict['feature']
            teacher_feature = output_dict['teacher_feature']
            N, C, H, W = teacher_feature.shape


            kd_loss_feature = self.FeatureDistillLoss(feature, teacher_feature)
            kd_loss = kd_loss_feature * self.kd['weight']

        if self.kd.get('decoder_kd', False):
            N, C, H, W = teacher_rm.shape
            teacher_rm = teacher_rm.permute(0,2,3,1).reshape(N*H*W, C)
            student_rm = rm.permute(0,2,3,1).reshape(N*H*W, C)
            kd_loss_rm = kl_loss_mean(
                    F.log_softmax(student_rm, dim=1), F.softmax(teacher_rm, dim=1)
                )

            N, C, H, W = teacher_psm.shape
            teacher_psm = teacher_psm.permute(0,2,3,1).reshape(N*H*W, C)
            student_psm = psm.permute(0,2,3,1).reshape(N*H*W, C)
            kd_loss_psm = kl_loss_mean(
                    F.log_softmax(student_psm, dim=1), F.softmax(teacher_psm, dim=1)
                )

            kd_loss += kd_loss_rm * self.kd['weight_rm'] + kd_loss_psm * self.kd['weight_psm']

        #kd_loss *= self.kd['weight']
        total_loss += kd_loss
        self.loss_dict.update({'total_loss': total_loss.item(),
                              'kd_loss': kd_loss.item()})


        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=''):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        kd_loss = self.loss_dict.get('kd_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || KD Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, kd_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Kd_loss'+suffix, kd_loss,
                            epoch*batch_len + batch_id)
  

    def FeatureDistillLoss(
        feature_lidar, feature_fuse, gt_boxes_bev_coords, gt_boxes_indices
    ):
        
        '''
        feature_lidar: [B, C, H, W]
        gt_boxes_bev_coords: [B, N, 4, 2]
        '''
        h, w = feature_lidar.shape[-2:]
        gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2) #[B, N, 1, 2]
        gt_boxes_bev_edge_1 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_2 = torch.mean(
            gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_3 = torch.mean(
            gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
        ).unsqueeze(2)
        gt_boxes_bev_edge_4 = torch.mean(
            gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
        ).unsqueeze(2)
        #[B,N,9,2]
        gt_boxes_bev_all = torch.cat(
            (
                gt_boxes_bev_coords,
                gt_boxes_bev_center,
                gt_boxes_bev_edge_1,
                gt_boxes_bev_edge_2,
                gt_boxes_bev_edge_3,
                gt_boxes_bev_edge_4,
            ),
            dim=2,
        )
        gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
        gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
        gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
        feature_lidar_sample = torch.nn.functional.grid_sample(
            feature_lidar, gt_boxes_bev_all
        )
        feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
        feature_fuse_sample = torch.nn.functional.grid_sample(
            feature_fuse, gt_boxes_bev_all
        )
        feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
        criterion = nn.L1Loss(reduce=False)
        loss_feature_distill = criterion(
            feature_lidar_sample[gt_boxes_indices], feature_fuse_sample[gt_boxes_indices]
        )
        loss_feature_distill = torch.mean(loss_feature_distill, 2)
        loss_feature_distill = torch.mean(loss_feature_distill, 1)
        loss_feature_distill = torch.sum(loss_feature_distill)
        weight = gt_boxes_indices.float().sum()
        #weight = reduce_mean(weight)
        loss_feature_distill = loss_feature_distill / (weight + 1e-4)
        return loss_feature_distill
