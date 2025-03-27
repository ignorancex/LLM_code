import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy, smooth_l1_loss
import torch.nn.functional as F


@HEADS.register_module()
class OKPDBBoxHead(BBoxHead):
    def __init__(self,
                 num_shared_convs=2,
                 num_reg_convs=0,
                 global_sp_len=5,
                 global_ch=64,
                 num_kp=16,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(OKPDBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs > 0)
        assert num_kp > 0
        self.num_shared_convs = num_shared_convs
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.global_sp_len = min(global_sp_len, self.roi_feat_area)
        self.global_ch = global_ch 
        self.num_kp = num_kp 
        self.global_sp_pool = nn.AdaptiveAvgPool2d((self.global_sp_len,self.global_sp_len))
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # add concentrate convs 
        self.shared_convs_1, self.shared_convs_2, last_layer_dim = \
            self._add_okpd_branch(
                self.num_shared_convs, self.in_channels)
        self.shared_out_channels = last_layer_dim

        self.dropout = nn.Dropout(p=0.01)
        self.use_fc = (fc_out_channels > 0)
        if self.use_fc:
            self.feat_fc = nn.Linear(last_layer_dim, fc_out_channels)
            last_layer_dim = fc_out_channels

        self.fc_cls = nn.Linear(last_layer_dim, self.num_classes + 1)
        out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                       self.num_classes)
        self.fc_reg = nn.Linear(last_layer_dim, out_dim_reg)

    def _add_okpd_branch(self,
                         num_branch_convs,
                         in_channels,
                         is_shared=False):
        last_layer_dim = in_channels
        branch_convs_1 = nn.ModuleList()
        branch_convs_2 = nn.ModuleList()
        max_dilation = int(self.roi_feat_size[0] / 2)
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                branch_convs_1.append(
                    ConvModule(
                        in_channels,
                        int(in_channels/8),
                        3,
                        conv_cfg=None,
                        norm_cfg=None,
                        padding=min(2*i+1,max_dilation),
                        dilation=min(2*i+1,max_dilation),
                        bias=True, groups=int(in_channels/8)//8))
                branch_convs_2.append(
                    ConvModule(
                        int(in_channels/8),
                        in_channels,
                        1,
                        padding=0,
                        conv_cfg=None,
                        norm_cfg=None,
                        bias=True))

        self.kp_discover_conv = ConvModule(in_channels,
                                           self.num_kp,
                                           1,
                                           padding=0,
                                           conv_cfg=None,
                                           norm_cfg=None,
                                           bias=True)

        self.reduce_ch = self.global_ch != in_channels
        if self.reduce_ch:
            self.global_ch_conv = ConvModule(in_channels,
                                     self.global_ch,
                                     1,
                                     padding=0,
                                     conv_cfg=self.conv_cfg,
                                     norm_cfg=self.norm_cfg,
                                     bias=True)
        last_layer_dim = int(self.global_ch * self.global_sp_len**2 + self.num_kp * self.roi_feat_area + in_channels * self.num_kp)
        return branch_convs_1, branch_convs_2, last_layer_dim

    def init_weights(self):
        if self.use_fc:
            nn.init.kaiming_uniform_(self.feat_fc.weight)
            nn.init.constant_(self.feat_fc.bias, 0)
        ##
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for i in range(self.num_shared_convs):
                x_out = self.shared_convs_1[i](F.relu(x)) #
                x_out = self.shared_convs_2[i](F.relu(x_out))
                x = x + x_out

        global_x = self.global_sp_pool(x) #
        global_x = self.global_ch_conv(global_x) if self.reduce_ch else global_x
        global_x = global_x.reshape(global_x.size(0), -1)

        # kp gen
        kp_out = self.kp_discover_conv(F.relu(x)) 
        kp_gen_data = kp_out.reshape(x.size(0), -1)

        # truncated maximum regularization
        kp_out_prior = F.relu(kp_out + 0.5)
        max_kp_prior = self.global_max_pool(kp_out_prior)
        kp_gen_rst = kp_out_prior / (F.relu(max_kp_prior - 1.0) + 1.1) #

        sp_max_kp_gen_rst, sp_max_kp_gen_inds = torch.max(kp_gen_rst.view(kp_gen_rst.size(0), kp_gen_rst.size(1), -1), dim=2)
        ch_sum_kp_gen_rst = torch.sum(kp_gen_rst, 1)
        ch_sum_kp_gen_rst = ch_sum_kp_gen_rst[:,None,:,:]
        max_ch_sum_kp_gen_rst = self.global_max_pool(ch_sum_kp_gen_rst)
 
        with torch.no_grad():
            inc_inds = torch.arange(x.size(0),device=x.device).repeat_interleave(self.num_kp)
            max_kp_gen_inds = sp_max_kp_gen_inds.view(-1)
            inc_inds = inc_inds.view(-1)
            max_kp_gen_inds_g = inc_inds * self.roi_feat_area + max_kp_gen_inds #global indices
        kp_feat = x.permute((1,0,2,3)).reshape(x.size(1), -1)
        kp_feat = torch.index_select(kp_feat, 1, max_kp_gen_inds_g)

        kp_feat = kp_feat.reshape( (x.size(1), x.size(0), -1) )
        kp_feat = kp_feat.permute( (1,2,0) )
        kp_feat = kp_feat.reshape(x.size(0), -1)
        fc_feat = torch.cat( (global_x, kp_feat, kp_gen_data), 1) #
        fc_feat = self.dropout(fc_feat)
        if self.use_fc:
            fc_feat = F.relu(self.feat_fc(fc_feat))

        # separate branches
        x_cls = fc_feat
        x_reg = fc_feat

        cls_score = self.fc_cls(x_cls) #if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) #if self.with_reg else None
        if self.training:
            return (sp_max_kp_gen_rst, max_ch_sum_kp_gen_rst, cls_score, bbox_pred)
        else:
            return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             okpd_outs,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        cls_score = okpd_outs[-2]
        bbox_pred = okpd_outs[-1]
        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        if cls_score.numel() > 0:
            losses['loss_okpd_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['okpd_acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_okpd_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0

        if self.num_kp > 0:
            kp_max_target = torch.zeros_like(okpd_outs[0])
            kp_max_target[pos_inds,...] = 1.
            kp_max_wts = torch.zeros_like(okpd_outs[0]) + 0.5 
            fn_inds = (kp_max_target > 0.) & (okpd_outs[0] < 0.5)
            fp_inds = (kp_max_target < 1.) & (okpd_outs[0] > 0.5)
            kp_max_wts[fp_inds|fn_inds] = 1.0 
            Ld_loss = smooth_l1_loss(okpd_outs[0] * kp_max_wts, kp_max_target * kp_max_wts, beta=0.05)

            pos_mask = pos_inds.float()[:,None,None,None]
            kp_ch_sum_target = torch.zeros_like(okpd_outs[1]) 
            kp_ch_sum_target[pos_inds,...] = 1.
            Lu_loss = smooth_l1_loss(okpd_outs[1] * pos_mask, kp_ch_sum_target * pos_mask, beta=1.0)
            losses['kp_loss'] = 0.03 * (Ld_loss + Lu_loss)

        return losses



