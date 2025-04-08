import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
import cv2
import os
FLAGS = flags.FLAGS

from network.backbone import convnext_backbone
from network.xyz_head import TopDownMaskXyzHead
from network.conv_pnp_net import ConvPnPNet
from network.pose_head import SizeHead
from tools.rot_utils import get_rot_mat_y_first
from tools.umeyama import pose_from_umeyama
import torch.nn.functional as F
from .pose_utils.pose_from_pred import pose_from_pred
from .pose_utils.pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_utils.pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
import copy
import torch
import torch.nn as nn
import numpy as np
from .pose_utils.pose_error import re, te
from .pose_utils.pose_utils import quat2mat_torch
from .pose_utils.rot_reps import rot6d_to_mat_batch
from .extractor_dino import *
from network.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from network.att_mask_head import AttentionMaskHead
from torchvision.transforms import Resize, InterpolationMode
import itertools


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def get_mask_prob(pred_mask, mask_loss_type):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"Unknown mask loss type: {mask_loss_type}")
    return mask_prob


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()

def symmetry_rotation_matrix_y(number=30):
    result = []
    for i in range(number):
        theta = 2 * np.pi / number * i
        r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        result.append(r)
    result = np.stack(result)
    return result

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        featdim = 128
        pnp_input_feat = 3 + 2
        if FLAGS.coor_type == 'both':
            pnp_input_feat += 3
            if FLAGS.pnp_att_mask:
                assert FLAGS.mask_attention_type == 'concat'
                pnp_input_feat += FLAGS.mask_dim
        self.pnp_net = ConvPnPNet(pnp_input_feat, featdim=featdim, mask_attention_type=FLAGS.mask_attention_type)
        add_dim = 0
        if FLAGS.dinov2 and FLAGS.coor_type == 'dino':
            self.dino_xyz_head = TopDownMaskXyzHead(in_dim=384+add_dim, up_types=['deconv'],
                                                    num_conv_per_block=6)
            self.size_head = SizeHead(in_dim=384+add_dim)
        elif FLAGS.dinov2 and FLAGS.coor_type == 'both':
            self.backbone = convnext_backbone()
            self.xyz_head = TopDownMaskXyzHead(in_dim=1024+add_dim)
            self.size_head = SizeHead(in_dim=1024+add_dim)
            self.dino_xyz_head = TopDownMaskXyzHead(in_dim=384+add_dim, up_types=['deconv'],
                                                    num_conv_per_block=6)
        else:
            self.backbone = convnext_backbone()
            self.xyz_head = TopDownMaskXyzHead(in_dim=1024+add_dim)
            self.size_head = SizeHead(in_dim=1024+add_dim)
        self.out_res = FLAGS.out_res
        self.use_pnp = FLAGS.use_pnp > 0
        self.COORD_2D_TYPE = "abs"
        self.ROT_TYPE = "allo_rot6d"
        self.TRANS_TYPE = "centroid_z"
        self.Z_TYPE = "REL"  # REL | ABS | LOG | NEG_LOG  (only valid for centroid_z)

        if FLAGS.dinov2:
            self.extractor = ViTExtractor('dinov2_vits14', 14)

        self.use_attention_mask = FLAGS.pnp_att_mask
        self.att_use_rgb = FLAGS.att_use_rgb
        if self.use_attention_mask:
            dino_feature_dim = 384+add_dim if FLAGS.dinov2 and FLAGS.att_use_dino else 0
            self.attention_mask_head = AttentionMaskHead(dino_feature_dim=dino_feature_dim, use_rgb_feature=FLAGS.att_use_rgb, add_dim=add_dim, out_dim=FLAGS.mask_dim)
            assert FLAGS.att_type in ['mask', 'var']

        self.resize_func_out = Resize(FLAGS.out_res, interpolation=InterpolationMode.NEAREST)



    def forward(self, data, device, do_loss=False):
        img = data['roi_img'].to(device)
        img_pth = data["img_path"]
        if do_loss:
            mask = data['roi_mask_deform'].to(device)
        else:
            mask = data['roi_mask'].to(device)
        if FLAGS.mask_input:
            img = mask * img
        bs = img.shape[0]
        one_hot = data['one_hot'].to(device)
        # x.shape [bs, 3, 256, 256]
        if FLAGS.dinov2:
            dino_img = data['dino_img'].to(device)
            with torch.no_grad():
                descriptors = self.extractor.extract_descriptors(dino_img, 11, 'token')
                # [bs, 1, 32*32, 384]
                dino_feat = descriptors.view(bs, 32, 32, 384)
                dino_feat = dino_feat.permute(0,3,1,2)
                dino_feat_out = self.resize_func_out(dino_feat)
        else:
            dino_feat = None
            dino_feat_out = None
        if FLAGS.coor_type == 'convnext':
            conv_feat = self.backbone(img)  # [bs, c, 8, 8]
            pred_mask, coor_x, coor_y, coor_z = self.xyz_head(conv_feat)
            pred_size, nocs_scale = self.size_head(conv_feat)
            coor_xyz = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
            if not FLAGS.pre_norm:
                norm = pred_size.norm(dim=1)
                coor_xyz = coor_xyz*norm.view(norm.shape[0],1,1,1)


        elif FLAGS.coor_type == 'dino':
            conv_feat = None
            pred_mask, coor_x, coor_y, coor_z = self.dino_xyz_head(dino_feat)
            pred_size, nocs_scale = self.size_head(dino_feat)
            coor_xyz = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
        elif FLAGS.coor_type == 'both':
            conv_feat = self.backbone(img)  # [bs, c, 8, 8]
            _, coor_x, coor_y, coor_z = self.xyz_head(conv_feat)
            coor_xyz_conv = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
            _, coor_x_dino, coor_y_dino, coor_z_dino = self.dino_xyz_head(dino_feat)
            coor_xyz_dino = torch.cat([coor_x_dino, coor_y_dino, coor_z_dino], dim=1)  # BCHW
            pred_size, nocs_scale = self.size_head(conv_feat)

        else:
            raise NotImplementedError
        mean_size = data['mean_size'].to(pred_size.device)
        mean_scale = mean_size.norm(dim=1)
        if FLAGS.pre_norm:
            pred_size = pred_size + mean_size / mean_scale.unsqueeze(-1)
        else:
            pred_size = pred_size + mean_size

        mask_out = self.resize_func_out(mask)
        if 'roi_depth' in data.keys():
            depth_gt = self.resize_func_out(data['roi_depth'])
            depth_gt = depth_gt.squeeze()

        att_mask_out = None
        if FLAGS.coor_type == 'both':
            roi_coord_2d = data['roi_coord_2d'].to(device)
            if self.use_attention_mask:
                coor_feat_conv = torch.cat([coor_xyz_conv, roi_coord_2d], dim=1)
                att_mask_conv, log_var_conv = self.attention_mask_head(coor_feat_conv, mask_out, dino_feat_out, conv_feat)
                coor_feat_dino = torch.cat([coor_xyz_dino, roi_coord_2d], dim=1)
                att_mask_dino, log_var_dino = self.attention_mask_head(coor_feat_dino, mask_out, dino_feat_out, conv_feat)
                att_mask = torch.cat([att_mask_conv, att_mask_dino], dim=1) if FLAGS.att_type == 'mask' \
                    else torch.cat([log_var_conv, log_var_dino], dim=1)
                att_mask = att_mask * mask_out
                log_var = [log_var_conv, log_var_dino]
                att_mask_out = [att_mask_conv, att_mask_dino]
            else:
                att_mask = mask_out
                log_var = None
            coor_xyz = [coor_xyz_conv, coor_xyz_dino]
            coor_feat = torch.cat([coor_xyz_conv, coor_xyz_dino, roi_coord_2d], dim=1)
        else:
            roi_coord_2d = data['roi_coord_2d'].to(device)
            coor_feat = torch.cat([coor_xyz, roi_coord_2d], dim=1)

            if self.use_attention_mask:
                att_mask, log_var = self.attention_mask_head(coor_feat, mask_out, dino_feat_out)
                att_mask = att_mask if FLAGS.att_type == 'mask' else log_var
                att_mask = att_mask * mask_out
            else:
                att_mask = mask_out
                log_var = None

        pred_rot_, pred_t_ = self.pnp_net(coor_feat, mask_attention=att_mask)

        # convert pred_rot to rot mat -------------------------
        rot_type = self.ROT_TYPE
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        roi_cams = data['cam_K'].to(device)
        roi_whs = data['roi_wh'].to(device)
        roi_centers = data['bbox_center'].to(device)
        resize_ratios = data['resize_ratio'].to(device)

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if self.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=self.Z_TYPE,
                # is_train=True
                is_train=do_loss,
            )
        elif self.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=do_loss,
            )
        elif self.TRANS_TYPE == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in rot_type, is_train=do_loss
            )
        else:
            raise ValueError(f"Unknown trans type: {self.TRANS_TYPE}")
        out_dict = {"rot": pred_ego_rot, "trans": pred_trans, "size": pred_size}
        out_dict.update(
            {
                "mask": mask_out,
                "coor": coor_xyz,  # BCHW
                "log_var": log_var,
                "att_mask": att_mask,
                'att_mask_out': att_mask_out,
                'dino_feat': dino_feat
            }
        )
        return out_dict



    def build_params_optimizer(self, training_stage_freeze=None):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []
        if 'backbone' in training_stage_freeze:
            for param in self.backbone.parameters():
                with torch.no_grad():
                    param.requires_grad = False

        # backbone
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, self.parameters()),
                "lr": float(FLAGS.lr),
            }
        )

        return params_lr_list