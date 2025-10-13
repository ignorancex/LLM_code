import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_scatter

from util.ssl_match import FreeMatch

g = torch.Generator()
g.manual_seed(0)


def multi_class_loss_3d(cam_3d, cls_3D_label, inverse_map):
    cam_3d_feat = cam_3d.F[inverse_map]
    cam_3d_coord = cam_3d.C[inverse_map]
    bs = cam_3d_coord[-1, 0] + 1
    cls_feat = []
    cls_3D_label_all = torch.ones_like(cam_3d_feat).cuda()
    for b_i in range(bs):
        inds = cam_3d_coord[:, 0] == b_i
        cls_feat.append(torch.mean(cam_3d_feat[inds], dim=0, keepdim=True))
        cls_3D_label_all[inds] = cls_3D_label_all[inds] * cls_3D_label[b_i]
    cls_feat = torch.cat(cls_feat)
    loss_3d = F.multilabel_soft_margin_loss(cls_feat, cls_3D_label, weight=None)
    return loss_3d, cls_3D_label_all


def multi_class_loss_2d(cam_2d, cls_2D_label):
    predict_2d = F.adaptive_avg_pool2d(cam_2d.mean(-1), (1, 1)).squeeze(3).squeeze(2)
    loss_2d = F.multilabel_soft_margin_loss(predict_2d, cls_2D_label, weight=None)
    return loss_2d, cls_2D_label.unsqueeze(2).unsqueeze(3).unsqueeze(-1)


def cmg_loss(fea_3d, fea_2d, num_matches):
    loss_pp = 0
    for b in range(len(fea_3d)):
        fea_3d_v = fea_3d[b]
        fea_2d_v = fea_2d[b]
        if fea_3d_v.shape[0] > 0:
            idx = np.random.choice(fea_3d_v.shape[0], min(fea_3d_v.shape[0], num_matches), replace=False)
            k = fea_3d_v[idx]
            q = fea_2d_v[idx]
            logits = torch.mm(k, q.transpose(1, 0))
            logits = torch.div(logits, 0.07)
            logits = logits.contiguous()

            target = torch.arange(k.shape[0], device=k.device).long()
            target = F.one_hot(target, logits.shape[1])

            loss_pp += -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
    return loss_pp


def view_multi_view_feat(feat, V_NUM):
    V_B, C, H, W = feat.shape
    feat = feat.view(V_B//V_NUM, V_NUM, C, H, W)
    # feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    return feat


def negtive_softmax(cam_out, cls_label):
    cam_out_neg = cam_out + (1-cls_label) * -1000000
    return F.softmax(cam_out_neg, dim=1)


def convert_tensor_to_str(tensor):
    line_print = ""
    for i in range(len(tensor)):
        line_print += " & {:.2f}".format(float(tensor[i]))
    return line_print


class WSegPC_Loss(nn.Module):

    def __init__(self, args):
        super(WSegPC_Loss, self).__init__()
        self.loss_weight = {
            "cam_2d": 1.0,
            "cam_3d": 1.0,
            "cmg_2d": 1.0,
            "cmg_3d": 1.0,
            "rpc_2d": 1.0,
            "rpc_3d": 1.0,
            "seg_2d": 1.0,
            "seg_3d": 1.0,            
        }
        self.classes = args.classes
        self.arch = args.arch
        
        # cam learning
        self.loss_weight["cam_2d"] = args.cam_2d_weight
        self.loss_weight["cam_3d"] = args.cam_3d_weight

        # cross-modal feature guidance learning
        self.cmg_loss = args.cmg_loss

        self.cmg_2d_num_matches = args.cmg_2d_num_matches
        self.cmg_3d_num_matches = args.cmg_3d_num_matches

        self.loss_weight["cmg_2d"] = args.cmg_2d_weight
        self.loss_weight["cmg_3d"] = args.cmg_3d_weight

        # region-point consistency learning
        self.rpc_loss = args.rpc_loss

        self.loss_weight["rpc_2d"] = args.rpc_2d_weight
        self.loss_weight["rpc_3d"] = args.rpc_3d_weight

        self.rpc_match_alg_2d = FreeMatch(args.rpc_p_cutoff, args.classes)
        self.rpc_match_alg_3d = FreeMatch(args.rpc_p_cutoff, args.classes)

        self.pseudo_label_2d = args.get("pseudo_label_2d", False)
        self.loss_weight["seg_2d"] = args.get("pseudo_label_2d_weight", 1.0)
        self.pseudo_label_3d = args.get("pseudo_label_3d", False)
        self.loss_weight["seg_3d"] = args.get("pseudo_label_3d_weight", 1.0)

        self.warm_iter_cam = args.warm_iter_cam
        self.current_iter_cam = 0

        self.warm_iter_cmg = args.warm_iter_cmg
        self.current_iter_cmg = 0

        self.warm_iter_rpc = args.warm_iter_rpc
        self.current_iter_rpc = 0

        self.warm_iter_ps = args.warm_iter_ps
        self.current_iter_ps = 0

        self.ignore_label = args.ignore_label
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    def forward(self, output_3d, output_2d, sinput_student, output_3d_ema, output_2d_ema, label_3d, label_2d, labels_cls, links):
        # print(label_3d.shape)
        inverse_map = sinput_student.inverse_mapping
        unique_index = sinput_student.unique_index
        
        if self.pseudo_label_3d:
            label_3d_ps = label_3d[:,2]

        if self.pseudo_label_2d:
            label_2d_ps = label_2d[:,2]

        label_3d, label_3d_sp = label_3d[:,0], label_3d[:,1]
        label_2d, label_2d_sp = label_2d[:,0], label_2d[:,1]

        labels_cls_3d = labels_cls[:,:,0]
        labels_cls_2d = labels_cls[:,:,1]

        loss_dict = {}

        V_NUM = links.shape[-1]

        # CAM Loss
        warm_cam_weight = min(1.0*self.current_iter_cam/self.warm_iter_cam, 1.0)
        self.current_iter_cam += 1

        cam_3d_up = output_3d[0]
        loss_cam_3d, cls_3D_label = multi_class_loss_3d(cam_3d_up, labels_cls_3d, inverse_map)
        result_3d = negtive_softmax(cam_3d_up.F[inverse_map], cls_3D_label)
        loss_dict["cam_3d"] = loss_cam_3d * warm_cam_weight

        cam_2d_up = view_multi_view_feat(output_2d[0], V_NUM).permute(0, 2, 3, 4, 1).contiguous()
        loss_cam_2d, cls_2D_label = multi_class_loss_2d(cam_2d_up, labels_cls_2d)
        result_2d = negtive_softmax(cam_2d_up, cls_2D_label)
        loss_dict["cam_2d"] = loss_cam_2d * warm_cam_weight

        # CMG Loss
        if self.cmg_loss:
            self.current_iter_cmg += 1
            warm_cmg_weight = min(1.0*self.current_iter_cmg/self.warm_iter_cmg, 1.0)

            feat_proj_3d = output_3d[1]
            feat_detach_2d = output_2d[2]
            loss_cmg_3d = cmg_loss(feat_proj_3d, feat_detach_2d, self.cmg_3d_num_matches)
            loss_dict["cmg_3d"] = loss_cmg_3d * warm_cmg_weight

            feat_detach_3d = output_3d[2]
            feat_proj_2d = output_2d[1]
            loss_ppsl = cmg_loss(feat_proj_2d, feat_detach_3d, self.cmg_2d_num_matches)
            loss_dict["cmg_2d"] = loss_ppsl * warm_cmg_weight

        # RPC Loss
        if self.rpc_loss:
            self.current_iter_rpc += 1
            warm_rpc_weight = min(1.0*self.current_iter_rpc/self.warm_iter_rpc, 1.0)

            # super-point pre-process
            label_3d_sp_region = label_3d_sp[unique_index]
            label_2d_sp_region = label_2d_sp.permute(0, 3, 1, 2).contiguous()

            # get output and target
            cam_3d_up_region = cam_3d_up.F
            result_3d_region = output_3d_ema[0].F[inverse_map]

            cam_2d_up_region = view_multi_view_feat(output_2d[0], V_NUM).permute(0, 1, 3, 4, 2).contiguous()
            result_2d_region = view_multi_view_feat(output_2d_ema[0], V_NUM).permute(0, 1, 3, 4, 2).contiguous()

            result_3d_region = negtive_softmax(result_3d_region, cls_3D_label)
            result_3d_region = result_3d_region[unique_index]

            result_2d_region = result_2d_region + (1-cls_2D_label.permute(0, 4, 2, 3, 1).contiguous()) * -1000000
            result_2d_region = torch.softmax(result_2d_region, dim=-1)

            result_2d_region = result_2d_region.detach()
            result_3d_region = result_3d_region.detach()

            label_3d_sp_mask = label_3d_sp_region >= 0
            result_3d_region_self = torch.clone(result_3d_region)
            result_3d_region_self_mean = torch_scatter.scatter_mean(result_3d_region_self[label_3d_sp_mask], label_3d_sp_region[label_3d_sp_mask], dim=0)
            result_3d_region_self[label_3d_sp_mask] = result_3d_region_self_mean[label_3d_sp_region[label_3d_sp_mask]]
            loss_dict["rpc_3d"] = self.rpc_match_alg_3d.train_step(result_3d_region_self, cam_3d_up_region, None) * warm_rpc_weight

            label_2d_sp_mask = label_2d_sp_region >= 0
            result_2d_region_self = torch.clone(result_2d_region)
            result_2d_region_self_mean = torch_scatter.scatter_mean(result_2d_region_self[label_2d_sp_mask], label_2d_sp_region[label_2d_sp_mask], dim=0)
            result_2d_region_self[label_2d_sp_mask] = result_2d_region_self_mean[label_2d_sp_region[label_2d_sp_mask]]
            result_2d_region_self = result_2d_region_self.view(-1, self.classes)
            cam_2d_up_region = cam_2d_up_region.view(-1, self.classes)
            loss_dict["rpc_2d"] = self.rpc_match_alg_2d.train_step(result_2d_region_self, cam_2d_up_region, None) * warm_rpc_weight

        if self.pseudo_label_3d or self.pseudo_label_2d:
            self.current_iter_ps += 1
            warm_ps_weight = min(1.0*self.current_iter_ps/self.warm_iter_ps, 1.0)
            if self.pseudo_label_3d:
                loss_pseudo_3d = self.criterion(cam_3d_up.F, label_3d_ps)
                loss_dict["seg_3d"] = loss_pseudo_3d * warm_ps_weight
            if self.pseudo_label_2d:
                loss_pseudo_2d = self.criterion(cam_2d_up, label_2d_ps)
                loss_dict["seg_2d"] = loss_pseudo_2d * warm_ps_weight

        loss_sum = 0
        for k,v in loss_dict.items():
            loss_dict[k] = loss_dict[k] * self.loss_weight[k]
            loss_sum += loss_dict[k]
        loss_dict["loss"] = loss_sum
        return loss_dict, result_3d, result_2d