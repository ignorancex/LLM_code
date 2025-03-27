# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from network.head import *
from network.resnet import *
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import network.vision_transformer as vits
import numpy as np
import os
import torchvision

class WCL(nn.Module):
    def __init__(self, pretrained_path, device, patch_size=16, dim_hidden=384, dim=256):
        super(WCL, self).__init__()
        
        if os.path.exists(pretrained_path):
            
            if "moco" in pretrained_path:
                self.net = vits.__dict__['moco_vit_small']()
                state_dict = torch.load(pretrained_path, map_location="cpu")
                dim_input = 384
                state_dict = state_dict['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
            elif "mae" in pretrained_path:
                self.net = vits.__dict__['mae_vit_base']()
                dim_input = 768
                state_dict = torch.load(pretrained_path, map_location="cpu")
                state_dict = state_dict['model']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('decoder') or k.startswith('mask_token'):
                        del state_dict[k]
            else:
                self.net = vits.__dict__['vit_small'](patch_size)
                state_dict = torch.load(pretrained_path, map_location="cpu")
                dim_input = 384
                weights_dict = torch.load(pretrained_path, map_location=device)
                state_dict = {k: v for k, v in weights_dict.items()
                                if self.net.state_dict()[k].numel() == v.numel()}
            self.net.load_state_dict(state_dict, strict=True)

        print(f"Dim Hidden:{dim_hidden}")
        self.head1 = ProjectionHead(dim_in=dim_input, dim_out=dim, dim_hidden=dim_hidden)
        self.head2 = ProjectionHead(dim_in=dim_input, dim_out=dim, dim_hidden=dim_hidden)
        self.head2d = ProjectionHead2d(in_dim=dim_input, bottleneck_dim=dim, hidden_dim=dim_hidden)
        self.device = device

    @torch.no_grad()
    def build_connected_component(self, dist):
        b = dist.size(0)
        dist = dist - torch.eye(b, b, device='cuda') * 2
        x = torch.arange(b, device='cuda').unsqueeze(1).repeat(1,1).flatten()
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels, device='cuda')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def self_distill(self, q, k, t=0.2):
        q = F.log_softmax(q / t, dim=-1)
        k = F.softmax((k) / t, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()
        
    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def forward(self, x1, x2, coords, flags, t=0.1):

        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.net._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

        attentions = self.net.get_last_selfattention(x1)
        attentions = attentions.detach()

        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        
        b = x1.size(0)

        # <feat1> 
        bakcbone_feat1 = self.net(x1)

        qkv1 = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4).contiguous()
        )
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        k1 = k1.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q1 = q1.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v1 = v1.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        if self.patch_size == 16:
            patch_nb = 14
        else:
            patch_nb = 28
        
        feat1_2d = k1.permute(0,2,1).contiguous()[:,:,1:].view(nb_im, -1, patch_nb, patch_nb)

        feat1_2d = F.normalize(self.head2d(feat1_2d))
        # <feat1> 

        # <feat2>
        bakcbone_feat2 = self.net(x2)

        qkv2 = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4).contiguous()
        )
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        k2 = k2.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q2 = q2.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v2 = v2.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        feat2_2d = k2.permute(0,2,1).contiguous()[:,:,1:].view(nb_im, -1, patch_nb, patch_nb)

        feat2_2d = F.normalize(self.head2d(feat2_2d))
        # <feat2>

        # <self_distillation>
        f1_aligned, f2_aligned = self.invaug(feat1_2d, coords[0], flags[0]), self.invaug(feat2_2d, coords[1], flags[1])
        loss_ovlp = self.self_distill(f1_aligned.permute(0, 2, 3, 1).contiguous().flatten(0, 2), f2_aligned.permute(0, 2, 3, 1).contiguous().flatten(0, 2)) 
        # <self_distillation>
        
        # <ce loss>
        feat1 = F.normalize(self.head1(bakcbone_feat1))
        feat2 = F.normalize(self.head1(bakcbone_feat2))

        other1 = feat1
        other2 = feat2

        prob = torch.cat([feat1, feat2]) @ torch.cat([feat1, feat2, other1, other2]).T / t
        diagnal_mask = (1 - torch.eye(prob.size(0), prob.size(1), device='cuda')).bool()
        logits = torch.masked_select(prob, diagnal_mask).reshape(prob.size(0), -1)

        first_half_label = torch.arange(b-1, 2*b-1).long().cuda()
        second_half_label = torch.arange(0, b).long().cuda()
        labels = torch.cat([first_half_label, second_half_label])
        # <ce loss>

        # <graph loss>
        feat1 = F.normalize(self.head2(bakcbone_feat1))
        feat2 = F.normalize(self.head2(bakcbone_feat2))
        all_feat1 = feat1
        all_feat2 = feat2
        all_bs = all_feat1.size(0)
        
        mask1 = self.build_connected_component(all_feat1 @ all_feat1.T).float()
        mask2 = self.build_connected_component(all_feat2 @ all_feat2.T).float()

        diagnal_mask = torch.eye(all_bs, all_bs, device='cuda')
        graph_loss =  5*self.sup_contra(feat1 @ all_feat1.T / t, mask2, diagnal_mask)
        graph_loss += 5*self.sup_contra(feat2 @ all_feat2.T / t, mask1, diagnal_mask)
        # <graph loss>
        return logits, labels, graph_loss, loss_ovlp