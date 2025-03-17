import math
from typing import List

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modal_utils import TrajectoryDec, MVS
from .utils import Mlp, GraphAttentionLayer, Normalize, IWPA


# cls
class LinearClsHead(nn.Module):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        if isinstance(x, List):
            hidden_states = torch.zeros_like(x[0])
            for item in x:
                hidden_states.add_(item)
        else:
            hidden_states = x
        cls_score = self.fc(hidden_states.mean(1))
        return cls_score
    
    def loss(self, cls_score, target) -> torch.Tensor:
        loss = self.loss_fn(cls_score, target)
        return loss
    
# multi-label cls
class MultiLabelLinearClsHead(nn.Module):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
    
    def forward(self, x):
        if isinstance(x, List):
            hidden_states = torch.zeros_like(x[0])
            for item in x:
                hidden_states.add_(item)
        else:
            hidden_states = x
        cls_score = self.fc(hidden_states.mean(1))
        return cls_score
    
    def loss(self, cls_score, target) -> torch.Tensor:
        loss = self.loss_fn(cls_score, target)
        return loss


# Table RegressPredict
class RegressHead(nn.Module):
    def __init__(self, embed_dim=1408, in_len=None, out_len=None):
        super().__init__()

        self.out_len = out_len
        if self.out_len:
            self.adapter = nn.Sequential(
                nn.Linear(in_len, out_len),
                nn.GELU(),
                nn.Linear(out_len, out_len)
            )
        
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, x):   
        if self.out_len:
            x = einops.rearrange(x, "B N D -> B D N")
            x = self.adapter(x)
            x = einops.rearrange(x, "B D N -> B N D") 
        pred = self.fc(x)
        return pred


# Trajectory 
class TrajHead(nn.Module):
    def __init__(self, embed_size, obs_len, int_num_layers_list, pred_len):
        super().__init__()
        in_size = 2
        
        self.cls_head = nn.Linear(embed_size, 1)
        self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
        self.social_decoder = TrajectoryDec.Decoder(embed_size, int_num_layers_list[1], 4, 2, islinear=False)
        self.reg_head = nn.Linear(embed_size, in_size*pred_len)
        self.out_proj = nn.Linear(4096, embed_size)
        
        self.test = False
        
    def spatial_interaction(self, ped, neis, mask):
        
        # ped [B K embed_size]
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]
        
        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]
        int_feat = self.social_decoder(ped, nei_embeddings, mask)  # [B K embed_size]

        return int_feat # [B K embed_size]
        
    def forward(self, ped_feat, closest_mode_indices, neis_obs, mask, num_k=20):

        ped_feat = self.out_proj(ped_feat)
        
        scores = self.cls_head(ped_feat).squeeze()
        if not self.test:
            index1 = torch.LongTensor(range(closest_mode_indices.shape[0])).to(scores.device)  # [B]
            index2 = closest_mode_indices
            closest_feat = ped_feat[index1, index2].unsqueeze(1)  # [B 1 embed_size]

            int_feat = self.spatial_interaction(closest_feat, neis_obs, mask)  # [B 1 embed_size]
            pred_traj = self.reg_head(int_feat.squeeze())  # [B pred_len*2]

            return pred_traj, scores

        if self.test:
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices  # [B num_k]
            top_k_indices = top_k_indices.flatten()  # [B*num_k]
            index1 = torch.LongTensor(range(ped_feat.shape[0])).to(scores.device)  # [B]
            index1 = index1.unsqueeze(1).repeat(1, num_k).flatten() # [B*num_k]
            index2 = top_k_indices # [B*num_k]
            top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
            top_k_feat = top_k_feat.reshape(ped_feat.shape[0], num_k, -1)  # [B num_k embed_size]

            int_feats = self.spatial_interaction(top_k_feat, neis_obs, mask)  # [B num_k embed_size]
            pred_trajs = self.reg_head(int_feats)  # [B num_k pred_size*2]

            return pred_trajs, scores
    
    def set_test(self):
        self.test = True
        
    def set_train(self):
        self.test = False
        
        
# Graph
class GraphHead(nn.Module):
    def __init__(self, in_steps=12, num_nodes=207, out_steps=12, output_dim=1, model_dim=4096, num_latents=207, out_num=2484):
        super().__init__()
        self.in_steps = in_steps
        self.num_nodes = num_nodes
        self.out_steps = out_steps
        self.output_dim = output_dim
        self.model_dim = model_dim

        self.num_latents_proj = nn.Linear(num_latents, out_num)
        
        self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        
    def forward(self, x):

        x = self.num_latents_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        B, N, D = x.shape
        x = x.reshape(B, self.in_steps, self.num_nodes, D)
        
        out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = out.reshape(
            B, self.num_nodes, self.in_steps * D
        )
        out = self.output_proj(out).view(
            B, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)
        
        return out

        
        
