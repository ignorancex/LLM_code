import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
sys.path.append(ROOT.as_posix())

from torch import Tensor
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self,
                 inp_dim: int = 512,
                 d_model: int = 512,
    ):
        super(Projection, self).__init__()
        self.proj = nn.Linear(inp_dim, d_model)
        
    def forward(self, inp: Tensor) -> Tensor:
        return self.proj(inp)


class AVCrossAttn(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 dropout: float = 0.1
    ):
        super(AVCrossAttn, self).__init__()

        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_mask = None
        
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)
    
    def sub_forward(self, 
                src_q: Tensor,
                src_v: Tensor,
                query: Optional[Tensor] = None,
    ) -> Tensor:

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        
        slf_attn = self.slf_attn(src_q, src_q, src_q)[0]
        crs_attn = self.crs_attn(src_q, src_v, src_v)[0]
        src_q = src_q + \
                self.dropout(slf_attn) + \
                self.dropout(crs_attn)
        src_q = self.norm1(src_q)
        
        src_q = src_q + \
                self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)
    
    def forward(self, 
                src_q: Tensor,
                src_v: Tensor,
                query: Optional[Tensor] = None,
                visualize: bool = False
    ) -> List[Tensor]:

        src1 = self.sub_forward(src_q, src_v)
        src2 = self.sub_forward(src_v, src_q)
        
        if visualize:
            return src1, src2, None
        return src1, src2


class AVQCrossAttn(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 dropout: float = 0.1
    ):
        super(AVQCrossAttn, self).__init__()

        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)
    
    def sub_forward(self, 
                src_q: Tensor,
                src_v: Tensor,
                query: Tensor,
                visualize: bool = False
    ) -> Tensor:

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        
        qst_attn, weight = self.qst_attn(src_q, query, query)
        slf_attn = self.slf_attn(src_q, src_q, src_q)[0]
        crs_attn = self.crs_attn(src_q, src_v, src_v)[0]
        src_q = src_q + \
                self.dropout(slf_attn) + \
                self.dropout(crs_attn) + \
                self.dropout(qst_attn)
        src_q = self.norm1(src_q)
        
        src_q = src_q + \
                self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2), weight
    
    def forward(self, 
                src_q: Tensor,
                src_v: Tensor,
                query: Tensor,
                visualize: bool = False
    ) -> List[Tensor]:

        src1, a_weight = self.sub_forward(src_q, src_v, query, visualize)
        src2, v_weight = self.sub_forward(src_v, src_q, query, visualize)
        
        if visualize:
            return src1, src2, [a_weight, v_weight]
        return src1, src2


class QstGrounding(nn.Module):
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 dropout: float = 0.1):
        super(QstGrounding, self).__init__()
        
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self,
                qst: Tensor,
                data: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        if isinstance(data, list):
            data = [d.permute(1, 0, 2) for d in data]
            data = torch.cat(data, dim=0)
        else:
            data = data.permute(1, 0, 2)
        qst = qst.unsqueeze(0)
        attn = self.attn(qst, data, data)[0].squeeze(0)
        feat = data.mean(dim=0) + self.dropout(self.mlp(attn))
        feat = self.norm(feat)
        return feat


class TempMoE(nn.Module):
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 topK: int = 5,
                 n_experts: int = 10,
                 sigma: int = 9,
                 dropout: float = 0.1,
                 vis_branch: bool = False,
    ):
        super(TempMoE, self).__init__()

        self.sigma = sigma
        self.topK = topK
        self.n_experts = n_experts
        
        if vis_branch:
            self.anorm = nn.LayerNorm(d_model)
            self.vnorm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.gauss_pred = nn.Sequential(
            nn.Linear(d_model, 2 * n_experts)
        )
        self.router = nn.Sequential(
            nn.Linear(d_model, n_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(d_model, int(d_model // 2)),
                nn.ReLU(),
                nn.Linear(int(d_model // 2), d_model)
            ])
            for _ in range(n_experts)
        ])
        self.experts.apply(self._init_weights)

        self.margin = (1 / (n_experts * 2)) # non overlapping center area
        self.center = torch.linspace(self.margin, 1-self.margin, self.n_experts)
        self.center.requires_grad_(False)
        self.router.apply(self._init_weights)
        self.gauss_pred.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_gaussian(self,
                          pred: torch.Tensor, 
                          topk_inds: torch.Tensor,
                          T: int = 60
    ) -> Tensor:
        # [refernce] https://github.com/minghangz/cpl
        weights = []
        centers = self.center.unsqueeze(0).repeat(pred.size(0), 1).to(pred.device)
        centers = centers + pred[:, :, 0]
        centers = torch.gather(centers, 1, topk_inds)
        widths = torch.gather(pred[:, :, 1], 1, topk_inds)
        
        for i in range(self.topK):
            center = centers[:, i]
            width = widths[:, i]
            
            weight = torch.linspace(0, 1, T)
            weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
            center = torch.clamp(center.unsqueeze(-1), min=0, max=1)
            width = torch.clamp(width.unsqueeze(-1), min=0.09) / self.sigma
            
            w = 0.3989422804014327
            weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
            weights.append(
                weight/weight.max(dim=-1, keepdim=True)[0]
            )
        return torch.stack(weights, dim=1)
    
    def get_output(self,
                   experts_logits: Tensor,
                   gauss_weight: Tensor,
                   topk_inds: Tensor,
                   topk_probs: Tensor,
                   shape: tuple,
    ) -> Tensor:
        B, T, C = shape
        
        experts_logits = torch.gather(
            experts_logits.permute(1, 0, 2, 3).reshape(B*T, self.n_experts, -1), 1, 
            topk_inds.repeat(T, 1).unsqueeze(-1).repeat(1, 1, C)
        )
        experts_logits = experts_logits.reshape(B, T, self.topK, -1).contiguous()
        output = [
            (gauss_weight[:, i, :].unsqueeze(1) @ experts_logits[:, :, i, :])
            for i in range(self.topK)
        ]
        output = torch.cat(output, dim=1) # [B, N_EXPERTS, C]
        output = topk_probs.unsqueeze(1) @ output
        return output
    
    def forward(self,
                qst: Tensor,                       # [B, D]
                data: Tensor,                      # [B, T, D] Audio | Video,
                sub_data: Optional[Tensor] = None, # [[B, T, D], [B, T, D]] Patch(Audio | Video)
    ) -> Union[Tensor, List[Tensor]]:
        B, T, C = data.size()
        data = data.permute(1, 0, 2)
        
        qst = qst.unsqueeze(0)                                              # [1, B, D]
        temp_w = self.qst_attn(qst, data, data)[0]                          # [1, B, C]
        temp_w = temp_w.squeeze(0)                                          # [B, C]
        router_logits = self.router(temp_w)                                 # [B, N_EXPERTS]
        router_probs = F.softmax(router_logits, dim=-1)                     # [B, N_EXPERTS]
        topk_probs, topk_inds = torch.topk(router_probs, self.topK, dim=-1) # [B, TOPK]
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)      # [B, TOPK]
        
        gauss_cw = self.gauss_pred(temp_w)                                  # [B, 2*N_EXPERTS]
        gauss_cw = gauss_cw.view(B, self.n_experts, 2)                      # [B, N_EXPERTS, 2]
        gauss_cw[:, :, 0] = torch.tanh(gauss_cw[:, :, 0]) * self.margin     
        gauss_cw[:, :, 1] = torch.sigmoid(gauss_cw[:, :, 1])
        gauss_weight = self.generate_gaussian(gauss_cw, topk_inds=topk_inds, T=T) # [B, TOPK, T]
        
        if sub_data is not None:
            a_data = sub_data[0].permute(1, 0, 2)                                             # [T, B, C]
            a_data = data + a_data
            a_outs = torch.stack([exprt(a_data) for exprt in self.experts], dim=2)            # [T, B, N_EXPERTS, C]
            a_outs = self.get_output(a_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))  # [B, 1, C]
            v_data = sub_data[1].permute(1, 0, 2)                                             # [T, B, C]
            v_data = data + v_data
            v_outs = torch.stack([exprt(v_data) for exprt in self.experts], dim=2)            # [T, B, N_EXPERTS, C]
            v_outs = self.get_output(v_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))  # [B, 1, C]
            return self.anorm(a_outs), self.vnorm(v_outs)
        else:
            main_outs = torch.stack([exprt(data) for exprt in self.experts], dim=2)                # [T, B, N_EXPERTS, C]
            main_outs = self.get_output(main_outs, gauss_weight, topk_inds, topk_probs, (B, T, C)) # [B, 1, C]
            return self.norm(main_outs)


class PatchSelecter(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 dropout: float = 0.1):
        super(PatchSelecter, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        
        self.vnorm = nn.LayerNorm(d_model)
        self.anorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        self.mlp.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                patch: Tensor, # [B, T, P, D]
                audio: Tensor, # [B, T, D]
                video: Tensor, # [B, T, D]
    ) -> List[Tensor]: 
        B, T, P, D = patch.size()
        audio = audio.reshape(B*T, 1, D) # [B*T, 1, D]
        video = video.reshape(B*T, 1, D) # [B*T, 1, D]
        patch = patch.reshape(B*T, P, D) # [B*T, P, D]
        video = video.permute(1, 0, 2)   # [1, B*T, D]
        audio = audio.permute(1, 0, 2)   # [1, B*T, D]
        patch = patch.permute(1, 0, 2)   # [P, B*T, D]

        patch = patch + self.slf_attn(patch, patch, patch)[0] # [P, B*T, D]
        data = torch.cat([video, audio], dim=0)               # [2, B*T, D]
        attn = self.crs_attn(data, patch, patch)[0]           # [2, B*T, D]
        attn = attn.permute(1, 0, 2)                          # [B*T, 2, D]
        attn = self.mlp(self.dropout(attn))                   # [B*T, 2, D]

        v, a = torch.chunk(attn, 2, dim=1)
        v = v.reshape(B, T, D).permute(1, 0, 2) # [T, B, D]
        a = a.reshape(B, T, D).permute(1, 0, 2) # [T, B, D]
        return [
            self.anorm(a.permute(1, 0, 2)),
            self.vnorm(v.permute(1, 0, 2)),
        ]


# [reference] TSPM -> https://github.com/GeWu-Lab/TSPM/blob/0106ce4127b8aa6728ee09439a59fbe7f19fe2f4/nets/net.py#L80-L146
class TSPM_topKSelection(nn.Module):

    def __init__(self, topK: int = 10):
        super(TSPM_topKSelection, self).__init__()
        
        self.topK = topK
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

    def QstQueryClipAttn(self, query_feat, kv_feat):
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0)
        attn_feat, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat, 
                                                      attn_mask=None, key_padding_mask=None)
        attn_feat = attn_feat.squeeze(0)
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)

        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)

        return attn, temp_weights

    def SelectTopK(self, temp_weights, audio_input, visual_input, patch_inputs, B, C):
        '''
            Top-k temporal segments selection
        '''

        # return temporal indices
        sort_index = torch.argsort(temp_weights, dim=-1)        # [B, 1, T]
        top_k_index = sort_index[:, :, -self.topK:]             # [B, 1, Top_K]
        top_k_index_sort, indices = torch.sort(top_k_index)     # [B, 1, Top_K]
        top_k_index_sort = top_k_index_sort.cpu().numpy()       # [B, 1, Top_K],

        output_audio = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_a_patches = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_v_patches = torch.zeros(B, self.topK, C).to(audio_input.device)
        for batch_idx in range(B):
            idx = 0
            for temp_idx in top_k_index_sort.tolist()[batch_idx][0]:
                output_audio[batch_idx, idx, :] = audio_input[batch_idx, temp_idx, :]
                out_a_patches[batch_idx, idx, :] = patch_inputs[0][batch_idx, temp_idx, :]
                out_v_patches[batch_idx, idx, :] = patch_inputs[1][batch_idx, temp_idx, :]
                idx = idx + 1
        return output_audio, (out_a_patches, out_v_patches)

    def forward(self, audio_input, visual_input, patch_inputs, qst_input):

        B, T, C = audio_input.size()
        temp_clip_attn_feat, temp_weights = self.QstQueryClipAttn(qst_input, visual_input)
        output_audio, output_patches = self.SelectTopK(temp_weights, audio_input, visual_input, patch_inputs, B, C)
        return output_audio, output_patches


