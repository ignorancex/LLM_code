import torch.nn as nn
import torch
from typing import Callable, Tuple, Any
from torch import Tensor


def bipartite_soft_matching1(
        metric: torch.Tensor,
        M: int,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]] | tuple[
    Callable[[Tensor], Tensor], Callable[[Tensor], Tensor], Any, Any]:
    B, N, C = metric.shape
    assert M >= N // 2, f"M must ≥ {N // 2}, but got {M}"
    r = N - M
    r = min(r, N // 2)

    if r <= 0:
        return lambda x: x, lambda x: x

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a = metric[:, ::2, :]
        b = metric[:, 1::2, :]
        scores = a @ b.transpose(-1, -2)

        max_sim, dst_idx = torch.max(scores, dim=-1)
        src_rank = torch.argsort(max_sim, dim=-1, descending=True)
        src_idx = src_rank[:, :r, None]

        dst_idx = dst_idx.unsqueeze(-1)
        dst_idx = dst_idx.gather(dim=1, index=src_idx)

        unm_idx = src_rank[:, r:, None]  # [B, N_even - r, 1]

    def merge(x: torch.Tensor) -> torch.Tensor:

        src = x[:, ::2, :]
        dst = x[:, 1::2, :]

        src_unm = src.gather(1, unm_idx.expand(-1, -1, C))
        src_merged = src.gather(1, src_idx.expand(-1, -1, C))

        dst_merged = dst.scatter_reduce(1, dst_idx.expand(-1, -1, C), src_merged, reduce='mean')

        return torch.cat([src_unm, dst_merged], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:

        src_unm = x[:, :unm_idx.shape[1], :]
        dst_merged = x[:, unm_idx.shape[1]:, :]

        restored = torch.zeros(B, N, C, device=x.device)

        restored.scatter_(
            dim=1,
            index=(2 * unm_idx).expand(-1, -1, C),
            src=src_unm
        )

        restored[:, 1::2, :] = dst_merged


        restored.scatter_(
            dim=1,
            index=(2 * src_idx).expand(-1, -1, C),
            src=dst_merged.gather(1, dst_idx.expand(-1, -1, C))
        )

        return restored

    return merge, unmerge, src_idx, dst_idx




class CAM(nn.Module):
    def __init__(self, dim, H, keep_ratio, merge_ratio, num_heads, qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_dim = dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, (num_heads * self.head_dim) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None
        self.seq_ranks = None
        self.cnt = 0
        self.keep_ratio = keep_ratio
        self.keep_num = int(H * H * self.keep_ratio)
        self.M = int((1 - merge_ratio) * self.keep_num)
        self.k = nn.Linear(dim, (num_heads * self.head_dim), bias=qkv_bias)
        self.global_token = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        B, N, C = x.shape

        # TCP
        global_token = self.global_token(x.mean(dim=1)).unsqueeze(1)
        k = self.k(x)  # B,N,C
        global_score = (global_token @ k.transpose(-2, -1)) * self.scale
        score, indices = torch.sort(global_score.squeeze(1).softmax(dim=-1), dim=-1, descending=True)
        pruned_indices = indices[:, self.keep_num:]
        mask = torch.ones((B, N), dtype=torch.float32, device=x.device)
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, pruned_indices.shape[1])
        mask[batch_indices, pruned_indices] = 0
        mask_inverted = 1 - mask
        x_prune = x * mask_inverted.unsqueeze(-1)
        reserve_indices = indices[:, :self.keep_num]

        reserve_batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.keep_num)
        x_reserve = x[reserve_batch_indices, reserve_indices, :]
        k_reserve = k[reserve_batch_indices, reserve_indices, :]
        merge, unmerge, src_index, dst_index = bipartite_soft_matching1(k_reserve, self.M)

        x_merge = merge(x_reserve)
        merge_num = x_merge.shape[1]

        # top-k sparse attention
        qkv = self.qkv(x_merge).reshape(B, merge_num, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        top_k = attn.shape[2] // 8
        # ===========================================
        # 1.A more efficient approach we designed
        # score2, index = attn.topk(top_k, dim=-1)
        # v_un = v.unsqueeze(-2).expand(-1, -1, -1, v.shape[2], -1)
        # idx = index.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        # v_g = torch.gather(v_un, dim=3, index=idx)
        # score2 = score2.unsqueeze(-2)
        # score2 = torch.softmax(score2, dim=-1)
        # score2 = self.attn_drop(score2)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # =================================================
        # 2.Mask-based approach
        mask = torch.zeros(B, self.num_heads, attn.shape[2], attn.shape[2], device=x.device, requires_grad=False)
        index = torch.topk(attn, k=top_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # TDP
        restored_x = torch.zeros_like(x)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, merge_num, self.num_heads * self.head_dim)
        x_recover = unmerge(x_attn)
        restored_x = restored_x.to(x.dtype)
        restored_x[reserve_batch_indices, reserve_indices, :] = x_recover
        x = restored_x + x_prune
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, H, W, C)

        return x
