import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal
    

def info_nce_loss(
    query: Tensor,
    pos_key: Tensor,
    neg_key: Tensor,
    temp: Optional[float] = 0.1,
    reduction: Literal['none', 'mean', 'sum'] = 'none'
) -> Tensor:
    """
    Args:
        query (torch.Tensor): queries (B, D).
        pos_key (torch.Tensor): positive keys (B, N, D).
        neg_key (torch.Tensor): negative keys (B, M, D). 
        temp (float, optional): temperature coefficient.
    """
    assert reduction in ['none', 'mean', 'sum']
    B, N, _ = pos_key.shape
    query = F.normalize(query, dim=-1)
    pos_key = F.normalize(pos_key, dim=-1)
    neg_key = F.normalize(neg_key, dim=-1)
    query = query.unsqueeze(1)
    pos_logit = torch.sum(query * pos_key, dim=-1, keepdim=True).reshape(B * N, 1)
    neg_logit = (query @ neg_key.transpose(-2, -1)).repeat(1, N, 1).reshape(B * N, -1)
    logit = torch.cat([pos_logit, neg_logit], dim=-1)  # (B * N, 1 + M)
    label = torch.zeros(B * N, dtype=torch.long, device=query.device)
    # Adapted from MoCo codebase
    loss = F.cross_entropy(logit / temp, label, reduction='none')
    loss = loss.reshape(B, N).mean(dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()