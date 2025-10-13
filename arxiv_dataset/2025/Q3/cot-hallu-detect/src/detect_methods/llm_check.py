
import torch
from torch import Tensor
import numpy as np

from utils import time_performance_decorator


@time_performance_decorator(enable=False)
@torch.no_grad()
def _attn_score(
    attns: Tensor,
    alpha=1e-6,
):
    """attns: (head_num, seq_len, seq_len)"""
    attention_matrix = attns.to(dtype=torch.float64).cpu().numpy()
    ker_jj = [
        np.diag(attn) for attn in attention_matrix
    ]
    return np.mean([
        np.sum(np.log(ker + alpha)).item()
        for ker in ker_jj
    ]).item()


@time_performance_decorator(enable=False)
@torch.no_grad()
def calcu_layers_attn_scores(
    answer_attns: tuple[Tensor],
    attn_layers: list[int],
    alpha=1e-6,
):
    scores = []
    for layer in attn_layers:
        attns = answer_attns[layer][0]  # 0 for no batch
        scores.append(_attn_score(attns, alpha))

    torch.cuda.empty_cache()

    return scores


@time_performance_decorator(enable=False)
@torch.no_grad()
def _hidden_score(
    hiddens: Tensor,
    alpha=1e-6,
):
    """hiddens: (seq_len, hidden_dim)"""
    H = hiddens.T
    Sigma2 = torch.mm(H.T, H).to(dtype=torch.float64).cpu().numpy()
    u, s, vh = np.linalg.svd(Sigma2, full_matrices=False)
    mean_log_det = 2 * np.sum(np.log(s+alpha)) / hiddens.shape[0]
    return mean_log_det.item()


@time_performance_decorator(enable=False)
@torch.no_grad()
def calcu_layers_hidden_scores(
    answer_hidden_states: tuple[Tensor],
    hidden_layers: list[int],
    alpha=1e-6,
):
    scores = []
    for layer in hidden_layers:
        hidden_states = answer_hidden_states[layer][0]  # 0 for no batch
        scores.append(_hidden_score(hidden_states, alpha))

    torch.cuda.empty_cache()

    return scores
