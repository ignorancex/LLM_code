
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import numpy as np

from utils import time_performance_decorator


@time_performance_decorator(enable=False)
@torch.no_grad()
def _eigen_score(
    hidden_states: Tensor,
    alpha=1e-3,
):
    """
    hidden_states: (seq_cnt, hidden_dim)
    """

    # transfer to numpy to avoid cuda runtime error.
    # may be caused by floating point precision.

    hidden_states_np = hidden_states.to(torch.float64).cpu().numpy()

    k, d = hidden_states_np.shape
    j = np.eye(d) - (1 / d) * np.ones((d, d))
    z = hidden_states_np
    sigma = z @ j @ z.T
    regularized_covariance = sigma + alpha * np.eye(k)
    u, s, vh = np.linalg.svd(regularized_covariance)
    lambda_ = np.log(s)
    eigen_score = np.sum(lambda_) / k

    return float(eigen_score)


@time_performance_decorator(enable=False)
@torch.no_grad()
def calcu_layers_eigen_scores(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generated: list[str],
    layers: list[int],
    alpha=1e-3,
):

    gen_inputs = tokenizer(
        generated,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(llm.device)

    outputs = llm.forward(
        **gen_inputs,
        return_dict=True,
        output_hidden_states=True
    )

    valid_token_indices: Tensor = gen_inputs.attention_mask.sum(dim=1) - 1
    valid_token_indices = valid_token_indices.unsqueeze(1).unsqueeze(1).repeat(
        (1, 1, outputs.hidden_states[0].shape[-1])
    )

    scores = []
    for layer in layers:
        hidden_states = outputs.hidden_states[layer]
        hidden_state_per_seqs = hidden_states.gather(
            dim=1, index=valid_token_indices
        ).squeeze(1)
        score = _eigen_score(hidden_state_per_seqs, alpha)
        scores.append(score)

    torch.cuda.empty_cache()

    return scores
