# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Optional

import torch


def alpha_blending(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Perform alpha blending between two tensors.

    Args:
    -----------
    a : torch.tensor
         First tensor to perform blending.
    b : torch.tensor
         Second tensor to perform blending.
    alpha : float
         Blending coefficient. Determines the contribution of second tensor towards the blended result. Range is [0, 1].

    Returns:
    --------
    torch.tensor
         Result of performing alpha blending between the two input tensors.

    """
    assert 0 <= alpha <= 1, alpha
    assert a.shape == b.shape, (a.shape, b.shape)
    return alpha * a + (1 - alpha) * b


def cache_aware_reweighting_current_cache(
    score: torch.Tensor,
    cache: torch.BoolTensor,
    gamma: float,
    fixed_top_n: Optional[int] = 0,
    warm_up: bool = False,
    use_alpha_blend=False,
    **kwargs,
) -> torch.Tensor:
    """
    Applies cache-aware reweighting to the input score tensor based on the current cache status.
    This function presumes that higher scores denote greater impacts on the decision-making process. Consequently,
    decreasing the score of items not in cache encourages cache hits. Nonetheless, it might yield
    inferior outcomes due to limited exploration of novel alternatives. Mitigation could involve setting `fixed_top_n`
    to retain a proportionate quantity of highly influent items unaffected by caching behaviour.

    Args:
    ------------
    score : torch.Tensor
        3D tensor storing the scores to be weighed. Expects shape [1, 1, N].
    cache : torch.ByteTensor
        Byte tensor signifying whether each item is currently in cache. Requires identical shape to `score`.
    gamma : float
        Factor controlling the degree of weight decrease for items not in cache. Larger values imply lesser influence.
        Range is [0, 1].
    fixed_top_n : int, optional
        Specifies the number of topmost influencing items whose weight remains invariant irrespective of caching
        behaviour. Default is 0.
    use_alpha_blend: bool, optional
        Whether to use gamma for de-weighting (default) or alpha-blending.
    warm_up : bool
        If true, warms-up the gamma values starting from zeros for the first token.
        This is implemented as gamma_hat = gamma * (1 - gamma ** t) where t is the token index in the sequence.
        This allows to avoid fixing the cache status too harshly with the first choices.

    Returns:
    ---------
    torch.Tensor
        Reweighed version of the input score tensor. Scores of items not in cache are decreased by `gamma` times.

    """
    original_shape = score.shape
    assert score.ndim == 3, score.shape
    assert score.shape[0] == score.shape[1] == 1, score.shape
    score = score[0, 0]
    assert score.shape == cache.shape, (score.shape, cache.shape)

    # Optionally warm-up gamma
    if warm_up:
        assert "t" in kwargs, kwargs
        gamma = 1 - (1 - gamma) * (1 - (gamma ** (kwargs["t"] / 4)))

    # Get mask for fixed neurons. By default, no neurons are fixed.
    mask_fixed_neurons = torch.zeros_like(cache)
    top_neurons_indices = torch.topk(score, fixed_top_n).indices
    mask_fixed_neurons[top_neurons_indices] = True

    # De-weight scores by gamma if neuron is not in cache and not fixed
    if use_alpha_blend:
        score[~mask_fixed_neurons] = alpha_blending(
            score[~mask_fixed_neurons],
            cache[~mask_fixed_neurons].to(dtype=score.dtype),
            gamma,
        )
    else:
        weight = torch.ones_like(score)
        weight[~cache & ~mask_fixed_neurons] = gamma
        score = score * weight

    score = score.unsqueeze(0).unsqueeze(0)
    assert score.shape == original_shape, (score.shape, original_shape)
    return score


def cache_aware_reweighting_approximate(
    score: torch.Tensor,
    cache: torch.BoolTensor,
    top_k: int,
    top_m: int,
    fixed_top_n: int = 0,
    gamma: float = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Implements the cache-aware masking based on the approximate caching idea explored in MoE, where neurons with top
    M scores and in cache are picked if some of the neurons in top K are not in cache.
    Neurons are selected based on the following priority until top_k neurons are picked:
    - 1) fixed_top_n
    - 2) top_k & cache
    - 3) top_m & cache
    - 4) top_k & not cache
    For each group, the neurons with highest score (activations) are selected first.
    If gamma is not zero, re-weighing based on normalized cache statistics is applied (except for the fixed_top_n).

    Args:
    -----------
    score : torch.Tensor
        Tensor of neuron importance scores.
    cache : torch.BoolTensor
        Tensor of layer caching status.
    top_k : int
        Number of neurons selected.
    top_m : int
        Upper limit of neurons considered for selection. (m > k)
    fixed_top_n : int, optional
        Number of top neurons guaranteed to be included in the selection. Default is 0. (n < k)
    gamma : float, optional
        Weight assigned to cache stats in re-weighting. Default is 0. Range is [0, 1].

    Returns:
    --------
    Tensor
        Mask selecting the approximately chosen neural network layers.

    """
    original_shape = score.shape
    assert score.ndim == 3, score.shape
    assert score.shape[0] == score.shape[1] == 1, score.shape
    score = score[0, 0]
    assert score.shape == cache.shape, (score.shape, cache.shape)
    assert fixed_top_n < top_k < top_m, (fixed_top_n, top_k, top_m)

    # Get fixed-top-n mask
    mask_top_n = torch.zeros_like(cache)
    idx_top_n = torch.topk(score, fixed_top_n).indices
    mask_top_n[idx_top_n] = True

    # Optionally, re-weight scores based on cache statistics
    if gamma > 0:
        assert "cache_stats" in kwargs
        cache_stats = kwargs["cache_stats"]
        assert score.shape == cache_stats.shape
        cache_stats = cache_stats / (cache_stats.max() + 1e-7)
        score = alpha_blending(cache_stats, score, gamma)
        assert torch.isnan(score).any() == False, torch.isnan(score).sum()

    # Get top-k and top-m masks
    mask_top_k = torch.zeros_like(cache)
    mask_top_m = torch.zeros_like(cache)
    idx_top_k = torch.topk(score, top_k).indices
    idx_top_m = torch.topk(score, top_m).indices
    mask_top_k[idx_top_k] = True
    mask_top_m[idx_top_m] = True

    # Initialize output mask
    mask = torch.zeros_like(cache)

    # Add fixed top_n neurons
    mask[mask_top_n] = True

    # Add best from top_k neurons that are in cache (if gamma > 0, then top_n might not be in top_k deweighted)
    count_remaining = top_k - mask.sum()
    if count_remaining > 0:
        mask_top_k_cache = ~mask * mask_top_k & cache
        if count_remaining > mask_top_k_cache.sum():
            # If still not enough, add all
            mask[mask_top_k_cache] = True
        else:
            # Otherwise, add the best from top_m until we have k neurons in total
            mask_top_k_until_full = torch.zeros_like(score[mask_top_k_cache], dtype=torch.bool)
            indices_top_k_until_full = torch.topk(
                score[mask_top_k_cache], k=count_remaining
            ).indices
            mask_top_k_until_full[indices_top_k_until_full] = True
            mask[mask_top_k_cache] = mask_top_k_until_full
    assert mask.sum() <= top_k, (mask.sum(), top_k)

    # Add best from top_m neurons that are in cache
    count_remaining = top_k - mask.sum()
    if count_remaining > 0:
        mask_top_m_cache = ~mask * mask_top_m & cache
        if count_remaining > mask_top_m_cache.sum():
            # If still not enough, add all
            mask[mask_top_m_cache] = True
        else:
            # Otherwise, add the best from top_m until we have k neurons in total
            mask_top_m_until_full = torch.zeros_like(score[mask_top_m_cache], dtype=torch.bool)
            indices_top_m_until_full = torch.topk(
                score[mask_top_m_cache], k=count_remaining
            ).indices
            mask_top_m_until_full[indices_top_m_until_full] = True
            mask[mask_top_m_cache] = mask_top_m_until_full
    assert mask.sum() <= top_k, (mask.sum(), top_k)

    # Add best from top_k neurons that are not in cache
    count_remaining = top_k - mask.sum()
    if count_remaining > 0:
        mask_top_k_not_cache = ~mask * mask_top_k & ~cache
        mask_top_k_until_full = torch.zeros_like(score[mask_top_k_not_cache], dtype=torch.bool)
        indices_top_k_until_full = torch.topk(
            score[mask_top_k_not_cache], k=count_remaining
        ).indices
        mask_top_k_until_full[indices_top_k_until_full] = True
        mask[mask_top_k_not_cache] = mask_top_k_until_full
    assert mask.sum() == top_k, (mask.sum(), top_k)

    mask = mask.to(dtype=score.dtype)  # cast to float (in theory these are scores)
    mask = mask.unsqueeze(0).unsqueeze(0)
    assert mask.shape == original_shape, (mask.shape, original_shape)
    return mask
