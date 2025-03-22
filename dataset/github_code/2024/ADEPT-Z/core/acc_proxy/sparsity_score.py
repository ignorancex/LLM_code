"""
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-24 00:27:28
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-24 00:30:10
"""
import torch


__all__ = ["compute_sparsity_score"]


def compute_sparsity_score(
    model=None,
) -> float:
    with torch.no_grad():
        super_layer = model.super_layer
        sigma = torch.ones(
            1, 1, super_layer.n_waveguides, dtype=torch.cfloat, device=model.device
        )
        super_ps_layers = super_layer.build_ps_layers(1, 1)
        weights = super_layer.get_weight_matrix(super_ps_layers, sigma)
        sparsity_score = torch.count_nonzero(weights) / weights.numel()

    return sparsity_score.detach().item()
