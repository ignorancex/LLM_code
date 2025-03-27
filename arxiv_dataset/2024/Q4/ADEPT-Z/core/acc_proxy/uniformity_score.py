"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-19 11:14:46
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:34:25
"""

import torch
import torch.nn.functional as F

from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear

__all__ = ["compute_uniformity_score_kl"]

_conv = (SuperBlockConv2d,)
_linear = (SuperBlockLinear,)


def compute_uniformity_score_kl(model=None, device="cuda:0") -> float:
    super_layer = model.super_layer
    super_ps_layers = super_layer.build_ps_layers(1, 1)
    for layer in super_ps_layers:
        layer.reset_parameters(alg="identity")
    U, V = super_layer.get_UV(super_ps_layers, 1, 1)

    U_abs = U.abs().square().clamp(min=1e-8)
    V_abs = V.abs().square().clamp(min=1e-8)
    # print(W_abs)

    W_abs = torch.cat((U_abs, V_abs), dim=0)

    # Compute the uniform distribution matrix, ensuring its elements sum to 1
    uniform_prob = torch.full(
        (super_layer.n_waveguides, super_layer.n_waveguides),
        fill_value=1 / super_layer.n_waveguides,
        device=device,
    )

    # Convert the predicted distribution to log form
    log_W_prob = torch.log(W_abs)

    # Calculate KL divergence
    kl_divergences = F.kl_div(log_W_prob, uniform_prob, reduction="batchmean")

    # kl_divergences = torch.nn.KLDivLoss(W_abs, uniform_prob)

    kl_divergences_mean = kl_divergences.mean()

    return kl_divergences_mean.item()


def compute_uniformity_score_js(model=None, device="cuda:0", num_samples=5) -> float:
    super_layer = model.super_layer

    super_ps_layers = super_layer.build_ps_layers(num_samples, 1)
    for layer in super_ps_layers:
        layer.reset_parameters(alg="uniform")

    U, V = super_layer.get_UV(super_ps_layers, num_samples, 1)

    U_abs = U.abs().square().clamp(min=1e-8)
    V_abs = V.abs().square().clamp(min=1e-8)

    W_abs = torch.cat((U_abs, V_abs), dim=0)
    # Compute the uniform distribution matrix, ensuring its elements sum to 1
    uniform_prob = torch.full(
        (super_layer.n_waveguides, super_layer.n_waveguides),
        fill_value=1 / super_layer.n_waveguides,
        device=device,
    )
    M = 0.5 * (W_abs + uniform_prob)

    # Convert the predicted distribution to log form
    log_W_prob = torch.log(W_abs)
    log_uniform_prob = torch.log(uniform_prob)

    # Calculate KL divergence
    kl_divergences1 = F.kl_div(log_W_prob, M, reduction="none")
    kl_divergences1_mean = kl_divergences1.mean(dim=1, keepdim=True).mean()

    kl_divergences2 = F.kl_div(log_uniform_prob, M, reduction="none")
    kl_divergences2_mean = kl_divergences2.mean(dim=1, keepdim=True).mean()

    js_divergence = (kl_divergences1_mean + kl_divergences2_mean) / 2
    js_divergences_mean = js_divergence.mean()

    return js_divergences_mean.item()
