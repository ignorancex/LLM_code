"""
Date: 2024-03-31 21:19:44
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:34:08
FilePath: /ADEPT_Zero/core/acc_proxy/robust_score.py
"""
import torch

from core.models.layers.super_mesh import (
    SuperBatchedPSLayer,
    SuperMeshADEPTZero,
    SuperZeroCRLayer,
    SuperZeroDCLayer,
)

__all__ = ["compute_exp_error_score"]


def compute_exp_error_score(
    super_layer: SuperMeshADEPTZero,
    super_ps_layer: SuperBatchedPSLayer,
    sigma,
    num_samples: 16,
    phase_noise_std: 0.2,
    dc_noise_std: 0.2,
    cr_tr_noise_std: 0.2,
    cr_phase_noise_std: 0.2,
    device="cuda:0",
) -> float:
    # compute the expected relative L1 error
    # evaluate the robustness for all phases
    # E[|w-w'|/|w|] for all noises and all weights
    # for MC sampling, need to sample different noises and different weights (i.e., phi, sigma)
    super_ps_layer = super_layer.build_ps_layers(
        num_samples, 1
    )  # resample phi: num_samples(pxq) blocks at one time
    for m in super_ps_layer:
        m.reset_parameters(alg="uniform")  # uniform initialization

    sigma = torch.randn(
        1, num_samples, super_layer.n_waveguides, dtype=torch.cfloat, device=device
    )  # resample sigma [num_samples, 1, n_waveguides]
    for m in super_layer.super_layers_all:
        if isinstance(m, SuperZeroDCLayer):
            m.set_dc_noise(noise_std=0)
        if isinstance(m, SuperZeroCRLayer):
            m.set_cr_noise(tr_noise_std=0, phase_noise_std=0)

    ideal_weights = super_layer.get_weight_matrix(
        super_ps_layer, sigma
    )  # get ideal weight matrix
    ideal_weights_real = torch.view_as_real(ideal_weights)  # [p,q,k,k,2]
    ideal_weights_l1norm = ideal_weights_real.norm(1, dim=(-3, -2, -1))  # [p,q]

    # turn on noise
    for m in super_ps_layer:
        # turn on phase noise (standard deviation also random)
        m.set_phase_noise(noise_std=phase_noise_std)
    for m in super_layer.super_layers_all:
        if isinstance(m, SuperZeroDCLayer):
            m.set_dc_noise(noise_std=dc_noise_std)  # turn on dc noise
        if isinstance(m, SuperZeroCRLayer):
            m.set_cr_noise(
                tr_noise_std=cr_tr_noise_std, phase_noise_std=cr_phase_noise_std
            )  # turn on cr noise

    noisy_weights = super_layer.get_weight_matrix(super_ps_layer, sigma)
    noisy_weights_real = torch.view_as_real(noisy_weights)  # [p,q,k,k,2]
    err_norm = (noisy_weights_real - ideal_weights_real).norm(1, dim=(-3, -2, -1))

    relative_error = err_norm / ideal_weights_l1norm  # [p,q]
    relative_error_avg = torch.mean(
        relative_error
    )  # Average error of <pxq> kxk weight matrices

    return relative_error_avg.item()
