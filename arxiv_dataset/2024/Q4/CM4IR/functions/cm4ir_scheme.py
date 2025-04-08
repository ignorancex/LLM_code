import numpy as np
import torch
from tqdm import tqdm

class_num = 951


def get_scalings_for_boundary_condition(sigma, sigma_data=0.5, sigma_min=0.002):
    c_skip = sigma_data ** 2 / (
            (sigma - sigma_min) ** 2 + sigma_data ** 2
    )
    c_out = (
            (sigma - sigma_min)
            * sigma_data
            / (sigma ** 2 + sigma_data ** 2) ** 0.5
    )
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out, c_in


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_next_alpha(prev_alpha, gamma):
    return torch.clamp((prev_alpha * (1 + gamma)), 0, 0.9999)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(1)
    return a


def cm4ir_restoration(
        x,
        model,
        A_funcs,
        y_0,
        sigma_y,
        betas,
        eta,
        deltas,
        deltas_injection_type,
        gamma,
        iN,
        zeta,
        classes,
        config,
        mu=1,
        mu_factor=1,
        sigma_t_min=0.002,
        sigma_t_max=80.0,
):
    with torch.no_grad():
        iter_ind = -1
        t = (torch.ones(1) * (iN + 1)).to(x.device)
        aN = compute_alpha(betas, t.long())
        alphas = [aN]
        for i in range(config.sampling.T_sampling - 1):
            alphas.append(get_next_alpha(alphas[-1], gamma).reshape(1))

        sigma_t = torch.sqrt(1 - alphas[0])
        xt = x + torch.randn_like(x) * sigma_t

        alphas_next = alphas[1:] + [-1]
        bp_eta_reg = sigma_y ** 2 * zeta

        for at, at_next in tqdm(zip(alphas, alphas_next)):
            iter_ind += 1

            sigma_t = torch.sqrt(1 - at)
            sigma_t = torch.clamp(sigma_t, sigma_t_min, sigma_t_max)

            if deltas_injection_type == 0:
                scaling_values = get_scalings_for_boundary_condition(sigma_t)
            elif deltas_injection_type == 1:
                scaling_values = get_scalings_for_boundary_condition((1 + deltas[iter_ind]) * sigma_t)
            else:
                raise NotImplementedError(f"Unsupported deltas_injection_type: {deltas_injection_type}."
                                          f"Please view README for optional values.")

            c_skip, c_out, c_in = [append_dims(s, xt.ndim) for s in scaling_values]
            rescaled_sigma_t = 1000 * 0.25 * torch.log((1 + deltas[iter_ind]) * sigma_t + 1e-44)

            if classes is None:
                et = model(c_in * xt, rescaled_sigma_t)
            else:
                et = model(c_in * xt, rescaled_sigma_t, classes)
            x0_t = c_out * et + c_skip * xt
            x0_t = torch.clamp(x0_t, -1.0, 1.0)

            z_hat_minus = (x0_t - xt) / sigma_t
            if sigma_y != 0 and at_next == -1:  # last iteration of noisy case, skip guidance
                break

            BP_guidance = A_funcs.A_pinv_add_eta(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y_0.reshape(y_0.size(0), -1), eta=bp_eta_reg).reshape(
                *x.size())
            xt = x0_t - (mu * mu_factor ** iter_ind) * BP_guidance

            if at_next != -1:
                next_sigma_t = torch.sqrt(1 - at_next)
                next_sigma_t = torch.clamp(next_sigma_t, sigma_t_min, sigma_t_max)

                c1 = next_sigma_t * eta
                c2 = next_sigma_t * ((1 - eta ** 2) ** 0.5)
                xt = xt + c1 * torch.randn_like(xt) + c2 * z_hat_minus

        if sigma_y != 0:
            return [x0_t.to('cpu')]
        return [xt.to('cpu')]
