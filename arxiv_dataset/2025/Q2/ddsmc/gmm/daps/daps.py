"""
MIT License

Copyright (c) 2025 Filip Ekström Kelvinius

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
import numpy as np

from ddsmc.utils.diffusion_utils import get_diffusion_coefficients


class DAPSSampler:
    def __init__(self, alphas_cumprod, ddim_timesteps, max_num_ode_steps=None, lr_0=None, lr_delta=None, num_lgvd_steps=None, lgvd_tau=None, ):
        self.alphas_cumprod = alphas_cumprod
        if ddim_timesteps is not None:
            self.alphas_cumprod = self.alphas_cumprod[ddim_timesteps]
        if max_num_ode_steps is not None:
            self.max_num_ode_steps = max_num_ode_steps
        else:
            self.max_num_ode_steps = len(self.alphas_cumprod)
        self.rt = torch.sqrt(1-self.alphas_cumprod) / 2**0.5
    
    @property
    def num_steps(self):
        return len(self.alphas_cumprod) - 1
        
    def sample(self, x_init, score_model, y, operator):
        assert len(y.shape) == 2
        if y.shape[0] == 1 and y.shape[0] != x_init.shape[0]:
            y = y.repeat_interleave(x_init.shape[0], dim=0)
        assert x_init.shape[0] == y.shape[0]
        xt = x_init
        for t in reversed(range(1, len(self.alphas_cumprod))):
            # reconstruct
            x0_hat = self.reconstruct(xt, score_model, t)

            # in case of linear-Gaussian, can sample exactly
            x0y = self.x0y_exact(x0_hat, y, self.rt[t], operator)

            # forward sampling
            xt = torch.sqrt(self.alphas_cumprod[t-1]) * x0y + torch.sqrt(1 - self.alphas_cumprod[t-1]) * torch.randn_like(x0y)
        return xt

    def reconstruct(self, xt, score_model, tstart):
        timesteps = torch.linspace(1, tstart, min(self.max_num_ode_steps, tstart)).long()
        timesteps = torch.cat([torch.zeros(1), timesteps]).long()
        for t, tprev in zip(reversed(timesteps[1:]), reversed(timesteps[:-1])):
            coef_xt, coef_score, _ = get_diffusion_coefficients(self.alphas_cumprod, t, tprev, 0.)
            xt = coef_xt * xt + coef_score * score_model(xt, self.alphas_cumprod[t])
        return xt

    def x0y_exact(self, x0hat, y, rt, operator):
        x0hat_prime = (operator.V.T @ x0hat.T).T
        y_prime = (operator.U.T @ y.T).T
        M_inv = self.get_M_inv(operator, rt)
        b = self.get_b(operator, y_prime, x0hat_prime, rt)
        x0y_prime = M_inv * b + torch.sqrt(M_inv) * torch.randn_like(x0hat_prime)
        return (operator.V @ x0y_prime.T).T
        
    @staticmethod
    def get_M_inv(operator, rt):
        return (operator.sigma**2*rt**2/(operator.STS_diag * rt**2 + operator.sigma**2)).unsqueeze(0)
    
    @staticmethod
    def get_b(operator, y, x0hat, rt):
        return 1/operator.sigma**2 * (operator.S.T @ y.T).T + (1/rt**2) * x0hat