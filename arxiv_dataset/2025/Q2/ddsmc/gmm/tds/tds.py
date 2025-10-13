"""
MIT License

Copyright (c) 2025 Filip Ekström Kelvinius

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from ddsmc.ddsmc_sampler import DDSMCDDPM
from ddsmc.utils.smc_utils import BatchedSMCHelper
from ddsmc.utils.diffusion_utils import get_diffusion_coefficients


class TDS:
    # TDS for DDPM/DDIM
    def __init__(self, alphas_cumprod, num_particles, resampling_fn, device):
        self.alphas_cumprod = alphas_cumprod
        self.num_particles = num_particles
        self.smc_helper = BatchedSMCHelper(resampling_fn, device, num_particles)

    def sample(self, model, x_start, operator, y, timesteps, eta):
        # setup
        num_samples, r = divmod(x_start.shape[0], self.num_particles)
        assert r == 0, "the number of samples in x_start is not divisible by number of particles"
        if len(y.shape) == 1 or (len(y.shape) == 3 and y.shape[0] == 3):
            # single measurement of 1 or 3 channels, add batch dimension
            y = y.unsqueeze(0)
        if y.shape[0] == 1:  # a single measurement for all chains
            y = y.repeat_interleave(self.num_particles*num_samples, dim=0)
        else:  # different measurements for each chain
            y = y.repeat_interleave(self.num_particles, 0)
        assert y.shape[0] == x_start.shape[0]

        # starting TDS
        xt = x_start.clone()
        xt = xt.requires_grad_(True)
        assert xt.shape[0] == self.num_particles*num_samples
        log_ptilde, s_uncond = self.log_ptilde(xt, timesteps[-1], model, operator, y)
        log_weights = self.smc_helper.normalize_log(log_ptilde).detach()
        for step_t in reversed(range(1, len(timesteps))):
            t = timesteps[step_t]
            tnew = timesteps[step_t-1]

            # compute gradient of log likelihood function
            grad_logp_tilde = torch.autograd.grad(log_ptilde.sum(), xt)[0]

            # resample
            ancestors = self.smc_helper.resample(log_weights, log=True)
            xt_prev = xt[ancestors]
            s_uncond_prev = s_uncond[ancestors]
            grad_logp_tilde_prev = grad_logp_tilde[ancestors]
            log_ptilde_prev = log_ptilde[ancestors]

            # propagate
            xt, log_prop = self.proposal(xt_prev, s_uncond_prev, grad_logp_tilde_prev, t, tnew, eta)
            log_trans = self.evaluate_target_transition(xt, xt_prev, s_uncond_prev, t, tnew, eta)
            
            # weight
            xt = xt.detach().requires_grad_(True)
            log_ptilde, s_uncond = self.log_ptilde(xt, tnew, model, operator, y)
            log_weights = self.smc_helper.normalize_log(log_ptilde - log_ptilde_prev + log_trans - log_prop).detach()

        return self.smc_helper.importance_sampling(xt, log_weights, log=True).detach(), None

    def log_ptilde(self, xt, t, model, operator, y):
        score = model.score(xt, self.alphas_cumprod[t])
        x0hat = self.reconstruct(xt, t, score)
        log_weight = operator.log_likelihood(x0hat, y)
        return log_weight, score

    def evaluate_target_transition(self, xt_new, xt, s_uncond, t, t_new, eta):
        # log scale
        if t_new == 0:
            return torch.zeros(xt.shape[0], device=xt_new.device)
        else:
            coef_xt, coef_score, sigma = get_diffusion_coefficients(self.alphas_cumprod, t, t_new, eta)
            mean = coef_xt * xt + coef_score * s_uncond
            return DDSMCDDPM.diag_gauss_logpdf(xt_new, mean, sigma**2)

    def proposal(self, xt, s_uncond, grad_loglik, t, t_new, eta):
        coef_xt, coef_score, sigma = get_diffusion_coefficients(self.alphas_cumprod, t, t_new, eta)
        s_tilde = s_uncond + grad_loglik
        mean = coef_xt * xt + coef_score * s_tilde
        xt_new = mean + sigma * torch.randn_like(xt)
        if t_new == 0:
            log_qt = torch.zeros(xt.shape[0], device=xt.device)
        else:
            log_qt = DDSMCDDPM.diag_gauss_logpdf(xt_new, mean, sigma**2)
        return xt_new, log_qt

    def reconstruct(self, xt, t, score):
        alpha_cumprod_t = self.alphas_cumprod[t]
        return (xt + (1-alpha_cumprod_t)*score)/torch.sqrt(alpha_cumprod_t)



            
