"""
This source code is licensed under an MIT license, found below.

MIT License

Copyright (c) 2025 Filip Ekström Kelvinius

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
import tqdm

from ddsmc.utils.smc_utils import BatchedSMCHelper
from ddsmc.utils.diffusion_utils import get_diffusion_coefficients

from ddsmc.ddsmc_image_utils.daps_utils import Scheduler, DiffusionSampler

from abc import ABC, abstractmethod

def diag_gauss_logpdf(x, mean, diag):
    return torch.sum(-1/2 * (x-mean)**2/diag, dim=-1)

class DDSMC(ABC):
    def __init__(self, num_particles, device, eta, resampling_fn="multinomial"):
        self.num_particles = num_particles
        self.smc_helper = BatchedSMCHelper(resampling_fn, device, num_particles)
        self.device = device
        self.eta = eta
    
    @property
    @abstractmethod
    def num_steps(self):
        pass

    def t_from_step(self, step):
        return self.num_steps - step
    
    def step_from_t(self, t):
        return self.num_steps - t

    @abstractmethod
    def evaluate_target_transition(self, xt, xtprev, x0hat_prev, t):
        pass

    @abstractmethod
    def proposal(self, x0hat_prev, xtprev, y, operator, tprev):
        pass

    @abstractmethod
    def reconstruct(self, xt, score_model, t):
        pass

    def sample(self, score_model, x_start, operator, y, **kwargs):
        # setup, need to separate batched (when we have multiple chains in same batch) and unbatched (only sample a single chain)
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
        
        xt = x_start
        assert xt.shape[0] == self.num_particles*num_samples

        if "verbose" in kwargs and kwargs["verbose"]:
            pbar = tqdm.trange(self.num_steps)
        else:
            pbar = range(self.num_steps)

        ancestors = torch.arange(self.num_particles, device=self.device).unsqueeze(0).repeat(num_samples, 1)
        assert ancestors.shape[0] == num_samples

        for step in pbar:
            t = self.t_from_step(step)
            # reconstruct
            x0hat = self.reconstruct(xt, score_model, t)

            # weight
            logp_y_x0_t = operator.log_likelihood(x0hat, y, self.rho_t[t])

            if step == 0:
                log_weights = self.smc_helper.normalize_log(logp_y_x0_t)
            else:
                log_q0 = self.evaluate_target_transition(xt, xtprev, x0hat_prev, t)
                log_weights = self.smc_helper.normalize_log(logp_y_x0_t + log_q0 - logp_y_x0_tplusone - log_qt_xt)
            ess = self.smc_helper.compute_ess(log_weights, log=True)

            # Resample
            ancestors = self.smc_helper.resample(log_weights, log=True)
            x0hat_prev = x0hat[ancestors]
            logp_y_x0_tplusone = logp_y_x0_t[ancestors]
            xtprev = xt[ancestors]

            # proposal
            xt, log_qt_xt = self.proposal(x0hat_prev, xtprev, y, operator, t)

        logp_y_x0_t = operator.log_likelihood(xt, y)
        log_q0 = self.evaluate_target_transition(xt, xtprev, x0hat_prev, 0)
        log_weights = self.smc_helper.normalize_log(logp_y_x0_t + log_q0 - logp_y_x0_tplusone - log_qt_xt)
        x0 = self.smc_helper.importance_sampling(xt, log_weights, log=True)
        return x0, None


class DDSMCDDPM(DDSMC):
    # Currently it only works for single "channel" data, i.e., a data batch
    # has shape [B, D]
    def __init__(self, alpha_bars, num_particles, device, eta, ddim_taus=None, resampling_fn="multinomial", use_ode=False, max_num_ode_steps=None):
        super().__init__(num_particles, device, eta, resampling_fn)
        assert torch.all(alpha_bars[:-1] >= alpha_bars[1:])
        if ddim_taus is not None:
            # As DDIM has the same marginals q(x_\tau|x_0) as DDPM, 
            # DDSMCDDIM is the same as DDSMCDDPM, but with a shorter list of alpha_bars
            assert torch.all(ddim_taus[:-1] <= ddim_taus[1:])
            alpha_bars =  alpha_bars[ddim_taus]
        self.alpha_bars = alpha_bars
        self.rho_t = (1-alpha_bars)**0.5/(2**0.5)
        self.use_ode = use_ode
        if max_num_ode_steps is not None:
            self.max_num_ode_steps = max_num_ode_steps
        else:
            self.max_num_ode_steps = len(self.alpha_bars)

    @property
    def num_steps(self):
        return len(self.alpha_bars) - 1
    
    def proposal(self, x0hat_prev, xtprev, y, operator, tprev):
        if operator.sigma != 0:
            M_inv = self.get_M_inv(operator, self.rho_t[tprev])
            b = self.get_b(operator, y, x0hat_prev, self.rho_t[tprev])
            abar_t = self.alpha_bars[tprev - 1]
            abar_tprev = self.alpha_bars[tprev] if tprev > 0 else torch.as_tensor(1.)
            beta_prev = 1 - abar_tprev / abar_t
            denominator = self.eta - self.eta * beta_prev - self.eta * abar_tprev + beta_prev
            x0_factor = torch.sqrt(abar_t) * beta_prev / denominator
            xtprev_factor = self.eta * torch.sqrt(1 - beta_prev) * (1 - abar_t) / denominator
            mean = x0_factor * M_inv * b + xtprev_factor * xtprev
            if tprev > 1:
                # sigma2 = 1 - self.alpha_bars[tprev - 1] * (1 + self.rho_t[tprev]**2) + self.alpha_bars[tprev - 1] * M_inv
                sigma2 = 1 - abar_t - (x0_factor * self.rho_t[tprev])**2 + x0_factor ** 2 * M_inv
            else:
                sigma2 = torch.zeros_like(M_inv)
        else:
            raise NotImplementedError("Noise-free proposal not implemented yet")
        xt = mean + torch.sqrt(sigma2) * torch.randn_like(x0hat_prev)
        if tprev > 1:
            log_qt = self.diag_gauss_logpdf(xt, mean, sigma2)
        else:
            log_qt = torch.zeros(xt.shape[0], device=xt.device)
        return xt, log_qt
    
    def reconstruct_tweedie(self, xt, score_model, t):
        alpha_bar_t = self.alpha_bars[t]
        score = score_model.score(xt, alpha_bar_t)
        return (xt + (1-alpha_bar_t)*score)/torch.sqrt(alpha_bar_t)
    
    def reconstruct_ode(self, xt, score_model, tstart):
        timesteps = torch.linspace(1, tstart, min(self.max_num_ode_steps, tstart)).long()
        timesteps = torch.cat([torch.zeros(1), timesteps]).long()
        for t, tprev in zip(reversed(timesteps[1:]), reversed(timesteps[:-1])):
            coef_xt, coef_score, _ = get_diffusion_coefficients(self.alpha_bars, t, tprev, 0.)
            xt = coef_xt * xt + coef_score * score_model.score(xt, self.alpha_bars[t])
        return xt                                
    
    def reconstruct(self, xt, score_model, t):
        if self.use_ode:
            return self.reconstruct_ode(xt, score_model, t)
        else:
            return self.reconstruct_tweedie(xt, score_model, t)

    def evaluate_target_transition(self, xt, xtprev, x0hat_prev, t):
        if t > 0:
            abar_t = self.alpha_bars[t]
            abar_tprev = self.alpha_bars[t + 1]
            beta_prev = 1 - abar_tprev / abar_t
            denominator = self.eta - self.eta * beta_prev - self.eta * abar_tprev + beta_prev
            x0_factor = torch.sqrt(abar_t) * beta_prev / denominator
            xtprev_factor = self.eta * torch.sqrt(1 - beta_prev) * (1 - abar_t) / denominator

            diag_sigma2 = beta_prev * (1-abar_t) / denominator
            mean = x0_factor * x0hat_prev + xtprev_factor * xtprev
            # diag_sigma2 = (1-self.alpha_bars[t])
            # mean = torch.sqrt(self.alpha_bars[t]) * x0hat
            return self.diag_gauss_logpdf(xt, mean, diag_sigma2)
        else:
            return torch.zeros(xt.shape[0], device=xt.device)
        
    @staticmethod
    def diag_gauss_logpdf(x, mean, diag_sigma2):
        return -1/2 * torch.sum((x-mean)**2 / diag_sigma2, dim=-1)
    
    @staticmethod
    def get_M_inv(operator, rt):
        return (operator.sigma**2*rt**2/(operator.STS_diag * rt**2 + operator.sigma**2)).unsqueeze(0)
    
    @staticmethod
    def get_b(operator, y, x0hat, rt):
        return 1/operator.sigma**2 * (operator.S.T @ y.T).T + (1/rt**2) * x0hat

        
class DDSMCImage(DDSMC):
    """
        Implementation of DDSMC for images
    """
    def __init__(self, smc_config, annealing_scheduler_config, diffusion_scheduler_config, device, eta, reconstruct_fn, rt_scheduler_config):
        """
            Initializes the DDSMC sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                device: device
        """
        super().__init__(smc_config["num_particles"], device, eta, smc_config["resampling_method"])
        annealing_scheduler_config, diffusion_scheduler_config, rt_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config,
                                                                             rt_scheduler_config)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.rho_t = Scheduler(**rt_scheduler_config).sigma_steps[::-1]
        
        
        recond_fns = {"ode": self.ode_reconstruct, "tweedie": self.tweedie_reconstruct}
        self._reconstruct = recond_fns[reconstruct_fn]
        
    def _check(self, annealing_scheduler_config, diffusion_scheduler_config, rt_scheduler_config):
        """
            Checks and updates the configurations for the schedulers.
        """
        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # sigma final of annealing scheduler should always be 0
        annealing_scheduler_config['sigma_final'] = 0
        rt_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config, rt_scheduler_config

    def get_start(self, ref, *args):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        ref = torch.repeat_interleave(ref, self.num_particles, 0)
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start
    
    def proposal(self, x0hat_prev, xtprev, y, operator, tprev):
        step_prev = self.step_from_t(tprev)
        sigma2_t = self.annealing_scheduler.sigma_steps[step_prev + 1]**2
        delta_sigma2 = self.annealing_scheduler.sigma_steps[step_prev]**2 - sigma2_t
        assert delta_sigma2 > 0
        mean_factor = delta_sigma2 / (self.eta * sigma2_t + delta_sigma2)
        prev_sample_factor = self.eta * sigma2_t / (self.eta * sigma2_t + delta_sigma2)
        rho_t = self.rho_t[tprev]
        M_inv, mean = self.get_Minv_Minvb(operator, rho_t**2, x0hat_prev, y)
        if tprev > 1:
            diag = (sigma2_t * mean_factor - (mean_factor * rho_t)**2) + mean_factor**2*M_inv
        else:
            diag = torch.zeros_like(M_inv)
        
        # TODO: should this use mean (M^-1b) or just x0hat for tprev=1?
        mean = mean_factor * mean + prev_sample_factor * xtprev.flatten(1, -1)
        z = torch.randn_like(mean)
        x_new = mean + torch.sqrt(diag) * z
        if tprev <= 1:
            log_r = torch.zeros(xtprev.shape[0], device=xtprev.device)
        else:
            log_r = diag_gauss_logpdf(x_new, mean, diag)

        return x_new, log_r
    
    @property
    def num_steps(self):
        return self.annealing_scheduler.num_steps
    
    def ode_reconstruct(self, x_t, model, start_step):
        sigma = self.annealing_scheduler.sigma_steps[start_step]
        diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
        sampler = DiffusionSampler(diffusion_scheduler)
        x0hat = sampler.sample(model, x_t, SDE=False, verbose=False)
        return x0hat
    
    def tweedie_reconstruct(self, x_t, model, current_step):
        sigma = self.annealing_scheduler.sigma_steps[current_step]
        score = model.score(x_t, sigma)
        return x_t + sigma**2 * score

    def reconstruct(self, x_t, model, t):
        step = self.step_from_t(t)
        return self._reconstruct(x_t, model, step)
    
    def evaluate_target_transition(self, xt, xtprev, x0hat_prev, t):
        if t > 0:
            sigma2_t = self.annealing_scheduler.sigma_steps[self.step_from_t(t)]**2
            delta_sigma2 = self.annealing_scheduler.sigma_steps[self.step_from_t(t) - 1]**2 - sigma2_t
            assert delta_sigma2 > 0.
            diag = delta_sigma2 * sigma2_t/(self.eta * sigma2_t + delta_sigma2) * torch.ones(xt.shape[-1], device=xt.device)
            mean = self.eta / (self.eta + delta_sigma2 / sigma2_t) * xtprev +  1 /(self.eta * sigma2_t / delta_sigma2 + 1) * x0hat_prev
            return diag_gauss_logpdf(xt, mean, diag)
        else:
            return torch.zeros(xt.shape[0], device=xt.device)

    @staticmethod
    def get_Minv_Minvb(operator, rho2_t, x0hat, y):
        num_unobserved = x0hat.shape[-1] - y.shape[-1]
        num_observed = y.shape[-1]
        denominator = operator.singulars**2 * rho2_t + operator.sigma**2
        Minv_top = rho2_t * operator.sigma**2 / denominator
        Minv_bottom = rho2_t * torch.ones(num_unobserved, device=Minv_top.device)
        Minv = torch.cat([Minv_top, Minv_bottom])

        Minvb_top = rho2_t * operator.singulars / denominator * y + operator.sigma**2 / denominator * x0hat[:, :num_observed]
        Minvb = torch.cat([Minvb_top, x0hat[:, num_observed:]], dim=-1)
        assert Minv.shape[-1] == Minvb.shape[-1] == x0hat.shape[-1]
        return Minv, Minvb