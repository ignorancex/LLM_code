"""
Much of this code was borrowed from the DAPS repository https://github.com/zhangbingliang2019/DAPS
This is under an MIT License:

MIT License

Original work Copyright (c) 2025 Bingliang Zhang, Wenda Chu, and all collaborators.
Modifications Copyright (c) 2025 Filip Ekström Kelvinius

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Scheduler(nn.Module):
    """
        Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7'):
        """
            Initializes the scheduler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the schedule.
                sigma_max (float): Maximum value of sigma.
                sigma_min (float): Minimum value of sigma.
                sigma_final (float): Final value of sigma, defaults to sigma_min.
                schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
                timestep (str): Type of timestep function ('log' or 'poly-n').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)

        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # factor = 2\dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps = sigma_steps, time_steps, factor_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
            Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        elif schedule == 'ddpm':
            beta_min = 0.1
            beta_d = 19.9
            sigma_fn = lambda t: np.sqrt(np.exp(1/2 * beta_d * t**2 + beta_min*t) - 1)
            sigma_derivative_fn = lambda t: (1 / 2 / np.sqrt(np.exp(1/2 * beta_d * t**2 + beta_min*t) - 1)) * np.exp(1/2 * beta_d * t**2 + beta_min*t)*(beta_d * t + beta_min)
            sigma_inv_fn = lambda sigma: -beta_min/beta_d + np.sqrt((beta_min/beta_d)**2 + 2*np.log(sigma**2+1)/beta_d)
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
            Returns the time step function based on the given timestep type.
        """
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        elif timestep == "ddpm":
            epsilon_s = 1e-3
            get_time_step_fn = lambda r: 1 + r * (epsilon_s - 1)
        else:
            raise NotImplementedError
        return get_time_step_fn


class DiffusionSampler(nn.Module):
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, SDE=False, record=False, verbose=False, inner_scheduler_config=None):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, record, verbose)
        elif self.solver == 'q0':
            if inner_scheduler_config is not None:
                if "sigma_max" in inner_scheduler_config:
                    inner_scheduler_config.pop("sigma_max")
                inner_scheduler_config["sigma_final"] = 0
            self.inner_scheduler_config = inner_scheduler_config
            print("using q0, using inner loop scheduler", self.inner_scheduler_config)
            return self._q0(model, x_start, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, sigma)
            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
        return x
    
    def _q0(self, model, x_start, record=False, verbose=False):
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)
        x = x_start
        for step in pbar:
            sigma, sigma_next = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step + 1]
            if self.inner_scheduler_config is not None:
                inner_diffusion_scheduler = Scheduler(**self.inner_scheduler_config, sigma_max=sigma)
                inner_sampler = DiffusionSampler(inner_diffusion_scheduler, "euler")
                x0 = inner_sampler.sample(model, x, SDE=False)
            else:
                score = model.score(x, sigma)
                x0 = x + sigma**2 * score
            x = x0 + sigma_next * torch.randn_like(x0)

            if record:
                self._record(x, score, sigma, None)
        return x
        
        
    def _record(self, x, score, sigma, factor, epsilon=None):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', x)
        self.trajectory.add_tensor(f'score', score)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'factor', factor)
        if epsilon is not None:
            self.trajectory.add_tensor(f'epsilon', epsilon)

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start
