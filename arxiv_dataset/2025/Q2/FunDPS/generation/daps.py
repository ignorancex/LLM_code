import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.dataset_utils import DatasetNormalizer

from .base import PDESolver


class Scheduler(nn.Module):
    """Scheduler for diffusion sigma(t) and discretization step size Delta t"""

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, rho=7, schedule="linear"):
        """Initialize the scheduler with the given parameters.

        Args:
            num_steps (int): Number of steps in the schedule
            sigma_max (float): Maximum value of sigma
            sigma_min (float): Minimum value of sigma
            sigma_final (float): Final value of sigma, defaults to sigma_min
            rho (float): Power parameter for sigma schedule
            schedule (str): Type of schedule for sigma ('linear' or 'sqrt')
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final if sigma_final is not None else sigma_min
        self.schedule = schedule
        self.p = rho

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.p, self.sigma_max, self.sigma_min)

        # Generate time steps and sigma values
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # Calculate factor = 2∇σ(t)σ(t)Δt
        factor_steps = np.array([2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])

        self.sigma_steps = sigma_steps
        self.time_steps = time_steps
        self.factor_steps = [max(f, 0) for f in factor_steps]

    def get_sigma_fn(self, schedule):
        """Returns the sigma function, its derivative, and inverse based on schedule.

        Args:
            schedule (str): The schedule type ('sqrt' or 'linear')

        Returns:
            tuple: (sigma_fn, sigma_derivative_fn, sigma_inv_fn)
        """
        if schedule == "sqrt":
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / (2 * np.sqrt(t))
            sigma_inv_fn = lambda sigma: sigma**2
        elif schedule == "linear":
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented")
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, p, sigma_max, sigma_min):
        """Returns the time step function based on parameters.

        Args:
            p (float): Power parameter
            sigma_max (float): Maximum sigma value
            sigma_min (float): Minimum sigma value

        Returns:
            callable: Time step function
        """
        return lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p


class PDESolverDAPS(PDESolver):
    """PDESolver implementation using Decoupled Annealing Posterior Sampling (DAPS)."""

    def __init__(self, config):

        super().__init__(config)
        self.normalizer = DatasetNormalizer(config["dataset"])

        self.annealing_config = config["guidance"]["annealing"].to_dict()
        self.diffusion_config = config["guidance"]["diffusion"].to_dict()
        self.langevin_config = config["guidance"]["langevin"].to_dict()

        self.lr = self.langevin_config["lr"]
        self.lr_min_ratio = self.langevin_config["lr_min_ratio"]
        self.langevin_steps = self.langevin_config["num_steps"]
        self.langevin_weights = torch.tensor(self.langevin_config["weights"], device=self.device).view(1, -1)

        assert self.num_steps == self.annealing_config["num_steps"]
        assert self.observations[0].type == "sparse"

    def generate_single_batch(self, observations):
        """Generate a single batch of samples using DAPS.

        Args:
            observations (list): List of observation objects

        Returns:
            tuple: (predictions, auxiliary_info)
        """
        # Initialize annealing scheduler
        annealing_scheduler = Scheduler(**self.annealing_config)

        # Initialize starting point
        latents = self.generate_latents()
        xt = latents.to(torch.float64) * annealing_scheduler.sigma_max

        # Store intermediates if requested
        intermediates = []

        for step in tqdm(range(annealing_scheduler.num_steps), unit="step"):
            sigma_t = annealing_scheduler.sigma_steps[step]
            sigma_t_next = annealing_scheduler.sigma_steps[step + 1]

            # 1. Reverse Diffusion Step
            diffusion_scheduler = Scheduler(**self.diffusion_config, sigma_max=sigma_t)
            x0 = self._reverse_diffusion(xt, diffusion_scheduler)

            # 2. Langevin Dynamics Step
            x0y = self._langevin_dynamics(x0, observations, sigma_t, step / annealing_scheduler.num_steps)

            # 3. Forward Diffusion Step
            xt = x0y + self.noise_sampler.sample(self.batch_size) * sigma_t_next

            # Save intermediates if requested
            if self.save_indices is not None and step in self.save_indices:
                denorm_x0 = self.normalizer.denormalize(x0)
                denorm_x0y = self.normalizer.denormalize(x0y)
                denorm_xt = self.normalizer.denormalize(xt)
                intermediates.append(torch.cat([denorm_x0, denorm_x0y, denorm_xt], dim=1))

            if xt.isnan().any():
                print(f"Step {step}: NaN detected!")
                break

        # Transform final result
        x_final = xt.detach()
        pred = self.normalizer.transform(x_final, denormalize=True)

        return pred, {"intermediates": intermediates}

    def _reverse_diffusion(self, x_cur, scheduler):
        """Perform reverse diffusion process.

        Args:
            x_cur (torch.Tensor): Current state
            scheduler (Scheduler): Diffusion scheduler

        Returns:
            torch.Tensor: Reversed state
        """
        for step in range(scheduler.num_steps):
            sigma_t = scheduler.sigma_steps[step]
            sigma_t_next = scheduler.sigma_steps[step + 1]
            sigma_t = torch.tensor(sigma_t, dtype=torch.float64, device=x_cur.device)
            sigma_t_next = torch.tensor(sigma_t_next, dtype=torch.float64, device=x_cur.device)

            # Euler step
            x_N = self.net(x_cur, sigma_t).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            # 2nd order correction
            if step < scheduler.num_steps - 1:
                x_N = self.net(x_next, sigma_t_next).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            x_cur = x_next

        return x_cur

    def _langevin_dynamics(self, x0hat: torch.Tensor, observations, sigma, ratio):
        """Perform Langevin dynamics sampling.

        Args:
            x0hat (torch.Tensor): Initial state
            observations (list): List of observation objects
            sigma (float): Current sigma value
            ratio (float): Current step ratio

        Returns:
            torch.Tensor: Updated state
        """
        x = x0hat.detach().clone()
        x.requires_grad_(True)

        rho = self.langevin_config["lr_rho"]
        eta = self.langevin_config["eta"]
        tau = self.langevin_config["tau"]

        # Calculate adaptive learning rate
        multiplier = (1 ** (1 / rho) + ratio * (self.lr_min_ratio ** (1 / rho) - 1 ** (1 / rho))) ** rho
        current_lr = multiplier * self.lr

        optimizer = optim.SGD([x], lr=current_lr)

        for _ in range(self.langevin_steps):
            optimizer.zero_grad()

            # Compute loss terms
            prior_loss = ((x - x0hat.detach()) ** 2).sum()

            obs_loss = []
            denorm_x = self.normalizer.denormalize(x)
            for obs in observations:
                loss = obs.get_observation_loss(denorm_x)
                obs_loss.append(loss)
            obs_loss = torch.cat(obs_loss, dim=1)
            weighted_obs_loss = (obs_loss * self.langevin_weights).sum()

            loss = weighted_obs_loss / (2 * tau**2) + prior_loss / (2 * sigma**2)

            # Update
            loss.backward()
            optimizer.step()

            # Add noise scaled by learning rate
            with torch.no_grad():
                noise = self.noise_sampler.sample(self.batch_size) * np.sqrt(2 * current_lr) * eta
                x.add_(noise)

            # Check for numerical stability
            if torch.isnan(x).any():
                print("NaN detected in Langevin dynamics")
                return torch.zeros_like(x)

        return x.detach()
