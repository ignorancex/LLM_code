import torch
import torch.nn as nn
import numpy as np

EPS = 1e-12

class DiffusionAwareNoiseScheduler:
    def __init__(self, sigma_meas_init: float, L: float = 1.0, device: str = "cpu"):
        log_sigma = torch.log(torch.tensor(sigma_meas_init + EPS, dtype=torch.float32))
        self.log_sigma_meas = nn.Parameter(log_sigma.to(device))
        self.L = float(L)
        self.device = device
        self.sigma_history = []
        self.update_count = 0

    @property
    def sigma_meas(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_meas)

    def get_effective_sigma(self, t: float, noise_schedule) -> float:
        t_tensor = torch.tensor(t, device=self.device, dtype=torch.float32)
        sigma_t = float(noise_schedule.sigma(t_tensor).item())
        sigma_eff = float(self.sigma_meas.item()) + self.L * sigma_t
        return sigma_eff

    def update_lipschitz_constant(self, L: float):
        self.L = float(L)

    def update_sigma(self, new_sigma: float, ema_decay: float):
        self.sigma_history.append(new_sigma)
        self.update_count += 1
        if len(self.sigma_history) > 5:
            self.sigma_history = self.sigma_history[-5:]
        median_sigma = np.median(self.sigma_history)
        new_sigma = 0.5 * new_sigma + 0.5 * median_sigma
        current = self.sigma_meas.item()
        updated = ema_decay * current + (1 - ema_decay) * new_sigma
        self.log_sigma_meas.data = torch.log(torch.tensor(updated + EPS, device=self.device))