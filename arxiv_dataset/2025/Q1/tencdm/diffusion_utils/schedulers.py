import torch
import numpy as np
from abc import ABCMeta, abstractmethod


def create_scheduler(config):
    if config.dynamic.scheduler == "cosine":
        return Cosine(config.dynamic.beta_min, config.dynamic.beta_max)
    elif config.dynamic.scheduler == "sd":
        return CosineSD(config.dynamic.coef_d)
    elif config.dynamic.scheduler == 'sqrt':
        return Sqrt()


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def params(self, t):
        pass

    def reverse(self, alpha):
        pass


class Cosine(Scheduler):
    def __init__(self, beta_0, beta_1):
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def beta_t(self, t):
        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def params(self, t):
        t = t[:, None, None]
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        alpha = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff))
        return alpha, std


class CosineSD(Scheduler):
    def __init__(self, d=1):
        self.d = d
        self.t_thr = 0.95

    def beta_t(self, t):
        t = torch.clip(t, 0, self.t_thr)
        tan = torch.tan(np.pi * t / 2)
        beta_t = np.pi * self.d ** 2 * tan * (1 + tan ** 2) / (1 + self.d ** 2 * tan ** 2)
        return beta_t

    def params(self, t):
        t = t[:, None, None]
        tan = torch.tan(np.pi * t / 2)
        alpha_t = 1 / torch.sqrt(1 + tan ** 2 * self.d ** 2)
        std_t = torch.sqrt(1 - alpha_t ** 2)
        return torch.clip(alpha_t, 0, 1), torch.clip(std_t, 0, 1)


class Sqrt(Scheduler):
    def __init__(self, ):
        self.s = 0.0001

    def beta_t(self, t):
        beta_t = 1 / 2 / ((torch.sqrt(t + self.s)) * (1 - torch.sqrt(t + self.s)))
        beta_t = torch.clip(beta_t, 0, 1000)
        return beta_t

    def params(self, t):
        t = t[:, None, None]
        
        alpha = 1 - torch.sqrt(t + self.s)
        alpha = torch.clip(alpha, 0, 1)
        mu = torch.sqrt(alpha)
        std = torch.sqrt(1. - alpha)
        return torch.clip(mu, 0, 1), torch.clip(std, 0, 1)
    