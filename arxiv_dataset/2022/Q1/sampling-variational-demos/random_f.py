from typing import Union
import numpy as np
import torch


def power_law(freqs: torch.Tensor, alpha: float, normalize: Union[None, str] = 'max') -> torch.Tensor:
    if normalize is None or normalize.lower() == 'none':
        return (freqs ** alpha)
    elif normalize.lower() == 'max':
        return (freqs ** alpha) / (freqs[0] ** alpha)
    elif normalize.lower() == 'mean':
        return (freqs ** alpha) / torch.sum(freqs[0] ** alpha)


class RandomMixtureOfSinusoids(object):
    def __init__(self, dim, freqs, alpha, norm='max', device=None):
        self._dim = dim
        self._omega = torch.tensor(freqs, device=device)
        self._norm = norm
        # Initialize weights
        self._alpha = None
        self._weights = None
        self.set_alpha(alpha)
        # Initialize random directions _t and phases
        self._phases = None
        self._t = None
        self.randomize()
    
    def set_alpha(self, a):
        self._alpha = a
        self._weights = power_law(self._omega, a, self._norm)
    
    def randomize(self, device=None):
        self._phases = torch.rand(len(self._omega), device=device) * 2 * np.pi
        self._t = torch.randn(len(self._omega), self._dim, device=device)
        self._t = self._t / torch.linalg.norm(self._t, dim=1, keepdim=True)
    
    def __call__(self, x, total=True):
        dot_tx = self._t @ x
        if total:
            return torch.sum(self._weights[:, None] * torch.sin(dot_tx*self._omega[:, None] + self._phases[:, None]), dim=0)
        else:
            return self._weights[:, None], np.sin(dot_tx*self._omega[:, None] + self._phases[:, None])

    def gauss_expectation(self, mean, cov, total=True):
        dot_tmu = self._t @ mean
        tCt = torch.sum(self._t * (self._t @ cov), dim=1)
        sin_part = torch.sin(dot_tmu*self._omega + self._phases)
        exp_part = torch.exp(-0.5 * tCt * self._omega**2)
        if total:
            return torch.sum(self._weights * sin_part * exp_part)
        else:
            return self._weights, sin_part * exp_part


class LazyMixtureOfSinusoids(object):
    def __init__(self, dim, freqs, x=None, mus=None, covs=None, t=None, device=None):
        self._dim = dim
        self._omega = torch.tensor(freqs, device=device)

        self._t = t
        self._sin_table = None
        self._cos_table = None
        if t is not None:
            self.set_t(t)
        else:
            self.randomize_t()
        
        if x is not None:
            self.update_table_x(x)
        elif mus is not None:
            self.update_table_gauss(mus, covs)
    
    def randomize_t(self, device=None):
        # Choose random unit directions
        self._t = torch.randn(len(self._omega), self._dim, device=device)
        self._t = self._t / torch.linalg.norm(self._t, dim=1, keepdim=True)
        # Invalidate precomputed table
        self._sin_table = None
        self._cos_table = None
        return self
    
    def set_t(self, t):
        self._t = t.clone()
        # Invalidate precomputed table
        self._sin_table = None
        self._cos_table = None
        return self

    def get_t(self):
        return self._t
    
    def update_table_x(self, x):
        dot_tx = self._t @ x
        self._sin_table = torch.sin(dot_tx*self._omega[:, None])
        self._cos_table = torch.cos(dot_tx*self._omega[:, None])
        return self
    
    def update_table_gauss(self, mus, covs):
        tmu = torch.einsum('fa,ax->fx', self._t, mus)
        tCt = torch.einsum('fa,fb,abx->fx', self._t, self._t, covs)
        self._sin_table = torch.sin(tmu*self._omega[:, None]) * torch.exp(-0.5 * tCt * self._omega[:, None]**2)
        self._cos_table = torch.cos(tmu*self._omega[:, None]) * torch.exp(-0.5 * tCt * self._omega[:, None]**2)
        return self
    
    def apply(self, weights, phases):
        """Use precomputed sin and cos tables to get sum of sin(omega*x+phase) using the identity
        sin(ax+b) = cos(ax)sin(b) + sin(ax)cos(b)
        """
        if self._sin_table is None:
            raise RuntimeError("Must call an update_table method first!")
        return weights @ (self._cos_table * torch.sin(phases[:, None]) + self._sin_table * torch.cos(phases[:, None]))
