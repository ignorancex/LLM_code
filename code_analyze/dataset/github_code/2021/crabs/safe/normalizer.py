import torch
import torch.nn as nn

import rlz


class Normalizer(nn.Module):
    def __init__(self, dim, *, clip=10):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('n', torch.tensor(0, dtype=torch.int64))
        self.placeholder = nn.Parameter(torch.tensor(0.), False)  # for device info (@maybe_numpy)
        self.clip = clip

    def forward(self, x, inverse=False):
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / self.std.clamp(min=1e-6)

    @rlz.torch_utils.maybe_numpy
    def update(self, data):
        data = data - self.mean

        m = data.shape[0]
        delta = data.mean(dim=0)
        new_n = self.n + m
        new_mean = self.mean + delta * m / new_n
        new_std = torch.sqrt((self.std**2 * self.n + data.var(dim=0) * m + delta**2 * self.n * m / new_n) / new_n)

        self.mean.set_(new_mean.data)
        self.std.set_(new_std.data)
        self.n.set_(new_n.data)

    @rlz.torch_utils.maybe_numpy
    def fit(self, data):
        n = data.shape[0]
        self.n.set_(torch.tensor(n, device=self.n.device))
        self.mean.set_(data.mean(dim=0))
        self.std.set_(data.std(dim=0))
