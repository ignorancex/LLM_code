import torch
import numpy as np
from torch.utils.data import Dataset


ftran = lambda x: x
fmult = lambda x: x


class Denoise(Dataset):

    def __init__(
            self,
            groundtruth,
            sigma
    ):

        self.sigma = sigma

        self.x = groundtruth['x']
        self.num_data = self.x.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        x = self.x[item]

        if x.dtype == torch.complex64:
            x_angle = torch.angle(x)
            x_abs = torch.abs(x)

            x_abs -= torch.min(x_abs)
            x_abs /= torch.max(x_abs)

            x = x_abs * np.exp(1j * x_angle)

        elif x.dtype == torch.float32:

            x -= torch.min(x)
            x /= torch.max(x)

        y = x + torch.randn(size=x.shape, dtype=x.dtype) * (self.sigma / 255)

        return y, x
