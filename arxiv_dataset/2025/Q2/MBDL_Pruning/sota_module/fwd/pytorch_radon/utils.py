import torch
import torch.nn.functional as F
import numpy as np

if torch.__version__ > '1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

PI = torch.tensor(np.float32(np.pi))
SQRT2 = torch.tensor(np.sqrt(np.float32(2)))

# constants
# PI = 4 * torch.ones(1, dtype=torch.double).atan()
# SQRT2 = (2 * torch.ones(1, dtype=torch.double)).sqrt()


def deg2rad(x, dtype=torch.float):
    return (x * PI / 180).to(dtype)


def rfft(tensor, axis=-1):
    ndim = tensor.ndim
    if axis < 0:
        axis %= ndim
    tensor = tensor.transpose(axis, ndim - 1)
    fft_tensor = torch.view_as_real(torch.fft.fft(
        tensor,
    ))  # WG: add .fft for compatibility to torch >= 1.8
    return fft_tensor.transpose(axis, ndim - 1)


def irfft(tensor, axis):
    assert 0 <= axis < tensor.ndim
    tensor = tensor.transpose(axis, tensor.ndim - 2)

    ifft_tensor = torch.fft.ifft(
        torch.view_as_complex(tensor),
    )  # WG: add .fft for compatibility to torch >= 1.8
    ifft_tensor = ifft_tensor.real
    ifft_tensor = ifft_tensor.transpose(axis, tensor.ndim - 2)

    return ifft_tensor
