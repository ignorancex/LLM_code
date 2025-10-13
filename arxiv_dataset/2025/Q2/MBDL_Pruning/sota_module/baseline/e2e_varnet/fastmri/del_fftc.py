"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch
import torch.fft

from torch.utils.data import Dataset

class fftinputDataDict(Dataset):
    def __init__(self, x: torch.Tensor, dim: Optional[List[int]] = None):
        self.x = x
        self.dim = dim

    def __len__(self):
        return 1

    def getData(self):
        return self.x, self.dim

class rollinputDataDict(Dataset):
    def __init__(self, x, shift, dim):
        self.x = x
        self.shift = shift
        self.dim = dim

    def __len__(self):
        return 1

    def getData(self):
        return self.x, self.shift, self.dim

def fft2c_new(data: torch.Tensor, norm: str = "ortho"):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    if data.shape[1] == 12:
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
            )
        )

    else:
        ifftdata = fftinputDataDict(data, dim=[-3, -2])
        data = ifftshift(ifftdata)
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
            )
        )
        fftdata = fftinputDataDict(data, dim=[-3, -2])
        data = fftshift(fftdata)

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho"):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")
    if data.shape[1] == 12:
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
            )
        )

    else:
        ifftdata = fftinputDataDict(data, dim=[-3, -2])
        data = ifftshift(ifftdata)
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
            )
        )
        fftdata = fftinputDataDict(data, dim=[-3, -2])
        data = fftshift(fftdata)

    return data


# Helper functions


def roll_one_dim(rollonedimdata):
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    x, shift, dim = rollonedimdata.getData()
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(rolldata):
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    x, shift, dim = rolldata.getData()
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        rollonedimdata = rollinputDataDict(x, s, d)
        x = roll_one_dim(rollonedimdata)

    return x



def fftshift(fftdata):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    x, dim = fftdata.getData()
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    rolldata = rollinputDataDict(x, shift, dim)
    a = roll(rolldata)
    return a


def ifftshift(ifftdata):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    x, dim = ifftdata.getData()
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2
    rolldata = rollinputDataDict(x, shift, dim)
    a = roll(rolldata)
    return a
