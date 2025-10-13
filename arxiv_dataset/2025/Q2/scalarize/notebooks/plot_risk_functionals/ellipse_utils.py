#!/usr/bin/env python3

r"""
Ellipse utilities.
"""

import math

from typing import Optional, Tuple

import matplotlib
import matplotlib.transforms as transforms
import torch
from botorch.utils.sampling import manual_seed
from matplotlib.patches import Ellipse
from torch import Tensor
from torch.quasirandom import SobolEngine


def sample_2d_circle(
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Uniformly sample from inside the two-dimensional circle using rejection
    sampling.

    Args:
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An  `n x 2` tensor of samples from the unit circle.
    """
    d = 2
    looking_for_samples = True
    samples = None
    while looking_for_samples:
        if qmc:
            sobol_engine = SobolEngine(d, scramble=True, seed=seed)
            rnd = sobol_engine.draw(n, dtype=dtype)
        else:
            with manual_seed(seed=seed):
                rnd = torch.rand(n, d, dtype=dtype)

        u = 2 * rnd - 1
        new_samples = u[u[:, 0] ** 2 + u[:, 1] ** 2 <= 1]
        if samples is None:
            samples = new_samples
        else:
            samples = torch.row_stack([samples, new_samples])

        if len(samples) > n:
            looking_for_samples = False

    return samples[:n].to(device)


def ellipsify(X: Tensor, radius: Tensor, angle: float, translate: Tensor) -> Tensor:
    r"""Transform a sample from the two-dimensional circle into a sample of a
    two-dimensional ellipse.

    Args:
        X: A `n x 2`-dim Tensor of samples.
        radius: A `2`-dim Tensor containing the radii.
        angle: A float determining the angle of rotation.
        translate: A `2`-dim Tensor containing the centre of the ellipse.

    Returns:
        An  `n x 2` tensor of samples from the ellipse.
    """
    rotation_matrix = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    ).to(X)

    ellipse_X = (
        torch.column_stack([radius[0] * X[:, 0], radius[1] * X[:, 1]]) @ rotation_matrix
    )

    return ellipse_X + translate


def get_ellipse_patch(
    radius: Tensor, angle: float, translate: Tensor, **kwargs
) -> Tuple[matplotlib.patches.Ellipse, matplotlib.transforms.Affine2D]:
    r"""Get the ellipse patch.

    Args:
        X: A `n x 2`-dim Tensor of samples.
        radius: A `2`-dim Tensor containing the radii.
        angle: A float determining the angle of rotation.
        translate: A `2`-dim Tensor containing the centre of the ellipse.

    Returns:
        The ellipse patch and the transform.
    """
    ellipse = Ellipse((0, 0), width=2 * radius[0], height=2 * radius[1], **kwargs)
    deg = -angle * 180 / torch.pi
    t = translate
    transform = transforms.Affine2D().rotate_deg(deg).translate(t[0], t[1])

    return ellipse, transform
