#!/usr/bin/env python3

r"""
Utilities for MC and qMC sampling.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.utils.sampling import manual_seed, sample_hypersphere
from torch import Tensor
from torch.quasirandom import SobolEngine

from scalarize.utils.scalarization_parameters import (
    OrderedUniform,
    SimplexWeight,
    UnitVector,
)


def sample_unit_vector(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Uniformly sample from the non-negative surface of unit d-sphere.

    Args:
        d: The dimension of the hypersphere.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An  `n x d` tensor of samples from the positive surface of the d-hypersphere.

    """
    return abs(
        sample_hypersphere(d=d, n=n, qmc=qmc, seed=seed, device=device, dtype=dtype)
    )


def sample_ordered_uniform(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    ordered: bool = True,
    descending: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Sample uniformly from the d-uniform (order) statistics.

    Args:
        d: The dimension of the simplex.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        ordered: If True, we generate from the ordered uniform, else we sample from
            the standard uniform.
        descending: If True, sort in descending order, else sort in ascending order.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An `n x d` tensor of samples from the d-uniform (order) statistics.

    """
    d_eff = d + 1 if ordered else d
    dtype = torch.float if dtype is None else dtype
    if qmc:
        sobol_engine = SobolEngine(d_eff, scramble=True, seed=seed)
        rnd = sobol_engine.draw(n, dtype=dtype)
    else:
        with manual_seed(seed=seed):
            rnd = torch.rand(n, d_eff, dtype=dtype)

    if ordered:
        samples = OrderedUniform.exponential_spacing(X=rnd, descending=descending)
    else:
        samples = rnd

    if device is not None:
        samples = samples.to(device)
    return samples


def sample_ordered_simplex(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    ordered: bool = True,
    descending: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Sample uniformly from the (ordered) d-simplex.

    Args:
        d: The dimension of the simplex.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        ordered: If True, we generate from the ordered uniform, else we sample from
            the standard uniform.
        descending: If True, sort in descending order, else sort in ascending order.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An `n x d` tensor of samples from the uniform (ordered) d-simplex.

    """
    ordered_rnd = sample_ordered_uniform(
        d=d,
        n=n,
        qmc=qmc,
        seed=seed,
        ordered=ordered,
        descending=descending,
        device=device,
        dtype=dtype,
    )

    # inverse transform
    return SimplexWeight.log_normalize(ordered_rnd)


def sample_ordered_unit_vector(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    ordered: bool = True,
    descending: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Sample from the uniform (ordered) non-negative surface of unit d-sphere.

    Args:
        d: The dimension of the simplex.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        ordered: If True, we generate from the ordered uniform, else we sample from
            the standard uniform.
        descending: If True, sort in descending order, else sort in ascending order.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An `n x d` tensor of samples from the uniform (ordered) non-negative surface
            of the unit d-sphere.

    """
    ordered_rnd = sample_ordered_uniform(
        d=d,
        n=n,
        qmc=qmc,
        seed=seed,
        ordered=ordered,
        descending=descending,
        device=device,
        dtype=dtype,
    )

    # inverse transform
    return UnitVector.erf_normalize(ordered_rnd)


def sample_permutations(
    weights: Tensor,
    n: int = 1,
    qmc: bool = False,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Sample from the space of permutations without replacement.

    Args:
        weights: A `d`-dim Tensor containing the probability weights.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An `n x d` tensor of samples containing the permutation indices.

    """
    d = len(weights)
    if not torch.all(weights > 0):
        raise ValueError("The weights need to be positive.")
    w = weights / torch.sum(weights)

    dtype = torch.float if dtype is None else dtype
    if qmc:
        sobol_engine = SobolEngine(d, scramble=True, seed=seed)
        rnd = sobol_engine.draw(n, dtype=dtype)
    else:
        with manual_seed(seed=seed):
            rnd = torch.rand(n, d, dtype=dtype)

    _, samples = torch.sort(torch.pow(rnd, 1 / w), descending=True)

    if device is not None:
        samples = samples.to(device)

    return samples
