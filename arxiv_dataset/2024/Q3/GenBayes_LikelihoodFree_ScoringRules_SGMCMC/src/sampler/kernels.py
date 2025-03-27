# This file is modified from the original file located at:
# https://github.com/jeremiecoullon/SGMCMCJax/blob/master/sgmcmcjax/kernels.py
# The modifications include changes to the kernel adapted to our custom gradient estimation functions
# Original file copyright (c) 2022 Jeremie Coullon
# Licensed under the Apache 2.0 License.

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import jit, lax, random

from sgmcmcjax.diffusions import sgnht, psgld
from sgmcmcjax.kernels import _build_langevin_kernel
from .gradient_estimation import build_gradient_estimation_fn

def build_sgnht_lfi_kernel(
    dt: float,
    data,
    joint_log_prob,
    grad_est_func=build_gradient_estimation_fn,
    a: float = 0.01
) -> Tuple[Callable, Callable, Callable]:
    """build stochastic gradient Nose Hoover Thermostats kernel. From http://people.ee.duke.edu/~lcarin/sgnht-4.pdf

    Args:
        dt (float): step size
        data : Observations
        joint_log_prob (Callable): Function returning the joint log-probability of all the data and parameters
        Note: This function is passed to the grad_est_func, which usually accepts a function that has arguments (params, data, n_samples_per_param), although this can be adapted depending on the gradient estimation function.
        grad_est_func (Callable, optional): Function returning the gradient estimation function and its initialisation function
        n_samples_per_param (int, optional): Number of samples to use for estimating the gradient. Defaults to 500.
        a (float, optional): diffusion factor. Defaults to 0.01.

    Returns:
        Tuple[Callable, Callable, Callable]: An (init_fun, kernel, get_params) triple.
    """
    estimate_gradient, init_gradient = grad_est_func(
       joint_log_prob, data 
    )
    init_diff, update_diff, get_p_diff = sgnht(dt, a)
    update_diff = jit(update_diff)
    init_fn, sgnht_kernel, get_params = _build_langevin_kernel(
        init_diff, update_diff, get_p_diff, estimate_gradient, init_gradient
    )
    return init_fn, sgnht_kernel, get_params


def build_psgld_lfi_kernel(
    dt: float,
    data,
    joint_log_prob,
    grad_est_func=build_gradient_estimation_fn,
    alpha: float = 0.99,
    eps: float = 1e-5,
) -> Tuple[Callable, Callable, Callable]:
    """build preconditioned SGLD kernel

    Args:
        dt (float): step size
        data : Observations
        joint_log_prob (Callable): Function returning the joint log-probability of all the data and parameters
        Note: This function is passed to the grad_est_func, which usually accepts a function that has arguments (params, data, n_samples_per_param), although this can be adapted depending on the gradient estimation function.
        grad_est_func (Callable, optional): Function returning the gradient estimation function and its initialisation function
        n_samples_per_param (int, optional): Number of samples to use for estimating the gradient. Defaults to 500.
        a (float, optional): diffusion factor. Defaults to 0.01.

    Returns:
        Tuple[Callable, Callable, Callable]: An (init_fun, kernel, get_params) triple.
    """
    estimate_gradient, init_gradient = grad_est_func(
       joint_log_prob, data 
    )
    init_diff, update_diff, get_p_diff = psgld(dt, alpha, eps)
    init_fn, sgld_kernel, get_params = _build_langevin_kernel(
        init_diff, update_diff, get_p_diff, estimate_gradient, init_gradient
    )
    return init_fn, sgld_kernel, get_params