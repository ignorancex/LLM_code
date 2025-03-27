# This file is modified from the original file located at:
# https://github.com/jeremiecoullon/SGMCMCJax/blob/master/sgmcmcjax/optimizer.py
# Original file copyright (c) 2022 Jeremie Coullon
# Licensed under the Apache 2.0 License.

from typing import Callable, Tuple
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import jit, lax, random
from jax.example_libraries.optimizers import adam
from .gradient_estimation import build_gradient_estimation_fn

def build_adam_lfi_optimizer(
    dt: float,
    data,
    joint_log_prob,
    grad_est_func=build_gradient_estimation_fn,
) -> Callable:
    """build adam optimizer using JAX `optimizers` module

    Args:
        dt (float): step size
        data : Observations
        joint_log_prob (Callable): Function returning the joint log-probability of all the data and parameters
        Note: This function is passed to the grad_est_func, which usually accepts a function that has arguments (params, data, n_samples_per_param), although this can be adapted depending on the gradient estimation function.
        grad_est_func (Callable, optional): Function returning the gradient estimation function and its initialisation function
        n_samples_per_param (int, optional): Number of samples to use for estimating the gradient. Defaults to 500.

    Returns:
        Callable: optimizer function with signature:
            Args:
                key (PRNGKey): random key

                Niters (int): number of iterations

                params_IC (PyTree): initial parameters
            Returns:
                PyTree: final parameters

                jnp.ndarray: array of log-posterior values during the optimization
    """
    estimate_gradient, init_gradient = grad_est_func(
       joint_log_prob, data 
    )

    opt_init, opt_update, get_params = adam(dt)
    opt_update = jit(opt_update)

    def run_adam(key, Niters, params_IC):
        key, subkey = jax.random.split(key)
        opt_state = opt_init(params_IC)
        state = init_gradient(subkey, get_params(opt_state))
        
        for i in tqdm(range(Niters)):
            key, subkey = jax.random.split(key)
            grads, state = estimate_gradient(i, key, get_params(opt_state), state)
            opt_state = opt_update(i, - grads, opt_state)

        return get_params(opt_state)

    return run_adam

