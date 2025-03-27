# This file is modified from the original file located at:
# https://github.com/jeremiecoullon/SGMCMC_bandit_tuning/blob/master/tuning/util.py
# Original file copyright (c) 2021 Jeremie Coullon

import numpy as np
from jax import random
from .sampler import timed_sampler
from jax.tree_util import tree_leaves, tree_flatten
from jax import jit, lax
import jax.numpy as jnp
from jax.experimental import host_callback
from tqdm.auto import tqdm

def build_metric_curve_fn(dict_params, list_time_budget, partial_build_kernel, params_IC, error_fn):
    "Build a function that calculates the ksd curves (or for any metric)"
    timed_sampler_factory = timed_sampler(partial_build_kernel)
    print(f"Running for {dict_params}")
    my_timed_sampler = timed_sampler_factory(**dict_params)
    def run_metric_curve(key):
        metric_curve = []
        for time_budget in list_time_budget:
            samples, _ = my_timed_sampler(key, time_budget, params_IC)
            print(f"Budget: {time_budget}, num_samples = {len(samples)}")
            metric_curve.append(error_fn(samples))
        return metric_curve
    return run_metric_curve


def get_mean_sd_ksd_curve(key, num_curves, get_metric_curve):
    "Get the mean and standard deviation of several realisations of ksd curves"
    list_ksd_curves = []
    keys = random.split(key, num_curves)
    for idx, k in enumerate(keys):
        print(f"\nIteration {idx}/{num_curves}")
        ksd_curve = get_metric_curve(k)
        list_ksd_curves.append(ksd_curve)

    mean_ksd_curve = np.mean(list_ksd_curves, axis=0)
    sd_ksd_curve = np.std(list_ksd_curves, axis=0)
    return mean_ksd_curve, sd_ksd_curve

def wait_until_computed(x):
    for leaf in tree_leaves(x):
        leaf.block_until_ready()


@jit
def flatten_param_list(samples):
    """
    Turns a list of PyTrees into an array of shape (Nsamples, dim)
    If the samples are already jnp.ndarray, then simply return them
    """
    if isinstance(samples, jnp.ndarray):
        return samples
    else:
        flattened_samples = []
        for lesam in samples:
            flat_sample, _ = tree_flatten(lesam)
            flattened_samples.append(jnp.concatenate([elem.flatten() for elem in flat_sample]))
        return jnp.array(flattened_samples)