"""Functions related to statistics"""

import numpy as np
from scipy import interpolate
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial


def sample_from_pdf(pdf_func, start, end, num_samples, num_partition=1000):
    """Note: pdf_func must be vectorized"""
    x = np.linspace(start, end, num_partition)
    cdf = np.cumsum(pdf_func(x) * np.diff(x, prepend=x[0]))
    cdf /= cdf[-1]
    inverse_cdf = interpolate.interp1d(cdf, x)
    return inverse_cdf(np.random.uniform(0, 1, num_samples))

@partial(vmap, in_axes=(0, None, None))
def interp1d_vmap(x, xp, fp):
    return jnp.interp(x, xp, fp)

def sample_from_pdf_jax(rng_key, pdf_func, start, end, num_samples, num_partition=1000):
    """Note: pdf_func must be vectorized"""
    x = jnp.linspace(start, end, num_partition)
    diff = jnp.diff(x)
    x0 = jnp.array([x[0]])
    diff = jnp.concatenate([x0, diff], axis=0)
    cdf = jnp.cumsum(pdf_func(x) * diff)
    cdf /= cdf[-1]
    return interp1d_vmap(jax.random.uniform(rng_key, shape=(num_samples,)), cdf, x)


def bounded_sample(sampler, lower_bound=None, upper_bound=None):
    while True:
        sample = sampler()
        if lower_bound is not None and sample < lower_bound:
            continue
        if upper_bound is not None and sample > upper_bound:
            continue
        return sample


def poisson_process(rate, total_time):
    """Returns the array of occurance time given rate and total time.
    [rate] * [total_time] = 1
    """
    ts = []
    t = np.random.exponential(1/rate)
    while t < total_time:
        ts.append(t)
        t += np.random.exponential(1/rate)
    return ts