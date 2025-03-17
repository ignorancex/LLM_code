import haiku as hk
import jax
import jax.numpy as jnp
import math
from typing import NamedTuple, List, Any
from functools import wraps

class Distribution:
    def sample(self, rng):
        raise RuntimeError("Not implemented")
    
    def log_prob(self, x):
        raise RuntimeError("Not implemented")

    def sample_with_prob(self, rng):
        x = self.sample(rng)
        pdf = self.log_prob(x)
        return (x, pdf)
    
    def multi_sample(self, rng, n):
        # vmap sample() function
        # split rng by n
        rngs = jax.random.split(rng, n)
        return jax.vmap(self.sample)(rngs)

    def multi_log_prob(self, x):
        return jax.vmap(self.log_prob)(x)

    def multi_sample_with_prob(self, rng, n):
        # vmap sample() function
        # split rng by n
        rngs = jax.random.split(rng, n)
        return jax.vmap(self.sample_with_prob)(rngs)


# Monte-carlo expectation of a function over a particular distributio n
def mc_expectation(distribution_over, expr_lam, rng, n=64):
    x, prob = distribution_over.multi_sample_with_prob(rng, n)
    exp = jnp.mean(expr_lam(x, prob), axis=0)
    return exp

# will assume that expr_lam is the log of what we want to integrate
# and so will use log_sum_exp
# the output is the log of the expectation
def mc_expectation_logs(distribution_over, expr_lam, rng, n=64):
    x, log_pdf = distribution_over.multi_sample_with_prob(rng, n)
    exp = jax.scipy.special.logsumexp(expr_lam(x, log_pdf), axis=0) - math.log(n)
    return exp

# computes  D_KL(P|Q)
def mc_kl_divergence(P, Q, rng, n=64):
    return mc_expectation(P, lambda s, p_log_pdf: p_log_pdf - Q.log_pdf(s), rng, n)

def mc_kl_divergence_scaled(P, q_scaled_log_pdf, rng, n=64):
    x, log_pdf = P.multi_sample_with_prob(rng, n)
    log_scaled_pdf = q_scaled_log_pdf(x)
    log_norm = jax.scipy.special.logsumexp(log_scaled_pdf - log_pdf, axis=0) - math.log(n)
    exp = jnp.mean(log_pdf - (log_scaled_pdf - log_norm), axis=0)
    return exp