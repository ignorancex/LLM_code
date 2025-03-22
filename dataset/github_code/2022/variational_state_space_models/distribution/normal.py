from distribution import Distribution
from typing import NamedTuple
import jax.numpy as jnp
import jax
import scipy
import numpy as np
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class DiagonalNormal(Distribution):
    def __init__(self, mean, std_diag):
        self.mean = mean
        self.std_diag = std_diag

    def sample(self, rng):
        z = self.std_diag*jax.random.normal(rng,
                            (self.mean.shape[-1], )) \
                    + self.mean
        return z

    def log_prob(self, z):
        return jnp.sum(jax.scipy.stats.norm.logpdf(z, self.mean, self.std_diag), axis=-1)

    # kl divergence
    @staticmethod
    def kl_divergence(p, q): 
        return 0.5 * jnp.sum(2*jnp.log(q.std_diag) - 2*jnp.log(p.std_diag) + \
                (jnp.square(p.std_diag) + jnp.square(p.mean -  q.mean))/jnp.square(q.std_diag) - 1, axis=-1)
    
    @staticmethod
    def pdf_prod(p, q):
        p_var = jnp.square(p.std_diag)
        q_var = jnp.square(q.std_diag)
        p_rec = jnp.reciprocal(p_var)
        q_rec = jnp.reciprocal(q_var)
        std_diag = jnp.sqrt(jnp.reciprocal(p_rec + q_rec))
        mean = (p_var*q.mean + q_var*p.mean)/(p_var + q_var)
        return DiagonalNormal(mean, std_diag)

    @staticmethod
    def pdf_combine(p, q, c):
        p_var = jnp.square(p.std_diag)
        q_var = jnp.square(q.std_diag)
        c_var = jnp.square(c.std_diag)
        p_rec = jnp.reciprocal(p_var)
        q_rec = jnp.reciprocal(q_var)
        c_rec = jnp.reciprocal(c_var)
        std_var = jnp.reciprocal(p_rec + q_rec - c_rec)
        mean = std_var*(q_rec*q.mean + p_rec*p.mean - c_rec*c.mean)
        return DiagonalNormal(mean, jnp.sqrt(std_var))

    @staticmethod
    def jensen_shannon(p, q):
        # calculate M_mean, M_std
        m_mean = (p.mean + q.mean)/2
        m_std = jnp.sqrt(jnp.square(p.std_diag) + jnp.square(q.std_diag))
        m = DiagonalNormal(m_mean, m_std)
        return (DiagonalNormal.kl_divergence(p, m) + DiagonalNormal.kl_divergence(q, m))/2

    def tree_flatten(self):
        return (self.mean, self.std_diag), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class MultivariateNormal(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def sample(self, rng):
        return jax.random.multivariate_normal(rng, self.mean, self.cov)
    
    def log_prob(self, x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def tree_flatten(self):
        return (self.mean, self.cov), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class ConcentrationNormal(Distribution):
    def __init__(self, inf, conc):
        self.inf = inf
        self.conc = conc

    def calc_mean(self):
        cov = jnp.linalg.inv(self.conc)
        mean = (cov @ jnp.expand_dims(self.inf, -1)).squeeze(-1)
        return mean


    def sample(self, rng):
        cov = jnp.linalg.inv(self.conc)
        mean = (cov @ jnp.expand_dims(self.inf, -1)).squeeze(-1)
        return jax.random.multivariate_normal(rng, mean, cov)

    def log_prob(self, x):
        cov = jnp.linalg.inv(self.conc)
        mean = (cov @ jnp.expand_dims(self.inf, -1)).squeeze(-1)
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)

    @staticmethod
    def convert(dist):
        if isinstance(dist, MultivariateNormal):
            conc = jnp.linalg.inv(dist.cov)
            inf = (conc @ jnp.expand_dims(dist.mean, -1)).squeeze(-1)
            return ConcentrationNormal(inf, conc)
        elif isinstance(dist, DiagonalNormal):
            inv_diagonal = jnp.reciprocal(jnp.square(dist.std_diag))
            conc = jnp.diag(inv_diagonal)
            inf = inv_diagonal * dist.mean
            return ConcentrationNormal(inf, conc)
        elif isinstance(dist, ConcentrationNormal):
            return dist
        else:
            raise ValueError(f"Unknown dist type {dist}")

    @staticmethod
    def pdf_prod_multi(dists):
        conc = jnp.sum(dists.conc, 0) # add concentration matrix
        inf = jnp.sum(dists.inf, 0)
        return ConcentrationNormal(inf, conc)

    @staticmethod
    def pdf_prod_update(d, dists):
        conc = d.conc + jnp.sum(dists.conc, 0) # add concentration matrix
        inf = d.inf + jnp.sum(dists.inf, 0)
        return ConcentrationNormal(inf, conc)

    @staticmethod
    def pdf_prod(p, q):
        conc = p.conc + q.conc
        inf = p.inf + q.inf
        return ConcentrationNormal(inf, conc)

    @staticmethod
    def pdf_div(A, B):
        conc = A.conc - B.conc
        inf = A.inf - B.inf
        return ConcentrationNormal(inf, conc)

    @staticmethod
    def kl_divergence(p, q):
        Sigma_1 = jnp.linalg.inv(p.conc)
        Sigma_2 = jnp.linalg.inv(q.conc)
        Sigma_2_inv = q.conc

        mu_1 = (Sigma_1 @ jnp.expand_dims(p.inf, -1)).squeeze(-1)
        mu_2 = (Sigma_2 @ jnp.expand_dims(q.inf, -1)).squeeze(-1)

        _, log_det_Sigma_1 = jnp.linalg.slogdet(Sigma_1)
        _, neg_log_det_Sigma_2 = jnp.linalg.slogdet(Sigma_2_inv)
        log_det_Sigma_2 = -neg_log_det_Sigma_2

        d = Sigma_1.shape[-1]

        tr = jnp.trace(Sigma_2_inv @ Sigma_1)
        diff = jnp.expand_dims(mu_2 - mu_1, -1)
        quad = (diff.T @ (Sigma_2_inv @ diff)).squeeze(-1).squeeze(-1)
        return (log_det_Sigma_2 - log_det_Sigma_1 - d + tr + quad)/2

    def tree_flatten(self):
        return (self.inf, self.conc), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)