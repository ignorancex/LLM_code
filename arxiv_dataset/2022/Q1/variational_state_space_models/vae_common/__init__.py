from distribution import Distribution
from distribution.normal import ConcentrationNormal
from distribution.logistic import ImageDiscreteLogisticMixture
from models.conv_64 import conv_64_decoder, conv_64_encoder
import haiku as hk
from haiku.initializers import TruncatedNormal
import math
import jax
import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.experimental.host_callback import id_print

class LogisticMixtureDecoder(hk.Module):
    def __init__(self, channels, mixture_parts, fun=conv_64_decoder):
        super().__init__()
        self.channels = channels
        self.mixture_parts = mixture_parts
        self.fun = fun

    def __call__(self, z, is_training, cross_axes):
        nn_out = self.fun(z, self.mixture_parts*(3*self.channels + 1), 
                                is_training, cross_axes)
        return ImageDiscreteLogisticMixture.unpack(nn_out, self.channels)

@register_pytree_node_class
class ImageNormal(Distribution):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    @property
    def mode(self):
        x = np.clip(self.mean, -1, 1)
        # convert back to the correct range
        x = (x + 1)*127.5
        x = x.astype(jnp.uint8)
        return x

    def sample(self, rng):
        noise = self.sigma*jax.random.normal(rng, self.mean.shape)
        x = np.clip(self.mean + noise, -1, 1)
        # convert back to the correct range
        x = (x + 1)*127.5
        x = x.astype(jnp.uint8)
        return x

    def log_prob(self, x):
        # convert x to float in the range -1, 1
        m = x.astype(jnp.float32)/127.5 - 1
        diff = (self.mean - m).reshape((-1,))
        log_pdf = jnp.sum(jax.scipy.stats.norm.logpdf(diff, 0, self.sigma), axis=-1)
        return log_pdf

    def tree_flatten(self):
        return (self.mean, self.sigma), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class ConvImageNormalDecoder(hk.Module):
    def __init__(self, channels, sigma, fun=conv_64_decoder):
        super().__init__(name='decoder')
        self.channels = channels
        self.sigma = sigma
        self.fun = fun

    def __call__(self, z, is_training, cross_axes):
        z_dim = z.shape[-1]
        proc = hk.Sequential([
            hk.Linear(4*z_dim*z_dim),
            jax.nn.gelu,
            hk.Linear(4*z_dim*z_dim),
            jax.nn.gelu,
            hk.Linear(4*z_dim*z_dim)
        ])(z)
        nn_out = self.fun(proc, self.channels, is_training, cross_axes)
        mean = jax.nn.tanh(nn_out)
        return ImageNormal(nn_out, self.sigma)

class ConvImageNormalPosDecoder(hk.Module):
    def __init__(self, waves, channels, sigma, fun=conv_64_decoder):
        super().__init__(name='decoder')
        self.waves = waves
        self.wave_factors = jnp.array([2**k * math.pi for k in range(self.waves)])
        self.channels = channels
        self.sigma = sigma
        self.fun = fun

    def __call__(self, z, is_training, cross_axes):
        z_dim = z.shape[-1]
        # hit z with the waves per-channel
        z_with_waves = jnp.expand_dims(z, -1) * self.wave_factors
        z_sines = jnp.sin(z_with_waves)
        z_cos = jnp.cos(z_with_waves)

        # stack and reshape into final
        # position-encoded vector
        z_pos = jnp.concatenate((z_sines, z_cos), -1).reshape(-1)

        # do the positional encoding
        proc = hk.Sequential([
            hk.Linear(z_pos.shape[0]),
            jax.nn.gelu,
        ])(z_pos)
        nn_out = self.fun(proc, self.channels, is_training, cross_axes)
        mean = jax.nn.tanh(nn_out)
        return ImageNormal(nn_out, self.sigma)

class ConvImageConcentrationEncoder(hk.Module):
    def __init__(self, z_dim, Prior_inv, init_L, min_cov):
        super().__init__(name='encoder')
        self.z_dim = z_dim
        self.Prior_inv = Prior_inv
        self.init_L = init_L
        self.min_cov = min_cov
    
    def __call__(self, x, is_training, cross_axes):
        # convert x to range [-1, 1]
        m = x.astype(jnp.float32)/127.5 - 1
        p = conv_64_encoder(m, self.z_dim*self.z_dim*4, is_training, cross_axes)
        mu = hk.Sequential([
            hk.Linear(4*self.z_dim*self.z_dim),
            jax.nn.gelu,
            hk.Linear(4*self.z_dim*self.z_dim),
            jax.nn.gelu,
            hk.Linear(self.z_dim)
        ])(p)
        mu = 10*jax.nn.tanh(mu)
        L_vec = hk.get_parameter("L", (self.z_dim,self.z_dim,), jnp.float32,
            lambda s, d: self.init_L*jnp.eye(self.z_dim))
        L = L_vec.reshape((self.z_dim, self.z_dim))

        L_conc = jnp.linalg.inv(L @ L.T + \
                self.min_cov*jnp.eye(self.z_dim))
        # L_conc = jnp.linalg.inv(jnp.array(
        #     [[0.0001, 0],
        #      [0, 10]]
        # ))
        #L_conc = jnp.diag(jnp.exp(-L))
        # L_conc = jnp.array([[0.001, 0], [0, 10]])
        # cov = jnp.array([[0.001, 0], [0, 10]])
        # L_conc = jnp.linalg.inv(cov)
        # make sure q(z|x) has more information than the prior
        conc = L_conc + self.Prior_inv
        inf = L_conc @ mu
        return ConcentrationNormal(inf, conc)