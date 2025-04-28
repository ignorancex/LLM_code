from typing import Callable, List
import numpy as onp
import jax.numpy as np
import jax
from flax import linen as nn
from flax.core import freeze
import optax
import functools
from tqdm import trange

from utils import *


def get_activation(a: str) -> Callable[[np.ndarray], np.ndarray]:
    if a == 'id':
        return lambda x: x
    return getattr(nn, a)


class MLP(nn.Module):
    n_layers: List[int]
    activation: str
    dropout_rate: float = 0.
    base_layer: nn.Module = nn.Dense
    bn: bool = False

    @nn.compact
    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        act = get_activation(self.activation)
        for i, n_h in enumerate(self.n_layers):
            x = self.base_layer(n_h, name=f'fc_{i}')(x)
            if i+1 != len(self.n_layers):
                x = act(x)
                if self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.bn:
            x = nn.BatchNorm(
                use_running_average=not train, name=f'bn_{i}',
                use_bias=False, use_scale=False)(x)
        return x


class SmallConvNet(nn.Module):

    out_dims: int
    n_hidden: int = 128
    dropout_rate: float = 0.05
    activation: str = 'gelu'

    @nn.compact
    def __call__(self, inp: np.ndarray, train: bool) -> np.ndarray:
        act = get_activation(self.activation)
        assert len(inp.shape) == 4 and inp.shape[-1] <= 3, inp.shape  # NHWC
        h = nn.Conv(features=self.n_hidden, kernel_size=(3, 3), strides=(1, 1), name='conv1')(inp)
        h = act(h)
        h = nn.Conv(features=self.n_hidden, kernel_size=(3, 3), strides=(1, 1), name='conv2')(inp)
        h = act(h)
        h = nn.max_pool(h, (2, 2))
        h = h.reshape((h.shape[0], -1))
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
        h = act(nn.Dense(features=self.n_hidden*2)(h))
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
        h = act(nn.Dense(features=self.n_hidden)(h))
        h = nn.Dense(features=self.out_dims)(h)
        return h


class HLLTModel(nn.Module):

    is_s1: bool  # if true, the input is (C, T, S); otherwise it is (T, S)
    im_feature: bool
    out_dims: int
    activation: str
    dropout_rate: float = 0.05

    def setup(self):
        if self.im_feature:
            self.tnet = SmallConvNet(32, 32, self.dropout_rate, self.activation)
        self.rnet = MLP(n_layers=[128, 64, 32, self.out_dims], activation=self.activation)

    def __call__(self, inp: np.ndarray, train: bool) -> np.ndarray:
        i_s = 1 if self.is_s1 else 0
        if self.im_feature:
            assert inp.shape[1] == 28*28 + 1 + int(self.is_s1), inp.shape
            inp_im = inp[:, i_s: i_s+28*28].reshape((-1, 28, 28, 1))
            inp_t = get_activation(self.activation)(self.tnet(inp_im, train))
        else:
            assert inp.shape[1] == 2 + int(self.is_s1), inp.shape
            inp_t = inp[:, i_s: i_s+1]
        inp = np.concatenate([inp[:, :i_s], inp_t, inp[:, -1:]], 1)
        return self.rnet(inp, train)


def parameterize(manif_type: str, eps: np.ndarray) -> np.ndarray:
    """
    maps an arbitrary eps \in R^d to a point on the corresponding manifold
    eps is expected to satisfy E\|eps\|_2^2 ~ 1.
    """
    if manif_type == 'sph':
        return eps / (eps**2).sum(1, keepdims=True)**0.5
    elif manif_type == 'cube':
        d = eps.shape[-1]
        ret = np.tanh(eps * d**0.5)  # scale input to reach the nonlinear region
        return ret * (3/d)**0.5  # scale down to match prior as below
    raise NotImplementedError(manif_type)


def sample_from_prior(manif_type: str, rkey: np.ndarray, shape: Any) -> np.ndarray:
    """ all priors are expected to satisfy Ez = 0 and E\|z\|_2^2 ~ 1. """
    if manif_type == 'cube':
        # E\|Unif[-1,1]^d\|^2 ~ d / 3
        return (jax.random.uniform(rkey, shape) * 2 - 1) * (3/shape[-1])**0.5
    else:
        return parameterize(manif_type, jax.random.normal(rkey, shape))


def l2_norm(inp: np.ndarray, axis=None, eps=1e-10):
    return inp / ((inp**2).sum(axis) + eps)**0.5


def spectral_normalization(A: np.ndarray, sn_u: np.ndarray, n_iter: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    assert len(sn_u.shape) == len(A.shape) == 2
    for _ in range(n_iter):
        sn_v = l2_norm(A.T @ sn_u)
        sn_u = l2_norm(A @ sn_v)
    # valid for first order derivative
    sn_u, sn_v = map(jax.lax.stop_gradient, (sn_u, sn_v))
    return (sn_u * (A @ sn_v)).sum(), sn_u


class SNDense(nn.Module):

    features: int
    use_bias: bool = True
    dtype: Any = np.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        is_initialized = self.has_variable('state', 'kernel_sn_u')
        inputs = np.asarray(inputs, self.dtype)
        kernel = self.param('kernel',
                            self.kernel_init,
                            (inputs.shape[-1], self.features))
        kernel = np.asarray(kernel, self.dtype)
        kernel_sn_u = self.variable('state', 'kernel_sn_u', np.ones, (inputs.shape[-1], 1))
        y = jax.lax.dot_general(
            inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision)

        spec_norm, new_sn_u = spectral_normalization(kernel, kernel_sn_u.value, 3)
        if inputs.shape[-1] < self.features:
            # for row-normalized D_out*D_in matrices (i.e. fan-in init) the smallest possible 
            # spectral norm is sqrt(D_inp/D_fea) < 1. This may mess with subsequent activations,
            # so we re-scale the matrix here
            spec_norm /= (self.features/inputs.shape[-1])**0.5

        y = y / spec_norm

        if is_initialized:
            # in val mode this update is discarded, and spec_norm is calculated with old u
            kernel_sn_u.value = new_sn_u
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
            bias = np.asarray(bias, self.dtype)
            y += np.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class WAE(nn.Module):
    z_dims: int
    x_dims: int
    dec_layers: list[int]
    disc_layers: list[int]
    activation: str
    prior_type: str    # 'sph', 'torus'
    wae_lam: float

    def setup(self):
        enc_layers = list(reversed(self.dec_layers))
        self.encoder = MLP(enc_layers + [self.z_dims], self.activation)
        self.decoder = MLP(list(self.dec_layers) + [self.x_dims], self.activation)
        self.discriminator = MLP(list(self.disc_layers) + [1], self.activation)

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        z_raw = self.encoder(x, train)
        z = parameterize(self.prior_type, z_raw)
        recox_dims = self.decoder(z, train)
        return recox_dims, z

    def sample(self, batch_size, train: bool):
        eps = jax.random.normal(self.make_rng('z'), (batch_size, self.z_dims))
        z = parameterize(self.prior_type, eps)
        return self.decoder(z, train)
    
    def get_loss(self, x, train: bool):
        recox_dims, qz = self(x, train)
        # draw z_neg ~ P(z)
        eps = jax.random.normal(self.make_rng('z'), qz.shape)
        pz = parameterize(self.prior_type, eps)
        # discriminator: CE loss
        qz_logits, pz_logits = self.discriminator(qz, train), self.discriminator(pz, train)
        disc_loss = -np.mean(nn.log_sigmoid(pz_logits) + np.log(1+1e-7 - nn.sigmoid(qz_logits)))
        # AE: recon_loss + asymmetric GAN loss
        recon_loss = ((recox_dims - x)**2).mean()
        ae_loss = recon_loss - self.wae_lam * np.mean(nn.log_sigmoid(qz_logits))
        #
        lcs = locals()
        stats = dict((k, lcs[k]) for k in ['disc_loss', 'recon_loss', 'ae_loss'])
        return disc_loss, ae_loss, stats

