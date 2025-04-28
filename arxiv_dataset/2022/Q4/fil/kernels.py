import dataclasses
import functools
import jax
from jax import numpy as np
from flax import struct, linen as nn

import imq_rf
from utils import *


# ====== wraps jax objects ====== 

class KernelSpec(struct.PyTreeNode):

    name: str = 'rbf'
    bandwidth: float = 0.01
    bw_med_multiplier: float = -1
    poly_k: int = 5
    matern_k: int = 3
    imq_amp: Any = -1
    var: float = 1.

    def instantiate(self, x_train=None):
        ret = self
        if self.bw_med_multiplier > 0:
            h = median_sqdist(x_train) * self.bw_med_multiplier**2
            ret = dataclasses.replace(ret, bw_med_multiplier=-1, bandwidth=h)
        if self.name == 'imq':
            ret = dataclasses.replace(
                ret,
                imq_amp=imq_rf.imq_amplitude_frequency_and_probs(x_train.shape[1]))
        return ret

    def create(self, x_train=None):
        if self.bw_med_multiplier > 0:
            h = median_sqdist(x_train) * self.bw_med_multiplier**2
        else:
            h = self.bandwidth

        if self.name == 'rbf':
            return RBFKernel(var=self.var, h=h)
        if self.name == 'imq':
            return IMQKernel(var=self.var, h=h, amp_tuple=self.imq_amp)
        # if self.name == 'matern':
        #     return MaternKernel(k=self.matern_k, h=h)
        if self.name == 'poly':
            return PolynomialKernel(x_train=None, h=h, k=self.poly_k, var=self.var)
        raise NotImplementedError(self.name)


class RFExpander(nn.Module):

    def init_and_call(self, _: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @nn.compact
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.init_and_call(inp)


class GPSampler(object):

    """ helper class that converts flax classes back """

    def __init__(self, _, out_dims: int, n_rf: int, kspec: KernelSpec, pkey: np.ndarray):
        if kspec.name == 'poly':
            n_rf = kspec.poly_k
        self.n_rf = n_rf
        self.rf_key, k2 = jax.random.split(pkey)
        self.kern = kspec.create()
        self.W = jax.random.normal(k2, (n_rf, out_dims))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_rf = self.kern.rf_expand(self.n_rf, self.rf_key, x)
        return x_rf @ self.W


# ====== old kernel classes follow ====== 


class Kernel(object):

    def __init__(self):
        pass

    def __call__(self, x1: np.ndarray, x2: np.ndarray, lhs_deriv=False, rhs_deriv=False
                 ) -> np.ndarray:
        assert not (lhs_deriv or rhs_deriv), NotImplementedError()
        return self._call(x1, x2)

    def kdiag(self, x: np.ndarray, is_deriv=False) -> np.ndarray:
        assert not is_deriv, NotImplementedError()
        return np.diag(self(x, x))

    def rf_expand(self, n_rf: int, rng: np.ndarray, inp: np.ndarray) -> np.ndarray:
        # the subsequent layer will handle scaling
        raise NotImplementedError()


class IMQKernel(Kernel):

    def __init__(self, h, amp_tuple, var=1.):
        self.h, self.var = h, var
        self.amp, self.amp_probs = amp_tuple

    def _call(self, x1, x2):
        sqd = (x1[:,None,0] - x2[None,:,0])**2 / self.h
        return self.var * (1 + sqd)**-0.5

    def rf_expand(self, n_rf, rkey, inp) -> np.ndarray:
        return self.var**0.5 * imq_rf.imq_rff_features(
            n_rf, rkey, inp, self.h**0.5, self.amp, self.amp_probs)


class CircularMaternKernel(Kernel):
    r"""
    Borovitskiy et al (2021), Matern Gaussian processes on Riemannian manifolds. Example 8.
    The spectral density is rho_nu(n) \propto (2nu/kappa^2 + 4pi^2 n^2)^{-nu-1/2}, 
    so we approximate the kernel with truncations
    
    The K-L expansion of the GP is (Eq.(13))
        sqrt{rho_nu(0)} eps_{0} * 1 + \sum_{n>0} \sqrt{2rho_nu(n)} (
            eps_{n,1} cos(2pi n x) + eps_{n,2} sin(2pi n x))
    (the sqrt 2 comes from summing two i.i.d. N(0,1) rvs.)
    So the kernel is 
        rho(0) + \sum_{n>0} 2\rho(n)[cos(2pi n x)cos(2pi n x') + sin(2pi n x)sin(2pi n x')]
      = rho(0) + \sum_{n>0} 2\rho(n) cos(2pi n (x-x'))
    we store a modified version of rho, so that k(x,x') = sum_{n>=0} rho_n cos (2pi n (x-x')),
    and sum_n rho_n = 1.
    """
    def __init__(self, k, trunc_order=400, kappa=None, x_train=None, var=1.):
        if x_train is None:
            assert kappa is not None
        else:
            assert x_train.shape[-1] == 1
            kappa = median_sqdist(x_train[:5000])
        self.kappa = kappa
        self.nu = k/2
        self.var = var
        self.trunc_order = trunc_order
        self.ns = ns = np.arange(trunc_order)
        rho_unnormalized = np.power(2*self.nu/self.kappa**2 + (2*np.pi*ns)**2, -(self.nu+1/2))
        rho_unnormalized = jax.ops.index_mul(rho_unnormalized, 0, 0.5)
        self.rho = rho_unnormalized / rho_unnormalized.sum()
    
    def __call__(self, x1, x2, lhs_deriv=False, rhs_deriv=False):
        assert len(x1.shape) == len(x2.shape) == 2, 'expect shape [N,1]'
        x_diff = x1[:,None,0] - x2[None,:,0]

        n_2pi = 2 * np.pi * self.ns
        rho, fn = {
            (False, False): (self.rho, np.cos),
            (True, False): (-self.rho * n_2pi, np.sin),
            (False, True): (self.rho * n_2pi, np.sin),
            (True, True): (self.rho * n_2pi**2, np.cos)
        }[(lhs_deriv, rhs_deriv)]

        if max(x1.shape[0], x2.shape[0]) <= 500:
            x_diff = x_diff[:, :, None]
            ret = (fn(n_2pi[None, None] * x_diff) * rho[None, None]).sum(-1)
        else:
            # avoid putting x_diff into JIT buffer
            ret = jax.lax.fori_loop(
                0, self.trunc_order,
                lambda n, cur: (cur[0] + fn(2 * np.pi * n * cur[1]) * rho[n], cur[1]),
                (np.zeros(x_diff.shape[:2], dtype=x1.dtype), x_diff)
            )[0]
        return ret * self.var
    
    def kdiag(self, x, is_deriv=False):
        ret = np.ones_like(x)[:, 0] * self.var
        if is_deriv:
            assert self.nu > 1, ValueError("derivative functionals are unbounded in H")
            ret *= ((2*np.pi*self.ns)**2 * self.rho).sum()
        return ret


class LinearKernel(Kernel):
    
    def __init__(self, inp_stats=(0, 1), intercept=False, var=1.):
        self.inp_stats = inp_stats
        self.intercept = intercept
        self.var = var

    def _call(self, x1, x2):
        t1 = self.rf_expand(None, None, x1)
        t2 = self.rf_expand(None, None, x2)
        return t1 @ t2.T
    
    def kdiag(self, x, is_deriv=False):
        assert not is_deriv, NotImplementedError()
        return (self.rf_expand(None, None, x)**2).sum(axis=-1)

    def rf_expand(self, _, __, inp):
        ret = ((inp - self.inp_stats[0]) / (self.inp_stats[1]+1e-6)).astype('f')
        if self.intercept:
            ret = np.concatenate([ret, np.ones_like(ret[:, -1:])], -1)
        return ret * self.var**0.5


def poly_expand(inp, k):
    assert len(inp.shape) == 2 and inp.shape[-1] == 1
    return np.concatenate([inp**i for i in range(1, k+1)], 1)


class PolynomialKernel(LinearKernel):

    def __init__(self, x_train=None, h=None, k=2, var=1.):
        self.k = k
        if x_train is None:
            # crude estimate for the Hermite polynomial coefficients
            x_train = jax.random.normal(jax.random.PRNGKey(23), shape=(1000, 1)) * (h**0.5)
        xf = poly_expand(x_train, k)
        xm, xs = xf.mean(0), xf.std(0)
        super().__init__((xm, xs), intercept=(x_train is None), var=var)

    def rf_expand(self, _, __, inp):
        return super().rf_expand(None, None, poly_expand(inp, self.k)) * self.var**0.5


def median_sqdist(x, n_max=2500):
    x = x[:n_max]
    ret = []
    for i in range(x.shape[1]):
        ret.append(np.median((x[:,None,i] - x[None,:,i])**2))
    return np.array(ret)


@jax.jit
def get_sqdist(x1, x2, h):
    return (((x1[:,None] - x2[None,:])**2) / h).sum(-1)


class RBFKernel(Kernel):
    
    def __init__(self, var=1., h=None, x_train=None):
        self.var = var
        if h is not None:
            self.h = h
        else:
            self.h = median_sqdist(x_train[:2500])
    
    def _call(self, x1, x2):
        return self.var * np.exp(-get_sqdist(x1, x2, self.h) / 2)
    
    def kdiag(self, x, is_deriv=False):
        assert not is_deriv, NotImplementedError()
        return self.var * np.ones((x.shape[0],))

    def rf_expand(self, n_rf, rkey, inp):
        """ Rahimi and Recht (2007) """
        k1, k2 = jax.random.split(rkey)
        W = jax.random.normal(k1, [inp.shape[1], n_rf])
        b = jax.random.uniform(k2, [n_rf]) * 2 * np.pi
        inp = inp / self.h**0.5
        ret = (2*self.var)**0.5 * np.cos(inp @ W + b) / n_rf**0.5
        return ret.astype('f')


class ScaleMixtureKernel(Kernel):

    def __init__(self, x_train, scales=[0.1, 1, 10], KBase=RBFKernel, **kw):
        h = median_sqdist(x_train)
        self.ks = []
        for s in scales:
            kw['h'] = h * s
            self.ks.append(KBase(**kw))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _call(self, x1, x2):
        ret = self.ks[0](x1, x2)
        for k in self.ks[1:]:
            ret += k(x1, x2)
        return ret / len(self.ks)

    def kdiag(self, x, is_deriv=False):
        assert not is_deriv, NotImplementedError()
        return sum(k.kdiag(x) for k in self.ks) / len(self.ks)

    def rf_expand(self, n, r, inp):
        return (sum(k.rf_expand(n, r, inp) for k in self.ks) / len(self.ks)**0.5).astype('f')

    
class MaternKernel(Kernel):
    
    def __init__(self, k, var=1., h=None, x_train=None):
        self.k = k
        self.var = var
        assert k in [1, 3, 5, 7]
        if h is not None:
            self.h = h
        else:
            assert x_train is not None
            sqdist = (x_train[:,None]-x_train[None,:])**2
            self.h = np.median(sqdist.reshape((x_train.shape[0]**2, x_train.shape[-1])), axis=0)

    def _call(self, x1, x2):
        dist = ((x1[:,None] - x2[None,:])**2 / self.h).sum(-1) ** 0.5
        d = (self.k**0.5) * dist
        if self.k == 1:
            ret = np.exp(-d)
        elif self.k == 3:
            ret = (1 + d) * np.exp(-d)
        elif self.k == 5:
            ret = (1 + d + d**2/3) * np.exp(-d)
        elif self.k == 7:
            ret = (1 + d + 2/5*d**2 + d**3/15) * np.exp(-d)
        return self.var * ret
    
    def kdiag(self, x, is_deriv=False):
        assert not is_deriv, NotImplementedError()
        return self.var * np.ones((x.shape[0],))
