import numpy as np
import jax.numpy as jnp
from typing import NamedTuple
import jax

# utility function for unpacking outputs
# to a discrete logistic mixture
def _unpack_nn_output(nn_out, c): # k is the number of mixture parts
    # the last dimension should be mixture_parts * (3*c + 1)
    assert nn_out.shape[-1] % (3*c + 1) == 0
    k = nn_out.shape[-1] // (3*c + 1)
    *batch, h, w, _ = nn_out.shape
    logit_weights, nn_out = jnp.split(nn_out, [k], -1)
    m, s, t = jnp.moveaxis(
        jnp.reshape(nn_out, tuple(batch) + (h, w, c, k, 3)), (-2, -1), (-4, 0))
    assert m.shape == tuple(batch) + (k, h, w, c)
    inv_scales = jnp.maximum(jax.nn.softplus(s), 1e-7)
    return ImageDiscreteLogisticMixture(
        m=m,
        t=jnp.tanh(t),
        inv_scales=inv_scales,
        logit_weights=jnp.moveaxis(logit_weights, -1, -3)
    )

class ImageDiscreteLogisticMixture(NamedTuple):
    # m contains offsets
    m: np.array
    # t is for conditioning on previous sub-pixels
    t: np.array
    inv_scales: np.array
    logit_weights: np.array

    @staticmethod
    def unpack(nn_out, channels):
        return _unpack_nn_output(nn_out, channels)

    def sample(self, rng):
        pass

    def log_prob(self, x):
        assert x.dtype == jnp.uint8
        # convert x to -1 to 1 float
        x = x.astype(jnp.float32) / 127.5 - 1

        x = jnp.expand_dims(x, -4)  # Add mixture dimension
        if x.shape[-1] == 3:
            mean_red   = m[..., 0]
            mean_green = m[..., 1] + t[..., 0] * img[..., 0]
            mean_blue  = m[..., 2] + t[..., 1] * img[..., 0] + t[..., 2] * img[..., 1]
            means = jnp.stack((mean_red, mean_green, mean_blue), axis=-1)
        elif x.shape[-1] == 1:
            means = m
        else:
            raise ValueError("Expected 1 or 3 channels")
        logprobs = jnp.sum(_logistic_logpmf(img, means, inv_scales), -1)
        log_mix_coeffs = logit_weights - logsumexp(logit_weights, -3, keepdims=True)
        return jnp.sum(logsumexp(log_mix_coeffs + logprobs, -3), (-2, -1))

def _logistic_logpmf(img, means, inv_scales):
    centered = img - means
    top    = -jnp.logaddexp(0,  (centered - 1 / 255) * inv_scales)
    bottom = -jnp.logaddexp(0, -(centered + 1 / 255) * inv_scales)
    mid = _log1mexp(inv_scales / 127.5) + top + bottom
    return jnp.select([img == -1, img == 1], [bottom, top], mid)

@jax.custom_jvp
def _log1mexp(x):
  """Accurate computation of log(1 - exp(-x)) for x > 0."""

  # Method from
  # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  return jnp.where(x > jnp.log(2), jnp.log1p(-jnp.exp(-x)),
                   jnp.log(-jnp.expm1(-x)))


# log1mexp produces NAN gradients for small inputs because the derivative of the
# log1p(-exp(-eps)) branch has a zero divisor (1 + -jnp.exp(-eps)), and NANs in
# the derivative of one branch of a where cause NANs in the where's vjp, even
# when the NAN branch is not taken. See
# https://github.com/google/jax/issues/1052. We work around this by defining a
# custom jvp.
_log1mexp.defjvps(lambda t, _, x: t / jnp.expm1(x))