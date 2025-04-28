import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit)
def linear_kernel(x2: jnp.ndarray, x1: jnp.ndarray, y2: jnp.ndarray, y1: jnp.ndarray, scale: float = 1.0) -> float:
  return jnp.dot(scale * (x2-x1), scale * (y2-y1))

@partial(jax.jit)
def rbf_kernel(x2: jnp.ndarray, x1: jnp.ndarray, y2: jnp.ndarray, y1: jnp.ndarray, scale: float = 1.0) -> float:
  inc1 = jnp.exp(-jnp.sum((x2 - y2) ** 2) / (2.0 * scale ** 2))
  inc2 = jnp.exp(-jnp.sum((x1 - y1) ** 2) / (2.0 * scale ** 2))
  inc3 = jnp.exp(-jnp.sum((x2 - y1) ** 2) / (2.0 * scale ** 2))
  inc4 = jnp.exp(-jnp.sum((x1 - y2) ** 2) / (2.0 * scale ** 2))
  return inc1 + inc2 - inc3 - inc4
