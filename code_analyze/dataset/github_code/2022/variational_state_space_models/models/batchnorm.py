# From the original haiku source,
# modified so as to normalize across
# multiple cross-axes

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Sequence, List
from haiku._src import utils

class MultiBatchNorm(hk.Module):
  def __init__(
      self,
      create_scale: bool,
      create_offset: bool,
      decay_rate: float,
      eps: float = 1e-5,
      scale_init: Optional[hk.initializers.Initializer] = None,
      offset_init: Optional[hk.initializers.Initializer] = None,
      axis: Optional[Sequence[int]] = None,
      data_format: str = "channels_last",
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if not create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`")
    if not create_offset and offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`")
    self.create_scale = create_scale
    self.create_offset = create_offset
    self.eps = eps
    self.scale_init = scale_init or jnp.ones
    self.offset_init = offset_init or jnp.zeros
    self.axis = axis
    self.channel_index = utils.get_channel_index(data_format)
    self.mean_ema = hk.ExponentialMovingAverage(decay_rate, name="mean_ema")
    self.var_ema = hk.ExponentialMovingAverage(decay_rate, name="var_ema")

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool,
      cross_axes: List[str] = [],
      test_local_stats: bool = False,
      scale: Optional[jnp.ndarray] = None,
      offset: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    if self.create_scale and scale is not None:
      raise ValueError(
          "Cannot pass `scale` at call time if `create_scale=True`.")
    if self.create_offset and offset is not None:
      raise ValueError(
          "Cannot pass `offset` at call time if `create_offset=True`.")

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    if is_training or test_local_stats:
      mean = jnp.mean(inputs, axis, keepdims=True)
      mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)
      for a in cross_axes:
        mean = jax.lax.pmean(
            mean,
            axis_name=a)
        mean_of_squares = jax.lax.pmean(
            mean_of_squares,
            axis_name=a)
      var = mean_of_squares - jnp.square(mean)
    else:
      mean = self.mean_ema.average
      var = self.var_ema.average

    if is_training:
      self.mean_ema(mean)
      self.var_ema(var)

    w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
    w_dtype = inputs.dtype

    if self.create_scale:
      scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
    elif scale is None:
      scale = jnp.ones([], dtype=w_dtype)

    if self.create_offset:
      offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
    elif offset is None:
      offset = jnp.zeros([], dtype=w_dtype)

    eps = jax.lax.convert_element_type(self.eps, var.dtype)
    inv = scale * jax.lax.rsqrt(var + eps)
    return (inputs - mean) * inv + offset