#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Normalization Layers."""

from typing import Any, Tuple

from flax import linen as nn
from jax import lax
import jax.numpy as jnp
from layers import initializers

Initializer = initializers.Initializer


class RMSNorm(nn.Module):
  """RMS normalization."""

  epsilon: float = 1e-6
  weight_dtype: Any = jnp.float32
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[str, ...] = ()
  scale_init: Initializer = nn.initializers.ones # pax also ones init

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    if self.scale_init is None: return y  # XD
    scale = self.param(
        "scale",
        nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
        (features,),
        self.weight_dtype,
    )

    scale = jnp.asarray(scale, self.dtype)
    return y * scale

def get_rmsnorm(name, cfg=None, **kwargs):
  if cfg is None:
    dtype = kwargs.get('dtype', jnp.bfloat16)
    weight_dtype = kwargs.get('weight_dtype', jnp.float32)
    epsilon = kwargs.get('normalization_layer_epsilon', 1.e-6)
  else:
    dtype = getattr(cfg, 'dtype', jnp.bfloat16)
    weight_dtype = getattr(cfg, 'weight_dtype', jnp.float32)
    epsilon = getattr(cfg, 'normalization_layer_epsilon', 1.e-6)
  scale_init = kwargs['scale_init'] if 'scale_init' in kwargs else nn.initializers.ones
  # max_logging.log(f'\nnorm name: {name} dtype: {dtype} weight_dtype: {weight_dtype} epsilon: {epsilon} scale_init: {scale_init}\n')
  return RMSNorm(name=name,
                dtype=dtype,
                weight_dtype=weight_dtype,
                epsilon=epsilon,
                kernel_axes=("norm",),
                scale_init=scale_init) # use scale default is true.
