# Taken from https://github.com/deepmind/ssl_hsic/blob/main/ssl_hsic/kernels.py
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
from typing import Any, Dict, List, Optional, Text, Tuple

import jax
import jax.numpy as np
import numpy as onp
import os, pickle


def compute_prob(d: int, x_range: np.ndarray) -> np.ndarray:
  """Compute the probablity to sample the random fourier features."""
  import mpmath
  probs = [mpmath.besselk((d - 1) / 2, x) * mpmath.power(x, (d - 1) / 2)
           for x in x_range]
  normalized_probs = [float(p / sum(probs)) for p in probs]
  return np.array(normalized_probs)


def imq_amplitude_frequency_and_probs(d: int) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the range and probablity for sampling RFF."""
  if os.path.exists(f'cached/amp-{d}.pkl'):
    with open(f'cached/amp-{d}.pkl', 'rb') as fin:
      x, p = map(np.asarray, pickle.load(fin))
  else:
    x = onp.linspace(1e-12, 100, 10000)  # int(n * 10 / c)
    p = compute_prob(d, x)
  return np.asarray(x), p


def imq_rff_features(num_features: int, rng: np.DeviceArray, x: np.ndarray,
                     c: float, amp: np.ndarray,
                     amp_probs: np.ndarray) -> np.ndarray:
  """Returns the RFF feature for IMQ kernel with pre-computed amplitude prob."""
  d = x.shape[-1]
  rng1, rng2 = jax.random.split(rng)
  amp = jax.random.choice(rng1, amp, shape=[num_features, 1], p=amp_probs)
  directions = jax.random.normal(rng2, shape=(num_features, d))
  b = jax.random.uniform(rng2, shape=(1, num_features)) * 2 * np.pi
  w = directions / np.linalg.norm(directions, axis=-1, keepdims=True) * amp
  z_x = np.sqrt(2 / num_features) * np.cos(np.matmul(x / c, w.T) + b)
  return z_x
