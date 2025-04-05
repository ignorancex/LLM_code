# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
r"""
The GDP accountant is based on TensorFlow privacy repo: https://github.com/tensorflow/privacy

Implements privacy accounting for Gaussian Differential Privacy.

Applies the Dual and Central Limit Theorem (CLT) to estimate privacy budget of
an iterated subsampled Gaussian Mechanism (by either uniform or Poisson
subsampling).
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(epoch, noise_multi, n, batch_size):
  """Compute mu from uniform subsampling."""

  t = epoch * n / batch_size
  c = batch_size * np.sqrt(t) / n
  return np.sqrt(2) * c * np.sqrt(
      np.exp(noise_multi**(-2)) * norm.cdf(1.5 / noise_multi) +
      3 * norm.cdf(-0.5 / noise_multi) - 2)


def compute_mu_poisson(epoch, noise_multi, n, batch_size):
  """Compute mu from Poisson subsampling."""

  t = epoch * n / batch_size
  return np.sqrt(np.exp(noise_multi**(-2)) - 1) * np.sqrt(t) * batch_size / n


def delta_eps_mu(eps, mu):
  """Compute dual between mu-GDP and (epsilon, delta)-DP."""
  return norm.cdf(-eps / mu +
                  mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
  """Compute epsilon from mu given delta via inverse dual."""

  def f(x):
    """Reversely solve dual by matching delta."""
    return delta_eps_mu(x, mu) - delta

  return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def compute_eps_uniform(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of uniform subsampling."""

  return eps_from_mu(
      compute_mu_uniform(epoch, noise_multi, n, batch_size), delta)


def compute_eps_poisson(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of Poisson subsampling."""

  return eps_from_mu(
      compute_mu_poisson(epoch, noise_multi, n, batch_size), delta)

def compute_eps_uniform_iter(iteration, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of uniform subsampling."""

  return eps_from_mu(
      compute_mu_uniform_iter(iteration, noise_multi, n, batch_size), delta)

def compute_mu_uniform_iter(iteration, noise_multi, n, batch_size):
  """Compute mu from uniform subsampling."""
  t = iteration
  c = batch_size * np.sqrt(t) / n
  return np.sqrt(2) * c * np.sqrt(
      np.exp(noise_multi**(-2)) * norm.cdf(1.5 / noise_multi) +
      3 * norm.cdf(-0.5 / noise_multi) - 2)


def find_noise_multi_poi(feature_number,mixup_number,eps_re,n,delta=1e-6,max_iter=50000):
    if eps_re<0:
        return 0,eps_re
    noise_multi=10
    # epoch*n=batchsize*iteration
    epoch=feature_number*mixup_number/n
    print(epoch)
    eps=compute_eps_poisson(epoch=epoch, noise_multi=noise_multi, n=n, batch_size=mixup_number, delta=delta)
    count=0
    direct="s"
    step=0.05
    while abs(eps_re-eps)>eps_re*(1e-2):
        if eps_re>eps:
            noise_multi*=1-step
            if direct!="s":
                step*=0.5
            direct="s"
        else:
            noise_multi*=1+step
            if direct!="l":
                step*=0.5
            direct="l"
       
        eps=compute_eps_poisson(epoch=epoch, noise_multi=noise_multi, n=n, batch_size=mixup_number, delta=delta)
        count+=1
        if count>max_iter:
            break
    assert noise_multi>0.0
    print(eps_re,eps,noise_multi)
    return noise_multi,eps

