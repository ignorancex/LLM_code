"""
Copyright (c) 2021 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from . import llh
from . import coordinates

class Ln_p_cpp(object):
    def __init__(self, prior_transform, lambda_eps_map, data, eps, d_mu_eps_map):
        self.prior_transform = prior_transform
        self.lambda_eps_map = lambda_eps_map
        self.llh = llh.LLH(data, eps, d_mu_eps_map)

    def __call__(self, theta):
        if np.any(theta < 0) or np.any(theta > 1):
            return -np.inf
        A, Fbs, ns, lambda_F = self.prior_transform(theta)
        N = coordinates.A_to_N(A, Fbs, ns)
        val, bad = self.llh.eval(N, A, Fbs, ns, lambda_F*self.lambda_eps_map)
        bad = bad or np.isinf(val) or val > 0
        assert(not bad), "Likelihood eval failed for {}".format(dict(A=A, Fbs=Fbs, ns=ns))
        return val

class LnPJoint(object):
    def __init__(self, joint_dist):
        self.joint_dist = joint_dist

    def __call__(self, theta):
        if np.any(theta < 0) or np.any(theta > 1):
            return -np.inf
        val, bad =  self.joint_dist.ln_p(theta, return_bad=True)
        bad = bad or np.isinf(val) or val > 0
        assert(not bad), "Likelihood eval failed for {}, val = {}, bad = {}".format(self.joint_dist.get_params(theta), val, bad)
        return val

def ptlnprior(theta):
        if np.any(theta < 0) or np.any(theta > 1):
            return -np.inf
        else:
            return -np.log(1.0)

def run_emcee(joint_dist, n_walkers, n_steps, random, pool, progress=True):
    n_params = joint_dist.total_params
    if n_params >= n_steps:
        raise Exception("You should ask for more steps than parameters in your model.")

    initial = random.uniform(size=(n_walkers, n_params)).astype(np.float64)
    ln_p = LnPJoint(joint_dist)

    import emcee
    sampler = emcee.EnsembleSampler(n_walkers, n_params, ln_p, pool=pool)
    sampler.run_mcmc(initial, n_steps, progress=progress)
    samples = sampler.get_chain(flat=True)
    if samples.shape[0] == n_params:
        return samples[:].T
    else:
        return samples[:]

def run_ptemcee(joint_dist, n_walkers, n_steps, n_temps, random, pool, progress=True):
    n_params = joint_dist.total_params
    if n_params >= n_steps:
        raise Exception("You should ask for more steps than parameters in your model.")

    initial = random.uniform(size=(n_temps, n_walkers, n_params)).astype(np.float64)
    ln_p = LnPJoint(joint_dist)

    import ptemcee
    sampler = ptemcee.Sampler(n_walkers, n_params, ln_p, ptlnprior, n_temps, pool=pool)
    sampler.run_mcmc(initial, n_steps)
    samples = sampler.flatchain[:]
    samples = samples[0]
    if samples.shape[0] == n_params:
        return samples[:].T
    else:
        return samples[:]

def run_dynesty(joint_dist, sampler_opts=None, **kw_args):
    n_params = joint_dist.total_params

    #initial = random.uniform(size=(n_temps, n_walkers, n_params)).astype(np.float64)
    ln_p = LnPJoint(joint_dist)

    import dynesty
    sampler = dynesty.DynamicNestedSampler(ln_p, lambda x: x, n_params, **(sampler_opts if sampler_opts else {}))
    sampler.run_nested(**kw_args)
    results = sampler.results
    return results.samples, results.importance_weights()
