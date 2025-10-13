"""
This source code is licensed under an MIT license, found below.

MIT License

Copyright (c) 2025 Filip Ekström Kelvinius

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
import torch.nn.functional as F


class UnbatchedSMCHelper:
    def __init__(self, resampling_method: str, device: str):
        resampling_fns = {
            "multinomial": self.multinomial_resampling,
            "systematic": self.systematic_resampling,
            "stratified": self.stratified_resampling,
        }

        self._resample = resampling_fns[resampling_method]
        self.device = device

    @staticmethod
    def multinomial_resampling(nu):
        a = nu.multinomial(len(nu), replacement=True)
        return a

    def _sampling_help(self, offset, nu):
        """
        Helper function for systematic and stratified resampling
        """
        num_particles = len(nu)
        base = (1 / num_particles) * torch.arange(num_particles, device=self.device)
        p = base + offset
        p = p.unsqueeze(1)
        nu_cumsum = nu.cumsum(dim=0).unsqueeze(0)
        indices = num_particles - torch.sum(p < nu_cumsum, dim=-1)
        return indices

    def systematic_resampling(self, nu):
        num_particles = len(nu)
        offset = 1 / num_particles * torch.rand(1, device=self.device)
        return self._sampling_help(offset, nu)

    def stratified_resampling(self, nu):
        num_particles = len(nu)
        offset = (1 / num_particles) * torch.rand(num_particles, device=self.device)
        return self._sampling_help(offset, nu)

    def resample(self, nu, log=False):
        assert len(nu.shape) == 1
        if log:
            nu = self.normalize(nu, log=True)
        assert torch.abs(torch.sum(nu) - 1.0) < 1e-5
        return self._resample(nu)

    def compute_ess(self, w, log=False):
        assert len(w.shape) == 1
        if log:
            w = self.normalize(w, log=True)
        assert torch.abs(torch.sum(w) - 1.0) < 1e-5
        n_eff = 1 / torch.sum(w**2, dim=-1)
        return n_eff

    def normalize(self, vec, log=False):
        assert len(vec.shape) == 1
        if log:
            vec = torch.exp(vec - torch.max(vec))
        return F.normalize(vec, p=1, dim=-1)

    def normalize_log(self, vec):
        assert len(vec.shape) == 1
        return vec - vec.logsumexp(-1)
    
    def importance_sampling(self, x, w, log=False):
        assert len(w.shape) == 1
        if log:
            distr = torch.distributions.Categorical(logits=w)
        else:
            distr = torch.distributions.Categorical(probs=w)
        return x[distr.sample((1,))]


class BatchedSMCHelper:
    def __init__(self, resampling_method, device, num_particles):
        self.device = device
        self.num_particles = num_particles
        resampling_fns = {"multinomial": self.multinomial_resampling, "systematic": self.systematic_resampling,
                          "stratified": self.stratified_resampling}

        self._resample = resampling_fns[resampling_method]
        self.device = device

    def resample(self, nu, log=False):
        assert len(nu.shape) == 1
        if log:
            nu = self.normalize(nu, log=True)
        nu = nu.reshape(-1, self.num_particles)
        assert not torch.any(torch.abs(torch.sum(nu, dim=-1) - 1.0) > 1e-5)
        return self._resample(nu)

    def multinomial_resampling(self, nu):
        a = nu.multinomial(self.num_particles, replacement=True).flatten()
        particle_offset = (self.num_particles * torch.arange(nu.shape[0],
                                                             device=nu.device)).repeat_interleave(self.num_particles)
        a = a + particle_offset
        return a.flatten()

    def _sampling_help(self, p, nu):
        nu_cumsum = nu.cumsum(dim=-1)
        p = p.flatten().unsqueeze(-1)
        nu_cumsum = nu_cumsum.repeat_interleave(self.num_particles, dim=0)
        indices = self.num_particles - torch.sum(p < nu_cumsum, dim=-1)
        indices = indices + (self.num_particles * torch.arange(nu.shape[0], 
                                                             device=self.device)).repeat_interleave(self.num_particles)
        return indices.flatten()

    def systematic_resampling(self, nu):
        base = (1 / self.num_particles) * torch.arange(self.num_particles, device=self.device).reshape((1, self.num_particles))
        offset = (1 / self.num_particles) * torch.rand((nu.shape[0], 1), device=self.device)
        p = base + offset
        return self._sampling_help(p, nu)

    def stratified_resampling(self, nu):
        base = (1 / self.num_particles) * torch.arange(self.num_particles, device=self.device).reshape((1, self.num_particles))
        offset = (1 / self.num_particles) * torch.rand((nu.shape[0], self.num_particles), device=self.device)
        p = base + offset
        return self._sampling_help(p, nu)

    def compute_ess(self, w, log=False):
        if log:
            w = self.normalize(w, log=True)
        w = w.reshape(-1, self.num_particles)
        assert not torch.any(torch.abs(torch.sum(w, dim=-1) - 1.0) > 1e-5)
        n_eff = 1 / torch.sum(w ** 2, dim=-1)
        return n_eff.flatten()

    def normalize(self, vec, log=False):
        assert len(vec.shape) == 1
        vec = vec.reshape((-1, self.num_particles))
        if log:
            vec = torch.exp(vec - torch.max(vec, dim=-1, keepdim=True)[0])
        return F.normalize(vec, p=1, dim=-1).flatten()

    def normalize_log(self, vec):
        assert len(vec.shape) == 1
        vec = vec.reshape((-1, self.num_particles))
        return (vec - vec.logsumexp(-1, keepdim=True)).flatten()
    
    def importance_sampling(self, x, w, log=False):
        assert len(w.shape) == 1
        w = w.reshape(-1, self.num_particles)
        if log:
            distr = torch.distributions.Categorical(logits=w)
        else:
            distr = torch.distributions.Categorical(probs=w)
        s = distr.sample((1,)).flatten()
        s = s + self.num_particles * torch.arange(s.shape[0], device=x.device)
        return x[s]