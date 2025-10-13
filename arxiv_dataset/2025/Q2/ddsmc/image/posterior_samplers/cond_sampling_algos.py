"""
This source code is licensed under the MIT license found below

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
import numpy as np

from ddsmc.ddsmc_sampler import DDSMCImage
from ddsmc.ddsmc_image_utils.sampling_utils import DDSMCOperator, DDSMCScoreModel

class ScoreNetDDSMCSVD():
    def __init__(self, net, H_func, device):
        self.net = DDSMCScoreModel(net)
        self.H_func = H_func
        self.device = device
    
    def score(self, x, sigma):
        shape = (x.shape[0], 3, int(np.sqrt((x.shape[-1] // 3))), -1)
        x = self.H_func.V(x.to(self.device)).reshape(shape)
        return self.H_func.Vt(self.net.score(x, sigma))


def ddsmc(initial_noise,
    epsilon_net,
    obs,
    H_func,
    std_obs,
    device,
    smc_config,
    annealing_scheduler_config,
    rho_scheduler_config,
    diffusion_scheduler_config,
    eta,
    recon_fn):
    sampler = DDSMCImage(smc_config, annealing_scheduler_config, diffusion_scheduler_config, device, eta, recon_fn, rho_scheduler_config,)

    x_start = sampler.get_start(initial_noise)
    model = ScoreNetDDSMCSVD(epsilon_net, H_func, device)
    
    operator = DDSMCOperator(H_func, std_obs)
    
    x_start_prime = H_func.Vt(x_start)
    y_prime = H_func.Ut(obs)
    if len(y_prime.shape) == 2:
        y_prime = y_prime.unsqueeze(0).flatten(1, -1)

    samples_prime, state = sampler.sample(model, x_start_prime, operator, y_prime, verbose=True)
    samples = H_func.V(samples_prime).reshape(-1, *x_start.shape[1:])
    return samples