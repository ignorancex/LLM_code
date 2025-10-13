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
import torch.nn as nn
from ddsmc.ddsmc_image_utils.precond import VPPrecond

class DDSMCScoreModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = VPPrecond(model=model, conditional=False, learn_sigma=False)  # learn_sigma=False even if model has learn_sigma=True as the split is done somewhere else
        self.model.eval()
        self.model.requires_grad_(False)

    def score(self, x, sigma):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        return self.model(x, sigma)

class DDSMCOperator:
    def __init__(self, H_func, sigma_y):
        self.sigma = sigma_y
        self.singulars = H_func.singulars()

    def log_likelihood(self, x0hat_prime, y_prime, rho_t=0.):
        mean = self.singulars * x0hat_prime[:, :self.singulars.shape[0]]
        sigma2 = self.sigma**2 + rho_t**2 * self.singulars**2
        return torch.sum(-1/2 * (mean - y_prime)**2/sigma2, dim=-1)