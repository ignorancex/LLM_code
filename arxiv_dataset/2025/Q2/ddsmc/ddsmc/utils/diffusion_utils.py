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
def get_diffusion_coefficients(abars, t, t_1, eta, use_score=True):
    # Backward diffusion coefficients from DDIM paper https://arxiv.org/abs/2010.02502
    # eta=1: DDPM
    # eta=0: DDIM
    # use_score switches between score and epsilon in the update
    abar_t = abars[t]
    abar_t_1 = abars[t_1]

    # Equation 16 in DDIM paper
    sigma = eta * (((1-abar_t_1)/(1-abar_t)) * (1 - abar_t / abar_t_1))**0.5

    # Equation 12 in DDIM paper
    coef_xt = (abar_t_1 / abar_t)**0.5
    coef_score = (1-abar_t_1 - sigma**2)**0.5 - ((abar_t_1 / abar_t)*(1-abar_t))**0.5
    if use_score:
        coef_score = -(1-abar_t)**0.5 * coef_score
    return coef_xt, coef_score, sigma