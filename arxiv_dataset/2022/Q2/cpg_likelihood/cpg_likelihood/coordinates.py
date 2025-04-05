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

def scan(f, init, xs, length=None):  
    if xs is None:  
      xs = range(length) 
    carry = init  
    ys = []  
    for x in xs:  
      carry, y = f(carry, x)  
      ys.append(y)  
    return carry, np.zeros((0,) + init.shape) if len(ys) == 0 else np.stack(ys)

def break_coeffs(Fbs, ns):
    # c_i = (Fbi/Fbim1)**-nim1 * c_im1
    ln_Fbs = np.log(Fbs)
    ln_c2 = np.zeros(Fbs.shape[1:])
    def ci_f(ln_ci, idx):
        result = -ns[idx+1] * (ln_Fbs[idx+1] - ln_Fbs[idx]) + ln_ci
        return result, result
    _, ln_coeffs = scan(ci_f, ln_c2, range(len(ns)-2))
    return np.concatenate((np.ones((2,) + Fbs.shape[1:]), np.exp(ln_coeffs)))

def A_to_N(A, Fbs, ns):
    coeffs = break_coeffs(Fbs, ns)
    mid = coeffs[1:-1] * Fbs[:-1] * (np.power(Fbs[:-1]/Fbs[1:], ns[1:-1]) - 1) / (1 - ns[1:-1])
    return A * (coeffs[0] * Fbs[0] / (ns[0] - 1) + mid.sum() + coeffs[-1] * Fbs[-1] / (1 - ns[-1]))

def Ftot_over_Fb2sq(A, betas, ns):
    coeffs = break_coeffs(betas, ns) # Although this function takes a list of Fbs, it only uses ratios, so we can feed it betas instead
    mid = coeffs[1:-1] * betas[:-1] * betas[:-1] * (np.power(betas[:-1]/betas[1:], ns[1:-1]) - 1) / (2 - ns[1:-1])
    return A * (coeffs[0] * betas[0] * betas[0] / (ns[0] - 2) + mid.sum() + coeffs[-1] * betas[-1] * betas[-1] / (2 - ns[-1]))


def Ftot_betas_to_A_Fbs(N, Ftot, betas, ns):
    ln_betas = np.log(np.concatenate([ np.ones((1,) + betas.shape[1:]), betas]))
    ln_alphas = np.cumsum(ln_betas, axis=0)
    alphas = np.exp(ln_alphas)

    A_times_Fb2 = N / ( A_to_N(1.0, alphas, ns) )

    Ftot_over_Fb2sq_over_A = Ftot_over_Fb2sq(1.0, alphas, ns)

    Ftot_over_Fb2 = Ftot_over_Fb2sq_over_A * A_times_Fb2

    Fb2 = Ftot/Ftot_over_Fb2

    return A_times_Fb2/Fb2, alphas * Fb2

def A_Fbs_to_Ftot_betas(A, Fbs, ns):
    N = A_to_N(A, Fbs, ns)
    alphas = Fbs/Fbs[0]
    ln_alphas = np.log(alphas)
    ln_betas = ln_alphas[1:] - ln_alphas[:-1]
    betas = np.exp(ln_betas)

    Ftot = Ftot_over_Fb2sq(A, alphas, ns) * (Fbs[0] * Fbs[0])

    return N, Ftot, betas
