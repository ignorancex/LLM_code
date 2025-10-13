"""
Code adapted from https://github.com/bahjat-kawar/ddrm to work with the GMM experiment where data has a single channel. 
The original code and the modifications done here are licensed under an MIT license:

MIT License

Modifications Copyright (c) 2025 Filip Ekström Kelvinius
Original work Copyright (c) 2022 Bahjat Kawar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from tqdm import tqdm

def compute_alpha(beta, t):
    if t[0] < 0:
        return torch.ones((t.shape[0], 1, 1, 1))
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0)
    a = a.index_select(0, t).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    #setup vectors used in the algorithm
    singulars = H_funcs.singulars()
    x = x.unsqueeze(1).unsqueeze(-1)  # expects BxCxD1xD2 data...
    y_0 = y_0.repeat_interleave(x.shape[0], dim=0)
    y_0 = y_0.unsqueeze(1).unsqueeze(-1)
    Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
    Sigma[:singulars.shape[0]] = singulars
    U_t_y = H_funcs.Ut(y_0)
    Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

    #initialize x_T as given in the paper
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
    large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
    inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
    inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
    inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)     

    # implement p(x_T | x_0, y) as given in the paper
    # if eigenvalue is too small, we just treat it as zero (only for init) 
    init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
    init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
    init_y = init_y.view(*x.size())
    remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
    remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
    init_y = init_y + remaining_s * x
    init_y = init_y / largest_sigmas
    
    #setup iteration variables
    x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    #iterate over the timesteps
    #for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to(x.device)
        if cls_fn == None:
            et = model(xt.reshape(xt.shape[0], -1), t[0].long()).reshape(xt.shape)
        else:
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
        
        # if et.size(1) == 6:
        #    et = et[:, :3]
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        #variational inference conditioned on y
        sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
        sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
        xt_mod = xt / at.sqrt()[0, 0, 0, 0]
        V_t_x = H_funcs.Vt(xt_mod)
        SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
        V_t_x0 = H_funcs.Vt(x0_t)
        SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

        falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
        cond_before_lite = singulars * sigma_next > sigma_0
        cond_after_lite = singulars * sigma_next < sigma_0
        cond_before = torch.hstack((cond_before_lite, falses))
        cond_after = torch.hstack((cond_after_lite, falses))

        std_nextC = sigma_next * etaC
        sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

        std_nextA = sigma_next * etaA
        sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
        
        diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

        #missing pixels
        Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

        #less noisy than y (after)
        Vt_xt_mod_next[:, cond_after] = \
            V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
        
        #noisier than y (before)
        Vt_xt_mod_next[:, cond_before] = \
            (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

        #aggregate all 3 cases and give next prediction
        xt_mod_next = H_funcs.V(Vt_xt_mod_next)
        xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))


    return xs, x0_preds

def ddrm(initial_noise, model, inverse_problem, betas, device, num_timesteps):
    obs, H_func, std = inverse_problem
    seq = list(range(0, 1001, 1001 // num_timesteps))
    seq[-1] = seq[-1] - 1
    ddrm_samples, _ = efficient_generalized_steps(
        x=initial_noise,
        b=betas,
        seq=seq,
        model=model,
        y_0=obs,
        H_funcs=H_func,
        sigma_0=std,
        etaB=1.0,
        etaA=0.85,
        etaC=1.0,
        #device=device,
        #classes=None,
        #cls_fn=None,
    )
    return ddrm_samples[-1].reshape(initial_noise.shape)