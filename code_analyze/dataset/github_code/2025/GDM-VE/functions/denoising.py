import torch
import numpy as np


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # [1, alphas_cumprod]
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            # Equation (12)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def sr_generalized_steps(x, x_bw, x_fw, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(torch.cat([x_bw, x_fw, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            # Equation (12)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def sr_ddpm_steps(x, x_bw, x_fw, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(torch.cat([x_bw, x_fw, x], dim=1), t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def sg_generalized_steps(x, x_img, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next =  list(seq[1:]) + [torch.tensor(1.0, dtype=torch.float32)]

        x0_preds = []
        xs = [x]
        
        for i, j in zip(seq, seq_next):
            at = (torch.ones(n) * i).to(x.device)
            at_next = (torch.ones(n) * j).to(x.device)

            t = (at*1000).to(x.device)

            xt = xs[-1].to(x.device)
            et = model(torch.cat([x_img, xt], dim=1), t)

            at = at.view(-1, 1, 1, 1)
            at_next = at_next.view(-1, 1, 1, 1)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            # Equation (12)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds


def sg_ddpm_steps(x, x_img, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(torch.cat([x_img, x], dim=1), t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def sg_fastddpm_steps(x, x_img, a_index, model, a, **kwargs):
    with torch.no_grad():
        x = x.float()
        x_img = x_img.float()
        a_index = a_index.float()
        a = a.float() 
        n = x.size(0)
        a = a.to(x.device) 
        a_index = a_index.to(x.device)
        a_next = torch.cat([torch.ones(1).to(a.device), a[:-1]])
        x0_preds = []
        xs = [x]
        
        for i, j, k in zip(reversed(a_index), reversed(a), reversed(a_next)):
            i = i.to(x.device)  
            index = (torch.ones(n, device=x.device) * i)
            alpha = j.view(1, 1, 1, 1).expand(n, -1, -1, -1)
            alpha_next = k.view(1, 1, 1, 1).expand(n, -1, -1, -1)
            sigma = (1 - alpha ** 2).sqrt()
            sigma_next = (1 - alpha_next ** 2).sqrt()
            
            
            xt = xs[-1].to(x.device)
            et = model(torch.cat([x_img, xt], dim=1), index)
            
            x0_t = (xt - et * sigma) / alpha
            x0_preds.append(x0_t.to('cpu'))
            
            c1 = alpha_next / alpha
            c2 = sigma_next - c1 * sigma
            xt_next = c1 * xt + c2 * et            
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def sg_geodesic_fisherve_steps(x, x_img, t_idx, t, model, sig, **kwargs):
    with torch.no_grad():
        x = x.float().to(x.device) 
        x_img = x_img.float().to(x.device)
        n = x.size(0)
        sig = sig.float().to(x.device)
        t_idx = t_idx.float().to(x.device)
        t = t.float().to(x.device)
        t_next = torch.cat([torch.zeros(1).to(t.device), t[:-1]])
        k = np.log(80 / 0.002)
        x0_preds = []
        xs = [x]
        
        for i, j, l, m in zip(reversed(t_idx), reversed(sig), reversed(t), reversed(t_next)):
            i = i.to(x.device)  
            idx = (torch.ones(n, device=x.device) * i)
            sigma = j.view(1, 1, 1, 1).expand(n, -1, -1, -1)
            
            delta_t = l - m
            delta_t = delta_t.view(1, 1, 1, 1).expand(n, -1, -1, -1)
            
            xt = xs[-1].to(x.device)
            et = model(torch.cat([x_img, xt], dim=1), idx)
            
            x0_t = xt - et * sigma
            x0_preds.append(x0_t.to('cpu'))
            
            # c1 = k*σ(t)*Δt
            c1 = k * sigma * delta_t
            xt_next = xt - c1 * et            
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def sr_geodesic_fisherve_steps(x, x_bw, x_fw, t_idx, t, model, sig, **kwargs):
    with torch.no_grad():
        x = x.float().to(x.device) 
        x_bw = x_bw.float().to(x.device)
        x_fw = x_fw.float().to(x.device)
        n = x.size(0)
        sig = sig.float().to(x.device)
        t_idx = t_idx.float().to(x.device)
        t = t.float().to(x.device)
        t_next = torch.cat([torch.zeros(1).to(t.device), t[:-1]])
        k = np.log(80 / 0.002)
        x0_preds = []
        xs = [x]
        
        for i, j, l, m in zip(reversed(t_idx), reversed(sig), reversed(t), reversed(t_next)):
            i = i.to(x.device)  
            idx = (torch.ones(n, device=x.device) * i)
            sigma = j.view(1, 1, 1, 1).expand(n, -1, -1, -1)
            
            delta_t = l - m
            delta_t = delta_t.view(1, 1, 1, 1).expand(n, -1, -1, -1)
            
            xt = xs[-1].to(x.device)
            et = model(torch.cat([x_bw, x_fw, xt], dim=1), idx)
            
            x0_t = xt - et * sigma
            x0_preds.append(x0_t.to('cpu'))
            
            # c1 = k*σ(t)*Δt
            c1 = k * sigma * delta_t
            xt_next = xt - c1 * et            
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds



