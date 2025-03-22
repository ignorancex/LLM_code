import torch
import numpy as np


def perturb(x, mu,device):
    b,c,h,w = x.size()
    mask = torch.rand(b,c,h,w)<mu
    mask = mask.float().to(device)
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = x*255
    perturbed_x = ((1-mask)*x + mask*noise)/255.
    return perturbed_x


def random_masking(x, mask_ratio, p):
    """
    used for mask operation, from mask-autoencoder
    https://github.com/facebookresearch/mae
    """

    c = 3
    h = w = x.shape[2] // p
    x_p = x.reshape(shape=(x.shape[0], c, h, p, w, p))
    x_p = torch.einsum('nchpwq->nhwpqc', x_p)
    x_p = x_p.reshape(shape=(x.shape[0], h * w, p**2 * c))
    N, L, D = x_p.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)


    x_p *= mask.unsqueeze(-1).repeat(1,1,p**2 * c)
    h = w = int(x_p.shape[1]**.5)
    x_p = x_p.reshape(shape=(x.shape[0], h, w, p, p, c))
    x_p = torch.einsum('nhwpqc->nchpwq', x_p)
    x_masked = x_p.reshape(shape=(x.shape[0], c, h * p, h * p))

    return x_masked


def KL_div(mu,logvar,reduction = 'avg'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1) 
        return KL
    

