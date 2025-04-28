import torch
from fast_soft_sort.pytorch_ops import soft_rank, soft_sort

SQRT_CONST = 1e-10

def torch_safe_sqrt(x, sqrt_const=SQRT_CONST):
    return torch.sqrt(torch.clamp(x, min=sqrt_const))

def torch_pdist2(X, Y):
    nx = torch.sum(torch.square(X), dim=1, keepdim=True)
    ny = torch.sum(torch.square(Y), dim=1, keepdim=True)
    C = -2 * torch.matmul(X, torch.t(Y))
    D = (C + torch.t(ny)) + nx
    return D
    
def invCDF_GPD(z,mu,sigma,xi):
    sigma = torch.abs(sigma)
    return mu + sigma/xi*((1-z)**(-xi) - 1)

def cutoff(r,lb,ub):
    return torch.min(torch.max(r,lb),ub)

def smooth_heaviside(x,thres,method='tanh',k=1):
    if method == 'tanh':
        return 0.5 * (1 + torch.tanh(k * (x - thres)))
    if method == 'soft_rank':
        t = torch.full(x.shape, thres)
        r = soft_rank(torch.cat([x,t], 1))[:,0].unsqueeze(1)
        return r - 1
    if method == 'sigmoid':
        return 1 / (1 + torch.exp(- k * (x - thres)))
