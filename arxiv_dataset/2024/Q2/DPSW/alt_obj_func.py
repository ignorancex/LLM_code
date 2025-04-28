import sys
import numpy as np
import torch
from fast_soft_sort.pytorch_ops import soft_rank
from util import *

def obj_pi0(pi_0):
    # compute 'cross entropy ross' (just take negative log since pi_0 is sigmoid output)
    m = torch.nn.LogSigmoid()
    crossentropy_loss = torch.mean(-m(pi_0))
    return crossentropy_loss    

def calc_weights(a, p, pi_0, weight_method='PS', reg_soft=0.01, estim_pareto_params='PWM', heavi_model='sigmoid', heavi_k=0.1):
    # compute IPW weights (if not weight_method == 'NoWeighting')
    if weight_method == 'truncate': # heuristic proposed by [Crump+; Biometrika2009]
        pi_0 = torch.clamp(pi_0, min=0.1, max=0.9)
    w = sample_weights(a, p, pi_0) if weight_method != 'None' else 1.0

    if weight_method in ['IPW', 'truncate', 'None']:
        return w
    if weight_method == 'normalize':
        _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
        _c_ind = torch.where(a == 0)[0]
        w_mean_t = torch.mean(w[_t_ind])
        w_mean_c = torch.mean(w[_c_ind])
        w_norm = w
        w_norm[_t_ind] /= w_mean_t
        w_norm[_c_ind] /= w_mean_c
        return w_norm
    if weight_method == 'PS':
        w = ParetoSmoothed_IPW(w, reg_soft, estim_pareto_params, heavi_model, heavi_k, pi_0)
        return w
    if weight_method == 'PS-norm':
        w = ParetoSmoothed_IPW(w, reg_soft, estim_pareto_params, heavi_model, heavi_k, pi_0)
        _t_ind = torch.where(a == 1)[0]
        _c_ind = torch.where(a == 0)[0]
        w_mean_t = torch.mean(w[_t_ind])
        w_mean_c = torch.mean(w[_c_ind])
        w_norm = w
        w_norm[_t_ind] /= w_mean_t
        w_norm[_c_ind] /= w_mean_c
        return w_norm



def objective(y, y_hat, X, a, p, w, reg_alpha, loss_func='l1', imbalance_func='mmd2_lin'):
    # compute prediction loss
    if loss_func == 'l1':
        pred_loss = torch_l1_loss(y, y_hat, w)
    elif loss_func == 'l2':
        pred_loss = torch_l2_loss(y, y_hat, w)    
    elif loss_func == 'ce':
        pred_loss = torch_ce_loss(y, y_hat, w)
    else:
        print("Error: loss_func = " + str(loss_func) + " is not implemented!", file=sys.stderr)
        sys.exit(1)

    # compute 'imbalance loss' used to balance feature representations
    if imbalance_func == 'lin_disc': 
        imbalance_loss = lin_disc(X, a, p)
    elif imbalance_func == 'mmd2_lin':
        imbalance_loss = mmd2_lin(X, a, p)
    elif imbalance_func == 'mmd2_rbf':
        sig = 0.1
        imbalance_loss = mmd2_rbf(X, a, p, sig)
    elif imbalance_func == 'wasserstein':
        imbalance_loss = wasserstein(X, a, p)
    else:
        print("Error: imbalance_func = " + str(imbalance_func) + " is not implemented", file=sys.stderr)
        sys.exit(1)


    return pred_loss + reg_alpha * imbalance_loss 

# Pareto smoothing for weight
def estimate_params_GPD(w, soft_ind, mu, rank, M, n_data, method = 'SolveLin'):
    # soft_ind ~= 1(rank >= M) 
    # use M largest weights (in the sense of soft rank) to estimate sigma and xi 
    soft_n = soft_ind.sum()

    if method == 'SolveLin':
        mean = torch.sum(soft_ind*w)/soft_n
        wmu = mean - mu
        var = torch.sum(soft_ind*w*w)/soft_n - mean*mean
        sigma = 0.5*wmu*(1 + wmu*wmu/var)
        xi = 0.5*(1 - wmu*wmu/var)
        return sigma, xi

    elif method == 'PWM':
        coef = n_data - rank
        wmu = w - mu    ##[Hosking+87] x -> w-mu (shift) 
        a0 = torch.sum(soft_ind*wmu)/soft_n    #[Hosking+87] below of eq(5); E[X]
        a1 = torch.sum(soft_ind*coef*wmu)/(soft_n - 1)/soft_n; # \sum (n-j)/(n-1) * x_j:n
        sigma = 2*a0*a1/(a0 - 2*a1)    #[Hosking+87] below of eq(5) sigma-> alpha
        xi = 2 - a0/(a0 - 2*a1)    #xi -> k
        return sigma, xi

def ParetoSmoothed_IPW(w, reg_soft, Pareto_est, heavi_method, heavi_k, pi_0):
    n_data = torch.tensor(w.shape[0])
    M = torch.min(0.2*n_data, 3*torch.sqrt(n_data))
    # input of soft_rank is 2d-tensor
    # rank the given tensor along the second axis in ASCENDING order
    w = w.reshape((1, w.shape[0]))
    rank = soft_rank(w,regularization_strength=reg_soft)
    rank = rank.reshape((rank.shape[1], 1))
    w = w.reshape((w.shape[1], 1))
    soft_ind = smooth_heaviside(rank, n_data - M, method=heavi_method, k=heavi_k)

    #if len(torch.where(rank[:,0] < n_data - M)[0]) == 0:
    if torch.isnan(rank[:, 0]) is True:
        print("No need to adjust weights")
        sys.exit()
        return w
    else:
        mu = torch.max(w[torch.where(rank[:,0] < n_data - M)[0],0])
        sigma, xi = estimate_params_GPD(w, soft_ind, mu, rank, M, n_data, method=Pareto_est)
        eps = torch.tensor((0.5))
        z = (cutoff(rank - n_data + M, eps, M + eps) - eps) / M
        w_PS_ = invCDF_GPD(z, mu, sigma, xi)
        w_PS = (1 - soft_ind) * w + soft_ind * w_PS_

        if len(torch.where(w_PS < 0.0)[0]) > 0:
            sys.exit()
        return w_PS

##### Loss functions
### (weighted) L1 loss for continuous-valued outcome
def torch_l1_loss(y, y_hat, w):
    res = w * torch.abs(y_hat - y)
    return torch.mean(res)
### (weighted) L1 loss for continuous-valued outcome
def torch_l2_loss(y, y_hat, w):
    res = w * torch.square(y_hat - y)
    return torch.mean(res)
### (weighted) cross entropy loss for binary outcome
def torch_ce_loss(y, y_hat, w):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    y = y.reshape((y.shape[0])).long()
    ce_loss = criterion(y_hat, y)
    res = w * ce_loss
    if torch.mean(res) < 0:
        print("negative CE loss")
        sys.exit()
    return torch.mean(res)
### sample weighting [Hassanpour+; Eq. (5), Sec.3, IJCAI2019]
CONST_EPS = 1e-20
def sample_weights(a, p, pi):
    pi_clipped = torch.clamp(pi, min=CONST_EPS, max=1.0-CONST_EPS)
    pi_ratio = 1.0 / (pi_clipped) - 1.0
    if len(torch.where(pi_clipped <= 0.0)[0]):
        print("ERROR: propensity score pi is not within [0.0, 1.0]")
        sys.exit()
    w1 = a * (1 + (p / (1 - p)) * pi_ratio)
    w2 = (1 - a) * (1 + ((1 - p) / p) * pi_ratio)
    return w1 + w2

##### Weight decay regularization [Johansson+; Sec.6, ICML2016], [Shalit+; Sec.5, ICML2017]

##### Balancing penalty functions
### Linear Discrepancy [Johansson+; Eq. (8), Sec. 4.1, ICML2016]
def lin_disc(X, a, p):
    mmd = mmd2_lin(X, a, p, const=1.0)
    sign_p = np.sign(p - 0.5)
    disc = sign_p * (p - 0.5) + torch_safe_sqrt(np.square(2 * p - 1) * 0.25 + mmd)
    return disc

### Linear MMD [Johansson+; ||v|| const=2.0 leads to ||v|| in Eq. (8), Sec. 4.1, ICML2016]

def mmd2_lin(X, a, p, const=2.0):
    _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
    _c_ind = torch.where(a == 0)[0]
    X1 = X[_t_ind]
    X0 = X[_c_ind]
    X1_mean = torch.mean(X1, dim=0)
    X0_mean = torch.mean(X0, dim=0)
    mmd = torch.sum(torch.square(const * p * X1_mean - const * (1.0 - p) * X0_mean))
    return mmd

### MMD with Gaussian kernel [Shalit+; Appendix B.2, ICML2017]
def mmd2_rbf(X, a, p, sig):
    _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
    _c_ind = torch.where(a == 0)[0]
    X1 = X[_t_ind]
    X0 = X[_c_ind]
    num_samples_0 = float(X0.shape[0])
    num_samples_1 = float(X1.shape[0])
    # compute gram matrices
    Gx_00 = torch.exp(-torch_pdist2(X0, X0) / (sig ** 2))
    Gx_01 = torch.exp(-torch_pdist2(X0, X1) / (sig ** 2))
    Gx_11 = torch.exp(-torch_pdist2(X1, X1) / (sig ** 2))
    mmd = np.square(1.0 - p) / (num_samples_0 * (num_samples_0 - 1.0)) * (torch.sum(Gx_00) - num_samples_0)
    + np.square(p) / (num_samples_1 * (num_samples_1 - 1.0)) * (torch.sum(Gx_11) - num_samples_1)
    - 2.0 * p * (1.0 - p) / (num_samples_0 * num_samples_1) * torch.sum(Gx_01)
    mmd = 4.0 * mmd
    return torch.square(mmd)   

### Approximated Wasserstein distance with Sinkhorn [Shalit+; Appendix B.1, ICML2017]
def wasserstein(X, a, p, lamb=10, its=10, sq=False, backpropT=False):
    _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
    _c_ind = torch.where(a == 0)[0]
    X1 = X[_t_ind]
    X0 = X[_c_ind]
    num_samples_0 = float(X0.shape[0])
    num_samples_1 = float(X1.shape[0])
    # compute distance matrix
    Mx_10 = torch_pdist2(X1, X0)
    if sq is False:
        Mx_10 = torch_safe_sqrt(Mx_10)
    # estimate lambda & delta
    Mx_10_mean = torch.mean(Mx_10)
    delta = torch.max(Mx_10).detach() # detach() = no gradient computed
    eff_lamb = (lamb / Mx_10_mean).detach()
    # compute new distance matrix
    Mx_10_new = Mx_10
    row = delta * torch.ones(Mx_10[0:1,:].shape)
    col = torch.cat([delta * torch.ones(Mx_10[:,0:1].shape), torch.zeros((1,1))], 0)
    Mx_10_new = torch.cat([Mx_10, row], 0)
    Mx_10_new = torch.cat([Mx_10_new, col], 1)
    # compute marginal vectors
    marginal1 = torch.cat([p * torch.ones(torch.where(a == 1)[0].reshape((-1, 1)).shape) / num_samples_1, (1-p) * torch.ones((1,1))], 0)
    marginal0 = torch.cat([(1 - p) * torch.ones(torch.where(a == 0)[0].reshape((-1, 1)).shape) / num_samples_0, p * torch.ones((1,1))], 0)
    # compute kernel matrix
    Mx_10_lamb = eff_lamb * Mx_10_new
    Kx_10 = torch.exp(- Mx_10_lamb) + 1.0e-06 ## constant added to avoid nan
    U = Kx_10 * Mx_10_new
    marginal1invK = Kx_10 / marginal1
    # fixed-point iterations of Sinkhorn algorithm [Cuturi+; NeurIPS2013]
    u = marginal1
    for i in range(0, its):
        u = 1.0 / (torch.matmul(marginal1invK, (marginal0 / torch.t(torch.matmul(torch.t(u), Kx_10)))))
    v = marginal0 / (torch.t(torch.matmul(torch.t(u), Kx_10)))

    T = u * (torch.t(v) * Kx_10)

    if backpropT is False:
        T = T.detach()

    E = T * Mx_10_new
    D = 2 * torch.sum(E)
    return D#, Mx_10_lamb
