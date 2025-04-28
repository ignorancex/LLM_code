"""Distribution helpers for PyTorch.

Implementations are largely based on the scipy.stats implementations:

https://github.com/scipy/scipy/blob/df134eab5a500c2146ed4552c8674a78d8154ee9/scipy/stats/_multivariate.py#L4331
"""
import torch


MAX_SAMPLES=10000


def multivariate_t_logpdf(x, df, mean, chol, covariance_type):
    if covariance_type == 'full':
        return multivariate_t_logpdf_full(x, df,  mean, chol)
    elif covariance_type == 'tied':
        return multivariate_t_logpdf_tied(x, df,  mean, chol)
    else:
        return multivariate_t_logpdf_diag(x, df, mean, chol, covariance_type)


def multivariate_t_logpdf_diag(x, df, mean, chol, covariance_type):
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # df = (K,)
    # mean = (K,D) or (D,)
    # chol =
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    df = torch.as_tensor(df)
    if isinstance(chol, float) or chol.ndim == 0:
        chol = torch.tensor([chol,])
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0) # (K,D)
        chol = chol.unsqueeze(0) # (K,D) or (K,1)
    if chol.ndim == 1:
        chol = chol.unsqueeze(-1)
    K = mean.shape[0]
    D = mean.shape[-1]
    # (K,D) or (K,1) => (K,)
    halflogdet = torch.log(chol).sum(-1)
    if covariance_type == 'spherical' and chol.shape[1] == 1:
        halflogdet *= D

    t = 0.5 * (df + D)
    A = torch.special.gammaln(t)
    B = torch.special.gammaln(0.5 * df)
    C = D/2. * torch.log(df * torch.pi)

    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            maha = (((x - mean[k]) / chol[k]) ** 2).sum(-1)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet[k] - t[k] * torch.log(maha)
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N, 1, D).repeat(1, K, 1)
        # float * (((N,K,D) - (K,D)) / (K, D or 1) ** 2) sumlast => (N,K)
        maha = (((x_NKD - mean) / chol) ** 2).sum(-1)
        maha = 1. + (1. / df) * maha

        t = 0.5 * (df + D)
        A = torch.special.gammaln(t)
        B = torch.special.gammaln(0.5 * df)
        C = D/2. * torch.log(df * torch.pi)
        out = A - B - C - halflogdet - t * torch.log(maha)
    return out.squeeze()


def multivariate_t_logpdf_full(x, df, mean, scale_tril):
    """Compute the logpdf of a multivariate t distribution.

    Parameters
    ----------
    x : FloatTensor
        Batch of samples
    mean : FloatTensor
    scale_tril : FloatTensor
        Lower Cholesky of the covariance.
    df : int or Tensor
    """
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
        scale_tril = scale_tril.unsqueeze(0)
    df = torch.as_tensor(df)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1) # (K,)
    t = 0.5 * (df + D)
    A = torch.special.gammaln(t)
    B = torch.special.gammaln(0.5 * df)
    C = D/2. * torch.log(df * torch.pi)
    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            dev = x - mean[k]
            dev = dev.T # (D, N)
            maha = torch.linalg.solve_triangular(scale_tril[k], dev, upper=False)
            maha = torch.square(maha).sum(0) # (N,)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet[k] - t[k] * maha.log()
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N, 1, D).repeat(1, K, 1)
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D)
        dev_KDN = dev.permute(1,2,0)
        maha = torch.linalg.solve_triangular(scale_tril, dev_KDN,  upper=False)
        if maha.ndim == 3:
            maha = torch.square(maha).sum(1).T # (N,K)
        else:
            maha = torch.square(maha).sum(0) # (N,)
        maha = 1. + (1. / df) * maha
        out = A - B - C - halflogdet - t * torch.log(maha)
    return out.squeeze()


def multivariate_t_logpdf_tied(x, df, mean, scale_tril):
    """Compute the logpdf of a multivariate t distribution.

    Parameters
    ----------
    x : FloatTensor
        Batch of samples
    mean : FloatTensor
    scale_tril : FloatTensor
        Lower Cholesky of the covariance.
    df : int or Tensor
    """
    # N: Batch dim; K: cluster dim; D: feature dim
    # x = (N, D) or (D,)
    # mean = (K,D) or (D,)
    # scale_tril =
        # full : (K,D,D) or (D,D)
        # diag : (K,D) or (D)
        # spherical : (K,) or float
    if x.ndim == 1:
        x = x.unsqueeze(0)
    N = x.shape[0]
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
    df = torch.as_tensor(df)
    K = mean.shape[0]
    D = mean.shape[-1]
    halflogdet = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1)
    t = 0.5 * (df + D)
    A = torch.special.gammaln(t)
    B = torch.special.gammaln(0.5 * df)
    C = D/2. * torch.log(df * torch.pi)
    if N > MAX_SAMPLES and K > 1:
        out = []
        for k in range(K):
            # float * ((((N,D) - (D)) / (D or 1)) ** 2) sumlast => (N)
            dev = x - mean[k]
            dev = dev.T # (D, N)
            maha = torch.linalg.solve_triangular(scale_tril, dev, upper=False)
            maha = torch.square(maha).sum(0) # (N,)
            maha = 1. + (1. / df[k]) * maha
            outk = A[k] - B[k] - C[k] - halflogdet - t[k] * maha.log()
            out.append(outk)
        out = torch.stack(out).T
    else:
        x_NKD = x.view(N, 1, D).repeat(1, K, 1)
        # (N,K,D) - (K, D) = (N,K,D)
        #  (N,K,n)
        dev = x_NKD - mean
        # (N,K,D)/(K,D,D)
        dev_KDN = dev.permute(1,2,0)
        maha = torch.linalg.solve_triangular(scale_tril, dev_KDN,  upper=False)
        if maha.ndim == 3:
            maha = torch.square(maha).sum(1).T # (N,K)
        else:
            maha = torch.square(maha).sum(0) # (N,)
        maha = 1. + (1. / df) * maha
        out = A - B - C - halflogdet - t * torch.log(maha)
    return out.squeeze()
