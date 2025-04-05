import numpy as np


from spn_c2 import *
#from argparse import ArgumentParser
from train_fde import *


def psvd_cnl(sample_mat, reg_coef=0, minibatch_size=1, NORMALIZE=True,\
base_lr=1e-4, base_scale=1e-1):
    """
        Probabilistic Singular Value Decomposition
        by minimizing Cauchy Noise Loss
        @param  minibatch_size: the number of singular values used at once
                                in estimaing the signal part
                                of one sample matrix.
        @param normalize:       if True, normalize sample matrix so that
                                its operator norm becomes less than 1.
    """
    p_dim = sample_mat.shape[0]
    dim = sample_mat.shape[1]
    if p_dim < dim:
        sample_mat = sample_mat.T
        p_dim = sample_mat.shape[0]
        dim = sample_mat.shape[1]

    U, D, V = np.linalg.svd(sample_mat)
    norm = max(D)
    NORMALIZE = (NORMALIZE and norm > 1)
    if NORMALIZE:
        D /= norm

    sample = D**2

    max_epoch = (20000/dim)  * (p_dim/dim)
    assert(minibatch_size >= 1)
    max_epoch = int(max_epoch)

    if NORMALIZE:
        edge = 1.01
    else:
        edge = norm*1.01

    result =  train_fde_spn(dim, p_dim, sample,\
    max_epoch=max_epoch, edge=edge, reg_coef=reg_coef,\
    base_lr=base_lr, base_scale=base_scale,\
    dim_cauchy_vec=minibatch_size)

    out_D = result["diag_A"]
    out_D = np.sort( abs(out_D))[::-1] ### Decreasing order
    out_sigma = result["sigma"]


    if NORMALIZE:
        out_D *= norm
        out_sigma *= norm

    z = z_value_spn(sample_mat, out_D, out_sigma)
    logging.info("z_value = {}".format(z))

    return U, out_D, V, out_sigma



def rank_estimation(sample_mat,reg_coef=1e-3, minibatch_size=1, NORMALIZE=True):
    """
    Rank esitmation by CNL with L1-regularization
    """
    p_dim = sample_mat.shape[0]
    dim = sample_mat.shape[1]
    if p_dim < dim:
        sample_mat = sample_mat.T
        p_dim = sample_mat.shape[0]
        dim = sample_mat.shape[1]

    U, D, V = np.linalg.svd(sample_mat)
    norm = max(D)
    if NORMALIZE:
        D /= norm

    sample = D**2

    max_epoch = (20000/dim)  * (p_dim/dim)
    max_epoch = int(max_epoch)

    if NORMALIZE:
        edge = 1.01
    else:
        edge = norm*1.01

    list_zero_thres = [reg_coef]
    result =  train_fde_spn(dim, p_dim, sample, max_epoch=max_epoch, edge=edge, reg_coef=reg_coef,\
    dim_cauchy_vec=minibatch_size, list_zero_thres=list_zero_thres)

    diag_A = result["diag_A"]
    sigma = result["sigma"]
    num_zero_array = result["num_zero"]

    out_D = np.sort(diag_A)[::-1]
    out_sigma = sigma

    if NORMALIZE:
        out_D *= norm
        out_sigma *= norm


    logging.info("list_zero_thres= {}".format( list_zero_thres))
    estimaed_ranks = dim - num_zero_array[-1]

    return estimaed_ranks[0], out_D, out_sigma



def z_value_spn(sample_mat, a, s):
    """
    z_test with the assumption that entries of  the noise part
    are independently distributed with N(0, 1/d).
    """
    assert s != 0
    U, singular, V = np.linalg.svd(sample_mat)
    p = sample_mat.shape[0]
    d = sample_mat.shape[1]
    a = np.sort(abs(a))[::-1]
    Diff = U @ rectangular_diag(singular - a, p, d) @ V
    z = np.sum(Diff)/ (sp.sqrt(p)*s)
    return z
