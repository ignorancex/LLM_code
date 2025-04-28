import numpy as np
from ResSR.utils import image_utils, ressr_utils
from cProfile import Profile
from pstats import SortKey, Stats

def res_sr(y, sigma, N_s, K, gamma_hr, lam, residual_correction_flag = True):
    """
    Super-resolve the input MSI, y, using ResSR as proposed in the paper. Namely, the optimization problem is solved 
    using an approximate closed-form solution.

    Args:
        y (list of numpy.ndarrays): The input MSI consisting of a list of arrays which contain all bands at distinct spatial resolutions, 
            in decreasing order of resolution. For example, a Sentinel-2 MSI would be of the form y = [y_10m, y_20m, y_60m].
        sigma (float): The noise standard deviation.
        N_s (int): The number of sub-sampled pixels to use for subspace estimation.
        K (int): The number of singular values to keep during SVD.
        gamma_hr (float): The impact of the high-resolution bands on the reconstruction (between 0 and 1).
        lam (float): The regularization parameter.

    Returns:
        numpy.ndarrays (N_rows x N_cols x N_b): The super-resolved MSI 

    """
    y, range = ressr_utils.normalize_MSI(y)

    D = ressr_utils.replicate_and_subsample(y, N_s)

    mu_hat, V_hat, Lambda_hat = ressr_utils.SVD(D, K)

    Z_hat = ressr_utils.approx_closed_form_solution(y, mu_hat, V_hat, Lambda_hat, gamma_hr, lam, sigma)

    X_hat_uncorrected = np.matmul(Z_hat, V_hat.T) + mu_hat.reshape((1, -1))
    X_hat_uncorrected = image_utils.unrasterize_img(X_hat_uncorrected, y[0].shape[0])
    
    if residual_correction_flag:
        X_hat = ressr_utils.residual_correction(X_hat_uncorrected, y)
    else:
        X_hat = X_hat_uncorrected
        
    X_hat = ressr_utils.unnormalize_MSI(X_hat, range)

    return X_hat

def iterative_res_sr(y, sigma, N_s, K, gamma_hr, lam, rho, conv_thresh=5e-8, residual_correction_flag = True):
    """
    Super-resolves the input MSI, y, using ResSR with an exact iterative solution. Namely, the optimization problem is solved 
    using an exact iterative method (ADMM).

    Args:
        y (list of numpy.ndarrays): The input MSI consisting of a list of arrays which contain all bands at distinct spatial resolutions, 
            in decreasing order of resolution. For example, a Sentinel-2 MSI would be of the form y = [y_10m, y_20m, y_60m].
        sigma (float): The noise standard deviation.
        N_s (int): The number of sub-sampled pixels to use for subspace estimation.
        K (int): The number of singular values to keep during SVD.
        gamma_hr (float): The impact of the high-resolution bands on the reconstruction (between 0 and 1).
        lam (float): The regularization parameter.

    Returns:
        numpy.ndarrays (N_rows x N_cols x N_b): The super-resolved MSI 

    """
    
    y, range = ressr_utils.normalize_MSI(y)

    D = ressr_utils.replicate_and_subsample(y, N_s)

    mu_hat, V_hat, Lambda_hat = ressr_utils.SVD(D, K)

    Z_init = ressr_utils.approx_closed_form_solution(y, mu_hat, V_hat, Lambda_hat, gamma_hr, lam, sigma)
    Z_hat, admm_error = ressr_utils.exact_iterative_solution(Z_init, y, mu_hat, V_hat, Lambda_hat, gamma_hr, lam, sigma, rho, conv_thresh)

    X_hat_uncorrected = np.matmul(Z_hat, V_hat.T) + mu_hat.reshape((1, -1))
    X_hat_uncorrected = image_utils.unrasterize_img(X_hat_uncorrected, y[0].shape[0])
    
    if residual_correction_flag:
        X_hat = ressr_utils.residual_correction(X_hat_uncorrected, y)
    else:
        X_hat = X_hat_uncorrected
        
    X_hat = ressr_utils.unnormalize_MSI(X_hat, range)

    return X_hat