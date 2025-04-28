import numpy as np
import scipy
from ResSR.utils import image_utils, decimation_utils

def normalize_MSI(y):
    """
    Normalize the MSI.

    Args:
        y (list of np.ndarray): A list of numpy arrays representing the unnormalized MSI.

    Returns:
        list of np.ndarray:  A list of numpy arrays representing the normalized MSI.
        list of tuples: A list of tuples containing the 2nd and 98th percentile pixel value for each band of the MSI.

    """
    y_normalized = len(y) * [None]
    y_new_range = np.zeros((0, 2))
    perc = 98
    skip = 4
    for j in range(len(y)):
        y_top = np.percentile(y[j][::skip, ::skip], perc, axis=[0, 1])
        y_bottom = np.percentile(y[j][::skip, ::skip], 100 - perc, axis=[0, 1])
        y_new_range = np.concatenate((y_new_range, np.array([y_bottom, y_top]).T))
        y_normalized[j] = (y[j] - y_bottom.reshape((1, 1, -1))) / (y_top.reshape((1, 1, -1)) - y_bottom.reshape((1, 1, -1)))

    return y_normalized, y_new_range

def unnormalize_MSI(x, y_range):
    """
    Unnormalize the MSI.
    
    Args:
        x (np.ndarray): N x M x B array representing the normalized super-resolved MSI.
        y_range (list of tuples): A list of tuples containing the 2nd and 98th percentile pixel value for each band of the measured MSI.
    
    Returns:
        list of np.ndarray: A list of numpy arrays representing the unnormalized super-resolved MSI.
        
    """
    img_range = y_range[:, 1] - y_range[:, 0]
    x = x * img_range.reshape((1, 1, -1)) + y_range[:, 0].reshape((1, 1, -1))
    return x

def replicate_and_subsample(y, N_s):
    """
    Replicates and subsamples the input data.

    Args:
        y (list): A list of input data arrays.
        N_s (int): The number of samples to be generated.

    Returns:
        ndarray: The replicated and subsampled data array.

    Raises:
        ValueError: If the input list `y` is empty, if `N_s` is 0, or if 'N_s' is greater than the number of pixels
    """

    if len(y) == 0:
        raise ValueError("y cannot be an empty list.")

    if N_s == 0:
        raise ValueError("N_s cannot be 0.")
        
    N_p = y[0].shape[0] * y[0].shape[1]

    if N_s > N_p:
        raise ValueError("N_s cannot be greater than the number of pixels.")

    N_bands = np.sum([y[j].shape[2] for j in range(len(y))])

    X_tilde = np.zeros((N_p, N_bands))

    idx = 0
    for j in range(len(y)):
        L_j = np.sqrt(N_p / (y[j].shape[0] * y[j].shape[1]))
        for i in range(y[j].shape[2]):
            y_upsampled =  decimation_utils.bicubic_upsample(y[j][:, :, i], int(L_j))
            X_tilde[:, idx] = y_upsampled.flatten()
            idx += 1

    D = X_tilde[np.random.randint(N_p, size=N_s), :]
    
    return D

def svd_flip(u, v):
    """ (from scikit-learn.decomposition)
    
    Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        u can be None if `u_based_decision` is False.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`. The input v should
        really be called vt to be consistent with scipy's output.
        v can be None if `u_based_decision` is True.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """

    # rows of v, columns of u
    max_abs_v_rows = np.argmax(np.abs(v), axis=1)
    shift = np.arange(v.shape[0])
    indices = max_abs_v_rows + shift * v.shape[1]
    signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
    if u is not None:
        u *= signs[np.newaxis, :]
    v *= signs[:, np.newaxis]

    return u, v
    
def SVD(D, K):
    """
    Perform trunctated Singular Value Decomposition (SVD) on the input matrix D.

    Parameters:
        D (np.ndarray): Input matrix of shape (N, M), where N is the number of samples and M is the number of features.
        K (int): Number of singular values and corresponding eigenvectors to keep.

    Returns:
        np.ndarray: Mean of the input matrix D.
        np.ndarray: Matrix of shape (M, K) containing the top K eigenvectors of D.
        np.ndarray: Diagonal matrix of shape (K, K) containing the reciprocals of the squared singular values of D.

    """

    mu_hat = np.mean(D, axis=0)

    D_centered = D - mu_hat
    U, S, Vt = np.linalg.svd(D_centered, full_matrices=False)
    U, Vt = svd_flip(U, Vt)

    V_hat = Vt[:K].T
    singular_values = S[:K]

    Lambda_hat = np.zeros((K, K))
    for i in range(K):
        Lambda_hat[i, i] = 1 / (singular_values[i] ** 2)

    return mu_hat, V_hat, Lambda_hat

def approx_closed_form_solution(y, mu_hat, V_hat, Lambda_hat, gamma_hr, lam, sigma):
    """
    Computes the approximate closed-form solution for a given set of parameters, as described in the paper.

    Args:
        y (list of numpy.ndarrays): The input MSI consisting of a list of arrays which contain all bands at distinct spatial resolutions, 
            in decreasing order of resolution. For example, a Sentinel-2 MSI would be of the form y = [y_10m, y_20m, y_60m].
        mu_hat (numpy.ndarray): The mean of the input data.
        V_hat (numpy.ndarray): The top K eigenvectors of the input data.
        Lambda_hat (numpy.ndarray): The diagonal matrix containing the reciprocals of the squared singular values of the input data.
        gamma_hr (float): The impact of the high-resolution bands on the reconstruction (between 0 and 1).
        lam (float): The regularization parameter.
        sigma (float): The noise standard deviation.

    Returns:
        numpy.ndarray: The representation coefficients of the super-resolved image.
    """
    
    N_p = y[0].shape[0] * y[0].shape[1]
    K = V_hat.shape[1]
    
    L = [1]
    for i in range(1, len(y)):
        L.append(int(np.sqrt(N_p / (y[i].shape[0] * y[i].shape[1]))))

    gamma = [gamma_hr]
    gamma_constant = (1-gamma_hr)/(np.sum([1/ell for ell in L])-1)
    for i in range(1, len(L)):
        gamma.append(gamma_constant * 1/L[i])

    A = (lam * (sigma ** 2) / K) * Lambda_hat
    b = np.zeros((N_p, K))

    idx = 0
    for j in range(len(y)):
        for i in range(y[j].shape[2]):
            y_centered = y[j][:, :, i] - mu_hat[idx]
            upsampled_rast_y = image_utils.rasterize_img(decimation_utils.A_transpose(y_centered, L[j])).squeeze()
            b += gamma[j] * L[j]**2 * np.matmul(upsampled_rast_y[:, None], V_hat[idx, :][None, :])
            A += gamma[j] * np.outer(V_hat[idx, :], V_hat[idx, :].T)

            idx += 1

    Z_hat_transpose = np.linalg.solve(A.T, b.T)
    Z_hat = Z_hat_transpose.T

    return Z_hat

def exact_iterative_solution(Z_init, y, mu_hat, V_hat, Lambda_hat, gamma_hr, lam, sigma, rho, conv_thresh = 5e-8):
    """
    Computes the exact iterative solution for a given set of parameters.

    Args:
        Z_init (np.ndarray): The initial guess for the representation coefficients of the super-resolved image.
        y (list of numpy.ndarrays): The input MSI consisting of a list of arrays which contain all bands at distinct spatial resolutions, 
            in decreasing order of resolution. For example, a Sentinel-2 MSI would be of the form y = [y_10m, y_20m, y_60m].
        mu_hat (numpy.ndarray): The mean of the input data.
        V_hat (numpy.ndarray): The top K eigenvectors of the input data.
        Lambda_hat (numpy.ndarray): The diagonal matrix containing the reciprocals of the squared singular values of the input data.
        gamma_hr (float): The impact of the high-resolution bands on the reconstruction (between 0 and 1).
        lam (float): The regularization parameter.
        sigma (float): The noise standard deviation.
        rho (float): The ADMM penalty parameter.
        conv_thresh (float): The convergence threshold.

    Returns:
        numpy.ndarray: The super-resolved image.
    """
    
    N_p = y[0].shape[0] * y[0].shape[1]
    N_b = np.sum([y[j].shape[2] for j in range(len(y))])
    K = V_hat.shape[1]
    
    L = [1]
    for i in range(1, len(y)):
        L.append(int(np.sqrt(N_p / (y[i].shape[0] * y[i].shape[1]))))

    gamma = [gamma_hr]
    gamma_constant = (1-gamma_hr)/(np.sum([1/ell for ell in L])-1)
    for i in range(1, len(L)):
        gamma.append(gamma_constant * 1/L[i])

    Z = Z_init
    W = np.zeros((N_p, N_b))
    X = np.matmul(Z, V_hat.T) + mu_hat.reshape((1, -1))

    admm_error = []
    admm_error_iter = 100
    i = 0
    while admm_error_iter > conv_thresh:
        print(" Iteration " + str(i + 1))

        X = data_fitting_proximal_map(y, Z, V_hat, mu_hat, W, gamma, sigma, rho)

        Z = spectral_regularizer_proximal_map(W, X, np.tile(mu_hat, (N_p, 1)), V_hat, Lambda_hat, lam, rho)

        W = W + X - (np.matmul(Z, V_hat.T) + np.tile(mu_hat, (N_p, 1)))

        admm_error_iter = (
            1 / (N_p * N_b) * np.linalg.norm(X - (np.matmul(Z, V_hat.T) + np.tile(mu_hat, (y[0].shape[0] * y[0].shape[1], 1))), ord="fro")
        )
        print("     Current ADMM error: ", admm_error_iter)
        admm_error.append(admm_error_iter)
        i += 1

    return Z, admm_error

def data_fitting_proximal_map(y, Z, V_hat, mu_hat, W, gamma, sigma, rho):
    """
    Compute the data fitting proximal map for a given set of inputs.

    Args:
        y (list): The input MSI consisting of a list of arrays which contain all bands at distinct spatial resolutions, 
            in decreasing order of resolution. For example, a Sentinel-2 MSI would be of the form y = [y_10m, y_20m, y_60m].
        Z (numpy.ndarray): The representation coefficients of the super-resolved image.
        V_hat (numpy.ndarray): The top K eigenvectors of the input data.
        mu_hat (numpy.ndarray): The mean of the input data.
        W (numpy.ndarray): 
        gamma (list): 
        sigma (float): The noise standard deviation.
        rho (float): The ADMM penalty parameter.

    Returns:
        numpy.ndarray: The data fitting proximal map.
    """

    (rows_10m, cols_10m, num_10m_bands) = y[0].shape
    (rows_20m, cols_20m, num_20m_bands) = y[1].shape
    (rows_60m, cols_60m, num_60m_bands) = y[2].shape
    num_bands = num_10m_bands + num_20m_bands + num_60m_bands

    gamma_10m = gamma[0]
    gamma_20m = gamma[1]
    gamma_60m = gamma[2]

    X = np.zeros((rows_10m * cols_10m, num_bands))
    X_init = np.matmul(Z, V_hat.T) + np.tile(mu_hat, (rows_10m * cols_10m, 1))
    constraint_error = X_init - W

    for i in range(num_10m_bands):
        A = ( 1 + (rho * sigma**2) / (num_bands * gamma_10m) ) * scipy.sparse.eye(rows_10m * cols_10m)
        A = A.tocsr()
        y_band = image_utils.rasterize_img(y[0][:, :, i][:, :, None]).squeeze()
        b = y_band + (rho * sigma**2) / (num_bands * gamma_10m) * constraint_error[:, i]
        X[:, i] = scipy.sparse.linalg.spsolve(A, b)

    for i in np.arange(num_10m_bands, num_10m_bands + num_20m_bands):
        ones_matrix = scipy.sparse.csr_array(np.ones((2, 2)))
        Psi_T_Psi = (
            1
            / 2**4
            * scipy.sparse.kron(
                scipy.sparse.kron(
                    scipy.sparse.kron(scipy.sparse.eye(rows_20m), ones_matrix), scipy.sparse.eye(cols_20m)
                ),
                ones_matrix,
            )
        )
        A = Psi_T_Psi + (rho * sigma**2) / (num_bands * gamma_20m * 2**2) * scipy.sparse.eye(rows_10m * cols_10m)
        A = A.tocsr()
        y_band_block_rep = image_utils.rasterize_img(
            decimation_utils.A_transpose(y[1][:, :, i - num_10m_bands], 2)
        ).squeeze()
        b =  y_band_block_rep + (rho * sigma**2) / (num_bands * gamma_20m * 2**2) * constraint_error[:, i]
        X[:, i] = scipy.sparse.linalg.spsolve(A, b)

    for i in np.arange(num_10m_bands + num_20m_bands, num_bands):
        ones_matrix = scipy.sparse.csr_array(np.ones((6, 6)))
        Psi_T_Psi = (
            1
            / 6**4
            * scipy.sparse.kron(
                scipy.sparse.kron(
                    scipy.sparse.kron(scipy.sparse.eye(rows_60m), ones_matrix), scipy.sparse.eye(cols_60m)
                ),
                ones_matrix,
            )
        )
        A = Psi_T_Psi + (rho * sigma**2) / (num_bands * gamma_60m * 6**2) * scipy.sparse.eye(rows_10m * cols_10m)
        A = A.tocsr()
        y_band_block_rep = image_utils.rasterize_img( 
           decimation_utils.A_transpose(y[2][:, :, i - num_10m_bands - num_20m_bands], 6)
        ).squeeze()
        b = y_band_block_rep + (rho * sigma**2) / (num_bands * gamma_60m * 6**2) * constraint_error[:, i]
        X[:, i] = scipy.sparse.linalg.spsolve(A, b)

    return X

def spectral_regularizer_proximal_map(W, X, mu_tiled, V_hat, Lambda_hat, lam, rho):
    """
    Calculates the proximal map of the spectral regularizer.
    Args:
        W (numpy.ndarray): 
        X (numpy.ndarray): The current super-resolved image.
        mu_tiled (numpy.ndarray): The mean matrix tiled to match the shape of X.
        V_hat (numpy.ndarray): The top K eigenvectors of the input data.
        Lambda_hat (numpy.ndarray): The diagonal matrix containing the reciprocals of the squared singular values of the input data.
        lam (float): The regularization parameter.
        rho (float): The ADMM penalty parameter.
    Returns:
        ndarray: The proximal map of the spectral regularizer.
    """

    n_components = V_hat.shape[1]
    
    n_bands = X.shape[1]

    constant = (rho * n_components) / (n_bands * lam)

    A = Lambda_hat ** 2 + constant * np.eye(n_components)

    b = constant * np.matmul((X + W - mu_tiled), V_hat)

    Z_T = np.linalg.solve(A.T, b.T)

    Z = Z_T.T

    return Z

def residual_correction(X_hat_uncorrected, y):
    """
    Corrects the residual of the reconstructed image by adding the upsampled low-resolution image.

    Args:
        X_hat_uncorrected (np.ndarray): The uncorrected super-resolved MSI.
        y (list of numpy.ndarrays): The input MSI consisting of a list of arrays which contain all bands at distinct spatial resolutions, 
            in decreasing order of resolution. For example, a Sentinel-2 MSI would be of the form y = [y_10m, y_20m, y_60m].

    Returns:
        np.ndarray: The corrected super-resolved MSI.
    """

    N_p = y[0].shape[0] * y[0].shape[1]

    L = [1]
    for i in range(1, len(y)):
        L.append(int(np.sqrt(N_p / (y[i].shape[0] * y[i].shape[1]))))

    idx = 0
    corrected_img = np.zeros_like(X_hat_uncorrected)
    for j in range(len(y)):
        for i in range(y[j].shape[2]):
            decimated_recon = decimation_utils.A(X_hat_uncorrected[..., idx], L[j])
            difference =  decimation_utils.bicubic_upsample(y[j][..., i] - decimated_recon, L[j])
            corrected_img[..., idx] = X_hat_uncorrected[..., idx] + difference
            idx += 1

    return corrected_img
