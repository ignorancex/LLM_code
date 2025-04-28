import numpy as np
import torch
import time
import scipy.optimize as opt
from functools import partial
import pywph as pw


def compute_autocovariance(data):
    """
    Compute the autocovariance matrix of the input data.
    
    Works for any dimension.

    Parameters
    ----------
    data : array
        Input data.

    Returns
    -------
    array
        Autocovariance matrix.

    """
    return np.real(np.fft.ifftn(np.absolute(np.fft.fftn(data - data.mean())) ** 2) / np.prod(data.shape))


def generate_grf(cov, mean=0.0):
    """
    Generate a realization of a 2D real-valued GRF with periodic boundary conditions
    for a given mean and autocovariance matrix.

    Parameters
    ----------
    cov : array
        2D autocovariance matrix.
    mean : float, optional
        Mean value of the GRF. The default is 0.0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    array
        Realization of the corresponding GRF.

    """
    if cov.ndim != 2:
        raise Exception("This function expects cov to be a 2D array.")
    M, N = cov.shape
    gamma = np.fft.fft2(cov)
    Z = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    X = np.fft.fft2(np.multiply(np.sqrt(gamma), Z / np.sqrt(M * N))).real
    return X + mean


def get_initialization(x, nsynth=1):
    nchan, M, N = x.shape[-3:]
    assert x.ndim == 3

    autocov = []
    for i in range(nchan):
        autocov.append(compute_autocovariance(x[i]))
    autocov = np.array(autocov)

    x0 = np.zeros((nsynth, nchan, M, N))
    for i in range(nsynth):
        for j in range(nchan):
            x0[i, j, :, :] = generate_grf(autocov[j])
                    
    return x0


def _objective(x, M, N, nchan, nsynth, wph_op, wph_model, dn, cross_pairs, coeffs, coeffs_norm, device):
    start_time = time.time()

    # Unpack target coeffs
    coeffs_auto, coeffs_cross = coeffs
    coeffs_auto_norm, coeffs_cross_norm = coeffs_norm

    # Reshape x
    x_curr = x.reshape((nsynth, nchan, M, N))
    x_curr = pw.to_torch(x_curr, device=device)
    x_curr.requires_grad = True

    loss_details = []

    # Compute the loss (squared 2-norm)
    loss_tot = torch.zeros(1)
    wph_op.load_model(wph_model, cross_moments=False, dn=dn)
    for i in range(nchan):
        wph_op.set_normalization(*coeffs_auto_norm[i])
        x_curr_i, nb_chunks = wph_op.preconfigure(x_curr[:, i])
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(x_curr_i, j, norm='auto', ret_indices=True)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_auto[i][indices]) ** 2) / (nsynth * coeffs_auto[i].shape[0])
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    loss_details.append(loss_tot.numpy()[0])
    wph_op.load_model(wph_model, cross_moments=True, dn=0)
    for pair_index, (i1, i2) in enumerate(cross_pairs):
        wph_op.set_normalization(*coeffs_cross_norm[pair_index])
        x_curr_i, nb_chunks = wph_op.preconfigure([x_curr[:, i1], x_curr[:, i2]], cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(x_curr_i, j, norm='auto', ret_indices=True, cross=True)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_cross[pair_index][indices]) ** 2) / (nsynth * coeffs_cross[pair_index].shape[0])
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    wph_op.load_model(wph_model, cross_moments=False, dn=dn)
    loss_details.append(loss_tot.numpy()[0] - sum(loss_details))
    
    # Get the gradient
    x_grad = x_curr.grad.cpu().numpy().astype(x.dtype).ravel()
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
    return loss_tot.item(), x_grad


def synthesis(x, x0, J, L=4, dn=0, cross_pairs=[], wph_model=["S11", "S00", "S01", "C01", "Cphase", "L"], device="cpu", optim_params={}):
    """Generate a multi-channel synthesis based on a WPH model of x.

    Parameters
    ----------
    x : array
        Input data.
    x0 : array
        Initialization.
    J : int
        Number of scales.
    L : int, optional
        Number of angles, by default 4
    dn : int, optional
        Number of radial steps for tau, by default 0
    cross_pairs : list, optional
        List of channel pairs to compute cross-stats, by default []
    wph_model : list, optional
        WPH model, by default ["S11", "S00", "S01", "C01", "Cphase"]
    device : str or int, optional
        Device for stats computation, by default "cpu"
    optim_params : dict, optional
        Optimization parameters, by default {}

    Returns
    -------
    array
        Synthesis.
    """
    ## Get shape and type of input data
    nchan, M, N = x.shape[-3:]
    assert x.ndim == 3

    ## Shape initialization map
    assert (nchan, M, N) == x0.shape[-3:]
    if x0.ndim == 3:
        nsynth = 1
    else:
        assert x0.ndim == 4
        nsynth = x0.shape[0]

    ## Optimization parameters
    optim_params_base = {"maxiter": 100, "gtol": 1e-12, "ftol": 1e-12, "maxcor": 20}
    optim_params_merged = {**optim_params_base, **optim_params}

    ## Load WPHOp object
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(wph_model)

    ## Compute stats of target
    print("Computing WPH stats of target image...")
    start_time = time.time()
    coeffs_auto, coeffs_auto_norm = [], []
    coeffs_cross, coeffs_cross_norm = [], []
    wph_op.load_model(wph_model, cross_moments=False, dn=dn)
    for i in range(nchan):
        coeffs_auto.append(wph_op.apply(x[i], norm='auto'))
        coeffs_auto_norm.append(wph_op.get_normalization())
        wph_op.clear_normalization()
    wph_op.load_model(wph_model, cross_moments=True, dn=0)
    for i, j in cross_pairs:
        coeffs_cross.append(wph_op.apply([x[i], x[j]], norm='auto', cross=True))
        coeffs_cross_norm.append(wph_op.get_normalization())
        wph_op.clear_normalization()
    wph_op.load_model(wph_model, cross_moments=False, dn=dn)
    print(f"Done! (in {time.time() - start_time}s)")
    print(len(coeffs_auto), coeffs_auto[0].shape if len(coeffs_auto) != 0 else None)
    print(len(coeffs_cross), coeffs_cross[0].shape if len(coeffs_cross) != 0 else None)

    dim_stats = 2*len(coeffs_auto)*coeffs_auto[0].shape[0]
    if len(coeffs_cross) != 0: dim_stats += 2*len(coeffs_cross)*coeffs_cross[0].shape[0]
    print(f"(Approximate) ratio n / m: {(dim_stats / (M*N*nchan) * 100):.2f}%")

    ## Optimization
    total_start_time = time.time()
    _objective_loc = partial(_objective, M=M, N=N,
                            nchan=nchan, nsynth=nsynth, wph_op=wph_op, wph_model=wph_model, dn=dn, cross_pairs=cross_pairs,
                            coeffs=(coeffs_auto, coeffs_cross), coeffs_norm=(coeffs_auto_norm, coeffs_cross_norm),
                            device=device)
    result = opt.minimize(_objective_loc, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params_merged)
    _, x_synth, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
    print(f"Synthesis time: {time.time() - total_start_time}s")

    ## Reshape and return result
    x_synth = x_synth.reshape((nsynth, nchan, M, N)).astype(np.float32)
    return x_synth
