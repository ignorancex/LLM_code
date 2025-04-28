"""
Gaussian likelihood module for shear bandpowers.
"""

import glob
import os.path
import time
import warnings

import numpy as np


def mvg_logpdf_fixedcov(x, mean, inv_cov):
    """
    Log-pdf of the multivariate Gaussian distribution where the determinant and inverse of the covariance matrix are
    precomputed and fixed.

    Note that this neglects the additive constant: -0.5 * (len(x) * log(2 * pi) + log_det_cov), because it is irrelevant
    when comparing pdf values with a fixed covariance, but it means that this is not the normalised pdf.

    Args:
        x (1D numpy array): Vector value at which to evaluate the pdf.
        mean (1D numpy array): Mean vector of the multivariate Gaussian distribution.
        inv_cov (2D numpy array): Inverted covariance matrix.

    Returns:
        float: Log-pdf value.
    """

    dev = x - mean
    return -0.5 * (dev @ inv_cov @ dev)


def setup(obs_bp_path, n_zbin, cov_path, mixmat_path, binmat_path, she_nl_path, noise_lmin, lmax_in, lmin_in, lmax_like,
          lmin_like):
    """
    Load/precompute anything fixed across parameter space.

    Args:
        obs_bp_path (str): Path to observed bandpowers in .npz numpy array of shape (n_spectra, n_bandpowers).
        n_zbin (int): Number of redshift bins; it is assumed 1 shear field per z bin.
        cov_path (str): Path to precomputed covariance in .npz numpy array
                        of shape (n_spectra * n_bandpowers, n_spectra * n_bandpowers).
        mixmat_path (str): Path to mixing matrix.
        binmat_path (str): Path to bandpower binning matrix.
        she_nl_path (str): Path to noise power spectrum for shear auto-spectra.
        noise_lmin (int): Input lmin for the noise power spectra.
        lmax_in (int): Maximum l pre-mixing.
        lmin_in (int): Minimum l pre-mixing.
        lmax_like (int): Maximum l used for the likelihood.
        lmin_like (int): Minimum l used for the likelihood.

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # Load observed bandpowers and prepare into a vector
    with np.load(obs_bp_path) as data:
        obs_bp = data['obs_bp']
    n_spec, n_bandpower = obs_bp.shape
    assert n_spec == n_zbin * (n_zbin + 1) // 2
    n_data = n_spec * n_bandpower
    obs_bp = np.reshape(obs_bp, n_data)

    # Load and invert covariance
    with np.load(cov_path) as data:
        cov = data['cov']
    assert cov.shape == (n_data, n_data)
    inv_cov = np.linalg.inv(cov)

    # Load mixing matrix, allowing for it to run from 0
    with np.load(mixmat_path) as data:
        mixmat = data['mixmat_ee_to_ee']
    n_ell_in = lmax_in - lmin_in + 1
    n_ell_like = lmax_like - lmin_like + 1
    if mixmat.shape == (n_ell_like + lmin_like, n_ell_in + lmin_in):
        mixmat = mixmat[lmin_like:, lmin_in:]
    assert mixmat.shape == (n_ell_like, n_ell_in)

    # Load binning matrix
    with np.load(binmat_path) as data:
        binmat = data['pbl']
    assert binmat.shape == (n_bandpower, n_ell_like)

    # Multiply binning and mixing matrices
    binmixmat = binmat @ mixmat
    assert binmixmat.shape == (n_bandpower, n_ell_in)

    # Load noise and trim/pad to correct length for input to mixing matrices,
    # truncating power above input_lmax
    she_nl = np.loadtxt(she_nl_path, max_rows=(lmax_in - noise_lmin + 1))
    she_nl = np.concatenate((np.zeros(noise_lmin), she_nl))[lmin_in:(lmax_in + 1)]
    assert she_nl.shape == (n_ell_in, )

    # Prepare config dictionary
    config = {
        'obs_bp': obs_bp,
        'n_zbin': n_zbin,
        'inv_cov': inv_cov,
        'n_spec': n_spec,
        'she_nl': she_nl,
        'lmax_in': lmax_in,
        'lmin_in': lmin_in,
        'n_bandpower': n_bandpower,
        'binmixmat': binmixmat
    }
    return config


def execute(theory_cl, theory_lmin, config):
    """
    Calculate the joint log likelihood at a particular point in parameter space.

    Args:
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        theory_lmin (int): l corresponding to the first Cl in each power spectrum (must be consistent between spectra).
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Unpack config dictionary
    obs_bp = config['obs_bp']
    n_zbin = config['n_zbin']
    inv_cov = config['inv_cov']
    n_spec = config['n_spec']
    she_nl = config['she_nl']
    lmax_in = config['lmax_in']
    lmin_in = config['lmin_in']
    n_bandpower = config['n_bandpower']
    binmixmat = config['binmixmat']

    # Trim/pad theory Cls to correct length for input to mixing matrices, truncating power above lmax_in:
    # 1. Trim so power is truncated above lmax_in
    theory_cl = theory_cl[:, :(lmax_in - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0
    theory_cl = np.concatenate((np.zeros((n_spec, theory_lmin)), theory_cl), axis=-1)
    # 3. Truncate so it runs from lmin_in
    theory_cl = theory_cl[:, lmin_in:]
    n_ell_in = lmax_in - lmin_in + 1
    assert theory_cl.shape == (n_spec, n_ell_in)

    # Add noise to auto-spectra
    theory_cl[:n_zbin] += she_nl

    # Apply combined mixing and binning matrix and vectorise
    exp_bp = np.einsum('bl,sl->sb', binmixmat, theory_cl) # b = bandpower, s = spectrum, l = ell
    exp_bp = np.reshape(exp_bp, n_spec * n_bandpower)

    # Evalute log pdf
    return mvg_logpdf_fixedcov(obs_bp, exp_bp, inv_cov)


def load_cls(n_zbin, she_she_dir, lmax=None, lmin=0):
    """
    Load shear-shear power spectra in the correct order (diagonal / healpy new=True ordering).

    Args:
        n_zbin (int): Number of redshift bins.
        she_she_dir (str): Path to directory containing shear-shear power spectra.
        lmax (int, optional): If supplied, maximum l to be read in.
        lmin (int, optional): If > 0, output will be padded to start at l = 0.

    Returns:
        2D numpy array: All Cls, with different spectra along the first axis and increasing l along the second, with \
                        shape (n_spectra, lmax + 1), where n_spectra = n_zbin * (n_zbin + 1) / 2.
    """

    # Calculate number of fields assuming 1 shear field per redshift bin
    n_field = n_zbin

    # Load power spectra in 'diagonal order'
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Form the path to the Cls - higher bin index goes first
            bins = (row + 1, col + 1)
            bin1 = max(bins)
            bin2 = min(bins)
            cl_path = os.path.join(she_she_dir, f'bin_{bin1}_{bin2}.txt')

            # Load with appropriate ell range
            max_rows = None if lmax is None else (lmax - lmin + 1)
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

    return np.asarray(spectra)


def run_likelihood(grid_dir, varied_params, save_path, obs_bp_path, n_zbin, cov_path, mixmat_path, binmat_path,
                   she_nl_path, noise_lmin, lmax_in, lmin_in, lmax_like, lmin_like):
    """
    Evaluate the likelihood on a precomputed theory grid as produced using CosmoSIS, and save the result as a text file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        varied_params (list): List of varied parameter names as they appear in the cosmological_parameters/values.txt
                              file.
        save_path (str): Path to save output text file to.
        obs_bp_path (str): Path to observed bandpowers in .npz format as produced by simulation.get_obs.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one shear field per redshift bin.
        cov_path (str): Path to covariance matrix, as produced by post_processing.get_composite_covs.
        mixmat_path (str): Path to mixing matrix, as produced by post_processing.get_mixmat.
        binmat_path (str): Path to bandpower binning matrix.
        she_nl_path (str): Path to shear noise power spectrum as a text file.
        noise_lmin (int): Minimum l in the input shear noise power spectrun.
        lmax_in (int): Maximum l pre-mixing.
        lmin_in (int): Minimum l pre-mixing.
        lmax_like (int): Maximum l to use in the likelihood.
        lmin_like (int): Minimum l to use in the likelihood.
    """

    print(f'Starting at {time.strftime("%c")}')

    # Setup the likelihood module
    print(f'Setting up likelihood module at {time.strftime("%c")}')
    config = setup(obs_bp_path, n_zbin, cov_path, mixmat_path, binmat_path, she_nl_path, noise_lmin, lmax_in,
                   lmin_in, lmax_like, lmin_like)
    print(f'Setup complete at {time.strftime("%c")}')

    # Loop over every input directory
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    if n_dirs == 0:
        warnings.warn(f'No matching directories. Terminating at {time.strftime("%c")}')
        return
    n_params = len(varied_params)
    if n_params == 0:
        warnings.warn(f'No parameters specified. Terminating at {time.strftime("%c")}')
        return
    first_dir = True
    res = []
    for i, source_dir in enumerate(source_dirs):
        print(f'Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}')

        # Extract cosmological parameters
        params = [None]*n_params
        values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
        with open(values_path) as f:
            for line in f:
                for param_idx, param in enumerate(varied_params):
                    param_str = f'{param} = '
                    if param_str in line:
                        params[param_idx] = float(line[len(param_str):])
        err_str = f'Not all parameters in varied_params found in {values_path}'
        assert np.all([param is not None for param in params]), err_str

        # Check the ells for consistency
        if first_dir:
            ell = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            theory_lmin = int(ell[0])
            first_dir = False
        else:
            ell_test = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            assert np.allclose(ell, ell_test)

        # Load theory Cls
        she_she_dir = os.path.join(source_dir, 'shear_cl/')
        theory_cl = load_cls(n_zbin, she_she_dir, lmax_in, theory_lmin)

        # Evaluate likelihood - theory lmin is now 0 because padded by the above step
        log_like_gauss = execute(theory_cl, 0, config)

        # Store cosmological params & likelihood
        res.append([*params, log_like_gauss])

    # Save results to file
    res_grid = np.asarray(res)
    param_names = ' '.join(varied_params)
    header = (f'Output from {__file__}.run_likelihood for parameters:\ngrid_dir = {grid_dir}\n'
              f'obs_bp_path = {obs_bp_path}\n,n_zbin = {n_zbin}\ncov_path = {cov_path}\nmixmat_path = {mixmat_path}\n'
              f'binmat_path = {binmat_path}\n,she_nl_path = {she_nl_path}\nnoise_lmin = {noise_lmin}\n'
              f'lmax_in = {lmax_in}\nlmin_in = {lmin_in}\nlmax_like = {lmax_like}\nlmin_like = {lmin_like}\n'
              f'at {time.strftime("%c")}\n\n'
              f'{param_names} log_like_gauss')
    np.savetxt(save_path, res_grid, header=header)
    print('Saved ' + save_path)

    print(f'Done at {time.strftime("%c")}')
