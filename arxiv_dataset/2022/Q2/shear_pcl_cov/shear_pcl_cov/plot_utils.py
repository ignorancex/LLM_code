"""
Utility functions for plotting and preparing data for plots.
"""

import time

import numpy as np
import scipy.interpolate


def get_3d_post(log_like_path, save_path):
    """
    Form 3D posterior grid from a log-likelihood file and save to disk, ready for plotting.

    Args:
        log_like_path (str): Path to log-likelihood text file.
        save_path (str): Path to save 3D posterior grid as .npz file.
    """

    # Load data
    print('Loading')
    data = np.loadtxt(log_like_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    log_like = data[:, 3]

    # Convert log-likelihood to unnormalised posterior (assuming flat prior)
    print('Converting to posterior')
    log_like -= np.amax(log_like) - 100
    post = np.exp(log_like)

    # Form grids
    print('Forming grids')
    x_range = np.unique(x)
    y_range = np.unique(y)
    z_range = np.unique(z)
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Grid the data
    print('Gridding data')
    post_grid = scipy.interpolate.griddata((x, y, z), post, (x_grid, y_grid, z_grid), fill_value=0)

    # Save to file
    print('Saving')
    header = f'3D posterior grid output from {__file__}.get_3d_post for input {log_like_path} at {time.strftime("%c")}'
    np.savez_compressed(save_path, x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, post_grid=post_grid, header=header)
    print('Saved ' + save_path)


def get_cov_diags(cov_cng_fullsky_path, theory_cl_path, per_mask_data, lmin, lmax, lmax_mix, diags, save_path):
    """
    Extract diagonals of the covariance matrix for plotting with plotting.cov_diags.

    Note that this is all for a single block, which in the paper is the auto-power in the lowest redshift bin.

    Args:
        cov_cng_fullsky_path (str): Path to the full-sky connected non-Gaussian covariance matrix.
        theory_cl_path (str): Path to theory power spectrum.
        per_mask_data (list): List of dictionaries, one dictionary per mask, each containing fields:
                              ``mask_label`` used for the column headers,
                              ``fsky`` sky fraction,
                              ``mixmat_path`` path to mixing matrix or None for full sky,
                              ``sim_cl_path`` path to simulated Cls as output by simulation.combine_sim_cl,
                              ``cov_g_path`` path to Gaussian covariance,
                              ``cov_ss_path`` path to super-sample covariance.
        lmin (int): Minimum l.
        lmax (int): Maximum l post-mixing.
        lmax_mix (int): Maximum l pre-mixing.
        diags (list): List of diagonals to extract, e.g. [0, 2, 10, 100].
        save_path (str): Path to save output data to.
    """

    # Load fixed things: full-sky CNG cov and theory Cls
    print('Loading full-sky connected non-Gaussian matrix')
    with np.load(cov_cng_fullsky_path) as per_mask_data:
        cov_cng_fullsky = per_mask_data['cov']
    print('Loading theory Cls')
    theory_cl_unmixed = np.loadtxt(theory_cl_path, max_rows=(lmax_mix - lmin + 1))

    # Loop over masks
    results = []
    n_masks = len(per_mask_data)
    for mask_idx, mask in enumerate(per_mask_data, 1):

        # Load sim Cls, calculate covariance and extract variance and correlation
        print(f'Mask {mask_idx} / {n_masks}: Loading sim Cls')
        with np.load(mask['sim_cl_path']) as per_mask_data:
            sim_cl = per_mask_data['cls'][0, :, :] # bin 1 auto-power
        print(f'Mask {mask_idx} / {n_masks}: Calculating sim covariance')
        sim_cov = np.cov(sim_cl, rowvar=True)
        sim_var = np.diag(sim_cov)
        sim_std = np.sqrt(sim_var)
        sim_corr = sim_cov / np.outer(sim_std, sim_std)

        # Load Gaussian covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading Gaussian covariance')
        with np.load(mask['cov_g_path']) as per_mask_data:
            cov_g = per_mask_data['cov']

        # Load and trim mixing matrix
        if mask['mixmat_path'] is not None and mask['fsky'] < 1:
            print(f'Mask {mask_idx} / {n_masks}: Loading mixing matrix')
            with np.load(mask['mixmat_path']) as per_mask_data:
                mixmat = per_mask_data['mixmat_ee_to_ee']
            mixmat = mixmat[lmin:, lmin:]
        elif mask['mixmat_path'] is None and mask['fsky'] == 1:
            print(f'Mask {mask_idx} / {n_masks}: Full sky')
            mixmat = np.identity(lmax_mix - lmin + 1)[:(lmax - lmin + 1), :]
        else:
            raise ValueError('Invalid combination of mixmat_path and fsky')

        # Load and mix super-sample covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading super-sample covariance')
        with np.load(mask['cov_ss_path']) as per_mask_data:
            cov_ss_unmixed = per_mask_data['cov']
        print(f'Mask {mask_idx} / {n_masks}: Mixing super-sample covariance')
        cov_ss_mixed = mixmat @ cov_ss_unmixed @ mixmat.T

        # Rescale full-sky connected non-Gaussian matrix to mimic CosmoLike output and apply mixing matrix
        cov_cng_unmixed = cov_cng_fullsky / mask['fsky']
        print(f'Mask {mask_idx} / {n_masks}: Mixing connected non-Gaussian covariance')
        cov_cng_mixed = mixmat @ cov_cng_unmixed @ mixmat.T

        # Extract variance and correlation from each theory covariance matrix
        print(f'Mask {mask_idx} / {n_masks}: Calculating correlation matrices')
        var_g = np.diag(cov_g)
        var_ss = np.diag(cov_ss_mixed)
        var_cng = np.diag(cov_cng_mixed)
        std_tot = np.sqrt(var_g + var_ss + var_cng)
        std_mat = np.outer(std_tot, std_tot)
        corr_g = cov_g / std_mat
        corr_ss = cov_ss_mixed / std_mat
        corr_cng = cov_cng_mixed / std_mat

        # Extract out the required diagonals
        mask_results = {
            'mask_label': mask['mask_label'],
            'results_per_diag': []
        }
        n_diags = len(diags)
        for diag_idx, diag in enumerate(diags, 1):
            print(f'Mask {mask_idx} / {n_masks}: Extracting diagonal {diag_idx} / {n_diags}')
            diag_results = {}

            if diag == 0: # Var(Cl) / Cl^2
                sim_cl_squared = np.mean(sim_cl, axis=1) ** 2
                diag_results['sim'] = sim_var / sim_cl_squared
                theory_cl_mixed = mixmat @ theory_cl_unmixed
                theory_cl_squared = theory_cl_mixed ** 2
                diag_results['g'] = var_g / theory_cl_squared
                diag_results['ss'] = var_ss / theory_cl_squared
                diag_results['cng'] = var_cng / theory_cl_squared

            else: # Corr(Cl)
                diag_results['sim'] = np.diag(sim_corr, k=diag)
                diag_results['g'] = np.diag(corr_g, k=diag)
                diag_results['ss'] = np.diag(corr_ss, k=diag)
                diag_results['cng'] = np.diag(corr_cng, k=diag)

            mask_results['results_per_diag'].append(diag_results)

        results.append(mask_results)

    # Save to disk
    header = (f'Intermediate output from {__file__} function to_file for input '
              f'cov_cng_fullsky_path = {cov_cng_fullsky_path}, theory_cl_path = {theory_cl_path}, lmin = {lmin}, '
              f'lmax = {lmax}, lmax_mix = {lmax_mix}, diags = {diags}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, results=results, input_data=per_mask_data, lmin=lmin, lmax=lmax, diags=diags,
                        header=header)
    print('Saved ' + save_path)


def get_cov_mats(cov_cng_fullsky_path, theory_cl_path, per_mask_data, lmin, lmax, lmax_mix, save_path):
    """
    Form correlation matrices for plotting with plotting.cov_mats.

    Note that this is all for a single block, which in the paper is the auto-power in the lowest redshift bin.

    Args:
        cov_cng_fullsky_path (str): Path to the full-sky connected non-Gaussian covariance matrix.
        theory_cl_path (str): Path to theory power spectrum.
        per_mask_data (list): List of dictionaries, one dictionary per mask, each containing fields:
                              ``mask_label`` used for the column headers,
                              ``fsky`` sky fraction,
                              ``mixmat_path`` path to mixing matrix or None for full sky,
                              ``sim_cl_path`` path to simulated Cls as output by simulation.combine_sim_cl,
                              ``cov_g_path`` path to Gaussian covariance,
                              ``cov_ss_path`` path to super-sample covariance.
        lmin (int): Minimum l.
        lmax (int): Maximum l post-mixing.
        lmax_mix (int): Maximum l pre-mixing.
        save_path (str): Path to save output data to.
    """

    # Load fixed things: full-sky CNG cov and theory Cls
    print('Loading full-sky connected non-Gaussian matrix')
    with np.load(cov_cng_fullsky_path) as data:
        cov_cng_fullsky = data['cov']

    # Loop over masks
    results = []
    n_masks = len(per_mask_data)
    for mask_idx, mask in enumerate(per_mask_data, 1):

        # Load sim Cls, calculate covariance and extract variance and correlation
        print(f'Mask {mask_idx} / {n_masks}: Loading sim Cls')
        with np.load(mask['sim_cl_path']) as data:
            sim_cl = data['cls'][0, :, :] # bin 1 auto-power
        print(f'Mask {mask_idx} / {n_masks}: Calculating sim covariance')
        sim_cov = np.cov(sim_cl, rowvar=True)
        sim_var = np.diag(sim_cov)
        sim_std = np.sqrt(sim_var)
        sim_corr = sim_cov / np.outer(sim_std, sim_std)

        # Load Gaussian covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading Gaussian covariance')
        with np.load(mask['cov_g_path']) as data:
            cov_g = data['cov']

        # Load and trim mixing matrix
        if mask['mixmat_path'] is not None and mask['fsky'] < 1:
            print(f'Mask {mask_idx} / {n_masks}: Loading mixing matrix')
            with np.load(mask['mixmat_path']) as data:
                mixmat = data['mixmat_ee_to_ee']
            mixmat = mixmat[lmin:, lmin:]
        elif mask['mixmat_path'] is None and mask['fsky'] == 1:
            print(f'Mask {mask_idx} / {n_masks}: Full sky')
            mixmat = np.identity(lmax_mix - lmin + 1)[:(lmax - lmin + 1), :]
        else:
            raise ValueError('Invalid combination of mixmat_path and fsky')

        # Load and mix super-sample covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading super-sample covariance')
        with np.load(mask['cov_ss_path']) as data:
            cov_ss_unmixed = data['cov']
        print(f'Mask {mask_idx} / {n_masks}: Mixing super-sample covariance')
        cov_ss_mixed = mixmat @ cov_ss_unmixed @ mixmat.T

        # Rescale full-sky connected non-Gaussian matrix to mimic CosmoLike output and apply mixing matrix
        cov_cng_unmixed = cov_cng_fullsky / mask['fsky']
        print(f'Mask {mask_idx} / {n_masks}: Mixing connected non-Gaussian covariance')
        cov_cng_mixed = mixmat @ cov_cng_unmixed @ mixmat.T

        # Extract variance and correlation from each theory covariance matrix
        print(f'Mask {mask_idx} / {n_masks}: Calculating correlation matrices')
        var_g = np.diag(cov_g)
        var_ss = np.diag(cov_ss_mixed)
        var_cng = np.diag(cov_cng_mixed)
        std_tot = np.sqrt(var_g + var_ss + var_cng)
        std_mat = np.outer(std_tot, std_tot)
        corr_g = cov_g / std_mat
        corr_ss = cov_ss_mixed / std_mat
        corr_cng = cov_cng_mixed / std_mat

        # Calculate totals, minima and maxima
        print(f'Mask {mask_idx} / {n_masks}: Calculating totals, minima and maxima')
        corr_tot = corr_g + corr_ss + corr_cng
        min_tot = np.amin(corr_tot)
        max_tot = np.amax(corr_tot)
        min_sim = np.amin(sim_corr)
        max_sim = np.amax(sim_corr)

        # Save the required matrices
        mask_results = {
            'mask_label': mask['mask_label'],
            'sim_corr': sim_corr,
            'corr_g': corr_g,
            'corr_ss': corr_ss,
            'corr_cng': corr_cng,
            'corr_tot': corr_tot,
            'min_tot': min_tot,
            'max_tot': max_tot,
            'min_sim': min_sim,
            'max_sim': max_sim
        }

        results.append(mask_results)

    # Save to disk
    print('Saving')
    header = (f'Output from {__file__}.get_cov_mats for input cov_cng_fullsky_path = {cov_cng_fullsky_path}, '
              f'theory_cl_path = {theory_cl_path}, lmin = {lmin}, lmax = {lmax}, lmax_mix = {lmax_mix}, '
              f'at {time.strftime("%c")}')
    np.savez_compressed(save_path, results=results, input_data=per_mask_data, header=header)
    print('Saved ' + save_path)


def cov_pool(arr, axis=None, threshold=0.6, **kwargs):
    """
    Pooling function for downsampling correlation matrices for plotting, which can be passed as the ``func`` argument to
    skimage.measure.block_reduce.

    The aim is to strike a balance between max pooling, which preserves important features (particularly the diagonal)
    but also artificially amplifies noise, and mean pooling which averages out noise but also washes out features.
    This function works by max pooling if the max is above some threshold value, and mean pooling otherwise.

    Args:
        arr (ND numpy array): Array to downsample.
        axis (int, optional): Axis to downsample, which is passed to np.max and np.mean. Default is None, which will
                              cause those functions to use the flattened input.
        threshold (float, optional): Threshold for the max value in a pool, above which to use max pooling and below
                                     which to use mean pooling (default 0.6).
        **kwargs: Additional keyword arguments to be passed to both np.max and np.mean.
    """

    arr_max = np.max(arr, axis=axis, **kwargs)
    arr_mean = np.mean(arr, axis=axis, **kwargs)
    return np.where(arr_max > threshold, arr_max, arr_mean)


def get_cov_diags_withnoise(cov_cng_fullsky_path, per_mask_data, lmin, lmax, lmax_mix, diags, save_path):
    """
    Extract diagonals of the covariance matrix with noise (no simulations), for plotting with
    plotting.cov_withnoise.

    Note that this is all for a single block, which in the paper is the auto-power in the lowest redshift bin.

    Args:
        cov_cng_fullsky_path (str): Path to the full-sky connected non-Gaussian covariance matrix.
        per_mask_data (list): List of dictionaries, one dictionary per mask, each containing fields:
                              ``mask_label`` used for the column headers,
                              ``fsky`` sky fraction,
                              ``mixmat_path`` path to mixing matrix or None for full sky,
                              ``cov_g_path`` path to Gaussian covariance,
                              ``cov_ss_path`` path to super-sample covariance.
        lmin (int): Minimum l.
        lmax (int): Maximum l post-mixing.
        lmax_mix (int): Maximum l pre-mixing.
        diags (list): List of diagonals to extract, e.g. [0, 2, 10, 100].
        save_path (str): Path to save output data to.
    """

    # Load fixed things: full-sky CNG cov
    print('Loading full-sky connected non-Gaussian matrix')
    with np.load(cov_cng_fullsky_path) as data:
        cov_cng_fullsky = data['cov']

    # Loop over masks
    results = []
    n_masks = len(per_mask_data)
    for mask_idx, mask in enumerate(per_mask_data, 1):

        # Load Gaussian covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading Gaussian covariance')
        with np.load(mask['cov_g_path']) as data:
            cov_g = data['cov']

        # Load and trim mixing matrix
        if mask['mixmat_path'] is not None and mask['fsky'] < 1:
            print(f'Mask {mask_idx} / {n_masks}: Loading mixing matrix')
            with np.load(mask['mixmat_path']) as data:
                mixmat = data['mixmat_ee_to_ee']
            mixmat = mixmat[lmin:, lmin:]
        elif mask['mixmat_path'] is None and mask['fsky'] == 1:
            print(f'Mask {mask_idx} / {n_masks}: Full sky')
            mixmat = np.identity(lmax_mix - lmin + 1)[:(lmax - lmin + 1), :]
        else:
            raise ValueError('Invalid combination of mixmat_path and fsky')

        # Load and mix super-sample covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading super-sample covariance')
        with np.load(mask['cov_ss_path']) as data:
            cov_ss_unmixed = data['cov']
        print(f'Mask {mask_idx} / {n_masks}: Mixing super-sample covariance')
        cov_ss_mixed = mixmat @ cov_ss_unmixed @ mixmat.T

        # Rescale full-sky connected non-Gaussian matrix to mimic CosmoLike output and apply mixing matrix
        cov_cng_unmixed = cov_cng_fullsky / mask['fsky']
        print(f'Mask {mask_idx} / {n_masks}: Mixing connected non-Gaussian covariance')
        cov_cng_mixed = mixmat @ cov_cng_unmixed @ mixmat.T

        # Extract variance and correlation from each theory covariance matrix
        print(f'Mask {mask_idx} / {n_masks}: Calculating correlation matrices')
        var_g = np.diag(cov_g)
        var_ss = np.diag(cov_ss_mixed)
        var_cng = np.diag(cov_cng_mixed)
        std_tot = np.sqrt(var_g + var_ss + var_cng)
        std_mat = np.outer(std_tot, std_tot)
        corr_g = cov_g / std_mat
        corr_ss = cov_ss_mixed / std_mat
        corr_cng = cov_cng_mixed / std_mat

        # Extract out the required diagonals
        mask_results = {
            'mask_label': mask['mask_label'],
            'results_per_diag': []
        }
        n_diags = len(diags)
        for diag_idx, diag in enumerate(diags, 1):
            print(f'Mask {mask_idx} / {n_masks}: Extracting diagonal {diag_idx} / {n_diags}')
            diag_results = {}

            if diag == 0: # Var(Cl)
                diag_results['g'] = var_g
                diag_results['ss'] = var_ss
                diag_results['cng'] = var_cng

            else: # Corr(Cl)
                diag_results['g'] = np.diag(corr_g, k=diag)
                diag_results['ss'] = np.diag(corr_ss, k=diag)
                diag_results['cng'] = np.diag(corr_cng, k=diag)

            mask_results['results_per_diag'].append(diag_results)

        results.append(mask_results)

    # Save to disk
    header = (f'Output from {__file__}.get_cov_diags_withnoise for input '
              f'cov_cng_fullsky_path = {cov_cng_fullsky_path}, lmin = {lmin}, lmax = {lmax}, lmax_mix = {lmax_mix}, '
              f'diags = {diags}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, results=results, input_data=per_mask_data, lmin=lmin, lmax=lmax, diags=diags,
                        header=header)
    print('Saved ' + save_path)


def get_cov_diags_gaussian(per_mask_data, diags, save_path):
    """
    Extract diagonals of the Gaussian covariance matrix for plotting with plotting.cov_gaussian.

    Note that this is all for a single block, which in the paper is the auto-power in the lowest redshift bin.

    Args:
        per_mask_data (list): List of dictionaries, one dictionary per mask, each containing fields:
                              ``mask_label`` used for the column headers,
                              ``sim_cl_path`` path to simulated Cls as output by simulation.gaussian_sim,
                              ``cov_g_path`` path to Gaussian covariance.
        diags (list): List of diagonals to extract, e.g. [0, 2, 10, 100].
        save_path (str): Path to save output data to.
    """

    # Loop over masks
    results = []
    n_masks = len(per_mask_data)
    for mask_idx, mask in enumerate(per_mask_data, 1):

        # Load sim Cls, calculate covariance and extract variance and correlation
        print(f'Mask {mask_idx} / {n_masks}: Loading sim Cls')
        with np.load(mask['sim_cl_path']) as data:
            sim_cl = data['obs_cls']
        print(f'Mask {mask_idx} / {n_masks}: Calculating sim covariance')
        sim_cov = np.cov(sim_cl, rowvar=False)
        sim_var = np.diag(sim_cov)
        sim_std = np.sqrt(sim_var)
        sim_corr = sim_cov / np.outer(sim_std, sim_std)

        # Load Gaussian covariance
        print(f'Mask {mask_idx} / {n_masks}: Loading Gaussian covariance')
        with np.load(mask['cov_g_path']) as data:
            cov_g = data['cov']

        # Extract variance and correlation
        print(f'Mask {mask_idx} / {n_masks}: Calculating correlation matrices')
        var_g = np.diag(cov_g)
        std_g = np.sqrt(var_g)
        std_mat = np.outer(std_g, std_g)
        corr_g = cov_g / std_mat

        # Extract out the required diagonals
        mask_results = {
            'mask_label': mask['mask_label'],
            'results_per_diag': []
        }
        n_diags = len(diags)
        for diag_idx, diag in enumerate(diags, 1):
            print(f'Mask {mask_idx} / {n_masks}: Extracting diagonal {diag_idx} / {n_diags}')
            diag_results = {}

            if diag == 0: # Var(Cl)
                diag_results['sim'] = sim_var
                diag_results['g'] = var_g

            else: # Corr(Cl)
                diag_results['sim'] = np.diag(sim_corr, k=diag)
                diag_results['g'] = np.diag(corr_g, k=diag)

            mask_results['results_per_diag'].append(diag_results)

        results.append(mask_results)

    # Save to disk
    header = (f'Intermediate output from {__file__} function to_file at {time.strftime("%c")}')
    np.savez_compressed(save_path, results=results, input_data=per_mask_data, diags=diags, header=header)
    print('Saved ' + save_path)
