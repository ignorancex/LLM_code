"""
Contains functions relating to the connected non-Gaussian covariance approximation.
"""

import glob
import os.path
import time

import numpy as np


def get_avg_l(lmin, lmax, n_bandpower):
    """
    Determine the weighted average l per bandpower.

    Args:
        lmin (int): Minimum l.
        lmax (int): Maximum l.
        n_bandpower (int): Number of bandpowers.
    """

    # Calculate bin boundaries (add small fraction to lmax to include it in the end bin)
    edges = np.logspace(np.log10(lmin), np.log10(lmax + 1e-5), n_bandpower + 1)

    # Loop over bins
    for bin_idx, (lo_edge, hi_edge) in enumerate(zip(edges[:-1], edges[1:])):

        # Determine ells in bin
        bin_lmin = np.ceil(lo_edge)
        bin_lmax = np.floor(hi_edge)
        bin_ell = np.arange(bin_lmin, bin_lmax + 1)

        # Calculate weights = l(l+1)/2π
        weights = bin_ell * (bin_ell + 1) / (2 * np.pi)

        # Calculate weighted average l over bin
        weighted_avg_l = np.average(bin_ell, weights=weights)

        # Round to the nearest integer and print
        weighted_avg_l = int(np.around(weighted_avg_l))
        print(f'Bin {bin_idx}: {weighted_avg_l}')


def get_bin_weights(full_cov_path, binmat_path, lmax, lmin, ells, save_path):
    """
    Obtain the binning weights used in the connected non-Gaussian approximation.

    Args:
        full_cov_path (str): Path to full connected non-Gaussian covariance block.
        binmat_path (str): Path to binning matrix.
        lmax (int): Maximum l.
        lmin (int): Minimum l.
        ells (list): List of ells to evaluate the weights for, as given by get_avg_l.
        save_path (str): Path to save output .npz file to.
    """

    # Load unmixed covariance and truncate to the lmax
    print('Loading covariance')
    n_ell = lmax - lmin + 1
    with np.load(full_cov_path) as data:
        cov_unbinned = data['cov'][:n_ell, :n_ell]

    # Load binning matrix
    print('Loading binning matrix')
    with np.load(binmat_path) as data:
        binmat = data['pbl']

    # Apply binning matrix
    print('Applying binning matrix')
    cov_binned = binmat @ cov_unbinned @ binmat.T

    # Extract out the sampled ells
    print('Extracting sampled ells')
    ell_idx = np.subtract(ells, lmin)
    cov_sampled = cov_unbinned[ell_idx, :][:, ell_idx]

    # Calculate ratio between sampled and real binned
    ratio = cov_sampled / cov_binned

    # Calculate and save weights to go from sampled to binned
    weights = 1 / ratio
    assert np.allclose(cov_sampled * weights, cov_binned, atol=0)

    header = ('Weights to go from covariance block sampled at ells given in ells array to approximate binned '
              f'covariance. Output from {__file__}.get_bin_weights for params '
              f'full_cov_path = {full_cov_path}, binmat_path = {binmat_path}, lmax = {lmax}, lmin = {lmin}, '
              f'ells = {ells}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, weights=weights, ells=ells, header=header)
    print('Saved ' + save_path)


def get_mix_weights(full_cov_path, binmat_path, bin_weights_path, mixmat_path, lmin, save_path, fsky=None):
    """
    Obtain the mixing weights used in the connected non-Gaussian approximation.

    Args:
        full_cov_path (str): Path to full connected non-Gaussian covariance block.
        binmat_path (str): Path to binning matrix.
        bin_weights_path (str): Path to binning weights obtained with get_bin_weights.
        mixmat_path (str): Path to mixing matrix.
        lmin (int): Minimum l.
        save_path (str): Path to save output .npz file to.
        fsky (float, optional): Sky fraction - if supplied, will multiply input covariance block by 1/fsky. This is only
                                necessary if the full_cov_path is the path to the full-sky connected non-Gaussian
                                covariance and hasn't already received the 1/fsky factor.
    """

    # Load unmixed, unbinned block
    print('Loading original block')
    with np.load(full_cov_path) as data:
        unmixed_unbinned_block = data['cov']

    # Load binning weights
    print('Loading binning weights')
    with np.load(bin_weights_path) as data:
        part1_weights = data['weights']
        sampled_ell = data['ells']

    # Load binning matrix
    print('Loading binning matrix')
    with np.load(binmat_path) as data:
        pbl = data['pbl']

    # Load mixing matrix
    print('Loading mixing matrix')
    with np.load(mixmat_path) as data:
        mixmat = data['mixmat_ee_to_ee'][lmin:, lmin:]

    # Adjust unmixed, unbinned block for fsky
    if fsky is not None:
        print('Applying fsky correction')
        unmixed_unbinned_block /= fsky

    # Select sampled ells and apply part 1 weights
    print('Applying part 1 weights')
    sampled_ell_idx = sampled_ell - lmin
    part1_input = unmixed_unbinned_block[sampled_ell_idx, :][:, sampled_ell_idx]
    part1_output = part1_weights * part1_input

    # As a check, apply binning matrix to truncated unmixed unbinned matrix
    # and confirm that it gives an identical result
    n_ell = mixmat.shape[0]
    part1_check = pbl @ unmixed_unbinned_block[:n_ell, :n_ell] @ pbl.T
    print('Part 1 check:', np.allclose(part1_output, part1_check, atol=0))

    # Apply mixing matrix to unmixed unbinned matrix, followed by binning matrix, to obtain binned mixed block
    print('Applying mixing matrix')
    mixed_unbinned_block = mixmat @ unmixed_unbinned_block @ mixmat.T
    print('Applying binning matrix')
    mixed_binned_block = pbl @ mixed_unbinned_block @ pbl.T

    # Elementwise divide binned mixed block by binned unmixed block to give effective fsky^2 matrix
    print('Calculating effective fsky^2 matrix')
    eff_fsky2 = mixed_binned_block / part1_output

    # As a check, apply effective fsky^2 matrix to binned unmixed block and check identical to binned mixed block
    part2_check = eff_fsky2 * part1_output
    print('Check:', np.allclose(part2_check, mixed_binned_block, atol=0))

    # Save effective fsky^2 matrix to disk
    header = (f'Mixing weights for CNG approximation. Output from {__file__}.get_mix_weights for params '
              f'full_cov_path = {full_cov_path}, bin_weights_path = {bin_weights_path}, '
              f'binmat_path = {binmat_path}, mixmat_path = {mixmat_path}, fsky = {fsky}, lmin = {lmin}, '
              f'at {time.strftime("%c")}')
    np.savez(save_path, eff_fsky2=eff_fsky2, header=header)
    print('Saved ' + save_path)


def test_bin_weights(ss_block_filemask, binmat_path, lmax, lmin, ells, n_spec, save_path):
    """
    Test the approach of binning weights by applying them to super-sample covariance blocks and measuring the ratio of
    approximate to exact (full treatment) covariance.

    Args:
        ss_block_filemask (str): Path to input (unmixed, unbinned) super-sample covariance blocks, with {spec1_idx}
                                 and {spec2_idx} placeholders.
        binmat_path (str): Path to binning matrix.
        lmax (int): Maximum l.
        lmin (int): Minimum l.
        ells (list): List of ells to evaluate the weights for, as given by get_avg_l.
        n_spec (int): Number of power spectra.
        save_path (str): Path to save covariance ratios to, for later plotting using plotting.cng_approx.
    """

    # Load binning matrix
    with np.load(binmat_path) as data:
        pbl = data['pbl']

    # Load the first block and use it to calculate weights
    print('Calculating weights')
    n_ell = lmax - lmin + 1
    with np.load(ss_block_filemask.format(spec1_idx=0, spec2_idx=0)) as data:
        first_block_unbinned = data['cov'][:n_ell, :n_ell]
    first_block_binned = pbl @ first_block_unbinned @ pbl.T
    ell_idx = np.subtract(ells, lmin)
    first_block_sampled = first_block_unbinned[ell_idx, :][:, ell_idx]
    weights = first_block_binned / first_block_sampled
    assert np.allclose(first_block_sampled * weights, first_block_binned, atol=0)
    assert np.allclose(weights, weights.T, atol=0)

    # Loop over all other blocks and measure ratio between approximate and exact
    n_blocks = n_spec * (n_spec + 1) // 2
    n_bp = len(ells)
    ratios = np.full((n_blocks - 1, n_bp, n_bp), np.NINF)
    block_idx = 0
    for spec1_idx in range(1, n_spec):
        for spec2_idx in range(spec1_idx + 1):
            print(f'Validating weights, block {block_idx + 1} / {n_blocks - 1}')

            with np.load(ss_block_filemask.format(spec1_idx=spec1_idx, spec2_idx=spec2_idx)) as data:
                block_unbinned = data['cov'][:n_ell, :n_ell]
            block_binned = pbl @ block_unbinned @ pbl.T
            block_sampled = block_unbinned[ell_idx, :][:, ell_idx]
            ratio = weights * block_sampled / block_binned

            # For symmetric blocks use nan for the lower triangle so no double-counting
            if spec1_idx == spec2_idx:
                ratio[np.tril_indices(n_bp, k=-1)] = np.nan
            ratios[block_idx] = ratio

            block_idx += 1

    # Save ratios to disk
    assert not np.any(np.isneginf(ratios))
    header = (f'Output of {__file__}.test_bin_weights for ss_block_filemask = {ss_block_filemask}, '
              f'binmat_path = {binmat_path}, lmax = {lmax}, ells = {ells}, n_spec = {n_spec}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, ratios=ratios, header=header)
    print('Saved ' + save_path)


def test_mix_weights(unmixed_unbinned_ss_dir, mixed_binned_ss_dir, input_filemask, binmat_path, n_spec, save_path):
    """
    Test the approach of mixing weights by applying them to super-sample covariance blocks and measuring the ratio of
    approximate to exact (full treatment) covariance.

    Args:
        unmixed_unbinned_ss_dir (str): Path to directory containing unmixed unbinned super-sample covariance blocks.
        mixed_binned_ss_dir (str): Path to directory containing mixed binned super-sample covariance blocks.
        input_filemask (str): Filename of blocks within input directories, with {spec1_idx} and {spec2_idx}
                              placeholders.
        binmat_path (str): Path to binning matrix.
        n_spec (int): Number of power spectra.
        save_path (str): Path to save covariance ratios to, for later plotting using plotting.cng_approx.
    """

    # For the first block:
    print('Calculating weights')

    # Load unmixed unbinned block
    first_block_filename = input_filemask.format(spec1_idx=0, spec2_idx=0)
    with np.load(os.path.join(unmixed_unbinned_ss_dir, first_block_filename)) as data:
        unmixed_unbinned_first_block = data['cov']

    # Load mixed binned block
    with np.load(os.path.join(mixed_binned_ss_dir, first_block_filename)) as data:
        mixed_binned_first_block = data['cov_binned']

    # Load binning matrix
    with np.load(binmat_path) as data:
        pbl = data['pbl']

    # Apply binning matrix to truncated unmixed unbinned block to obtain unmixed binned block
    n_ell = pbl.shape[1]
    unmixed_unbinned_first_block = unmixed_unbinned_first_block[:n_ell, :n_ell]
    unmixed_binned_first_block = pbl @ unmixed_unbinned_first_block @ pbl.T

    # Divide mixed binned block by unmixed binned block to obtain effective fsky^2 matrix
    eff_fsky2 = mixed_binned_first_block / unmixed_binned_first_block

    # Loop over subsequent blocks
    n_blocks = n_spec * (n_spec + 1) // 2
    n_bp = pbl.shape[0]
    ratios = np.full((n_blocks - 1, n_bp, n_bp), np.NINF)
    block_idx = 0
    for spec1_idx in range(1, n_spec):
        for spec2_idx in range(spec1_idx + 1):
            print(f'Validating weights, block {block_idx + 1} / {n_blocks - 1}')

            # Load unmixed unbinned block and mixed binned block
            block_filename = input_filemask.format(spec1_idx=spec1_idx, spec2_idx=spec2_idx)
            with np.load(os.path.join(unmixed_unbinned_ss_dir, block_filename)) as data:
                unmixed_unbinned_block = data['cov']
            with np.load(os.path.join(mixed_binned_ss_dir, block_filename)) as data:
                mixed_binned_block = data['cov_binned']

            # Apply binning matrix to truncated unmixed unbinned block to obtain unmixed binned block
            unmixed_unbinned_block = unmixed_unbinned_block[:n_ell, :n_ell]
            unmixed_binned_block = pbl @ unmixed_unbinned_block @ pbl.T

            # Apply effective fsky^2 matrix to unmixed binned block and compare with true mixed_binned_block
            ratio = eff_fsky2 * unmixed_binned_block / mixed_binned_block

            # For symmetric blocks use nan for the lower triangle so no double-counting
            if spec1_idx == spec2_idx:
                ratio[np.tril_indices(n_bp, k=-1)] = np.nan
            ratios[block_idx] = ratio

            block_idx += 1

    # Save ratios to disk
    assert not np.any(np.isneginf(ratios))
    header = (f'Output of {__file__}.test_mix_weights for unmixed_unbinned_ss_dir = {unmixed_unbinned_ss_dir}, '
              f'mixed_binned_ss_dir = {mixed_binned_ss_dir}, input_filemask = {input_filemask}, '
              f'binmat_path = {binmat_path}, n_spec = {n_spec}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, ratios=ratios, header=header)
    print('Saved ' + save_path)


def apply_bin_weights(input_dir, output_dir, filemask, weights_path):
    """
    Apply binning weights to each block of the connected non-Gaussian covariance matrix.

    Args:
        input_dir (str): Path to input blocks, evaluated only at one element per bandpower pair.
        output_dir (str): Path to save output blocks to.
        filemask (str): Glob filemask for input files within input directory.
        weights_path (str): Path to binning weights as obtained using get_bin_weights.
    """

    # Load weights
    with np.load(weights_path) as data:
        weights = data['weights']

    # Loop over files matching mask
    input_files = glob.glob(os.path.join(input_dir, filemask))
    n_files = len(input_files)
    for file_no, input_file in enumerate(input_files, 1):

        # Load block and original header
        print(f'{file_no} / {n_files}: Loading...', end='\r')
        with np.load(input_file) as data:
            block = data['cov']
            prev_headers = [data['header'], data['orig_header']]

        # Apply weights
        print(f'{file_no} / {n_files}: Applying weights...', end='\r')
        block *= weights

        # Save to disk
        print(f'{file_no} / {n_files}: Saving...          ', end='\r')
        header = (f'Output from {__file__}.apply_bin_weights for input {input_file}, '
                  f'weights_path {weights_path}, at {time.strftime("%c")}')
        output_path = input_file.replace(input_dir, output_dir)
        np.savez_compressed(output_path, cov_block=block, header=header, prev_headers=prev_headers)

        print(f'{file_no} / {n_files}: Saved {output_path}')


def apply_mix_weights(input_dir, output_dir, filemask, weights_path):
    """
    Apply mixing weights to each block of the connected non-Gaussian covariance matrix.

    Args:
        input_dir (str): Path to input blocks, as output by apply_bin_weights.
        output_dir (str): Path to save output blocks to.
        filemask (str): Glob filemask for input files within input directory.
        weights_path (str): Path to mixing weights as obtained using get_mix_weights.
    """

    # Load weights
    with np.load(weights_path) as data:
        weights = data['eff_fsky2']

    # Loop over files matching mask
    input_files = glob.glob(os.path.join(input_dir, filemask))
    n_files = len(input_files)
    for file_no, input_file in enumerate(input_files, 1):

        # Load block and header trace
        print(f'{file_no} / {n_files}: Loading...', end='\r')
        with np.load(input_file) as data:
            block = data['cov_block']
            prev_headers = [data['header'], *data['prev_headers']]

        # Apply weights
        print(f'{file_no} / {n_files}: Applying weights...', end='\r')
        block *= weights

        # Save to disk
        print(f'{file_no} / {n_files}: Saving...          ', end='\r')
        header = (f'Output from {__file__}.apply_mix_weights for input {input_file}, '
                  f'weights_path {weights_path}, at {time.strftime("%c")}')
        output_path = input_file.replace(input_dir, output_dir)
        np.savez_compressed(output_path, cov_block=block, header=header, prev_headers=prev_headers)

        print(f'{file_no} / {n_files}: Saved {output_path}')
