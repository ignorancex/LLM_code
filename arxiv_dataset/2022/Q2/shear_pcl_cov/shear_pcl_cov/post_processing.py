"""
Contains functions for post-processing of covariance matrices/blocks.
"""

import glob
import os.path
import time

import healpy as hp
import numpy as np
import pymaster as nmt


def get_mixmat(mask_path, nside, lmax_mix, lmax_out, save_path):
    """
    Produce the EE->EE mixing matrix for a given mask.

    Args:
        mask_path (str): Path to mask fits file, or set to None for full sky.
        nside (int): Healpix map resolution to use - input mask will be up/downgraded to this resolution.
        lmax_mix (int): Maximum l to include mixing to/from.
        lmax_out (int): Maximum l to support output to.
        save_path (str): Path to save mixing matrix as a .npz file.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float, verbose=False), nside)
    else:
        print('Full sky')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))
    fsky = np.mean(mask)
    print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_mix, 1)

    # Calculate full mixing matrix for spin  2-2
    print(f'Calculating mixing matrix at {time.strftime("%c")}')
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True)
    workspace_spin22 = nmt.NmtWorkspace()
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)

    # Extract the relevant block
    # For 2-2 there are 4x4 elements per l, ordered EE, EB, BE, BB.
    # We only need EE->EE, so select every 4th row and 1st column from each block
    print('Extracting mixing matrix block')
    mixmats_spin22 = workspace_spin22.get_coupling_matrix()
    mixmat_ee_to_ee = mixmats_spin22[::4, ::4]

    # Trim to give lmax_out output
    assert mixmat_ee_to_ee.shape == (lmax_mix + 1, lmax_mix + 1)
    mixmat_ee_to_ee = mixmat_ee_to_ee[:(lmax_out + 1), :]

    # Check shape and save
    header = (f'EE->EE mixing matrix indexed as [l1, l2]. Output from {__file__}.get_mixmat for input parameters '
              f'mask_path = {mask_path}, nside = {nside}, lmax_mix = {lmax_mix}, lmax_out = {lmax_out} '
              f'at {time.strftime("%c")}')
    np.savez_compressed(save_path, mixmat_ee_to_ee=mixmat_ee_to_ee, header=header)
    print('Saved ' + save_path)


def mix_blocks(input_dir, input_filemask, mixmat_path, output_dir, lmin, lmax_in, lmax_out):
    """
    Apply a mixing matrix to all covariance blocks inside the input directory matching the input filemask.

    Args:
        input_dir (str): Path to input directory.
        input_filemask (str): Glob filemask for input files within input directory (excluding path to directory).
        mixmat_path (str): Path to mixing matrix.
        output_dir (str): Path to output directory.
        lmin (int): Minimum l in input covariance blocks.
        lmax_in (int): Maximum l in input covariance blocks.
        lmax_out (int): Maximum l in output.
    """

    n_ell_in = lmax_in - lmin + 1
    n_ell_out = lmax_out - lmin + 1

    # Load mixing matrix and trim to lmin
    print('Loading mixing matrix')
    with np.load(mixmat_path) as data:
        mixmat = data['mixmat_ee_to_ee'][lmin:, lmin:]
    assert mixmat.shape == (n_ell_out, n_ell_in)

    # Loop over files matching input mask
    input_files = glob.glob(os.path.join(input_dir, input_filemask))
    n_files = len(input_files)
    print(f'{n_files} found')
    for file_no, input_path in enumerate(input_files, 1):

        # Load unmixed covariance
        print(f'{file_no} / {n_files}: Loading')
        with np.load(input_path) as data:
            cov_unmixed = data['cov']
            prev_headers = [data['header'], data['orig_header']]
        assert cov_unmixed.shape == (n_ell_in, n_ell_in)

        # Apply mixing matrix
        print(f'{file_no} / {n_files}: Mixing')
        cov_mixed = mixmat @ cov_unmixed @ mixmat.T
        assert cov_mixed.shape == (n_ell_out, n_ell_out)
        assert np.all(np.isfinite(cov_mixed))

        # Save to disk
        print(f'{file_no} / {n_files}: Saving')
        output_path = input_path.replace(input_dir, output_dir)
        header = (f'Output from {__file__}.mix_blocks for input file {input_path}, mixing matrix path {mixmat_path}, '
                  f'lmin {lmin},  lmax_in {lmax_in}, lmax_out {lmax_out}, at {time.strftime("%c")}.')
        np.savez_compressed(output_path, cov_mixed=cov_mixed, header=header, prev_headers=prev_headers)
        print(f'{file_no} / {n_files}: Saved {output_path}')

    print('Done')


def bin_blocks(input_dir, input_filemask, input_label, binmat_path, output_dir):
    """
    Apply a binning matrix to all covariance blocks inside the input directory matching the input filemask.

    Args:
        input_dir (str): Path to input directory.
        input_filemask (str): Glob filemask for input files within input directory (excluding path to directory).
        input_label (str): Label for covariance block within input .npz file. Should be 'cov_block' for Cov_G blocks
                           output by gaussian_cov.get_cov_blocks, or 'cov_mixed' for the output from mix_blocks.
        binmat_path (str): Path to binning matrix.
        output_dir (str): Path to output directory.
    """

    # Load binning matrix
    print('Loading binning matrix')
    with np.load(binmat_path) as data:
        pbl = data['pbl']

    # Loop over files matching input mask
    input_files = glob.glob(os.path.join(input_dir, input_filemask))
    n_files = len(input_files)
    print(f'{n_files} found')
    for file_no, input_path in enumerate(input_files, 1):

        # Load unbinned covariance block
        print(f'{file_no} / {n_files}: Loading')
        with np.load(input_path) as data:
            cov_unbinned = data[input_label]
            prev_headers = [data['header']]
            if 'prev_headers' in data:
                prev_headers.extend(data['prev_headers'])

        # Apply binning matrix
        print(f'{file_no} / {n_files}: Binning')
        cov_binned = pbl @ cov_unbinned @ pbl.T
        assert np.all(np.isfinite(cov_binned))

        # Save to disk
        print(f'{file_no} / {n_files}: Saving')
        output_path = input_path.replace(input_dir, output_dir)
        header = (f'Output from {__file__}.bin_blocks for input file {input_path}, '
                  f'binning matrix path {binmat_path}, at {time.strftime("%c")}.')
        np.savez_compressed(output_path, cov_binned=cov_binned, header=header, prev_headers=prev_headers)
        print(f'{file_no} / {n_files}: Saved {output_path}')

    print('Done')


def combine_blocks(input_filemask, input_label, save_path, n_spec, n_bp):
    """
    Combine covariance blocks into a full covariance matrix.

    Args:
        input_filemask (str): Filemask for input blocks, with {spec1} and {spec2} placeholders for the indices of the
                              two power spectra.
        input_label (str): Label for covariance block within input .npz file. Should be 'cov_binned' for output from
                           bin_blocks.
        save_path (str): Path to save output covariance matrix to.
        n_spec (int): Total number of power spectra.
        n_bp (int): Number of bandpowers.
    """

    # Preallocate full matrix
    n_data = n_spec * n_bp
    cov = np.full((n_data, n_data), np.nan)

    # Loop over blocks and insert into matrix
    for spec1 in range(n_spec):
        for spec2 in range(spec1 + 1):
            print(f'Loading spec1 {spec1}, spec2 {spec2}')
            with np.load(input_filemask.format(spec1=spec1, spec2=spec2)) as data:
                block = data[input_label]
            cov[(spec1 * n_bp):((spec1 + 1) * n_bp), (spec2 * n_bp):((spec2 + 1) * n_bp)] = block

    # Reflect to fill remaining elements, and check symmetric
    cov = np.where(np.isnan(cov), cov.T, cov)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=0)

    # Save to disk
    header = f'Output from {__file__}.combine_blocks for input_filemask {input_filemask} at {time.strftime("%c")}'
    np.savez_compressed(save_path, cov=cov, header=header)
    print('Saved ' + save_path)


def get_composite_covs(cov_g_path, cov_ss_path, cov_cng_path, output_path):
    """
    Form composite covariances from different combinations of G, SS and CNG, and save to disk.

    Args:
        cov_g_path (str): Path to Gaussian covariance.
        cov_ss_path (str): Path to super-sample covariance.
        cov_cng_path (str): Path to connected non-Gaussian covariance.
        output_path (str): Output path, with {label} placeholder which will be replaced by g, g_ss, g_cng or tot.
    """

    if '{label}' not in output_path:
        raise ValueError('output_path should contain {label} placeholder')

    # Load the three covariance contributions
    with np.load(cov_g_path) as data:
        cov_g = data['cov']
    with np.load(cov_ss_path) as data:
        cov_ss = data['cov']
    with np.load(cov_cng_path) as data:
        cov_cng = data['cov']

    # Save the different composite covariances to disk
    header_base = (f'Output from {__file__}.get_composite_covs for input cov_g_path = {cov_g_path}, '
                   f'cov_ss_path = {cov_ss_path}, cov_cng_path = {cov_cng_path}, at {time.strftime("%c")}.')
    cov_g_save_path = output_path.format(label='g')
    np.savez_compressed(cov_g_save_path, cov=cov_g, header=('Cov_G. ' + header_base))
    print('Saved ' + cov_g_save_path)
    cov_g_ss_save_path = output_path.format(label='g_ss')
    np.savez_compressed(cov_g_ss_save_path, cov=(cov_g + cov_ss), header=('Cov_G + Cov_SS. ' + header_base))
    print('Saved ' + cov_g_ss_save_path)
    cov_g_cng_save_path = output_path.format(label='g_cng')
    np.savez_compressed(cov_g_cng_save_path, cov=(cov_g + cov_cng), header=('Cov_G + Cov_CNG. ' + header_base))
    print('Saved ' + cov_g_cng_save_path)
    cov_tot_save_path = output_path.format(label='tot')
    np.savez_compressed(cov_tot_save_path, cov=(cov_g + cov_ss + cov_cng),
                        header=('Cov_G + Cov_SS + Cov_CNG. ' + header_base))
    print('Saved ' + cov_tot_save_path)
