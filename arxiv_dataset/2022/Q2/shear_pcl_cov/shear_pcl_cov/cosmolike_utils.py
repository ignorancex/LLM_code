"""
Contains utility functions for `CosmoLike/CosmoCov`.
"""

import glob
import os
import time

import healpy as hp
import numpy as np
import scipy.stats


def get_mask_cl(mask_path, save_path, lmax, nside):
    """
    Calculate mask power spectrum and save in format suitable for input to CosmoLike.

    Args:
        mask_path (str): Path to mask fits file.
        save_path (str): Path to save mask power spectrum to.
        lmax (int): Maximum l.
        nside (int): Healpix resolution to use (mask will be up/downgraded to this).
    """

    print('Loading mask')
    mask = hp.fitsfunc.read_map(mask_path, dtype=float)

    print('Up/downgrading')
    mask = hp.pixelfunc.ud_grade(mask, nside)

    print('Calculating fsky')
    fsky = np.mean(mask)
    print(f'fsky = {fsky}')

    print('Measuring power spectrum')
    cl = hp.sphtfunc.anafast(mask, lmax=lmax)

    print('Saving power spectrum')
    ell = np.arange(lmax + 1)
    to_save = np.column_stack((ell, cl))
    fmt = ['%d', '%.6e'] # format for input to CosmoLike
    np.savetxt(save_path, to_save, fmt=fmt)
    print('Saved', save_path)


def create_nz(save_path, z_min=0, z_max=5.3423, z_steps=10000, z_mean=None, z_std=0.3):
    """
    Create a N(z) file suitable for input to CosmoLike, using a Gaussian redshift distribution. Parameters default to
    match the Takahashi et al. HSC simulations with 5 bins.

    Args:
        save_path (str): Path to save output to.
        z_min (float, optional): Minimum z value (default 0).
        z_max (float, optional): Maximum z value (default 5.3423).
        z_steps (int, optional): Number of steps to use between z_min and z_max (default 10000).
        z_mean (list, optional): Mean z value of each bin (default [0.65, 0.95, 1.25, 1.55, 1.85]).
        z_std (float, optional): Standard deviation in z of every bin (default 0.3).
    """

    # Default bin centres
    if z_mean is None:
        z_mean = [0.65, 0.95, 1.25, 1.55, 1.85]

    # Generate and save
    z = np.linspace(z_min, z_max, z_steps)
    nz = [scipy.stats.norm.pdf(z, loc=loc, scale=z_std) for loc in z_mean]
    to_save = np.column_stack((z, *nz))
    np.savetxt(save_path, to_save)
    print('Saved ' + save_path)


def txt_to_npz(input_filemask, fsky=None):
    """
    Convert all .txt files matching the input filemask to .npz files (as a copy, without deleting the original).
    Optionally apply a 1/fsky factor.

    Args:
        input_filemask (str): glob filemask matching input files.
        fsky (float, optional): Sky fraction - if supplied, matrix will be multiplied by 1/fsky. This should generally \
                                only be used for the connected non-Gaussian covariance, and care must be taken to not \
                                account for it twice (here and elsewhere).
    """

    # Loop over files matching mask
    input_files = glob.glob(input_filemask)
    n_files = len(input_files)
    for file_no, input_file in enumerate(input_files, 1):

        # Load original header and array
        print(f'{file_no} / {n_files}: Loading...', end='\r')
        with open(input_file) as f:
            orig_header = f.readline().strip('# |\n')
        cov = np.loadtxt(input_file)

        # Apply 1/fsky
        if fsky is not None:
            cov /= fsky

        # Save to disk
        print(f'{file_no} / {n_files}: Saving... ', end='\r')
        header = f'Output from {__file__} for input {input_file} at {time.strftime("%c")}'
        output_path = input_file.replace('.txt', '.npz')
        np.savez_compressed(output_path, cov=cov, header=header, orig_header=orig_header)

        print(f'{file_no} / {n_files}: Saved {output_path}')


def delete_txt(input_filemask):
    """
    Delete all .txt files matching the input filemask if the corresponding .npz file is present.

    Args:
        input_filmask (str): glob filemask matching input files.
    """

    # Loop over files matching mask
    txt_files = glob.glob(input_filemask)
    n_files = len(txt_files)
    not_deleted = []
    for file_no, txt_path in enumerate(txt_files, 1):

        # If .npz is present, delete
        npz_path = txt_path.replace('.txt', '.npz')
        if os.path.isfile(npz_path):
            print(f'{file_no} / {n_files}: Deleting {txt_path}...', end='\r')
            os.remove(txt_path)
            print(f'{file_no} / {n_files}: Deleted {txt_path}    ')
        else:
            print(f'{file_no} / {n_files}: {npz_path} is not present, so not deleting {txt_path}')
            not_deleted.append(txt_path)

    print('Done')
    print(f'{len(not_deleted)} files not deleted:', not_deleted)
