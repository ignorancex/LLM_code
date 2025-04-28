"""
Module to calculate Gaussian covariance using the Improved NKA method implemented using NaMaster.
"""

import time
from collections import namedtuple

import healpy as hp
import numpy as np
import pymaster as nmt


# Simple class to keep this understandable
PowerSpectrum = namedtuple('PowerSpectrum', ['zbin_1', 'zbin_2'])


def spectrum_from_fields(zbin_a, zbin_b):
    """
    Returns the PowerSpectrum of the supplied bins in the correct order for indexing within this module,
    i.e. lower redshift bin first.

    Note that this is unrelated to the ordering when loading from CosmoSIS output.

    Args:
        zbin_a (int): One redshift bin.
        zbin_b (int): The other redshift bin.

    Returns:
        PowerSpectrum: PowerSpectrum corresponding to the two given redshift bins.
    """
    if zbin_a < zbin_b:
        return PowerSpectrum(zbin_1=zbin_a, zbin_2=zbin_b)
    else:
        return PowerSpectrum(zbin_1=zbin_b, zbin_2=zbin_a)


def load_cls(signal_paths, noise_paths, lmax_in, lmax_out, signal_lmin, noise_lmin):
    """
    Load a list of power spectra, adding noise as appropriate. If any supplied path is None the corresponding
    signal or noise power spectrum is assumed to be zero.

    Args:
        signal_paths (list): List of paths to signal power spectra - if any is None it is taken to be zero.
        noise_paths (list): List of paths to noise power spectra - if any is None it is taken to be zero.
        lmax_in (int): Maximum l to read in.
        lmax_out (int): Maximum l to output. If > lmax_in, power spectra are padded with zeros.
        signal_lmin (int): If > 0, signal power spectra are padded with zeros to start at l = 0.
        noise_lmin (int): If > 0, noise power spectra are padded with zeros to start at l = 0.

    Returns:
        list: List of signal + noise power spectra in the same order they were supplied in.
    """

    # If a signal or noise path is None then just use zeros
    zero_cl = np.zeros(lmax_out + 1)
    zero_pad = np.zeros(lmax_out - lmax_in) if lmax_out > lmax_in else []

    if lmax_in > lmax_out:
        lmax_in = lmax_out

    # Load Cls with appropriate padding and add signal and noise
    combined_cls = []
    for signal_path, noise_path in zip(signal_paths, noise_paths):
        signal_cl = (np.concatenate((np.zeros(signal_lmin),
                                     np.loadtxt(signal_path, max_rows=(lmax_in - signal_lmin + 1)), zero_pad))
                     if signal_path else zero_cl)
        noise_cl = (np.concatenate((np.zeros(noise_lmin),
                                    np.loadtxt(noise_path, max_rows=(lmax_in - noise_lmin + 1)), zero_pad))
                    if noise_path else zero_cl)
        combined_cls.append(signal_cl + noise_cl)

    return combined_cls


def get_cov_blocks(n_zbin, she_she_input_filemask, lmax_in, lmin_in, she_nl_path, noise_lmin, mask_path, nside,
                   lmax_mix, lmax_out, lmin_out, save_path):
    """
    Use NaMaster with the "improved NKA" method (Nicola et al. arXiv:2010.09717) to calculate N-bin shear covariance,
    with each block saved separately to disk.

    Args:
        n_zbin (int): Number of redshift bins.
        she_she_input_mask (str): Path to input shear-shear noise power spectra,
                                  with placeholders for {hi_zbin} and {lo_zbin}.
        lmax_in (int): Maximum l to read from input power spectra.
        lmin_in (int): l corresponding to first line in input power spectra.
        she_nl_path (str): Path to shear noise power spectrum, or None for no noise.
        noise_lmin (int): Minimum l for the shear noise power spectrum (0 for no noise).
        mask_path (str): Path to mask fits file, or None for full sky.
        nside (int): Healpix mask resolution to use - mask will be up/downgraded to this resolution.
        lmax_mix (int): Maximum l to account for mixing to/from.
        lmax_out (int): Maximum l to include in the covariance.
        lmin_out (int): Minimum l to include in the covariance.
        save_path (str): Path to save each block to, with placeholders for {spec1_idx} and {spec2_idx}.
    """

    if '{spec1_idx}' not in save_path or '{spec2_idx}' not in save_path:
        raise ValueError('save_path must include {spec1_idx} and {spec2_idx}')

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

    # Calculate spin-2-2 mixing matrices
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrices at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)

    # Generate list of target power spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')
    z_bins = list(range(1, n_zbin + 1))
    spectra = [PowerSpectrum(z_bins[row], z_bins[row + diag])
               for diag in range(n_zbin) for row in range(n_zbin - diag)]

    # Generate list of sets of mode-coupled theory Cls corresponding to the target spectra
    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        zbins = (spec.field_1.zbin, spec.field_2.zbin)

        # Noise should only be applied to auto-spectra
        nl_path = she_nl_path if zbins[0] == zbins[1] else None

        # Get paths of signal and noise spectra to load: EE, EB, BE, BB
        signal_paths = [she_she_input_filemask.format(hi_zbin=max(zbins), lo_zbin=min(zbins)), None, None, None]
        noise_paths = [nl_path, None, None, nl_path]

        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_mix, lmin_in, noise_lmin)

        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')
        assert len(uncoupled_theory_cls) == 4
        coupled_cls = workspace_spin22.couple_cell(uncoupled_theory_cls)
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_mix)

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(spectra):
        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Identify the four power spectra we need to calculate this covariance
            a1b1 = spectrum_from_fields(spec_a.field_1, spec_b.field_1)
            a1b2 = spectrum_from_fields(spec_a.field_1, spec_b.field_2)
            a2b1 = spectrum_from_fields(spec_a.field_2, spec_b.field_1)
            a2b2 = spectrum_from_fields(spec_a.field_2, spec_b.field_2)

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]
            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            spin = 2
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin, spin, spin, spin, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_spin22, workspace_spin22, coupled=True)
            assert np.all(np.isfinite(cl_cov)), cl_cov

            # Extract the part of the covariance we want, which is the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            cl_cov = cl_cov.reshape((lmax_mix + 1, len(coupled_theory_cls[spec_a_idx]),
                                     lmax_mix + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov[:, 0, :, 0]
            cl_cov = cl_cov[lmin_out:(lmax_out + 1), lmin_out:(lmax_out + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            n_ell_out = lmax_out - lmin_out + 1
            assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)
            save_path = save_path.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_cov_blocks for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, she_she_input_filemask = {she_she_input_filemask}, lmax_in = {lmax_in}, '
                      f'lmin_in = {lmin_in}, she_nl_path = {she_nl_path}, mask_path = {mask_path}, nside = {nside}, '
                      f'lmax_mix = {lmax_mix}, lmin_out = {lmin_out}, lmax_out = {lmax_out}; at {time.strftime("%c")}')

            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx, header=header)
            print(f'Saved {save_path} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')
