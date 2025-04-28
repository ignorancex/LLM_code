"""
Functions relating to processing the Takahashi et al HSC simulations, available at
http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/.
"""

import signal
import os.path
import time
import traceback
import urllib.request
import warnings
from collections import namedtuple

import healpy as hp
import numpy as np
import scipy.stats

import likelihood


# Z slices: names, redshifts and URL
# from http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sourceplane_redshift.html
ZSlice = namedtuple('ZSlice', ['id', 'z'])
Z_SLICES = [
    ZSlice('zs1', 0.0506),
    ZSlice('zs2', 0.1023),
    ZSlice('zs3', 0.1553),
    ZSlice('zs4', 0.2097),
    ZSlice('zs5', 0.2657),
    ZSlice('zs6', 0.3233),
    ZSlice('zs7', 0.3827),
    ZSlice('zs8', 0.4442),
    ZSlice('zs9', 0.5078),
    ZSlice('zs10', 0.5739),
    ZSlice('zs11', 0.6425),
    ZSlice('zs12', 0.7140),
    ZSlice('zs13', 0.7885),
    ZSlice('zs14', 0.8664),
    ZSlice('zs15', 0.9479),
    ZSlice('zs16', 1.0334),
    ZSlice('zs17', 1.1233),
    ZSlice('zs18', 1.2179),
    ZSlice('zs19', 1.3176),
    ZSlice('zs20', 1.4230),
    ZSlice('zs21', 1.5345),
    ZSlice('zs22', 1.6528),
    ZSlice('zs23', 1.7784),
    ZSlice('zs24', 1.9121),
    ZSlice('zs25', 2.0548),
    ZSlice('zs26', 2.2072),
    ZSlice('zs27', 2.3704),
    ZSlice('zs28', 2.5455),
    ZSlice('zs29', 2.7338),
    ZSlice('zs30', 2.9367),
    ZSlice('zs31', 3.1559),
    ZSlice('zs32', 3.3932),
    ZSlice('zs33', 3.6507),
    ZSlice('zs34', 3.9309),
    ZSlice('zs35', 4.2367),
    ZSlice('zs36', 4.5712),
    ZSlice('zs37', 4.9382),
    ZSlice('zs38', 5.3423)
]
SLICE_URL = ('http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/{subdir}/nres12/'
             'allskymap_nres12{realisation_id}.{slice_id}.mag.dat')
NSIDE = 4096


class DLTimeOutError(Exception):
    """
    Exception to raise when a download times out.
    """
    pass


def timeout_download(*_):
    """
    Raise a download timeout exception.

    Raises:
        DLTimeOutError: download timeout exception.
    """
    raise DLTimeOutError('Download timed out')


def read_slice(input_path, verbose=True):
    """
    Read gamma1 and gamma2 from a slice of the Takahashi et al. HSC simulations.

    Based on the script by Toshiya Namikawa / Ken Osato available at
    http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/nres12.html.

    Args:
        input_path (str): Path to slice.
        verbose (bool, optional): Whether to output progress (default True).

    Returns:
        (1D numpy array, 1D numpy array): (gamma1, gamma2) healpix maps.
    """

    # Open file
    with open(input_path, 'rb') as f:

        # Skip 4 bytes, read nside and npix, skip another 2*4 bytes
        f.seek(4, 1)
        nside = np.squeeze(np.fromfile(f, dtype='int32', count=1))
        npix = np.squeeze(np.fromfile(f, dtype='int64', count=1))
        assert nside == 4096 # This logic only works for nside 4096
        assert hp.pixelfunc.nside2npix(nside) == npix
        f.seek(8, 1)
        if verbose:
            print(f'nside: {nside} npix: {npix}')

        # Next npix*4 bytes are kappa, so skip those
        f.seek(4 * npix, 1)

        # Skip 2*4 bytes between kappa and gamma1
        f.seek(8, 1)

        # Next npix*4 bytes are gamma1
        if verbose:
            print('Reading gamma1...')
        gamma1 = np.fromfile(f, dtype='float32', count=npix)

        # Skip 2*4 bytes between gamma1 and gamma2
        f.seek(8, 1)

        # Next npix*4 bytes are gamma2
        if verbose:
            print('Reading gamma2...')
        gamma2 = np.fromfile(f, dtype='float32', count=npix)

    return gamma1, gamma2


def process_realisation(realisation_id, z_mean, z_std, mask_path, slice_path, lmax, lmin, cl_save_path,
                        delete_after_use=True):
    """
    Process a single realisation of the Takhashi et al. HSC simulations.

    Download all slices, combine into tomographic bins and measure the power spectra.

    Args:
        realisation_id (string): ID of the realisation, from 'r000' to 'r107'.
        z_mean (list): List containing the mean redshift of each bin.
        mask_path (str): Path to mask fits file, or None for full sky.
        slice_path (str): Path to save each slice locally, including placeholders for {realisation_id} and {slice_id}.
        lmax (int): Maximum l to measure.
        lmin (int): Minimum l to measure.
        cl_save_path (str): Path to save measured power spectra, including placeholder for {realisation_id}.
        delete_after_use (bool, optional): If True, delete each downloaded slice file once it has been used
                                           (default True).
    """

    # Set timeout function as alarm signal handler
    signal.signal(signal.SIGALRM, timeout_download)

    # For each redshift bin, form a Gaussian distribution having mean of the relevant bin and a fixed std
    z_dists = [scipy.stats.norm(loc=mean, scale=z_std) for mean in z_mean]

    # Calculate weights for each of the 38 slices for each bin by evaluating the pdf at each slice redshift
    # Weights array is indexed [bin_idx, slice_idx]
    slice_z = np.array([z_slice.z for z_slice in Z_SLICES])
    slice_weights = np.array([z_dist.pdf(slice_z) for z_dist in z_dists])

    # Reweight so they sum to 1 per bin
    slice_weights /= np.sum(slice_weights, axis=1)[:, np.newaxis]
    assert np.allclose(np.sum(slice_weights, axis=1), 1)

    # Start with a zero spin-2 map of the relevant resolution, per bin
    # Maps are indexed [bin_idx, {0: gamma1, 1: gamma2}, healpix_pixel_idx]
    n_zbin = len(z_mean)
    npix = hp.pixelfunc.nside2npix(NSIDE)
    tomo_maps = np.zeros((n_zbin, 2, npix))

    # Loop over slices
    n_slice = len(Z_SLICES)
    for slice_idx, z_slice in enumerate(Z_SLICES):

        print(f'Processing slice {slice_idx + 1} / {n_slice} at {time.strftime("%c")}')

        # Work out remote and local paths
        slice_path = slice_path.format(realisation_id=realisation_id, slice_id=z_slice.id)
        realisation_no = int(realisation_id[1:])
        assert realisation_id == f'r{realisation_no:03d}'
        subdir = 'sub1' if realisation_no < 54 else 'sub2'
        slice_url = slice_url.format(subdir=subdir, realisation_id=realisation_id, slice_id=z_slice.id)

        # Loop indefinitely to attempt to download file
        complete = False
        attempt_count = 1
        while not complete:
            try:
                print(f'Attempt {attempt_count} to download {slice_url} at {time.strftime("%c")}')
                signal.alarm(30 * 60) # allow 30 mins for the download
                urllib.request.urlretrieve(slice_url, slice_path)
                signal.alarm(0)
                assert os.path.isfile(slice_path)
                print(f'Successfully downloaded to {slice_path} at {time.strftime("%c")}')
                complete = True
            except (urllib.error.ContentTooShortError, DLTimeOutError):
                warnings.warn(f'Exception while attempting download:\n{traceback.format_exc()}')
                print('Trying again...')
                attempt_count += 1

        # Read gamma1 and gamma2 from slice file
        print(f'Reading from {slice_path} at {time.strftime("%c")}')
        gamma1, gamma2 = read_slice(slice_path)

        # Add slice to the cumulative maps, weighted appropriately
        this_slice_weights = slice_weights[:, slice_idx][:, np.newaxis]
        print(f'Adding to maps... 1/2 at {time.strftime("%c")}')
        tomo_maps[:, 0, :] += this_slice_weights * gamma1[np.newaxis, :]
        print(f'Adding to maps... 2/2 at {time.strftime("%c")}')
        tomo_maps[:, 1, :] += this_slice_weights * gamma2[np.newaxis, :]

        # Delete from disk
        if delete_after_use:
            print(f'Deleting {slice_path} at {time.strftime("%c")}')
            os.remove(slice_path)
            print(f'Deleted {slice_path} at {time.strftime("%c")}')
        else:
            print('Not deleting local copy')

        print(f'Finished slice {slice_idx + 1} / {n_slice} at {time.strftime("%c")}')

    # Load and apply mask
    if mask_path is not None:
        print(f'Loading mask at {time.strftime("%c")}')
        mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float, verbose=False), NSIDE)
        print(f'Applying mask at {time.strftime("%c")}')
        tomo_maps *= mask[np.newaxis, np.newaxis, :]
    else:
        print('Not applying mask; using full sky')

    # Measure power spectra -
    # First convert to E-mode alms, indexed [zbin_idx, healpix_alm_idx]
    # For spin-2 SHTs in healpix a dummy 'T' field is required
    t_map = np.zeros(npix)
    alms = np.full((n_zbin, hp.sphtfunc.Alm.getsize(lmax)), np.nan, dtype=complex)
    for zbin_idx, (gamma1_map, gamma2_map) in enumerate(tomo_maps):
        print(f'Converting to alms {zbin_idx + 1} / {n_zbin} at {time.strftime("%c")}')
        _, e_alm, _ = hp.sphtfunc.map2alm([t_map, gamma1_map, gamma2_map], lmax=lmax, pol=True)
        alms[zbin_idx, :] = e_alm
    assert np.all(np.isfinite(alms))
    del tomo_maps
    print(f'Finished converting to alms at {time.strftime("%c")}')

    # Calculate all EE auto- and cross-spectra
    print(f'Calculating power spectra at {time.strftime("%c")}')
    cl = hp.sphtfunc.alm2cl(alms)[:, lmin:]
    del alms

    # Save Cls to disk
    print(f'Saving to disk at {time.strftime("%c")}')
    cl_save_path = cl_save_path.format(realisation_id=realisation_id)
    header = (f'Output from {__file__}.process_realisation for realisation_id {realisation_id}, nside {NSIDE}, '
              f'lmax {lmax}, lmin {lmin} at {time.strftime("%c")}')
    np.savez_compressed(cl_save_path, cl=cl, realisation_id=realisation_id, header=header)
    print(f'Saved {cl_save_path} at {time.strftime("%c")}')

    print(f'Done for realisation {realisation_id} at {time.strftime("%c")}')


def loop_realisations(first_real, last_real, z_mean, z_std, mask_path, slice_path, lmax, lmin, cl_save_path,
                      delete_after_use=True):
    """
    Loop over realisations in serial from first_real to last_real (inclusive).

    Args:
        first_real (int): First realisation to process (between 0 and 107 inclusive).
        last_real (int): Last realisation to process (in same range).
        z_mean (list): List containing the mean redshift of each bin.
        mask_path (str): Path to mask fits file, or None for full sky.
        slice_path (str): Path to save each slice locally, including placeholders for {realisation_id} and {slice_id}.
        lmax (int): Maximum l to measure.
        lmin (int): Minimum l to measure.
        cl_save_path (str): Path to save measured power spectra, including placeholder for {realisation_id}.
        delete_after_use (bool, optional): If True, delete each downloaded slice file once it has been used
                                           (default True).
    """

    assert 0 <= first_real <= last_real <= 107

    # Loop over realisations
    realisation_ids = [f'r{r_id:03d}' for r_id in range(first_real, last_real + 1)]
    for realisation_id in realisation_ids:
        print(f'Starting realisation {realisation_id} at {time.strftime("%c")}')
        process_realisation(realisation_id, z_mean, z_std, mask_path, slice_path, lmax, lmin, cl_save_path,
                            delete_after_use)
        print()

    print(f'All done at {time.strftime("%c")}')


def combine_sim_cl(input_filemask, n_real, n_zbin, lmax, lmin, output_path):
    """
    Combine all realisations into a single file.

    Args:
        input_filemask (str): Path to input files, with {realisation} placeholder.
        n_real (int): Number of realisations.
        n_zbin (int): Number of redshift bins.
        lmax (int): Maximum l.
        lmin (int): Minimum l.
        output_path (str): Path to save combined .npz file.
    """

    # Create array to hold all data, indexed [spec_idx, ell_idx, realisation_idx]
    n_spec = n_zbin * (n_zbin + 1) // 2
    n_ell = lmax - lmin + 1
    all_cls = np.full((n_spec, n_ell, n_real), np.nan)

    # Loop over all realisations and extract Cls
    for real in range(n_real):
        input_path = input_filemask.format(realisation=real)
        with np.load(input_path) as data:
            assert data['realisation_id'] == f'r{real:03d}'
            all_cls[:, :, real] = data['cl']
    assert np.all(np.isfinite(all_cls))

    # Save to disk
    header = (f'Output from {__file__}.combine_sim_cl for input parameters input_filemask = {input_filemask}, '
              f'n_real = {n_real}, n_zbin = {n_zbin}, lmax = {lmax}, lmin = {lmin}, output_path = {output_path} '
              f'at {time.strftime("%c")}')
    np.savez_compressed(output_path, cls=all_cls, header=header)
    print('Saved ' + output_path)


def gaussian_sim(cl_in_path, lmax_in, lmin_in, nside, mask_path, lmax_out, lmin_out, n_real, save_path):
    """
    Produce repeated simulated realisations of a single Gaussian field, and save the power spectra to disk.

    Args:
        cl_in_path (str): Path to input power spectrum.
        lmax_in (int): Maximum l to read in.
        lmin_in (int): If > 0, input power spectrum will be padded with zeros to start at l = 0.
        nside (int): Healpix map resolution to use.
        mask_path (str): Path to mask fits file, or None for full sky.
        lmax_out (int): Maximum l to measure.
        lmin_out (int): Minimum l to measure.
        n_real (int): Number of realisations to produce.
        save_path (str): Path to save all power spectra to.
    """

    # Load input Cls, assume zero B-mode and create TT and TE placeholders for healpy
    print('Loading input Cls')
    cl_ee_in = np.concatenate((np.zeros(lmin_in), np.loadtxt(cl_in_path, max_rows=(lmax_in - lmin_in + 1))))
    cl_bb_in = np.zeros_like(cl_ee_in)
    cl_tt_in = np.zeros_like(cl_ee_in)
    cl_te_in = np.zeros_like(cl_ee_in)
    input_cls = [cl_tt_in, cl_ee_in, cl_bb_in, cl_te_in]

    # Load mask
    if mask_path is not None:
        print('Loading mask')
        mask = hp.pixelfunc.ud_grade(hp.fitsfunc.read_map(mask_path, dtype=float, verbose=False), nside)
    else:
        print('Full sky')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))

    # Generate realisations in loop
    n_ell_out = lmax_out - lmin_out + 1
    obs_cls = np.full((n_real, n_ell_out), np.nan)
    for real_idx in range(n_real):
        print(f'Generating realisation {real_idx + 1} / {n_real} at {time.strftime("%c")}')

        # Generate spin-2 map
        t_map, shear1_map, shear2_map = hp.sphtfunc.synfast(input_cls, nside, pol=True, new=True, verbose=False)

        # Apply mask
        shear1_map *= mask
        shear2_map *= mask

        # Measure Cls
        maps = [t_map, shear1_map, shear2_map]
        obs_cl_ee = hp.sphtfunc.anafast(maps, lmax=lmax_out, pol=True)[1, lmin_out:]
        obs_cls[real_idx] = obs_cl_ee

    assert np.all(np.isfinite(obs_cls))

    # Save all Cls to disk
    header = (f'Output from {__file__}.gaussian_sim for nside = {nside}, lmax_out = {lmax_out}, lmin_out = {lmin_out}, '
              f'cl_in_path = {cl_in_path}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, mask_path = {mask_path}, '
              f'n_real = {n_real} at {time.strftime("%c")}')
    np.savez_compressed(save_path, obs_cls=obs_cls, header=header)
    print('Saved ' + save_path)


def get_obs(theory_cl_dir, nl_path, mixmat_path, binmat_path, cov_tot_path, n_zbin, lmax_in, lmin_in, lmax_obs,
            lmin_obs, n_bandpower, save_path):
    """
    Generate a mock observation by sampling from a Gaussian likelihood with the total covariance.

    Args:
        theory_cl_dir (str): Path to directory containing theory shear power spectra.
        nl_path (str): Path to noise power spectrum.
        mixmat_path (str): Path to mixing matrix.
        binmat_path (str): Path to binning matrix.
        cov_tot_path (str): Path to total covariance matrix.
        n_zbin (int): Number of redshift bins.
        lmax_in (int): Maximum l to use as input pre-mixing.
        lmin_int (int): Minimum l to use as input pre-mixing.
        lmax_obs (int): Maximum l to use in the observation.
        lmin_obs (int): Minimum l to use in the observation.
        n_bandpower (int): Number of bandpowers.
        save_path (str): Path to save observed bandpowers to.
    """

    # Load theory Cls
    cl_in = likelihood.load_shear_cls(n_zbin, theory_cl_dir, lmax_in)
    n_spec = n_zbin * (n_zbin + 1) // 2
    n_ell_in = lmax_in - lmin_in + 1
    assert cl_in.shape == (n_spec, n_ell_in)

    # Add noise to auto-spectra
    nl = np.loadtxt(nl_path, max_rows=n_ell_in)
    cl_in[:n_zbin, :] += nl

    # Load and apply mixing matrix
    with np.load(mixmat_path) as data:
        mixmat = data['mixmat_ee_to_ee'][lmin_obs:, lmin_in:] # this mixing matrix starts from 0
    n_ell_obs = lmax_obs - lmin_obs + 1
    assert mixmat.shape == (n_ell_obs, n_ell_in)
    cl_mixed = np.einsum('ij,kj->ki', mixmat, cl_in) # i = l, j = l', k = spec_idx
    assert cl_mixed.shape == (n_spec, n_ell_obs)

    # Load and apply binning matrix
    with np.load(binmat_path) as data:
        binmat = data['pbl']
    assert binmat.shape == (n_bandpower, n_ell_obs)
    bp_exp = np.einsum('bl,kl->kb', binmat, cl_mixed) # b = bandpower, l = l, k = spec_idx
    assert bp_exp.shape == (n_spec, n_bandpower)

    # Reshape into a vector to obtain the mean
    n_data = n_spec * n_bandpower
    bp_exp = np.reshape(bp_exp, (n_data,))

    # Load the total covariance
    with np.load(cov_tot_path) as data:
        cov_tot = data['cov']
    assert cov_tot.shape == (n_data, n_data)

    # Sample from a Gaussian with the mean and covariance to obtain the observation
    obs_bp = scipy.stats.multivariate_normal.rvs(mean=bp_exp, cov=cov_tot)

    # Reshape back to (n_spec, n_bandpower)
    obs_bp = np.reshape(obs_bp, (n_spec, n_bandpower))

    # Save to disk
    header = ('Mock observation drawn from a multivariate Gaussian with the total covariance. '
              f'Output from {__file__}.get_obs for params theory_cl_dir = {theory_cl_dir}, nl_path = {nl_path}, '
              f'mixmat_path = {mixmat_path}, binmat_path = {binmat_path}, cov_tot_path = {cov_tot_path}, '
              f'n_zbin = {n_zbin}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, lmax_obs = {lmax_obs}, '
              f'lmin_obs = {lmin_obs}, n_bandpower = {n_bandpower}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, obs_bp=obs_bp, header=header)
    print('Saved ' + save_path)
