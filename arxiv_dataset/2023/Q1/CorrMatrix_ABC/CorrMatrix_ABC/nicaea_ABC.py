#!/usr/bin/env python

"""
:Name: nicaea_ABC.py

:Description: Python interface to nicaea.
Functions are copied from ~/python/nicaea.py

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

:Date: 2017

:Package: ABC
"""

import numpy as np

from CorrMatrix_ABC import covest

from astropy.io import ascii


def run_nicaea(lmin, lmax, nell, par_name=None, par_val=None, verbose=False):
    """Calls nicaea.

    Parameters
    ----------
    lmin: double
        minimum ell
    lmax: double
        maximume ell
    nell: int
        number of ell modes
    par_name: array of string, optional, default=None
        parameter names for on-the-fly updates
    par_val: array of float, optiona, default=None
	    parameter values corresponding to par_nam 
    verbose : bool, optional, default=False
        verbose output if True

    Returns
    -------
    err: int
        error code
    C_ell_name: string
        name of C(ell) file = P_kappa<out_suf>
    """

    Lstr = '-L \'{} {} {}\''.format(lmin, lmax, nell)
    parstr = '' 
    if par_name is not None:
        out_suf = ''
        for i, name in enumerate(par_name):
            parstr = '{} --{} {}'.format(parstr, name, par_val[i])
            out_suf = '{}_{}'.format(out_suf, par_val[i])
        out_suf_str = ' --out_suf {}'.format(out_suf)
    else:
        out_suf     = ''
        out_suf_str = ''

    err = covest.run_cmd('lensingdemo -D 0 {} {} {} -q -H 1 --linlog LIN'.format(Lstr, parstr, out_suf_str), verbose=verbose, stop=True)

    C_ell_name = 'P_kappa{}'.format(out_suf)

    if err != 0:
        print('Nicaea returned with error code {}'.format(err))

    return err, C_ell_name


def read_Cl(path, fname):
    """Read and return theoretical power spectrum.

    Parameters
    ----------
    path: string
        path to nicaea output file
    fname: string
        file name

    Returns
    -------
    ell: array of double
        angular Fourier modes
    C_ell: array of double
        power spectrum
    """

    dat   = ascii.read('{}/{}'.format(path, fname))
    ell   = dat['l'].data
    c_ell = dat['P_k^00(l)'].data

    return ell, c_ell


def create_link(dst, path_cosmo):
    """Create link to file 'path_cosmo/dst'.

    Parameters
    ----------

    dst: string
        Destination file name
    path_cosmo: string
        Directory name

    Returns
    -------
    None
    """

    if os.path.isfile(dst):
        pass
    else:
        src = '{}/{}'.format(path_cosmo, dst)
        if os.path.isfile(src):
            os.symlink(src, dst)
        else:
            stuff.error('File {} not found at dir {}'.format(dst, path_cosmo))


def create_links_to_cosmo(path_to_cosmo):
    """Create links to all cosmo files.

    Parameters
    ----------
    path_to_cosmo: string
        Directory name

    Returns
    -------
    None
    """

    files = ['cosmo.par', 'cosmo_lens.par', 'nofz.par']
    for dst in files:
        create_link(dst, path_cosmo)

    files = glob.glob('{}/nofz_*'.format(path_cosmo))
    for dst in files:
        create_link(os.path.basename(dst), path_cosmo)


def Fisher_ana_wl(ell, f_sky, sigma_eps, nbar_rad2, Omega_m_fid, sigma_8_fid, cov_model,
                    ellmode='log', templ_dir='.'):
    """Return Fisher matrix for weak-lensing model with parameters Omega_m and sigma_8.
    """

    par_name = ['Omega_m', 'sigma_8']
    lmin = ell[0]
    lmax = ell[-1]
    nell = len(ell)

    h = 0.01

    # Perturbed models for derivatives
    par_val = [Omega_m_fid, sigma_8_fid + h]
    err, path_Cell_ps8 = run_nicaea(lmin, lmax, nell, par_name=par_name, par_val=par_val, verbose=True)
    ell, Cell_ps8 = read_Cl('.', path_Cell_ps8)

    par_val = [Omega_m_fid, sigma_8_fid - h]
    err, path_Cell_ms8 = run_nicaea(lmin, lmax, nell, par_name=par_name, par_val=par_val, verbose=True)
    ell, Cell_ms8 = read_Cl('.', path_Cell_ms8)

    par_val = [Omega_m_fid + h, sigma_8_fid]
    err, path_Cell_pOm = run_nicaea(lmin, lmax, nell, par_name=par_name, par_val=par_val, verbose=True)
    ell, Cell_pOm = read_Cl('.', path_Cell_pOm)

    par_val = [Omega_m_fid - h, sigma_8_fid]
    err, path_Cell_mOm = run_nicaea(lmin, lmax, nell, par_name=par_name, par_val=par_val, verbose=True)
    ell, Cell_mOm = read_Cl('.', path_Cell_mOm)

    # Derivatives
    dCell_dOm = (Cell_pOm - Cell_mOm) / (2 * h)
    dCell_ds8 = (Cell_ps8 - Cell_ms8) / (2 * h)

    # Fiducial model for Gaussian covariance
    par_val = [Omega_m_fid, sigma_8_fid]
    err, path_Cell = run_nicaea(lmin, lmax, nell, par_name=par_name, par_val=par_val, verbose=True)
    ell, Cell = read_Cl('.', path_Cell)
    D = covest.get_cov_Gauss(ell, Cell, f_sky, sigma_eps, nbar_rad2)
    D = np.diag(D)

    # MKDEBUG TODO: clean up this duplicate code, see Fisher_ana_quad...
    if cov_model == 'Gauss':
        Psi = np.diag([1.0 / d for d in D])
    elif cov_model == 'Gauss+SSC_BKS17':
        ell_mode = covest.get_ell_mode(ell)
        if ell_mode == 'log':
            cov_SSC_base = 'cov_SSC_rel_log'
        elif ell_mode == 'lin':
            cov_SSC_base = 'cov_SSC_rel_lin'
        cov_SSC_path = '{}/{}.txt'.format(templ_dir, cov_SSC_base)
        func_SSC = 'BKS17'
        cov_SSC = covest.get_cov_SSC(ell, Cell, cov_SSC_path, 'BKS17')
        cov = np.diag(D) + cov_SSC
        Psi = np.linalg.inv(cov)

    [da2, db2], det = covest.Fisher_num(dCell_dOm, dCell_ds8, Psi)

    return np.sqrt([da2, db2]), det

