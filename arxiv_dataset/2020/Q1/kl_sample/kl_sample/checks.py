"""

This module contains checks that needs to be performed
to ensure that the input is consistent.

Functions:
 - unused_params(cosmo, settings, path)
 - sanity_checks(cosmo, settings, path)
 - kl_consistent(E, S, N, L, eigval, tol)

"""

import re
import sys
import numpy as np
from astropy.io import fits
import kl_sample.settings as set


# ------------------- Preliminary --------------------------------------------#

def unused_params(cosmo, settings, path):
    """ Check if there are unused.
        In case it prints the parameter on the screen.

    Args:
        cosmo: dictionary containing names, values and mask for
        the cosmological parameters
        settings: dictionary with all the settings used
        path: dictionary containing the paths stored

    Returns:
        None if there are no unused parameters.

    """

    # Join together all the parameters read
    params = \
        np.hstack((cosmo['names'], list(settings.keys()), list(path.keys())))

    # Scroll all the parameter to see if there are unused ones
    with open(path['params']) as fn:
        for line in fn:
            line = re.sub('#.+', '', line)
            if '=' in line:
                name, _ = line.split('=')
                name = name.strip()
                if name not in params:
                    print('Unused parameter: {}'.format(line).rstrip('\n'))
                    sys.stdout.flush()
    return


def sanity_checks(cosmo, settings, path):
    """ Perform different sanity checks on the input
        parameters and data.

    Args:
        cosmo: dictionary containing names, values and mask for
        the cosmological parameters
        settings: dictionary with all the settings used
        path: dictionary containing the paths stored

    Returns:
        None if the checks are passed. Otherwise it raises
        an error.

    """

    # Function to check if a string can be a natural number
    def is_natural(str):
        try:
            if int(str) > 0:
                return True
        except ValueError:
            return False
        return False

    # Checks on cosmological parameters
    for n, par in enumerate(cosmo['params']):
        # Check that the central value is a number
        test = par[1] is not None
        assert test, 'Central value for ' + cosmo['names'][n] + ' is None!'
        # If left and right are the same the prior is considered Gaussian
        # Check that the left bound is either None or smaller than central
        test = par[0] is None or par[0] <= par[1] or par[0] == par[2]
        assert test, 'Central value for {} is lower than the left bound!' \
            ''.format(cosmo['names'][n])
        # Check that the right bound is either None or larger than central
        test = par[2] is None or par[1] <= par[2] or par[0] == par[2]
        assert test, 'Central value for {} is larger than the right bound!' \
            ''.format(cosmo['names'][n])

    # Check sampler options
    test = settings['sampler'] in ['emcee', 'single_point']
    assert test, 'sampler not recognized! Options: emcee, single_point'
    # Check space options
    test = settings['space'] in ['real', 'fourier']
    assert test, 'space not recognized! Options: real, fourier'
    # Check method options
    test = settings['method'] in ['full', 'kl_off_diag', 'kl_diag']
    assert test, 'method not recognized! Options: full, kl_off_diag, kl_diag'
    # Check ell_max
    test = settings['ell_max'] > 100
    assert test, 'Unless you are crazy, choose a larger ell_max!'
    # Check n_sims
    test = \
        settings['n_sims'] in ['auto', 'all'] or is_natural(settings['n_sims'])
    assert test, 'n_sims not recognized! Options: auto, all, positive integer'

    # Checks related to the emcee sampler
    if settings['sampler'] == 'emcee':
        # Check that there are at least 2 varying parameters
        test = len(cosmo['mask'][cosmo['mask']]) > 1
        assert test, 'For emcee the minimum number of varied parameters is 2!'
        # Check n_walkers
        test = settings['n_walkers'] > 0
        assert test, 'n_walkers should be at least 1!'
        # Check n_steps
        test = settings['n_steps'] > 0
        assert test, 'n_steps should be at least 1!'
        # Check n_threads
        test = settings['n_threads'] > 0
        assert test, 'n_threads should be at least 1!'

    # Check data existence
    with fits.open(path['data']) as hdul:
        imgs = [hdul[n].name for n in range(1, len(hdul))]
        # Checks common to real and fourier spaces
        for name in ['PHOTO_Z']:
            test = name in imgs
            assert test, name + ' was not found in the data file!'
        # Checks related to real space
        if settings['space'] == 'real':
            for name in ['XIPM_OBS', 'XIPM_SIM']:
                test = name in imgs
                assert test, name + ' was not found in data file!'
        # Checks related to fourier space
        elif settings['space'] == 'fourier':
            for name in ['ELL', 'CL_EE', 'CL_EE_NOISE', 'CL_SIM_EE']:
                test = name in imgs
                assert test, name + ' was not found in data file!'

        # Check data dimensions
        n_bins = hdul['PHOTO_Z'].shape[0]-1
        # Photo_z
        test = len(hdul['PHOTO_Z'].shape) == 2
        assert test, 'photo_z has wrong dimensions!'
        if settings['space'] == 'real':
            n_theta = len(set.THETA_ARCMIN)
            n_fields = hdul['XIPM_SIM'].shape[0]
            n_sims = hdul['XIPM_SIM'].shape[1]
            # xipm_obs
            test = hdul['XIPM_OBS'].shape == (2, n_theta, n_bins, n_bins)
            assert test, 'xipm_obs has wrong dimensions!'
            test = hdul['XIPM_SIM'].shape == \
                (n_fields, n_sims, 2, n_theta, n_bins, n_bins)
            assert test, 'xipm_sim has wrong dimensions!'
        elif settings['space'] == 'fourier':
            n_ell = hdul['ELL'].shape[0]
            n_fields = hdul['CL_SIM_EE'].shape[0]
            n_sims = hdul['CL_SIM_EE'].shape[1]
            # cl_obs
            test = hdul['CL_EE'].shape == (n_fields, n_ell, n_bins, n_bins)
            assert test, 'CL_EE has wrong dimensions!'
            test = hdul['CL_EE_NOISE'].shape == \
                (n_fields, n_ell, n_bins, n_bins)
            assert test, 'CL_EE_NOISE has wrong dimensions!'
            test = hdul['CL_SIM_EE'].shape == \
                (n_fields, n_sims, n_ell, n_bins, n_bins)
            assert test, 'CL_SIM_EE has wrong dimensions!'

        # if n_sims is natural check that it is smaller than
        # the max n_sims we have
        if is_natural(settings['n_sims']):
            test = int(settings['n_sims']) <= n_sims and \
                int(settings['n_sims']) > 1
            assert test, 'n_sims has to be smaller than {} and larger than 1' \
                ''.format(n_sims)

    # Checks related to the KL transform
    if settings['method'] in ['kl_diag', 'kl_off_diag']:
        # n_kl
        test = settings['n_kl'] > 0 and settings['n_kl'] <= n_bins
        assert test, 'n_kl should be at least 1 and at most {}'.format(n_bins)

    return


# ------------------- Calculations related -----------------------------------#

def kl_consistent(E, S, N, L, eigval, tol):
    """ Check if the calculated KL transform is consistent.

    Args:
        E, S, N, eigval: KL transform, signal, noise,
        eivenvalues respectively.
        tol: tolerance.

    Returns:
        None if the checks are passed. Otherwise it raises
        a warning.

    """

    # Calculate the KL transformed Cl's
    angular_cl = np.array([np.diag(eigval[x]) for x in range(len(S))])

    # First test
    test1 = np.array([np.dot(E[x], S[x]+N[x]) for x in range(len(S))])
    test1 = np.array([np.dot(test1[x], E[x].T) for x in range(len(S))])
    test1 = np.array([abs(test1[x]-angular_cl[x]) for x in range(len(S))])
    test1 = test1[2:].max()/abs(angular_cl[2:]).max()

    # Second test
    test2 = np.array([np.dot(L[x].T, E[x].T) for x in range(len(S))])
    test2 = np.array([np.dot(test2[x], E[x]) for x in range(len(S))])
    test2 = np.array([np.dot(test2[x], L[x]) for x in range(len(S))])
    test2 = np.array([abs(test2[x] - np.identity(len(range(len(S[0])))))
                     for x in range(len(S))])
    test2 = test2[2:].max()

    # Warning message
    if test1 > tol or test2 > tol:
        print('WARNING: the transformation matrix does not reproduce the '
              'correct Cl\'s. The relative difference is {:1.2e} and the '
              'maximum accepted is {:1.2e}.'.format(max(test1, test2), tol))
        sys.stdout.flush()

    return
