"""

This module contains all the samplers implemented.

Functions:
 - run_emcee()
 - run_single_point()

"""

import sys
import numpy as np
import emcee
import kl_sample.likelihood as lkl
import kl_sample.cosmo as cosmo_tools


# ------------------- emcee --------------------------------------------------#

def run_emcee(args, cosmo, data, settings, path):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        file with chains.

    """

    # Local variables
    full = cosmo['params']
    mask = cosmo['mask']
    ns = settings['n_steps']
    nw = settings['n_walkers']
    nt = settings['n_threads']
    nd = len(mask[mask])

    # Print useful stuff
    print('Starting the chains!')
    for key in settings.keys():
        print(key + ' = ' + str(settings[key]))
    sys.stdout.flush()

    # Initialize sampler
    sampler = emcee.EnsembleSampler(nw, nd, lkl.lnprob,
                                    args=[full, mask, data, settings],
                                    threads=nt)

    if args.restart:
        # Initial point from data
        vars_0 = np.loadtxt(path['output'], unpack=True)
        vars_0 = vars_0[2:2+nd]
        vars_0 = vars_0[:, -nw:].T
    else:
        # Initial point
        vars_0 = \
            np.array([lkl.get_random(full[mask], 1.e3) for x in range(nw)])
        # Create file
        f = open(path['output'], 'w')
        f.close()

    for count, result in enumerate(sampler.sample(vars_0, iterations=ns,
                                   storechain=False)):
        pos = result[0]
        prob = result[1]
        f = open(path['output'], 'a')
        for k in range(pos.shape[0]):
            out = np.append(np.array([1., -prob[k]]), pos[k])
            out = np.append(out, cosmo_tools.get_sigma_8(pos[k], full, mask))
            f.write('    '.join(['{0:.10e}'.format(x) for x in out]) + '\n')
        f.close()
        print('----> Computed {0:5.1%} of the steps'
              ''.format(float(count+1) / ns))
        sys.stdout.flush()

    return


# ------------------- single_point -------------------------------------------#

def run_single_point(cosmo, data, settings):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        output in terminal likelihood.

    """

    # Local variables
    full = cosmo['params']
    mask = cosmo['mask']

    var = full[:, 1][mask]
    post = lkl.lnprob(var, full, mask, data, settings)
    sigma8 = cosmo_tools.get_sigma_8(var, full, mask)

    print('Cosmological parameters:')
    print('----> h             = ' + '{0:2.4e}'.format(full[0, 1]))
    print('----> Omega_c h^2   = ' + '{0:2.4e}'.format(full[1, 1]))
    print('----> Omega_b h^2   = ' + '{0:2.4e}'.format(full[2, 1]))
    print('----> ln(10^10 A_s) = ' + '{0:2.4e}'.format(full[3, 1]))
    print('----> n_s           = ' + '{0:2.4e}'.format(full[4, 1]))
    print('----> w_0           = ' + '{0:2.4e}'.format(full[5, 1]))
    print('----> w_A           = ' + '{0:2.4e}'.format(full[6, 1]))
    if settings['add_ia']:
        print('----> A_IA          = ' + '{0:2.4e}'.format(full[7, 1]))
        print('----> beta_IA       = ' + '{0:2.4e}'.format(full[8, 1]))
    print('Derived parameters:')
    print('----> sigma_8       = ' + '{0:2.4e}'.format(sigma8))
    print('Likelihood:')
    print('----> -ln(like)     = ' + '{0:4.4f}'.format(-post))

    return
