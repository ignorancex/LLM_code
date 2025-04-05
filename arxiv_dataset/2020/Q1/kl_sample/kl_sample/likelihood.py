"""

This module contains all the relevant functions
used to compute the likelihood.

Functions:
 - how_many_sims(data, settings)
 - select_sims(data, settings)
 - compute_kl(cosmo, data, settings)
 - apply_kl(kl_t, corr, settings)
 - compute_inv_covmat(data, settings)
 - lnprior(var, full, mask)
 - lnlike(var, full, mask, data, settings)
 - lnprob(var, full, mask, data, settings)
 - get_random(pars, squeeze)

"""

import numpy as np
import random
import sys
import kl_sample.cosmo as cosmo_tools
import kl_sample.checks as checks
import kl_sample.reshape as rsh
import kl_sample.settings as set


# ------------------- Simulations --------------------------------------------#

def how_many_sims(n_sims, n_sims_tot, n_data, n_data_tot):
    """ Compute how many simulations will be used.

    Args:
        data: dictionary containing the data stored.
        settings: dictionary with all the settings used.

    Returns:
        int: the number of simulations that will be used.

    """

    # If all simulations wanted, return it
    if n_sims == 'all':
        return n_sims_tot

    # If auto, calculate how many sims should be used
    elif n_sims == 'auto':
        ratio = (n_sims_tot-n_data_tot-2.)/(n_sims_tot-1.)
        return int(round((2.+n_data-ratio)/(1.-ratio)))

    # If it is a number, just return it
    else:
        return int(n_sims)


def select_sims(data, settings):
    """ Select simulations to use.

    Args:
        data: dictionary containing the data stored.
        settings: dictionary with all the settings used.

    Returns:
        array of simulations that will be used.
        array of weights for each simulation.

    """

    # Local variables
    n_sims = settings['n_sims']
    n_sims_tot = settings['n_sims_tot']

    # Select simulations
    rnd = random.sample(range(n_sims_tot), n_sims)
    # Generate arrays of simulations and weights
    sims = data['corr_sim'][:, rnd]

    return sims


# ------------------- KL related ---------------------------------------------#

def compute_kl(params, pz, noise, ell_min=2, ell_max=2000, scale_dep=False,
               bp=None):
    """ Compute the KL transform.

    Args:
        params: cosmological parameters.
        pz: photo-z
        noise: estimated noise.
        ell_min, ell_max: minimum, maximum ell.
        scale_dep: kl with scale dependence or not.

    Returns:
        array with the KL transform that will be used.

    """

    # Compute theory Cl's (S = signal)
    cosmo_ccl = cosmo_tools.get_cosmo_ccl(params)
    S = cosmo_tools.get_cls_ccl(params, cosmo_ccl, pz, ell_max)
    S = S[ell_min:ell_max+1]

    # Cholesky decomposition of noise (N=LL^+)
    N = noise[ell_min:ell_max+1]
    L = np.linalg.cholesky(N)
    if set.PINV:
        inv_L = np.linalg.pinv(L)
    else:
        inv_L = np.linalg.inv(L)

    # Matrix for which we want to calculate eigenvalues and eigenvectors
    M = np.array([np.dot(inv_L[x], S[x]+N[x]) for x in range(len(S))])
    M = np.array([np.dot(M[x], inv_L[x].T) for x in range(len(S))])

    # Calculate eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(M)
    # Re-order eigenvalues and eigenvectors sorted from smallest eigenvalue
    new_ord = np.array([np.argsort(eigval[x])[::-1] for x in range(len(S))])
    eigval = np.array([eigval[x][new_ord[x]] for x in range(len(S))])
    eigvec = np.array([eigvec[x][:, new_ord[x]] for x in range(len(S))])

    # Calculate transformation matrix (E) from eigenvectors and L^-1
    E = np.array([np.dot(eigvec[x].T, inv_L[x]) for x in range(len(S))])
    # Change sign to eigenvectors according to the first element
    signs = np.array([[np.sign(E[ell][x][0]/E[2][x][0])
                     for x in range(len(S[0]))] for ell in range(len(S))])
    E = np.array([(E[x].T*signs[x]).T for x in range(len(S))])

    # Test if the transformation matrix gives the correct new Cl's
    checks.kl_consistent(E, S, N, L, eigval, 1.e-12)

    # Return either the scale dependent or independent KL transform
    if scale_dep:
        return rsh.bin_cl(E, bp)

    else:
        E_avg = np.zeros((len(E[0]), len(E[0])))
        den = np.array([(2.*x+1) for x in range(2, len(E))]).sum()
        for n in range(len(E[0])):
            for m in range(len(E[0])):
                num = np.array([(2.*x+1)*E[:, n][:, m][x]
                               for x in range(2, len(E))]).sum()
                E_avg[n][m] = num/den
        return E_avg


def apply_kl(kl_t, corr, settings):
    """ Apply the KL transform to the correlation function
        and reduce number of dimensions.

    Args:
        kl_t: KL transform.
        corr: correlation function

    Returns:
        KL transformed correlation function.

    """

    kl_t_T = np.moveaxis(kl_t, [-1], [-2])

    # Apply KL transform
    corr_kl = np.dot(kl_t, corr)
    if settings['kl_scale_dep']:
        corr_kl = np.diagonal(corr_kl, axis1=0, axis2=-2)
        corr_kl = np.moveaxis(corr_kl, [-1], [-2])
    corr_kl = np.dot(corr_kl, kl_t_T)
    if settings['kl_scale_dep']:
        corr_kl = np.diagonal(corr_kl, axis1=-3, axis2=-2)
        corr_kl = np.moveaxis(corr_kl, [-1], [-2])
    corr_kl = np.moveaxis(corr_kl, [0], [-2])

    # Reduce dimensions of the array
    n_kl = settings['n_kl']
    corr_kl = np.moveaxis(corr_kl, [-2, -1], [0, 1])
    corr_kl = corr_kl[:n_kl, :n_kl]
    corr_kl = np.moveaxis(corr_kl, [0, 1], [-2, -1])
    if settings['method'] == 'kl_diag':
        corr_kl = np.diagonal(corr_kl,  axis1=-2, axis2=-1)

    return corr_kl


# ------------------- Covmat -------------------------------------------------#

def compute_inv_covmat(data, settings):
    """ Compute inverse covariance matrix.

    Args:
        data: dictionary containing the data stored
        settings: dictionary with all the settings used

    Returns:
        array with the inverse covariance matrix.

    """

    # Local variables
    n_fields = settings['n_fields']
    n_sims = settings['n_sims']
    n_data = data['corr_sim'].shape[2]
    A_c = set.A_CFHTlens
    A_s = set.A_sims
    corr = data['corr_sim']

    # Compute inverse covariance matrix
    cov = np.empty((n_fields, n_data, n_data))
    inv_cov_tot = np.zeros((n_data, n_data))
    for nf in range(n_fields):
        cov[nf] = np.cov(corr[nf].T)
        if set.PINV:
            inv_cov_tot = inv_cov_tot + A_c[nf]/A_s[nf]*np.linalg.pinv(cov[nf])
        else:
            inv_cov_tot = inv_cov_tot + A_c[nf]/A_s[nf]*np.linalg.inv(cov[nf])

    # Invert and mask
    if set.PINV:
        cov_tot = np.linalg.pinv(inv_cov_tot)
    else:
        cov_tot = np.linalg.inv(inv_cov_tot)
    cov_tot = rsh.mask_xipm(cov_tot, data['mask_theta_ell'], settings)
    cov_tot = rsh.mask_xipm(cov_tot.T, data['mask_theta_ell'], settings)

    # Add overall normalization
    n_data_mask = cov_tot.shape[0]
    factor = (n_sims-n_data_mask-2.)/(n_sims-1.)
    if set.PINV:
        inv_cov_tot = factor*np.linalg.pinv(cov_tot)
    else:
        inv_cov_tot = factor*np.linalg.inv(cov_tot)

    return inv_cov_tot


# ------------------- Likelihood ---------------------------------------------#

def lnprior(var, full, mask):
    """

    Function containing the prior.

    """

    var_uni = var[full[mask][:, 0] != full[mask][:, 2]]
    var_gauss = var[full[mask][:, 0] == full[mask][:, 2]]
    uni = full[mask][full[mask][:, 0] != full[mask][:, 2]]
    gauss = full[mask][full[mask][:, 0] == full[mask][:, 2]]

    is_in = (uni[:, 0] <= var_uni).all()
    is_in = is_in*(var_uni <= uni[:, 2]).all()

    if is_in:
        lp = (var_gauss-gauss[:, 1])**2./2./gauss[:, 0]**2.
        return lp.sum()
    return -np.inf


def lnlike(var, full, mask, data, settings, save_loc=None):
    """

    Function containing the likelihood.

    """

    # Get theory
    import signal
    tmout = 600

    def handler(signum, frame):
        raise Exception()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(tmout)
    try:
        th = cosmo_tools.get_theory(var, full, mask, data, settings)
    except Exception:
        print('Theory timeout with pars = ' + str(var))
        sys.stdout.flush()
        return -np.inf
    # except:
    #     print('Theory failure with pars = ' + str(var))
    #     sys.stdout.flush()
    #     return -np.inf
    signal.alarm(0)

    obs = data['corr_obs']
    icov = data['inv_cov_mat']

    if save_loc:
        z = np.array([(x[0]+x[1])/2. for x in set.Z_BINS])
        ell = np.array([(x[0]+x[1])/2. for x in set.BANDPOWERS[set.MASK_ELL]])
        np.savetxt(save_loc + 'z.txt', z)
        np.savetxt(save_loc + 'ell.txt', ell)
        np.savetxt(save_loc + 'cell_th.txt', th)
        np.savetxt(save_loc + 'cell_obs.txt', obs)
        np.savetxt(save_loc + 'cov_mat.txt', np.linalg.pinv(icov))

    # Get chi2
    chi2 = (obs-th).dot(icov).dot(obs-th)
    return -chi2/2.


def lnprob(var, full, mask, data, settings):
    """

    Function containing the posterior.

    """

    lp = lnprior(var, full, mask)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(var, full, mask, data, settings)


def get_random(pars, squeeze):
    """

    Get random initial points.

    """

    rnd_pars = np.array([])
    for count in range(len(pars)):
        if pars[count][2] is None:
            rb = np.inf
        else:
            rb = pars[count][2]
        if pars[count][0] is None:
            lb = -np.inf
        else:
            lb = pars[count][0]
        if (lb == -np.inf) and (rb == np.inf):
            rnd = pars[count][1] + 2.*(np.random.rand()-.5)/squeeze
        else:
            rnd = pars[count][1] + 2.*(np.random.rand()-.5)*min(
                rb-pars[count][1], pars[count][1]-lb)/squeeze
        rnd_pars = np.append(rnd_pars, rnd)

    return rnd_pars
