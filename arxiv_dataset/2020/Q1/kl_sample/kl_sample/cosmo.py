"""

Module containing all the relevant functions
to compute and manipulate cosmology.

Functions:
 - get_cosmo_mask(params)
 - get_cosmo_ccl(params)
 - get_cls_ccl(params, cosmo, pz, ell_max)
 - get_xipm_ccl(cosmo, cls, theta)

"""

import numpy as np
import pyccl as ccl
import kl_sample.reshape as rsh
import kl_sample.likelihood as lkl
import kl_sample.settings as set


# ------------------- Masks --------------------------------------------------#

def get_cosmo_mask(params):
    """ Infer from the cosmological parameters
        array which are the varying parameters.

    Args:
        params: array containing the cosmological parameters.

    Returns:
        mask: boolean array with varying parameters.

    """

    # Function that decide for a given parameter if
    # it is varying or not.
    def is_varying(param):
        if param[0] is None or param[2] is None:
            return True
        if param[0] < param[1] or param[1] < param[2]:
            return True
        return False

    return np.array([is_varying(x) for x in params])


# ------------------- CCL related --------------------------------------------#

def get_cosmo_ccl(params):
    """ Get cosmo object.

    Args:
        params: array with cosmological parameters.

    Returns:
        cosmo object from CCL.

    """

    cosmo = ccl.Cosmology(
        h=params[0],
        Omega_c=params[1]/params[0]**2.,
        Omega_b=params[2]/params[0]**2.,
        A_s=(10.**(-10.))*np.exp(params[3]),
        n_s=params[4],
        w0=params[5],
        wa=params[6],
        transfer_function='boltzmann_class'
        )

    return cosmo


def get_cls_ccl(params, cosmo, pz, ell_max, add_ia=False):
    """ Get theory Cl's.

    Args:
        cosmo: cosmo object from CCL.
        pz: probability distribution for each redshift bin.
        ell_max: maximum multipole.

    Returns:
        array with Cl's.

    """

    # Local variables
    n_bins = len(pz)-1
    n_ells = ell_max+1

    # z and pz
    z = pz[0].astype(np.float64)
    prob_z = pz[1:].astype(np.float64)

    # If add_ia
    if add_ia:
        f_z = np.ones(len(z))
        # Bias
        Omega_m = (params[1]+params[2])/params[0]**2.
        D_z = ccl.background.growth_factor(cosmo, 1./(1.+z))
        b_z = -params[7]*set.C_1*set.RHO_CRIT*Omega_m/D_z
        b_z = np.outer(set.L_I_OVER_L_0**params[8], b_z)
        # Tracers
        lens = np.array([
            ccl.WeakLensingTracer(
                cosmo,
                dndz=(z, prob_z[x]),
                ia_bias=(z, b_z[x]),
                red_frac=(z, f_z),
            ) for x in range(n_bins)])
    else:
        # Tracers
        lens = np.array([
            ccl.WeakLensingTracer(
                cosmo,
                dndz=(z, prob_z[x])
            ) for x in range(n_bins)])

    # Cl's
    ell = np.arange(n_ells)
    cls = np.zeros((n_bins, n_bins, n_ells))
    for count1 in range(n_bins):
        for count2 in range(count1, n_bins):
            cls[count1, count2] = \
                ccl.angular_cl(cosmo, lens[count1], lens[count2], ell)
            cls[count2, count1] = cls[count1, count2]
    cls = np.transpose(cls, axes=[2, 0, 1])

    return cls


def get_xipm_ccl(cosmo, cls, theta):
    """ Get theory correlation function.

    Args:
        cosmo: cosmo object from CCL.
        cls: array of cls for each pair of bins.
        theta: array with angles for the correlation function.

    Returns:
        correlation function.

    """

    # Local variables
    n_bins = cls.shape[-1]
    n_theta = len(theta)
    ell = np.arange(len(cls))

    # Main loop: compute correlation function for each bin pair
    xi_th = np.zeros((2, n_bins, n_bins, n_theta))
    for c1 in range(n_bins):
        for c2 in range(n_bins):
            for c3 in range(n_theta):
                xi_th[0, c1, c2, c3] = \
                    ccl.correlation(cosmo, ell, cls[:, c1, c2], theta[c3],
                                    corr_type='L+', method='FFTLog')
                xi_th[1, c1, c2, c3] = \
                    ccl.correlation(cosmo, ell, cls[:, c1, c2], theta[c3],
                                    corr_type='L-', method='FFTLog')

    # Transpose to have (pm, theta, bin1, bin2)
    xi_th = np.transpose(xi_th, axes=[0, 3, 1, 2])

    return xi_th


# ------------------- KL related ---------------------------------------------#

def get_theory(var, full, mask, data, settings):
    """ Get theory correlation function or Cl's.

    Args:
        var: array containing the varying cosmo parameters.
        full: array containing all the cosmo parameters.
        mask: array containing the mask for the cosmo parameters.
        data: dictionary with all the data used
        settings: dictionary with all the settings used

    Returns:
        array with correlation function or Cl's.

    """

    # Local variables
    pz = data['photo_z']
    theta = data['theta_ell']
    ell_max = settings['ell_max']
    bp = settings['bp_ell']
    ell = np.arange(bp[-1, -1] + 1)
    nf = settings['n_fields']
    nb = settings['n_bins']

    # Merge in a single array varying and fixed parameters
    pars = np.empty(len(mask))
    count1 = 0
    for count2 in range(len(pars)):
        if not mask[count2]:
            pars[count2] = full[count2][1]
        else:
            pars[count2] = var[count1]
            count1 = count1+1

    # Get corr
    cosmo = get_cosmo_ccl(pars)
    if set.THEORY == 'CCL':
        corr = get_cls_ccl(pars, cosmo, pz, ell_max, add_ia=settings['add_ia'])
        if settings['space'] == 'real':
            corr = get_xipm_ccl(cosmo, corr, theta)
    elif set.THEORY == 'Camera':
        corr = settings['cls_template']
        Om = (pars[1] + pars[2])/pars[0]**2.
        s8 = get_sigma_8(var, full, mask)
        Oms8 = np.zeros(len(set.Z_BINS))
        for nbin, bin in enumerate(set.Z_BINS):
            z = (bin[0] + bin[1])/2.
            D = cosmo.growth_factor(1./(1. + z))
            Oms8[nbin] = D*Om*s8
        # Multiply twice the template by Oms8 array along the last two axes
        corr = mult_elementwiselastaxis(corr, Oms8)
        corr = np.moveaxis(corr, [-2], [-1])
        corr = mult_elementwiselastaxis(corr, Oms8)

    # Keep cls coupled or not
    if set.KEEP_CELLS_COUPLED:
        corr = rsh.couple_cl(ell, corr, settings['mcm'], nf, nb, len(bp))
    else:
        corr = rsh.couple_decouple_cl(ell, corr, settings['mcm'], nf, nb,
                                      len(bp))

    # Apply KL
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        corr = lkl.apply_kl(data['kl_t'], corr, settings)
    if settings['method'] == 'kl_diag':
        is_diag = True
    else:
        is_diag = False

    # Reshape corr
    if settings['space'] == 'real':
        corr = rsh.flatten_xipm(corr, settings)
        corr = rsh.mask_xipm(corr, data['mask_theta_ell'], settings)
    else:
        corr = rsh.mask_cl(corr, is_diag=is_diag)
        corr = rsh.unify_fields_cl(corr, data['cov_pf'], is_diag=is_diag,
                                   pinv=set.PINV)
        # Apply BNT if required
        if set.BNT:
            corr = apply_bnt(corr, data['bnt_mat'])
        corr = rsh.flatten_cl(corr, is_diag=is_diag)

    return corr


def get_sigma_8(var, full, mask):
    # Merge in a single array varying and fixed parameters
    pars = np.empty(len(mask))
    count1 = 0
    for count2 in range(len(pars)):
        if not mask[count2]:
            pars[count2] = full[count2][1]
        else:
            pars[count2] = var[count1]
            count1 = count1+1
    # Cosmology
    cosmo = get_cosmo_ccl(pars)
    sigma8 = ccl.sigma8(cosmo)

    return sigma8


def mult_elementwiselastaxis(A, B):
    C = np.outer(A, B)
    C = C.reshape(A.shape+B.shape)
    C = np.diagonal(C, axis1=-2, axis2=-1)
    return C


class BNT(object):

    def __init__(self, params, photo_z):
        cosmo = get_cosmo_ccl(params[:, 1])
        self.z = photo_z[0]  # np.array of redshifts
        self.chi = cosmo.comoving_radial_distance(1./(1.+photo_z[0]))
        self.n_i_list = photo_z[1:]
        self.nbins = len(self.n_i_list)

    def get_matrix(self):
        A_list = []
        B_list = []
        for i in range(self.nbins):
            nz = self.n_i_list[i]
            A_list += [np.trapz(nz, self.z)]
            B_list += [np.trapz(nz / self.chi, self.z)]

        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1, 0] = -1.

        for i in range(2, self.nbins):
            mat = np.array([[A_list[i-1], A_list[i-2]],
                            [B_list[i-1], B_list[i-2]]])
            A = -1. * np.array([A_list[i], B_list[i]])
            soln = np.dot(np.linalg.inv(mat), A)
            BNT_matrix[i, i-1] = soln[0]
            BNT_matrix[i, i-2] = soln[1]

        return BNT_matrix


def apply_bnt(cl, bnt):
    bnt_cl = np.dot(cl, bnt)
    bnt_cl = np.moveaxis(bnt_cl, [-1], [-2])
    bnt_cl = np.dot(bnt_cl, bnt)
    return bnt_cl
