"""

This module contains functions to reshape and manipulate
the correlation function and power spectra.

Functions:
 - mask_cl(cl)
 - unify_fields_cl(cl, sims)
 - position_xipm(n, n_bins, n_theta)
 - unflatten_xipm(array)
 - flatten_xipm(corr, settings)
 - mask_xipm(array, mask, settings)
 - unmask_xipm(array, mask)

"""

import os
import numpy as np
import kl_sample.settings as set
import pymaster as nmt


# ------------------- Manipulate Cl's ----------------------------------------#

def mask_cl(cl, is_diag=False):
    if is_diag:
        idx = -2
    else:
        idx = -3
    mask_cl = np.moveaxis(cl, [idx], [0])
    mask_cl = mask_cl[set.MASK_ELL]
    mask_cl = np.moveaxis(mask_cl, [0], [idx])
    return mask_cl


def clean_cl(cl, noise):
    if cl.ndim == 4:
        return cl - noise
    elif cl.ndim == 5:
        clean = np.array([cl[:, x]-noise for x in range(len(cl[0]))])
        clean = np.transpose(clean, axes=(1, 0, 2, 3, 4))
        return clean
    else:
        raise ValueError('Expected Cl\'s array with dimensions 4 or 5. Found'
                         ' {}'.format(cl.ndim))


def flatten_cl(cl, is_diag=False):
    """
    Given a cl array with shape (A, ell, z_bin_1, z_bin_2), where A stands
    for additional shape indices (it can be the number of simulations,
    number of fields, ...), return a flattened one with shape (A, s).
    In practise only the ells and z_bins are flattened, piling triangle up
    indices for each ell, i.e.:
    cl[A, 0, 0, 0]
    cl[A, 0, 0, 1]
    ...
    cl[A, 0, 0, z_bins]
    cl[A, 0, 1, 1]
    ...
    cl[A, 0, z_bins, z_bins]
    cl[A, 1, 0, 0]
    cl[A, ell_bins, z_bins, z_bins]
    If is_diag, the input array should have this shape (A, ell, z_bin).
    In this case, the flatten process just flattens the last two indices.
    """
    flat_cl = cl
    if not is_diag:
        tr_idx = np.triu_indices(cl.shape[-1])
        flat_cl = np.moveaxis(flat_cl, [-2, -1], [0, 1])
        flat_cl = flat_cl[tr_idx]
        flat_cl = np.moveaxis(flat_cl, [0], [-1])
    flat_cl = flat_cl.reshape(flat_cl.shape[:-2] +
                              (flat_cl.shape[-2]*flat_cl.shape[-1],))
    return flat_cl


def unflatten_cl(cl, shape, is_diag=False):
    """
    Given a cl flattened array (see flatten_cl for details on the structure)
    return an array with shape (A, ell, z_bin_1, z_bin_2), where A stands
    for additional shape indices (it can be the number of simulations,
    number of fields, ...).
    If is_diag, the final shape will be (A, ell, z_bin).
    """
    if is_diag:
        unflat_cl = cl.reshape(shape)
    else:
        tr_idx = np.triu_indices(shape[-1])
        unflat_cl = np.zeros(shape)
        tmp_cl = cl.reshape(shape[:-2]+(-1,))
        tmp_cl = np.moveaxis(tmp_cl, [-1], [0])
        unflat_cl = np.moveaxis(unflat_cl, [-2, -1], [0, 1])
        unflat_cl[tr_idx] = tmp_cl
        unflat_cl = np.moveaxis(unflat_cl, [1], [0])
        unflat_cl[tr_idx] = tmp_cl
        unflat_cl = np.moveaxis(unflat_cl, [0, 1], [-2, -1])
    return unflat_cl


def flatten_covmat(cov, is_diag=False):
    """
    Given a cov_mat array with shape
    (A, ell_c1, ell_c2, z_bin_c1_1, z_bin_c2_1, z_bin_c1_2, z_bin_c2_2),
    where c1 and c2 stand for the first and second cls and  A
    for additional shape indices (it can be the number of simulations,
    number of fields, ...), return a A+square array with shape (A, s, s).
    This function just applies twice flatten_cl on the different indices of
    the covariance matrix.
    """
    if is_diag:
        flat_cov = np.moveaxis(cov, [-3, -2], [-2, -3])
        idx = 2
    else:
        flat_cov = np.moveaxis(cov, [-5, -4, -3, -2], [-3, -5, -2, -4])
        idx = 3
    flat_cov = flatten_cl(flat_cov, is_diag)
    flat_cov = np.moveaxis(flat_cov, [-1], [-1-idx])
    flat_cov = flatten_cl(flat_cov, is_diag)
    return flat_cov


def unflatten_covmat(cov, cl_shape, is_diag=False):
    """
    Given a covmat flattened array (see flatten_covmat for details on the
    structure) return an array with shape
    (A, ell_c1, ell_c2, z_bin_c1_1, z_bin_c2_1, z_bin_c1_2, z_bin_c2_2),
    where A stands for additional shape indices (it can be the number of
    simulations, number of fields, ...).
    This function just applies twice unflatten_cl on the different indices of
    the covariance matrix.
    """
    unflat_cov = np.apply_along_axis(unflatten_cl, -1, cov, cl_shape, is_diag)
    unflat_cov = np.apply_along_axis(unflatten_cl, -1-len(cl_shape),
                                     unflat_cov, cl_shape, is_diag)
    if is_diag:
        unflat_cov = np.moveaxis(unflat_cov, [-3, -2], [-2, -3])
    else:
        unflat_cov = np.moveaxis(unflat_cov,
                                 [-5, -4, -3, -2], [-4, -2, -5, -3])
    return unflat_cov


def get_covmat_cl(sims, is_diag=False):
    sims_flat = flatten_cl(sims, is_diag)
    if len(sims_flat.shape) == 2:
        cov = np.cov(sims_flat.T, bias=True)
    elif len(sims_flat.shape) == 3:
        cov = np.array([np.cov(x.T, bias=True) for x in sims_flat])
    else:
        raise ValueError('Input dimensions can be either 2 or 3, found {}'
                         ''.format(len(sims_flat.shape)))
    if is_diag:
        shape = sims.shape[-2:]
    else:
        shape = sims.shape[-3:]
    return unflatten_covmat(cov, shape, is_diag)


def unify_fields_cl(cl, cov_pf, is_diag=False, pinv=False):
    cl_flat = flatten_cl(cl, is_diag)
    cov = flatten_covmat(cov_pf, is_diag)
    if pinv:
        inv_cov = np.array([np.linalg.pinv(x) for x in cov])
    else:
        inv_cov = np.array([np.linalg.inv(x) for x in cov])
    tot_inv_cov = np.sum(inv_cov, axis=0)
    if pinv:
        tot_cov = np.linalg.pinv(tot_inv_cov)
    else:
        tot_cov = np.linalg.inv(tot_inv_cov)
    # keeping also original code just in case
    tot_cl = np.array([np.dot(inv_cov[x], cl_flat[x].T)
                      for x in range(len(cl))])
    # tot_cl = np.array([np.linalg.solve(cov[x], cl_flat[x].T)
    #                   for x in range(len(cl))])
    tot_cl = np.sum(tot_cl, axis=0)
    # tot_cl = np.linalg.solve(tot_inv_cov, tot_cl).T
    tot_cl = np.dot(tot_cov, tot_cl).T
    tot_cl = unflatten_cl(tot_cl, cl.shape[1:], is_diag=is_diag)
    return tot_cl


def debin_cl(cl, bp):
    if cl.shape[-3] != bp.shape[0]:
        raise ValueError('Bandpowers and Cl shape mismatch!')
    new_shape = list(cl.shape)
    new_shape[-3] = bp[-1, -1]
    new_shape = tuple(new_shape)
    cl_dbp = np.zeros(new_shape)
    cl_dbp = np.moveaxis(cl_dbp, [-3], [0])
    for count, range in enumerate(bp):
        n_rep = range[1]-range[0]
        cl_ext = np.repeat(cl[count], n_rep)
        cl_ext = cl_ext.reshape(cl.shape[1:]+(n_rep,))
        cl_ext = np.moveaxis(cl_ext, [-1], [0])
        cl_dbp[range[0]:range[1]] = cl_ext
    cl_dbp = np.moveaxis(cl_dbp, [0], [-3])
    return cl_dbp


def bin_cl(cl, bp):
    if cl.shape[-3] == bp[-1, -1] - bp[0, 0]:
        ell_min = bp[0, 0]
    elif cl.shape[-3] == bp[-1, -1] + 1:
        ell_min = 0
    else:
        raise ValueError('Bandpowers and Cl shape mismatch!')
    new_shape = list(cl.shape)
    new_shape[-3] = bp.shape[0]
    new_shape = tuple(new_shape)
    cl_bp = np.zeros(new_shape)
    cl_bp = np.moveaxis(cl_bp, [-3], [0])
    for count, range in enumerate(bp):
        cl_re = np.moveaxis(cl, [-3], [0])
        cl_bp[count] = np.average(cl_re[range[0]-ell_min:range[1]-ell_min],
                                  axis=0)
    cl_bp = np.moveaxis(cl_bp, [0], [-3])
    return cl_bp


def couple_cl(ell, cl, mcm_path, n_fields, n_bins, n_bp, return_BB=False):
    nmt_cl = np.moveaxis(cl, [0], [-1])
    nmt_cl = np.stack((nmt_cl, np.zeros(nmt_cl.shape), np.zeros(nmt_cl.shape),
                      np.zeros(nmt_cl.shape)))
    nmt_cl = np.moveaxis(nmt_cl, [0], [-2])
    final_cl = np.zeros((n_fields, n_bins, n_bins, n_bp))
    final_cl_BB = np.zeros((n_fields, n_bins, n_bins, n_bp))
    for nb1 in range(n_bins):
        for nb2 in range(nb1, n_bins):
            for nf in range(n_fields):
                wf = nmt.NmtWorkspaceFlat()
                wf.read_from(os.path.join(
                    mcm_path, 'mcm_W{}_Z{}{}.dat'.format(nf+1, nb1+1, nb2+1)))
                cl_pfb = wf.couple_cell(ell, nmt_cl[nb1, nb2])
                final_cl[nf, nb1, nb2] = cl_pfb[0]
                final_cl[nf, nb2, nb1] = cl_pfb[0]
                final_cl_BB[nf, nb1, nb2] = cl_pfb[-1]
                final_cl_BB[nf, nb2, nb1] = cl_pfb[-1]
    final_cl = np.moveaxis(final_cl, [-1], [-3])
    final_cl_BB = np.moveaxis(final_cl_BB, [-1], [-3])
    if return_BB:
        return final_cl, final_cl_BB
    else:
        return final_cl


def couple_decouple_cl(ell, cl, mcm_path, n_fields, n_bins, n_bp,
                       return_BB=False):
    nmt_cl = np.moveaxis(cl, [0], [-1])
    nmt_cl = np.stack((nmt_cl, np.zeros(nmt_cl.shape), np.zeros(nmt_cl.shape),
                      np.zeros(nmt_cl.shape)))
    nmt_cl = np.moveaxis(nmt_cl, [0], [-2])
    final_cl = np.zeros((n_fields, n_bins, n_bins, n_bp))
    final_cl_BB = np.zeros((n_fields, n_bins, n_bins, n_bp))
    for nb1 in range(n_bins):
        for nb2 in range(nb1, n_bins):
            for nf in range(n_fields):
                wf = nmt.NmtWorkspaceFlat()
                wf.read_from(os.path.join(
                    mcm_path, 'mcm_W{}_Z{}{}.dat'.format(nf+1, nb1+1, nb2+1)))
                cl_pfb = wf.couple_cell(ell, nmt_cl[nb1, nb2])
                cl_pfb = wf.decouple_cell(cl_pfb)
                final_cl[nf, nb1, nb2] = cl_pfb[0]
                final_cl[nf, nb2, nb1] = cl_pfb[0]
                final_cl_BB[nf, nb1, nb2] = cl_pfb[-1]
                final_cl_BB[nf, nb2, nb1] = cl_pfb[-1]
    final_cl = np.moveaxis(final_cl, [-1], [-3])
    final_cl_BB = np.moveaxis(final_cl_BB, [-1], [-3])
    if return_BB:
        return final_cl, final_cl_BB
    else:
        return final_cl


# ------------------- Flatten and unflatten correlation function -------------#

def position_xipm(n, n_bins, n_theta):
    """ Given the position in the array, find the
        corresponding position in the unflattened array.

    Args:
        n: position in the flattened array.
        n_bins: number of bins.
        n_theta: number of theta_ell variables.

    Returns:
        p_pm, p_theta, p_bin_1, p_bin_2.

    """

    # Check that the input is consistent with these numbers
    p_max = 2*n_theta*n_bins*(n_bins+1)/2
    if n >= p_max:
        raise ValueError("The input number is larger than expected!")
    # div: gives position of bins. mod: gives pm and theta
    div, mod = np.divmod(n, 2*n_theta)
    # Calculate position of pm and theta
    if mod < n_theta:
        p_pm = 0
        p_theta = mod
    else:
        p_pm = 1
        p_theta = mod-n_theta
    # Calculate position of bin1 and bin2
    intervals = np.flip(np.array([np.arange(x, n_bins+1).sum()
                                 for x in np.arange(2, n_bins+2)]), 0)
    p_bin_1 = np.where(intervals <= div)[0][-1]
    p_bin_2 = div - intervals[p_bin_1] + p_bin_1

    return p_pm, p_theta, p_bin_1, p_bin_2


def unflatten_xipm(array):
    """ Unflatten the correlation function.

    Args:
        array: correlation function.

    Returns:
        reshaped correlation function (pm, theta, bin1, bin2).

    """

    # Local variables
    n_bins = len(set.Z_BINS)
    n_theta = len(set.THETA_ARCMIN)

    # Initialize array with xipm
    xipm = np.zeros((2, n_theta, n_bins, n_bins))
    # Main loop: scroll all elements of the flattened array and reshape them
    for count in range(len(array)):
        # From position in flattened array, give position for each index
        p_pm, p_theta, p_bin_1, p_bin_2 = position_xipm(count, n_bins, n_theta)
        # Assign element to xipm (bin1 and bin2 are symmetric)
        xipm[p_pm, p_theta, p_bin_1, p_bin_2] = array[count]
        xipm[p_pm, p_theta, p_bin_2, p_bin_1] = array[count]

    return xipm


def flatten_xipm(corr, settings):
    """ Flatten the correlation function.

    Args:
        corr: correlation function.
        settings: dictionary with settings.

    Returns:
        flattened correlation function.

    """

    # Local variables
    n_theta_ell = settings['n_theta_ell']
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        n_bins = settings['n_kl']
    else:
        n_bins = settings['n_bins']

    # Flatten array
    if settings['method'] in ['full', 'kl_off_diag']:
        n_data = 2*n_theta_ell*n_bins*(n_bins+1)/2
        data_f = np.empty(n_data)
        for n in range(n_data):
            p_pm, p_tl, p_b1, p_b2 = position_xipm(n, n_bins, n_theta_ell)
            data_f[n] = corr[p_pm, p_tl, p_b1, p_b2]
    else:
        n_data = 2*n_theta_ell*n_bins
        data_f = np.empty(n_data)
        for n in range(n_data):
            div, mod = np.divmod(n, 2*n_theta_ell)
            if mod < n_theta_ell:
                p_pm = 0
                p_theta = mod
            else:
                p_pm = 1
                p_theta = mod-n_theta_ell
            p_bin = div
            data_f[n] = corr[p_pm, p_theta, p_bin]

    return data_f


# ------------------- Mask and Unmask correlation function -------------------#

def mask_xipm(array, mask, settings):
    """ Convert a unmasked array into a masked one.

    Args:
        array: array with the unmasked xipm.
        mask: mask that has been used.
        settings: dictionary with settings.

    Returns:
        array with masked xipm.

    """

    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        n_bins = settings['n_kl']
    else:
        n_bins = settings['n_bins']

    if settings['method'] in ['full', 'kl_off_diag']:
        mask_tot = np.tile(mask.flatten(), n_bins*(n_bins+1)/2)
    else:
        mask_tot = np.tile(mask.flatten(), n_bins)

    return array[mask_tot]


def unmask_xipm(array, mask):
    """ Convert a flatten masked array into
        an unmasked one (still flatten).

    Args:
        array: array with the masked xipm.
        mask: mask that has been used.

    Returns:
        array with unmasked xipm.

    """

    # Flatten mask and tile mask
    mask_f = mask.flatten()
    # Get number of times that theta_pm should be repeated
    div, mod = np.divmod(len(array), len(mask_f[mask_f]))
    if mod == 0:
        raise IOError('The length of the input array is not correct!')
    mask_f = np.tile(mask_f, div)

    # Find positions where to write values
    pos = np.where(mask_f)[0]

    # Define unmasked array
    xipm = np.zeros(len(mask_f))

    # Assign components
    for n1, n2 in enumerate(pos):
        xipm[n2] = array[n1]

    return xipm
