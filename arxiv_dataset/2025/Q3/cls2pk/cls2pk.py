import numpy as np
import matplotlib.pyplot as plt
import pyccl

def get_tracers(cosmo, z, nzs, delta_zs=None, ia_bias=None):
    """
    Creates weak lensing tracer objects for CCL
    """
    if delta_zs is None:
        delta_zs = [0.] * len(nzs)
        
    tracers = []
    for i, nz in enumerate(nzs):
        zz = z - delta_zs[i]
        w = zz > 0  # filter out potential z<0 points
        tracers.append(pyccl.WeakLensingTracer(cosmo, dndz=(zz[w], nz[w]), ia_bias=None if ia_bias is None else (zz[w], ia_bias[w])))

    return tracers


def compute_cl_lnk(cosmo, t1, t2, m1, m2, ell, k):
    """
    Computes the Limber integrand (over logk)
    """
    chi = (ell + 0.5) / k
    if chi > pyccl.comoving_angular_distance(cosmo, 1 / 2000.):
        return 0.
    a = pyccl.scale_factor_of_chi(cosmo, chi)

    res = sum(t1.get_kernel(chi) * t1.get_transfer(k, a)) * sum(t2.get_kernel(chi) * t2.get_transfer(k, a))  # sum potential shear + IA contributions
    res /= chi**2
    res *= pyccl.nonlin_matter_power(cosmo, k, a)
    res *= chi  # change of variables from chi to ln(k) (lots of term simplify)
    res *= (1. + m1) * (1. + m2)

    return res


compute_cl_lnk = np.vectorize(compute_cl_lnk)


def get_cl_lnk_dict(cosmo, ell, kk, tracers, bin_pairs, ms=None, bpws=None):
    """
    Casts Limber integrand into a dictionary over tomographic bins
    """
    dlogk = np.log(kk[1] / kk[0])
    if ms is None:
        ms = [0.] * len(tracers)
    if bpws is None:
        return {(i, j): compute_cl_lnk(cosmo, tracers[i], tracers[j], ms[i], ms[j], *np.meshgrid(ell, kk, indexing='ij')) for i, j in bin_pairs}
    else:
        assert len(ell) == bpws.shape[0]
        return {(i, j): np.matmul(bpws, compute_cl_lnk(cosmo, tracers[i], tracers[j], ms[i], ms[j], *np.meshgrid(np.arange(bpws.shape[1]), kk, indexing='ij')))
                for i, j in bin_pairs}


def get_cl_lnk_window_matrix(cosmo, ell, kk, tracers, bin_pairs, ms=None, bpws=None):
    """
    Casts Limber integrand into a window matrix for linear model
    """
    nell = len(ell)
    cl_lnk_dict = get_cl_lnk_dict(cosmo, ell, kk, tracers, bin_pairs, ms=ms, bpws=bpws)
    
    dlogk = np.log(kk[1] / kk[0])

    window_matrix = np.zeros((nell * len(bin_pairs), len(kk)))
    for ibin, (i, j) in enumerate(bin_pairs):
        w = slice(ibin * nell, (ibin + 1) * nell)
        window_matrix[w, :] = cl_lnk_dict[i, j] * dlogk

    return cl_lnk_dict, window_matrix


def predict_cls(cosmo, ell, tracers, bin_pairs, ms=None, return_dict=False):
    """
    Computes Cl's predictions from Limber equation (no linearization)
    """
    cls_pred = {}
    if ms is None:
        ms = [0.] * len(tracers)
    for i, j in bin_pairs:
        cls_pred[i, j] = (1 + ms[i]) * (1 + ms[j]) * np.array([pyccl.angular_cl(cosmo, tracers[i], tracers[j], l) for l in ell])

    if return_dict:
        return cls_pred
    else:
        return np.concatenate([cls_pred[i, j] for i, j in bin_pairs])


def compute_cov_syst(cosmo, ell, z, nzs, bin_pairs, muz, sigmaz, mus, sigmas):
    """
    Computes additive covariance term to analytically marginalize over shear and redshift biases
    """
    n = len(nzs)
    m = len(ell) * len(bin_pairs)
    cov_syst = np.zeros((m, m))

    eps = 0.5

    def mask(i):
        mask = np.zeros(n)
        mask[i] = 1.
        return mask

    def compute_dcldm(dm):
        tracers = get_tracers(cosmo, z, nzs, muz)
        dcldm = []
        for i in range(n):
            if dm[i] > 0.:
                dcldm_i = (predict_cls(cosmo, ell, tracers, bin_pairs, mus + eps * dm * mask(i)) -
                           predict_cls(cosmo, ell, tracers, bin_pairs, mus - eps * dm * mask(i))) / (2 * eps * dm[i])
            else:
                dcldm_i = np.zeros(m)
            dcldm.append(dcldm_i)
        return np.array(dcldm)

    def compute_dcldz(dz):
        dcldz = []
        for i in range(n):
            if dz[i] > 0.:
                tracers = get_tracers(cosmo, z, nzs, muz + eps * dz * mask(i))
                clup = predict_cls(cosmo, ell, tracers, bin_pairs, mus)
                tracers = get_tracers(cosmo, z, nzs, muz - eps * dz * mask(i))
                cllow = predict_cls(cosmo, ell, tracers, bin_pairs, mus)
                dcldz_i = (clup - cllow) / (2 * eps * dz[i])
            else:
                dcldz_i = np.zeros(m)
            dcldz.append(dcldz_i)
        return np.array(dcldz)

    if sigmas.ndim == 1:
        dcldm = compute_dcldm(sigmas)
        cov_syst += dcldm.T @ np.diag(sigmas**2) @ dcldm
    elif sigmas.ndim == 2:
        dcldm = compute_dcldm(np.sqrt(np.diagonal(sigmas)))
        cov_syst += dcldm.T @ sigmas @ dcldm
    else:
        raise ValueError

    if sigmaz.ndim == 1:
        dcldz = compute_dcldz(sigmaz)
        cov_syst += dcldz.T @ np.diag(sigmaz**2) @ dcldz
    elif sigmaz.ndim == 2:
        dcldz = compute_dcldz(np.sqrt(np.diagonal(sigmaz)))
        cov_syst += dcldz.T @ sigmaz @ dcldz
    else:
        raise ValueError

    return cov_syst