#############################################################################
##
##      Filename: generate_pop_draws.py
##
##      Author: Vijay Varma
##
##      Created: 21-07-2021
##
##      Description: Generates draws of spin and kick populations using results
##                   from arxiv:2107.09693
##
#############################################################################

import numpy as np
import matplotlib.pyplot as P
import os, sys
import bilby
import gwpopulation
from gwpopulation.utils import beta_dist, truncnorm
from gwpopulation.cupy_utils import to_numpy
from scipy import special
from scipy.interpolate import interp1d
import lal
import surfinBH
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
from multiprocessing import Pool
from functools import partial
import warnings
# Ignore warnings about surrogate extrapolation
warnings.filterwarnings("ignore",
                        message="Spin magnitude of BhA outside training range.")
warnings.filterwarnings("ignore",
                        message="Spin magnitude of BhB outside training range.")
warnings.filterwarnings("ignore",
                        message="Mass ratio outside training range.")



def mu_var_to_alpha_beta(mu, var):
    """ Transformation from mu-var to alpha-beta for Beta distribution."""
    # See https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
    nu = mu * (1 - mu) / var - 1
    alpha = mu * nu
    beta = (1-mu) * nu
    return alpha, beta


def tilt_marginalized_pdf(cos_tilt, xi_tilt, sigma_tilt):
    """
    The marginalized version of the joint cos_tilt1*cos_tilt2 distribution,
    assuming you use iid_spin_orientation_gaussian_isotropic in
    github.com/ColmTalbot/gwpopulation/blob/master/gwpopulation/models/spin.py
    """
    pdf = (1 - xi_tilt)/2. + xi_tilt * truncnorm(cos_tilt, 1, sigma_tilt, 1, -1)
    return pdf


def vonMises_pdf(phi, mu, sigma):
    """
    An approximation of a wrapped gaussian distribution.
    See https://en.wikipedia.org/wiki/Von_Mises_distribution

    PDF = exp(kappa * cos(phi - mu))/(2*pi*i0(kappa)),
    where i0 = Bessel function of the first kind of order 0.
    This can be evaluated using scipy.special.i0, but that can have issues
    when kappa is large. Instead, we use scipy.special.i0e, given by:
      i0e(kappa) = exp(-abs(kappa)) * i0(kappa)
                 = exp(-kappa) * i0(kappa), as kappa > 0 for vonMises.
    So,
    PDF = exp(kappa * (cos(phi - mu) - 1))/(2*pi*i0e(kappa))
    """
    kappa = 1/sigma**2      # This is approximate, but good enough
    norm = 2*np.pi*special.i0e(kappa)
    return np.exp( kappa * ( np.cos(phi - mu) - 1) )/norm


def evaluate_spin_pdfs(par):
    """
    Evaluates PDFs for spin distribution given hyperparameters.
    See Eq.S11 in the supplement of https://arxiv.org/abs/2107.09693
    """

    # Convert from mu-sigma to alpha-beta for the Beta dist
    par['alpha_chi'], par['beta_chi'] \
        = mu_var_to_alpha_beta(par['mu_chi'], par['var_chi'])

    # spin magnitude (same model for Bh1 and Bh2)
    prob_chimag = beta_dist(xvals_dict['chimag'],
                            par['alpha_chi'],
                            par['beta_chi'], scale=1)

    # cos(tilt) (same model for Bh1 and Bh2)
    prob_costheta = tilt_marginalized_pdf(xvals_dict['costheta'],
                                          par['xi_tilt'],
                                          par['sigma_tilt'])

    # phi1, the orbital-plane spin angle of Bh1, defined w.r.t
    # line-of-separation at tref=-100M
    prob_phi1 = vonMises_pdf(xvals_dict['phi'],
                             par['mu_phi1'],
                             par['sigma_phi1'])

    # delphi=phi1-phi2, at tref=-100M
    prob_delphi = vonMises_pdf(xvals_dict['phi'],
                               par['mu_delphi'],
                               par['sigma_delphi'])

    return prob_chimag, prob_costheta, prob_phi1, prob_delphi


def evaluate_mass_ratio_pdf(lvc_par):
    """
    Evaluates PDFs for mass ratio distribution given hyperparameters.
    Uses PowerLaw+Peak model from https://arxiv.org/abs/2010.14533.
    """

    # Do m1 first as q depends on this
    p_m1 = mass_model.p_m1(xvals_dict, lvc_par['alpha'], lvc_par['mmin'], \
        lvc_par['mmax'], lvc_par['lam'], lvc_par['mpp'], \
        lvc_par['sigpp'], lvc_par['delta_m'])
    prob_m1 = bilby.core.prior.interpolated.Interped( \
        to_numpy(xvals_dict['mass_1']), to_numpy(p_m1), minimum=2, maximum=100)
    m1 = prob_m1.sample()
    prob_q = mass_model.p_q({'mass_ratio': xvals_dict['mass_ratio'],
                            'mass_1': np.ones(500)*m1
                            },
                            lvc_par['beta'],
                            lvc_par['mmin'],
                            lvc_par['delta_m'])
    return prob_q


def evaluate_kick_surrogate(samples, max_inverse_mass_ratio=6):
    """
    Evaluates the NRSur7dq4Remnant model (https://arxiv.org/abs/1905.09300) to
    get the kick magnitude given the mass ratio and spins.
    NOTE: We are ignoring cases with 1/q < 1/6 as NRSur7dq4Remnant is not
    valid there. However, there should not be many such cases anyway.
    """

    def get_spin_vec(mag, th, ph):
        return mag * np.array([np.sin(th)*np.cos(ph), \
                               np.sin(th)*np.sin(ph), \
                               np.cos(th)])

    # NOTE: inverse_mass_ratio = m1/m2
    inverse_mass_ratio = 1./np.array(samples['mass_ratio'])
    chi1mag = np.array(samples['chi1mag'])
    chi2mag = np.array(samples['chi2mag'])
    tilt1 = np.arccos(np.array(samples['costheta1']))
    tilt2 = np.arccos(np.array(samples['costheta2']))
    phi1 = np.array(samples['phi1'])
    delphi = np.array(samples['delphi'])
    # because delphi = phi1 - phi2
    phi2 = phi1 - delphi

    vf_list = []
    for i in range(len(inverse_mass_ratio)):
        if inverse_mass_ratio[i] > max_inverse_mass_ratio:
            continue
        chi1 = get_spin_vec(chi1mag[i], tilt1[i], phi1[i])
        chi2 = get_spin_vec(chi2mag[i], tilt2[i], phi2[i])
        vf, _ = remnant_sur.vf(inverse_mass_ratio[i], chi1, chi2)
        vf_list.append(vf)
    # Get magnitude and convert to km/s
    vf_list = np.linalg.norm(np.array(vf_list), axis=1) * lal.C_SI/1e3
    return vf_list


def kde_helper(data, xlow=None, xhigh=None):
    """ Computes bouded 1d KDE.
    """
    # Initialize bounded KDE
    if xlow is None:
        xlow = np.amin(data)
    if xhigh is None:
        xhigh = np.amax(data)
    kde = bounded_1d_kde(data, xlow=xlow, xhigh=xhigh)
    pts = np.linspace(xlow, xhigh, num=500)
    pdf = kde(pts)
    return pts, pdf


def get_kick_pdf(prob_q, prob_chimag, prob_costheta, prob_phi1, prob_delphi):
    """
    Draw mass ratio and spin samples from a given PDF realization and
    evaluate the corresponding kick magnitude PDF.
    """

    # Get interpolated versions of the mass ratio and spin PDFs
    q_interp = bilby.core.prior.interpolated.Interped(
                        to_numpy(xvals_dict['mass_ratio']),
                        to_numpy(prob_q),
                        minimum=min(xvals_dict['mass_ratio']),
                        maximum = max(xvals_dict['mass_ratio']))
    chimag_interp = bilby.core.prior.interpolated.Interped(
                        to_numpy(xvals_dict['chimag']),
                        to_numpy(prob_chimag),
                        minimum = min(xvals_dict['chimag']),
                        maximum = max(xvals_dict['chimag']))
    costheta_interp = bilby.core.prior.interpolated.Interped(
                        to_numpy(xvals_dict['costheta']),
                        to_numpy(prob_costheta),
                        minimum = min(xvals_dict['costheta']),
                        maximum = max(xvals_dict['costheta']))
    phi1_interp = bilby.core.prior.interpolated.Interped(
                        to_numpy(xvals_dict['phi']),
                        to_numpy(prob_phi1),
                        minimum = min(xvals_dict['phi']),
                        maximum = max(xvals_dict['phi']))
    delphi_interp = bilby.core.prior.interpolated.Interped(
                        to_numpy(xvals_dict['phi']),
                        to_numpy(prob_delphi),
                        minimum = min(xvals_dict['phi']),
                        maximum = max(xvals_dict['phi']))

    # Draw nsamp_per_draw mass-spin samples for each draw from the
    # hyperposterior.
    # NOTE: We have the same PDF for chi1mag and chi2mag, but we draw samples
    # indepentenly for both. Same for costheta1 and costheta2. However, phi1
    # and delphi have separate models.
    nsamp_per_draw = 500
    samples = {'mass_ratio': q_interp.sample(nsamp_per_draw), \
               'chi1mag': chimag_interp.sample(nsamp_per_draw), \
               'chi2mag': chimag_interp.sample(nsamp_per_draw), \
               'costheta1': costheta_interp.sample(nsamp_per_draw), \
               'costheta2': costheta_interp.sample(nsamp_per_draw), \
               'phi1': phi1_interp.sample(nsamp_per_draw), \
               'delphi': delphi_interp.sample(nsamp_per_draw), \
              }


    # Get kick samples
    samples['vfmag'] = evaluate_kick_surrogate(samples)

    # Interpolate vfmag onto a uniform grid using kde.
    # But first work in logspace to capture the low kick limit well.
    logvfmag_min = np.log(1)
    logvfmag_max = np.log(5000)
    logvfmag_grid, logvfmag_kde = kde_helper(np.log(samples['vfmag']),
                                             xlow=logvfmag_min,
                                             xhigh=logvfmag_max)

    # Interpolate to uniform in vfmag grid and transfrom from P(log(v)) to P(v)
    prob_logvfmag_interp = interp1d(logvfmag_grid,
                                         logvfmag_kde,
                                         assume_sorted=True)
    vfmag_grid = xvals_dict['vfmag']
    prob_vfmag = prob_logvfmag_interp(np.log(vfmag_grid)) / vfmag_grid
    return prob_vfmag


def get_spin_pop(npop_draws=1):
    """
    Draw spin population PDFs.
    Each draw corresponds to the spin PDFs evaluated at a single hyperparameter
    value from the hyperposterior.
    """

    max_draws = spin_posterior.shape[0]
    if npop_draws > max_draws:
        raise Exception('npop_draws=%d > num samples=%d'%(npop_draws,
                                                          max_draws))

    # Draw hyperparameter samples for spin model from the spin hyperposterior.
    params = spin_posterior.sample(npop_draws, replace=False).to_dict('records')

    chimag_pdfs, costheta_pdfs, phi1_pdfs, delphi_pdfs = [], [], [], []
    for idx in range(npop_draws):
        prob_chimag, prob_costheta, prob_phi1, prob_delphi \
            = evaluate_spin_pdfs(params[idx])
        chimag_pdfs.append(prob_chimag)
        costheta_pdfs.append(prob_costheta)
        phi1_pdfs.append(prob_phi1)
        delphi_pdfs.append(prob_delphi)

    return np.array(chimag_pdfs), np.array(costheta_pdfs), \
           np.array(phi1_pdfs), np.array(delphi_pdfs)


def get_mass_ratio_pop(npop_draws=1):
    """
    Draw mass ratio population PDFs.
    Each draw corresponds to the mass ratio PDFs evaluated at a single
    hyperparameter value from the hyperposterior.
    """

    max_draws = mass_posterior.shape[0]
    if npop_draws > max_draws:
        raise Exception('npop_draws=%d > num samples=%d'%(npop_draws,
                                                          max_draws))

    # Draw hyperparameter samples for mass ratio model from the mass ratio
    # hyperposterior (PowerLaw+Peak model from
    # https://arxiv.org/abs/2010.14533).
    lvc_params = mass_posterior.sample(npop_draws,
                                       replace=False).to_dict('records')

    q_pdfs = []
    for idx in range(npop_draws):
        prob_q = evaluate_mass_ratio_pdf(lvc_params[idx])
        q_pdfs.append(prob_q)
    return np.array(q_pdfs)

def get_kick_pop_for_pool(idx, tmp_args):
    """ Version of get_kick_pdf for multiprocessing.
    """
    q_pdfs, chimag_pdfs, costheta_pdfs, phi1_pdfs, delphi_pdfs = tmp_args
    prob_vfmag = get_kick_pdf(q_pdfs[idx],
                              chimag_pdfs[idx],
                              costheta_pdfs[idx],
                              phi1_pdfs[idx],
                              delphi_pdfs[idx])
    return prob_vfmag

def get_kick_pop(q_pdfs, chimag_pdfs, costheta_pdfs, phi1_pdfs,
                 delphi_pdfs, nprocs=1):
    """
    Evaluate kick population PDFs, given mass ratio and spin population PDFs.
    """

    get_kick_pop_for_pool_partial = partial(get_kick_pop_for_pool,
                                            tmp_args = [q_pdfs,
                                                       chimag_pdfs,
                                                       costheta_pdfs,
                                                       phi1_pdfs,
                                                       delphi_pdfs],
                                           )

    pool = Pool(processes=nprocs)
    vfmag_pdfs = pool.map(get_kick_pop_for_pool_partial, range(len(q_pdfs)))
    pool.close()
    pool.join()

    return np.array(vfmag_pdfs)


if __name__ == "__main__":

    # Load PowerLaw+Peak model and hyperposteriors from arxiv:2010.14533
    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    mass_model_file = 'data/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json'
    mass_posterior = bilby.result.read_in_result(mass_model_file).posterior

    # Spin model and hyperposteriors from arxiv:2107.09693
    phi_sigma_prior = 'Jeffreys'    # Flat or Jeffreys, see arxiv:2107.09693
    spin_model_file = 'data/result_{0}sigma.json'.format(phi_sigma_prior)
    spin_posterior = bilby.result.read_in_result(spin_model_file).posterior

    # Load remnant surrogate model for evaluating kicks
    remnant_sur = surfinBH.LoadFits('NRSur7dq4Remnant')

    # xvals on which the PDFs are evaluated
    xvals_dict = {
        'mass_1': mass_model.m1s,
        'mass_ratio': mass_model.qs,
        'chimag': np.linspace(0, 1, 500),
        'costheta': np.linspace(-1, 1, 500),
        'phi': np.linspace(-np.pi, np.pi, 500),
        'vfmag': np.linspace(1, 5000, 500),
        }


    # Number of population realizations
    npop_draws = 5000

    # Evaluate spin population
    chimag_pdfs, costheta_pdfs, phi1_pdfs, delphi_pdfs \
        = get_spin_pop(npop_draws=npop_draws)

    # Evaluate mass ratio population
    q_pdfs = get_mass_ratio_pop(npop_draws=npop_draws)


    # Evaluate the corresponding kick populations
    nprocs = 24     # Number of parallel procs. Only needed for kick pop.
    print ("\nUsing {0} processors for kick population.".format(nprocs))
    vfmag_pdfs = get_kick_pop(q_pdfs, chimag_pdfs, costheta_pdfs, phi1_pdfs,
                              delphi_pdfs, nprocs=nprocs)

    # save to file
    savedir = phi_sigma_prior
    os.system('mkdir -p {0}'.format(savedir))
    np.save('{0}/q_pdfs.npy'.format(savedir), q_pdfs)
    np.save('{0}/chimag_pdfs.npy'.format(savedir), chimag_pdfs)
    np.save('{0}/costheta_pdfs.npy'.format(savedir), costheta_pdfs)
    np.save('{0}/phi1_pdfs.npy'.format(savedir), phi1_pdfs)
    np.save('{0}/delphi_pdfs.npy'.format(savedir), delphi_pdfs)
    np.save('{0}/vfmag_pdfs.npy'.format(savedir), vfmag_pdfs)
    np.save('{0}/xvals_dict'.format(savedir), xvals_dict)
