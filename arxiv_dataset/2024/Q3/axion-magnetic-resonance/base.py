"""This is a module to compute the axion-photon conversion in a rotating magnetic field.
"""

import numpy as np
from numpy.linalg import eig
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from numpy.random import normal
from scipy.integrate import solve_ivp

# constant numbers
_m_eV_ = 5067730.58270578
_G_over_GeV2_ = 1.95352783207652e-20
_GeV_over_eV_ = 1.e9
_one_over_nm_eV_ = 197.326985655594  # does not contain 2pi
_Tesla_over_Gauss_ = 1.e4
_m_MHz_ = 0.00333564
_eV_over_GHz_ = 1519267.40787114
_eV_over_MHz_ = 1519267407.87114


def M2_over_2om(m1, m2, m3):
    """The matrix M^2/(2 omega)
    """
    res = np.array([[m1/2, 0, 0], [0, m2/2, 0], [0, 0, m3/2]])
    return res


def Hint(cB, th, only_para=False):
    """ The interaction matrix that is responsible for
    the aovided level crossing
    """
    # debug the initial polarization direction
    # th = th + 2.3*np.pi/4.
    if not only_para:
        res = np.array([[0, 0, cB*np.sin(th)/2.], [0, 0, cB *
                                                   np.cos(th)/2.], [cB*np.sin(th)/2., cB*np.cos(th)/2., 0]])
    else:
        res = np.array([[0, 0, 0], [0, 0, cB *
                                    np.cos(th)/2.], [0, cB*np.cos(th)/2., 0]])
    return res


def diagonalize(hermitian_mtx, verify=False):
    """diagonalize a hermitian matrix and output
    the special unitary matrix that diagonalize it.
    The eivenvectors are sorted according to the size
    of the eigenvalues.
    """
    val, vec = eig(hermitian_mtx)
    sorted_idx_arr = val.argsort()

    # sort eigenvalues
    val = val[sorted_idx_arr]

    # note that the eigenvectors may not be properly normalized
    # for i, _ in enumerate(vec[0]):
    #     veci = vec[:, i]
    #     vec[:, i] = veci/np.sqrt(np.dot(veci, veci))
    unitary_mtx = vec[:, sorted_idx_arr]

    # correct the overall sign
    unitary_mtx = unitary_mtx*np.sign(np.linalg.det(unitary_mtx))

    if verify:
        print("eigenvalue:", val)
        print("hamiltonian:\n", hermitian_mtx)
        print("determinant: ", np.linalg.det(unitary_mtx))
        print("unitarity:", np.dot(
            unitary_mtx.transpose().conjugate(), unitary_mtx))
        print("U^+HU\n", np.dot(unitary_mtx.transpose().conjugate(),
              np.dot(hermitian_mtx, unitary_mtx)))
        print("check eigenvec norm:")
        for i, _ in enumerate(val):
            print(np.sqrt(np.dot(vec[:, i].conjugate(), vec[:, i])))

    return val, unitary_mtx


def derivs(x, y,
           ma,
           omega,
           cB,
           mg2_over_om_fn,
           theta_fn,
           only_para=False
           ):
    """The integrand to be evolved, corresponding to the coupled ODE in the notes.

    :param x: the distance propagated
    :param y: the array of gamma_perp, gamma_parallal, a
    :param ma: the axion mass
    :param omega: the energy of the axion-photon system
    :param cB: c_agamma * B
    :param mg2_over_om_fn: mgamma^2/omega as a function of distance
    :param theta_fn: theta(x)

    """

    ma2_over_om = ma**2/omega

    # integrand
    h_arr = np.zeros((3, 3), dtype='complex_')
    h_arr += np.array(M2_over_2om(mg2_over_om_fn(x),
                                  mg2_over_om_fn(x),
                                  ma2_over_om)
                      + Hint(cB, theta_fn(x), only_para=only_para)) * (-1.j)

    res = np.dot(h_arr, y)

    return res


def mixing_angle(x,
                 ma,
                 omega,
                 cB,
                 mg2_over_om_fn):
    """The mixing angle

    :param x: the distance propagated
    :param ma: the axion mass
    :param omega: the energy of the axion-photon system
    :param cB: c_agamma * B
    :param mg2_over_om_fn: mgamma^2/omega as a function of distance
    """

    ma2_over_om = ma**2/omega
    x_arr, is_scalar = treat_as_arr(x)

    sin_alpha = np.sqrt(
        4.*cB**2/(4.*cB**2+(mg2_over_om_fn(x_arr)-ma2_over_om)**2))

    if is_scalar:
        sin_alpha = np.squeeze(sin_alpha)

    return sin_alpha


def Pag_nr_analytical(x, dthetadx, cB):
    """This is the analytical expression from Seokhoon's notes

    :param ma: the axion mass
    :param mg: the photon plasma frequency
    :param om: the energy
    :param dthetadx: dtheta/dx

    """
    Delta_ag = cB/2
    Delta_phi = dthetadx
    k = np.sqrt(Delta_ag**2 + Delta_phi**2)
    prob = Delta_ag**2/(Delta_ag**2+Delta_phi**2)**2*np.sin(x/2.*k)**2\
        * (2.*Delta_phi**2 + Delta_ag**2*(1.+np.cos(x*k)))
    return prob


def treat_as_arr(arg):
    """A routine to cleverly return scalars as (temporary and fake) arrays. True arrays are returned unharmed.
    """

    arr = np.asarray(arg)
    is_scalar = False

    # making sure scalars are treated properly
    if arr.ndim == 0:  # it is really a scalar!
        arr = arr[None]  # turning scalar into temporary fake array
        is_scalar = True  # keeping track of its scalar nature

    return arr, is_scalar


def get_theta(x_arr, domain_size, rnd_seed=None, order=2, cache=None):
    """Generate a realization of the magnetic field


    """
    xi = x_arr[0]
    xe = x_arr[-1]

    if order == 0:
        # discontinuous orientations
        domain_arr = np.arange(xi, xe, domain_size)
        if rnd_seed is not None:
            np.random.seed(rnd_seed)
        domain_phase = np.random.rand(len(domain_arr)) * 2.*np.pi

        res = []
        for x in x_arr:
            idx = np.searchsorted(domain_arr, x, side='right')
            if idx == len(domain_arr):
                idx = idx - 1
            phase = domain_phase[idx]
            res.append(phase + (x-domain_arr[idx])/domain_size*2.*np.pi)

    elif order == 1:
        # first order
        raise Exception('first order orientation angles is not realized yet.')

    elif order == 2:
        # second order
        domain_arr = np.arange(xi, xe, domain_size)

        # average ddtheta: 2pi ~ .5*ddthate*domain_size**2
        ddtheta_max = (2.*np.pi)*2/domain_size**2
        if rnd_seed is not None:
            np.random.seed(rnd_seed)
        ddtheta_edge_arr = (np.random.rand(len(domain_arr))-0.5) * ddtheta_max

        # populate the denser array of x_arr
        dthetadx2_arr = interp1d(
            domain_arr, ddtheta_edge_arr, kind='previous', bounds_error=False, fill_value='extrapolate')(x_arr)

        dx_arr = np.diff(x_arr, prepend=x_arr[0])
        # dx_arr = np.diff(x_arr, prepend=0.)

        # first integral
        dthetadx_arr = np.cumsum(dthetadx2_arr*dx_arr)

        # second integral
        theta_arr = np.cumsum(dthetadx_arr*dx_arr)

        res = (dthetadx2_arr, dthetadx_arr, theta_arr)

    return np.array(res)


def get_dtheta_gaussian(num_of_domains, sigma, bg):
    """Generate a small perturbation to theta dot

    :param num_of_domains: number of domains. Inside each domain theta dot is constant.
    :param sigma: this is the standard deviation of the variable delta(\dot \theta)/ \bar \dot \theta
    :returns: the gaussian perturbation of theta dot

    """

    # num_of_domains = (xe-xs)/domain_size
    delta = normal(loc=0., scale=sigma, size=int(num_of_domains))
    # print("delta:", delta)
    dtheta_arr = bg*(1.+delta)
    # print("bg:", bg)

    return dtheta_arr


def integrate_theta(x_arr, dtheta_arr):

    dx_arr = np.diff(x_arr, prepend=x_arr[0])
    # dx_arr = np.diff(x_arr, append=x_arr[-1])
    theta_arr = np.cumsum(dtheta_arr*dx_arr)
    return theta_arr


def get_psurv(xi=1.e-4,
              xe=106.,
              ma=1.e-2,
              ga=1.e-9,
              # theta_dot=None,
              theta_dot_mean=1.,
              sigma=0.01,
              wavelength=1064,
              B=5.3,
              num_of_domains=10,
              seed=None,
              verbose=True,
              axion_init=True):
    """Generate one Gaussian realization of the conversion probability. The magnetic field is assumed to rotate with theta_dot centered around an average value, with 1 sigma deviation specified. 

    :param xi: initial point of the propagation [m] (Default: 1.e-3)
    :param xe: final point of the propagation [m] (Default: 106., from ALPS II)
    :param ma: axion mass [eV]
    :param ga: the axion photon coupling [GeV**-1] (Default: 1e-9)
    :param theta_dot_mean: if theta_dot is None, use this as the mean value of theta_dot in the unit of [-ma**2/(2.*omega)] (Default: 1.)
    :param sigma: the fluctuation of theta dot (Default: 1%)
    :param wavelength: the wave length of the laser [nm] (Default:  1064., from ALPS II)
    :param B: the magnitude of the magnetic field [T] (Default: 5.3)
    :param num_of_domains: the number of domains
    :param seed: random number generator seed used to reproduce the same realization
    :param verbose: if True output intermediate steps (Default: True)
    :param axion_init: if true, start with pure axion initial state (Default: True)

    """
    # helical numerical
    xi = xi * _m_eV_  # [1/eV]
    xe = xe * _m_eV_  # [1/eV]
    omega = 2.*np.pi/wavelength*_one_over_nm_eV_
    cB = ga*(B*_Tesla_over_Gauss_)*_G_over_GeV2_*_GeV_over_eV_  # [eV]
    # 1.000001 for numerical divergence
    mass_phase = (-ma**2/(2.*omega))
    # print("mass_phase:", mass_phase)
    # if theta_dot is None:
    theta_dot = theta_dot_mean * mass_phase * 1.000001

    if seed == 'constant':
        raise Exception(
            'seed==constant is deprecated. Use gaussian with small sigma instead.')

    else:
        # Gaussian realization

        if seed is not None:
            np.random.seed(seed)

        dtheta_arr = get_dtheta_gaussian(
            num_of_domains, sigma, bg=theta_dot)
        x_arr = np.linspace(xi, xe, num_of_domains)

        x_fine_arr = np.linspace(xi, xe, num_of_domains*100)
        dtheta_fn = interp1d(
            x_arr, dtheta_arr, kind='nearest')
        dtheta_fine_arr = dtheta_fn(x_fine_arr)

        theta_fine_arr = integrate_theta(x_fine_arr, dtheta_fine_arr)
        theta_fn = interp1d(x_fine_arr, theta_fine_arr)

        # since it's never in the maximal mixing + NL regime,
        # the final result can always be rescaled w.r.t. (gB)^2

        # make sure in each domain _cB_rescale_factor_ doesn't cause it to become NL
        _cB_rescale_factor_ = 0.2/(xe-xi)/cB
        # _cB_rescale_factor_ = 1e10
        # _cB_rescale_factor_ = 1.

        # make sure if there's oscillation, it is resolved
        # if it's in the linear regime, it's okay
        amplitude = 1.e-6
        for x in x_arr:
            k = np.sqrt((dtheta_fn(x)-mass_phase) **
                        2 + (cB*_cB_rescale_factor_)**2)
            phase = np.abs(k*x)
            if phase > 1.:
                amplitude = min(amplitude, (cB*_cB_rescale_factor_)**2/k**2)
        tolerance = amplitude*1e-5
        if verbose:
            print("(cB*_cB_rescale_factor_)**2/k**2",
                  (cB*_cB_rescale_factor_)**2/k**2)
            print("dtheta_arr", dtheta_arr)
            print("theta_dot=%e" % theta_dot)
            print("dtheta_fn(x)", dtheta_fn(x))
            print("mass_phase", mass_phase)
            print("ma2_over_om=%e" % mass_phase)
            print("cB=%e" % (cB))
            print("k=%e" % k)
            print("ma=%e" % ma)
            print("_cB_rescale_factor_: %e" % _cB_rescale_factor_)
            print("tolerance: ", tolerance)

        # TODO: adaptive tolerance until solution stablizes

    # Not including any plasma mass
    def mg2_over_om_fn(x):
        x, is_scalar = treat_as_arr(x)
        res = (x)*0.
        if is_scalar:
            res = np.squeeze(res)
        return res

    if axion_init:
        sol = solve_ivp(derivs,
                        [xi, xe],
                        [0.+0.j, 0.+0.j, 1.+0.j],
                        method='DOP853', vectorized=True,
                        # rtol=0.01,
                        # atol=1e-100,
                        rtol=tolerance,
                        atol=tolerance,
                        args=[ma, omega, cB*_cB_rescale_factor_,
                              mg2_over_om_fn, theta_fn],
                        dense_output=True)
        # get the final conversion probability
        psurv = ((1.-np.abs(sol.y[2])**2)/_cB_rescale_factor_**2)[-1]
        sol.psurv = psurv

    else:
        sol = solve_ivp(derivs,
                        [xi, xe],
                        # [-1./np.sqrt(2)+0.j, 1./np.sqrt(2)+0.j, 0.+0.j],
                        [0.j, 1.+0.j, 0.+0.j],
                        # [1.+0.j, 0.+0.j, 0.+0.j],
                        # [0.+1.j, 0.+0.j, 0.+0.j],
                        method='DOP853',
                        # method='RK45',
                        # method='BDF',
                        vectorized=True,
                        rtol=tolerance,
                        atol=tolerance,
                        # rtol=1e-13,
                        # atol=1e-13,
                        args=[ma, omega, cB*_cB_rescale_factor_,
                              mg2_over_om_fn, theta_fn],
                        dense_output=True)

        # get the final conversion probability
        psurv = (np.abs(sol.y[2])**2/_cB_rescale_factor_**2)[-1]
        sol.psurv = psurv

    # output
    sol._cB_rescale_factor_ = _cB_rescale_factor_
    sol.tolerance = tolerance

    return sol


def Pag_helical(ga, ma, B, omega, L, theta_dot=0):
    """The analytical expression of axion to photon conversion, with rotating B field

    :param ga: axion-photon coupling [GeV**-1]
    :param ma: axion mass [eV]
    :param B: amplitude of the magnetic field [T]
    :param omega: energy [eV]
    :param L: propagation length [m]
    :param theta_dot: rotation frequency of B [eV]

    """
    cB = ga * (B/np.sqrt(2)) * _Tesla_over_Gauss_ * \
        _G_over_GeV2_ * _GeV_over_eV_
    L_in_eV = L * _m_eV_  # [1/eV]
    k_p = (cB**2 + (ma**2/2./omega + theta_dot)**2)**0.5
    k_m = (cB**2 + (ma**2/2./omega - theta_dot)**2)**0.5
    amp_p = cB**2/k_p**2
    amp_m = cB**2/k_m**2
    return amp_p * np.sin(k_p*L_in_eV/2.)**2 + amp_m * np.sin(k_m*L_in_eV/2.)**2
    # Debug:
    # return amp_p * np.sin(k_p*L_in_eV/2.)**2
    # Debug:
    # return (np.sqrt(amp_p) * np.sin(k_p*L_in_eV/2.) + np.sqrt(amp_m) * np.sin(k_m*L_in_eV/2.))**2
