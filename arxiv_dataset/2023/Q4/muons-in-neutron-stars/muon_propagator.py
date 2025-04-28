# ---------------------------------------------------------------------------
# A class for propagating muons through a neutron star
# ---------------------------------------------------------------------------
# Author:  Joachim Kopp (CERN)
# Email:   jkopp@cern.ch
# Date:    2020
# ---------------------------------------------------------------------------

# Standard packages
import sys
import os
import pickle
import re
import time
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.special as sf
import scipy.stats as stats
import scipy.spatial.transform as trafo
import mpmath as mp
import multiprocess.pool

# Column indices for time-independent NSCool data
k_idx, k_r, k_emas, k_rho, k_pres, k_nb, k_kfe, k_kfmu, k_kfp, k_kfn, \
  k_kfla, k_kfSm, k_kfS0, k_kfSp, k_Tcn, k_Tcp, k_Tcla, k_Durca, k_mstn, k_mstp \
  = range(20)

# Column indices for time-dependent NSCool data
n_j = 15
j_zone, j_r, j_rho, j_expphi, j_dvol, j_T, j_L, j_QmUrca, \
  j_lambdaE, j_lambdaMu, j_QdUrca, \
  j_GammaMUrcaE, j_GammaMUrcaMu, \
  j_GammaDUrcaE, j_GammaDUrcaMu = range(n_j)

# Column indices for EOS data
n_l = 17
l_rho, l_p, l_nbar, \
  l_Ye, l_Ymu, l_Yn, l_Yp, l_Yla, l_Ysm, l_Ys0, l_Ysp, \
  l_mstp, l_mstn, l_mstla, l_mstsm, l_msts0, l_mstsp = range(n_l)


# Unit conversion and physical constants
# ---------------------------------------------------------------------------
class my_units:
    # Energy and mass
    eV     = 1.
    keV    = 1.e3
    MeV    = 1.e6
    GeV    = 1.e9
    TeV    = 1.e12
    PeV    = 1.e15
    kg     = 5.609588603e35*eV   # (exact)
    grams  = 0.001*kg
    tons   = 1000*kg
    Kelvin = 8.617333262e-5                  # PDG 2019
    Joule  = 1/1.602176634e-19               # PDG 2019 (exact)
    erg    = 1e-7*Joule
    
    # Length and time
    m     = 1/197.3269804e-9    # (exact)
    meter = m
    km    = 1000*m
    cm    = 0.01*m
    nm    = 1.e-9*m
    fm    = 1.e-15*m
    AU    = 1.4960e11*m
    pc    = 30.857e15*m
    kpc   = 1.e3*pc
    Mpc   = 1.e6*pc
    Gpc   = 1.e9*pc
    ly    = 9460730472580800*m  # light year (exact)
    sec   = 299792458*meter     # (exact)
    hours = 3600*sec
    days  = 24*hours
    yrs   = 365*days
    Hz    = 1./sec
    kHz   = 1.e3*Hz
    MHz   = 1.e6*Hz
    GHz   = 1.e9*Hz

    Tesla = 692.508 * eV**2
    Gauss = 1e-4 * Tesla
    dyne  = grams * cm / sec**2
    
    # particle physics
    alpha_em = (1./137.035999139)          # em fine structure constant (PDG 2018)
    m_e      = 0.5109989461 * MeV          # electron mass (PDG 2018)
    m_mu     = 105.6583745 * MeV           # muon mass (Wikipedia 28.10.2019)
    m_n      = 939.5654133 * MeV           # neutron mass (Wikipedia 29.10.2019)
    m_p      = 938.2720813 * MeV           # proton mass (Wikipedia 29.10.2019)
    m_pi     = 139.57039 * MeV             # charged pion mass (PDG 2022)
    tau_mu   = 2.1969811e-6 * sec          # muon lifetime (PDG 2019)
    GF       = 1.1663787e-5 / GeV**2       # Fermi constant (PDG 2019)
    sin_theta_C = 0.2243                   # sine of the Cabibbo angle (PDG 2022, eq. 67.18)
    gA       = 1.26                        # axial coupling of the nucleon

    # nuclear physics
    m_amu    = 931.49410242 * MeV          # atomic mass unit (PDG 2021)
    m_Ar40   = 39.948 * m_amu              # atomic mass of argon
    m_O16    = 15.999 * m_amu              # atomic mass of oxygen

    # Various astrophysical constants
    GN    = 6.708e-39/1e18  # eV^-2, Newton's constant
    MPl   = 1.22093e19*GeV   # Planck mass, PDG 2013
    Msun  = 1.989e30*kg
    Rsun  = 6.9551e8*meter
    
    # atomic physics
    a0      = 1. / (m_e * alpha_em)        # Bohr radius
    Ry      = 0.5 * m_e * alpha_em**2      # Rydberg constant
    
    # cosmology
    h       = 0.6766                       # (Planck 2018)
    H0      = h * 100. * km / sec/ Mpc     # Hubble parameter
    rho_c0  = 3. * H0**2/(8. * np.pi * GN) # critical density today, Kolb Turner eq. 3.14
    Omega_m = 0.14240 / h**2               # total matter density (Placnk 2018)
    Omega_Lambda = 0.6889 / h**2           # dark energy density (Planck 2018)

u = my_units()


# tidal deformabilities of a 1.4 M_sun neutron star for different equations of state
# based on tables VI/VII of https://arxiv.org/abs/2209.06052
tidal_deformabilities_14 = {
    'BSk20': 382.30,
    'BSk21': 533.99,
    'BSk22': 642.77,
    'BSk23': 642.77,
    'BSk24': 532.32,
    'BSk25': 495.04,
    'BSk26': 333.57
}


# ---------------------------------------------------------------------------
#             N U M E R I C A L   H E L P E R   F U N C T I O N S
# ---------------------------------------------------------------------------

# vectorized polylog function
v_polylog = np.vectorize(mp.fp.polylog, otypes=[complex])

def my_polylog_exp_3(x):
    """compute polylog(3, exp(x)), with proper approximation for large x"""
    if x < 500.:
        return v_polylog(3, np.exp(x))
    else:
        return -x**3/6 - 0.5j*np.pi*x**2 + np.pi**2*x/3

def my_polylog_exp_4(x):
    """compute polylog(4, exp(x)), with proper approximation for large x"""
    if x < 500.:
        return v_polylog(4, np.exp(x))
    else:
        return -x**4/24 - 1j/6*np.pi*x**3 + np.pi**2*x**2/6 + np.pi**4/45 

def my_polylog_exp_5(x):
    """compute polylog(5, exp(x)), with proper approximation for large x"""
    if x < 500.:
        return v_polylog(5, np.exp(x))
    else:
        return -x**5/120. - 1j/24*np.pi*x**4 + np.pi**2*x**3/18 + np.pi**4*x/45

def my_polylog_exp_6(x):
    """compute polylog(6, exp(x)), with proper approximation for large x"""
    if x < 500.:
        return v_polylog(6, np.exp(x))
    else:
        return - x**6/720 - 1j/120*np.pi*x**5 + np.pi**2*x**4/72 \
               + np.pi**4*x**2/90 + 2*np.pi**6/945


# ---------------------------------------------------------------------------
#              T H E R M A L   D I S T R I B U T I O N S
# ---------------------------------------------------------------------------

def f_FD(p, m, mu, T):
    '''Fermi Dirac distribution for a fermion of mass m, momentum p
       at chemical potential mu and temperature T'''
    if p > 50.*T or -mu > 50.*T:
        return 2. * np.exp( -(np.sqrt(p**2 + m**2) - mu) / T )  # factor 2 = dof
    else:
        return 2. / (np.exp( (np.sqrt(p**2 + m**2) - mu) / T ) + 1.)

def f_FD_E(E, mu, T):
    """the Fermi-Dirac distribution, evaluated at temperature T,
       energy E and chemical potential mu."""
    return 2. / (np.exp((E - mu) / T) + 1)



# ---------------------------------------------------------------------------
def n_FD(m, mu, T):
    '''Number density of a fermion of mass m
       at chemical potential mu and temperature T'''
    return (4.*np.pi)/(2*np.pi)**3 \
         * integ.quad(lambda p: p**2 * f_FD(p, m, mu, T), 0, 50.*max(T,mu))[0]

# ---------------------------------------------------------------------------
def chemical_potential(m, n, T):
    '''chemical potential of a species as a function of its number density'''
    mu_max_rel = 100. * T*np.log(n/T**3)
    mu_max_nr  = 100. * ( m - T*np.log((m*T/(2*np.pi))**1.5 / n) )
      # upper interval boundary for brentq in the relativistic/non-relativistic regime
    try:
        with np.errstate(over='ignore', divide='ignore'):
            if T < 0.3*m:
                return opt.brentq(lambda mu: n_FD(m, mu, T) - n, 0., mu_max_nr)
            else:
                return opt.brentq(lambda mu: n_FD(m, mu, T) - n, 0., mu_max_rel)
    except ValueError:   # if the boundaries don't straddle the root, there is none
        return np.nan


# ---------------------------------------------------------------------------
#                            U R C A   R A T E S
# ---------------------------------------------------------------------------

# ---------- Muon Production via Direct Urca Processes ----------
def I_DU_prime(B):
    """integral from A23 in the paper

       Arguments:
           B: \beta * (\mu_p - \mu_n + \mu_\mu), a measure for how far
              the system is out of equilibrium"""

    B_orig = B
    B = np.fmax(B, -600)
    z = -np.exp(-B)
    if B > 50.:
        ll = -B
    elif B < -50.:
        ll = B
    else:
        ll = np.log((1 + np.exp(-B)) / (1 + np.exp(B)))
    result = - 0.5 * B**2 * np.pi**2 * (B + ll) \
             - (B**2 + np.pi**2) * np.real(v_polylog(3, z)) \
             - 6 * B * np.real(v_polylog(4, z)) \
             - 12 * np.real(v_polylog(5, z))
    old_err = np.seterr(all='ignore')
    result = np.where(B != 0., result * (B_orig/B)**5, result)
    np.seterr(**old_err)
    return result
       
def Gamma_prod_DU(T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the muon production rate via direct Urca processes,
       see eq. A25 in the paper. It is assumed that electrons are in
       beta equilibrium (mu_e = mu_n - mu_p), while mu_mu may
       be away from beta equilibrium.

       Arguments:
           T:       temperature
           m_p_eff: effective proton mass
           m_n_eff: effective neutron mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    # note: the following is very sensitive to small deviations of B from zero.
    # NSCool results that are supposed to correspond to a star in equilibrium
    # have small differences between mu_e and mu_mu due to numerical error.
    # this needs to taken into account when comparing e.g. the in-equilibrium
    # Urca production and detection rates
    mu_e  = np.sqrt(pF_e**2  + u.m_e**2)
    mu_mu = np.sqrt(pF_mu**2 + u.m_mu**2)
    B    = (mu_mu - mu_e) / T
    result = np.where( (np.abs(pF_n-pF_mu)<pF_p) & (pF_p<np.abs(pF_n+pF_mu)),
                       m_n_eff * m_p_eff * mu_mu * T**5 / (4*np.pi**5) \
                     * u.GF**2 * (1 - u.sin_theta_C**2) * (1 + 3*u.gA**2) \
                     * I_DU_prime(B),
                      0. )
    return result


# ---------- Muon Production via Modified Urca Processes ----------
def I_MU_prime(B):
    """integral from eq. A40 in the paper"""

    B_orig = B
    B = np.fmax(B, -600)    
    z = -np.exp(-B)
    if B > 50.:
        ll = -B
    elif B < -50.:
        ll = B
    else:
        ll = np.log((1 + np.exp(-B)) / (1 + np.exp(B)))
    result = - 3/8 * B**2 * np.pi**4 * (B + ll) \
             - 1/12 * (B**2 + np.pi**2) * (B**2 + 9*np.pi**2) * np.real(v_polylog(3, z)) \
             - B * (B**2 + 5*np.pi**2) * np.real(v_polylog(4, z)) \
             - 2 * (3*B**2 + 5*np.pi**2) * np.real(v_polylog(5, z)) \
             - 20 * B * np.real(v_polylog(6, z)) \
             - 30 * np.real(v_polylog(7, z))
    old_err = np.seterr(all='ignore')
    result = np.where(B != 0., result * (B_orig/B)**7, result)
    np.seterr(**old_err)
    return result

def Gamma_prod_MU(T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the muon production rate via modified Urca processes,
       see eq. A42 in the paper.  It is assumed that electrons are in
       beta equilibrium (mu_e = mu_n - mu_p), while mu_mu may
       be away from beta equilibrium.

       Arguments:
           T:       temperature
           m_p_eff: effective proton mass
           m_n_eff: effective neutron mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    mu_e    = np.sqrt(pF_e**2  + u.m_e**2)
    mu_mu   = np.sqrt(pF_mu**2 + u.m_mu**2)
    n_n     = pF_n**3 / (3*np.pi**2)
    B       = (mu_mu - mu_e) / T
    n_0     = 0.16 / u.fm**3
    alpha_n = np.fmax(1.76 - 0.634 * (n_0/n_n)**(2/3), 0.0)
    beta_n  = 0.68
    g_piNN  = 1.
    return u.GF**2 * u.gA**2 * (1 - u.sin_theta_C**2) * m_n_eff**3 * m_p_eff * T**7 / np.pi**9 \
         * pF_p * pF_mu / mu_mu * alpha_n * beta_n * (g_piNN/u.m_pi)**4 \
         * I_MU_prime(B)


# ---------- Muon Absorption via Direct Urca Processes ----------
def I_DU_pprime(A):
    """integral from A60 in the paper (omitting the 1/|p_\mu| prefactor)"""

    A_orig = A
    A = np.fmax(A, -600)
    z = np.exp(-A)
    result = 2. * ( A * np.real(v_polylog(3, z))
                  + 3 * np.real(v_polylog(4, z) ))
    old_err = np.seterr(all='ignore')
    result = np.where(A != 0., result * (A_orig/A)**4, result)
    np.seterr(**old_err)
    return result

def Gamma_abs_DU(E_mu, T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the muon absorption rate via direct Urca processes,
       see eq. A61 in the paper.  It is assumed that electrons are in
       beta equilibrium (mu_e = mu_n - mu_p), while mu_mu may
       be away from beta equilibrium.

       Arguments:
           E_mu:    muon energy
           T:       temperature
           m_p_eff: effective proton mass
           m_n_eff: effective neutron mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    mu_e   = np.sqrt(pF_e**2  + u.m_e**2)
    mu_mu  = np.sqrt(pF_mu**2 + u.m_mu**2)
    A      = (mu_e - E_mu) / T
    result = np.where( (np.abs(pF_n-pF_mu)<pF_p) & (pF_p<np.abs(pF_n+pF_mu)),
                       0.5 * m_n_eff * m_p_eff * T**4 / (2*np.pi**3) * u.GF**2 \
                     * (1 - u.sin_theta_C**2) * (1 + 3*u.gA**2) \
                     * I_DU_pprime(A) / np.sqrt(np.fmax(1e-10, E_mu**2 - u.m_mu**2)),
                       0. )  # factor of 2 for the muon spin average
    return result

def Gamma_abs_DU_integ(T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the muon absorption rate via direct Urca processes,
       multiplied with the muon momentum distribution and integrated over
       muon momenta.

       Arguments:
           T:       temperature
           m_p_eff: effective proton mass
           m_n_eff: effective neutron mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    mu_mu    = np.sqrt(pF_mu**2 + u.m_mu**2)
    E_table  = np.linspace(mu_mu-10*T, mu_mu+10*T, 101)
    p_table  = np.sqrt(np.fmax(1e-10, E_table**2 - u.m_mu**2))
    f_table  = 2 * 4*np.pi/(2*np.pi)**3 * E_table * p_table / (np.exp((E_table - mu_mu)/T) + 1)
                                                 # factor 2 for two spin orientations
    Gamma_table = Gamma_abs_DU(E_table, T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu)
    return integ.trapz(x=E_table, y=f_table*Gamma_table)

    # the following is an alternative implementation using integ.quad
#    mu_mu = np.sqrt(pF_mu**2 + u.m_mu**2)
#    def abs_DU_integrand(E_mu):
#        p_mu = np.sqrt(np.fmax(1e-10, E_mu**2 - u.m_mu**2))
#        return 2 * 4*np.pi/(2*np.pi)**3 * E_mu*p_mu / (np.exp((E_mu - mu_mu)/T) + 1) \
#                * Gamma_abs_DU(E_mu, T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu)
#    return integ.quad(abs_DU_integrand, -np.inf, mu_mu)[0] \
#         + integ.quad(abs_DU_integrand,  mu_mu,  np.inf)[0]


# ---------- Muon Absorption via Modified Urca Processes ----------
def I_MU_pprime(A):
    """integral from A71 in the paper (omitting the 1/E_\mu^2 prefactor)"""

    A_orig = A
    A = np.fmax(A, -600)
    z = np.exp(-A)
    result = (     (A**3 + 4 * np.pi**2 * A) * np.real(v_polylog(3, z))
             + 3 * (3*A**2 + 4*np.pi**2) * np.real(v_polylog(4, z))
             + 36 * A * np.real(v_polylog(5, z))
             + 60 * np.real(v_polylog(6, z)) ) / 3.

    old_err = np.seterr(all='ignore')
    result = np.where(A != 0, result * (A_orig/A)**6, result)
    np.seterr(**old_err)
    return result

def Gamma_abs_MU(E_mu, T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the muon absorption rate via modified Urca processes,
       see eq. A73 in the paper.  It is assumed that electrons are in
       beta equilibrium (mu_e = mu_n - mu_p), while mu_mu may
       be away from beta equilibrium.

       Arguments:
           E_mu:    muon energy
           T:       temperature
           m_p_eff: effective proton mass
           m_n_eff: effective neutron mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    mu_e    = np.sqrt(pF_e**2  + u.m_e**2)
    mu_mu   = np.sqrt(pF_mu**2 + u.m_mu**2)
    n_n     = pF_n**3 / (3*np.pi**2)
    n_0     = 0.16 / u.fm**3
    alpha_n = np.fmax(1.76 - 0.634 * (n_0/n_n)**(2/3), 0.0)
#    uu      = u.m_pi / (2*pF_n)
#    alpha_n = 1 - 1.5 * uu * np.arctan(1/uu) + 0.5*uu**2/(1 + uu**2)
    beta_n  = 0.68
    g_piNN  = 1.
    A       = (mu_e - E_mu) / T
    return 0.5 * 2 * u.GF**2 * (1-u.sin_theta_C**2) * u.gA**2 * m_n_eff**3 * m_p_eff * T**6 / np.pi**7 \
         * pF_p * alpha_n * beta_n * (g_piNN / u.m_pi)**4 \
         * I_MU_pprime(A) / E_mu**2   # factor of 0.5 for the muon spin average

def Gamma_abs_MU_integ(T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the muon absorption rate via modified Urca processes,
       multiplied with the muon momentum distribution and integrated over
       muon momenta.

       Arguments:
           T:       temperature
           m_p_eff: effective proton mass
           m_n_eff: effective neutron mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    mu_mu    = np.sqrt(pF_mu**2 + u.m_mu**2)
    E_table  = np.linspace(np.fmax(u.m_mu,mu_mu-10*T), mu_mu+10*T, 101)
    p_table  = np.sqrt(E_table**2 - u.m_mu**2)
    f_table  = 2 * 4*np.pi/(2*np.pi)**3 * E_table * p_table / (np.exp((E_table - mu_mu)/T) + 1)
                                                 # factor 2 for two spin orientations
    Gamma_table = Gamma_abs_MU(E_table, T, m_p_eff, m_n_eff, pF_p, pF_n, pF_e, pF_mu)
    return integ.trapz(x=E_table, y=f_table*Gamma_table)

# -------------------- Assisted Muon Decay --------------------
def I_amd(C):
    """the integral appearing in the calculation of the assisted muon decay
       rate, see eq. A85 in the paper.

       Arguments:
           C: the deviation from equilibrium, beta*(mu_e - mu_mu)"""

    C_orig = C
    C = np.fmax(C, -600)
    z = -np.exp(-C)
    result = - 2. * (C**2 + np.pi**2) * np.real(v_polylog(6, z)) \
             - 24. * C * np.real(v_polylog(7, z)) \
             - 84. * np.real(v_polylog(8, z))
    old_err = np.seterr(all='ignore')
    result = np.where(C != 0., (C_orig/C)**8 * result, result)
    np.seterr(**old_err)
    return result

def Gamma_amd(E_mu, T, m_p_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the rate of assisted muon decay following eq. A86 in the paper.

       Arguments:
           E_mu:    muon energy
           T:       temperature
           m_p_eff: effective proton mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    mu_e  = np.sqrt(pF_e**2  + u.m_e**2)
    return np.where(pF_p <= 0., 0.,
                    16 * m_p_eff**2 * mu_e * E_mu**2 * u.alpha_em**2 * u.GF**2 \
                       / (np.pi**5/T**8 * u.m_mu**8) \
                       * I_amd((mu_e - E_mu)/T))
        # we set the assisted muon decay rate to zero in regions without protons.
        # the approximations made during the derivation of this expression
        # (notably extending the range of the x integrals to [-\infty, \infty])
        # are valid only if the proton Fermi surface lies well above the proton mass.

def Gamma_amd_integ(T, m_p_eff, pF_p, pF_n, pF_e, pF_mu):
    """compute the rate of assisted muon decay (following eq. A86 in the paper),
       multiplied with the muon momentum distribution and integrated over
       muon momenta.

       Arguments:
           T:       temperature
           m_p_eff: effective proton mass
           pF_p:    proton Fermi momentum
           pF_n:    neutron Fermi momentum
           pF_e:    electron Fermi momentum
           pF_mu:   muon Fermi momentum"""

    if pF_p <= 0.:
        return 0
    mu_mu   = np.sqrt(pF_mu**2 + u.m_mu**2)
    E_table = np.linspace(np.fmax(u.m_mu,mu_mu-10*T), mu_mu+10*T, 101)
    p_table = np.sqrt(E_table**2 - u.m_mu**2)
    f_table = 2 * 4*np.pi/(2*np.pi)**3 * E_table * p_table / (np.exp((E_table - mu_mu)/T) + 1)
    Gamma_table = Gamma_amd(E_table, T, m_p_eff, pF_p, pF_n, pF_e, pF_mu)
                                                 # factor 2 for two spin orientations
    return integ.trapz(x=E_table, y=f_table*Gamma_table)


# ---------------------------------------------------------------------------
#    M U O N   D E C A Y   R A T E   W I T H   P A U L I   B L O C K I N G
# ---------------------------------------------------------------------------

# muon width in the SM
mu_width_SM = u.GF**2 * u.m_mu**5 / (192.*np.pi**3)

# the following is based on Toby's worksheet muon-decay-v2.ipynb
def mu_width(p_mu, mu, T, use_quad=True):
    """compute the muon decay width in presence of a non-zero electron chemical potential.

       Arguments:
           p_mu:     muon momentum
           mu:       electron chemical potential
           T:        temperature
           use_quad: True  = use scipy.integrate.nquad for numerical integration
                             (fast, but less accurate)
                     False = use scipy.integrate.quadrature
                             (slow, but more accurate)"""
    
    Emu = np.sqrt(u.m_mu**2. + p_mu**2.)
    
    def diff_gamma(c_omega, m12_2, E_mu):
        res=u.GF**2./(192.*u.m_mu**2.*np.pi**3.*E_mu)
        res*=(u.m_mu**6.-3.*u.m_mu**2.*m12_2**2.+2.*m12_2**3.)
        return res

    def pe_boosted(c_omega, m12_2, p_mu) :
        Emu = np.sqrt(u.m_mu**2. + p_mu**2.)
#        res = np.sqrt(u.m_mu**2.-m12_2)*(Emu-p_mu*c_omega)/u.m_mu
        res = (u.m_mu**2.-m12_2)*(Emu-p_mu*c_omega)/(2.*u.m_mu**2.)
        return res

    def phase_space_integrand(c_omega, m12_2) :
        pe = pe_boosted(c_omega, m12_2, p_mu)
        x = np.clip((pe - mu)/T, -100., 100.)
        pblock = np.where( x > 20.,
                           1. - np.exp(-x),
                           1. - 1./(1. + np.exp(x)) )
        gamma = diff_gamma(c_omega, m12_2, Emu)
        return pblock*gamma
    
    # print(pe_boosted(1., u.m_mu**2./4, 0.1*u.m_mu)/u.MeV)
    # print(diff_gamma(1,u.m_mu**2./4,np.sqrt(u.m_mu**2.+(0.1*u.m_mu)**2.))*u.MeV)
    
    # Two different methods:
    #    -- quad is fast but with some bumps appearing (see below)
    #    -- quadrature is slow but reproduces exactly the mathematica result
    if use_quad == True :
        integral, err = integ.nquad(phase_space_integrand, [[-1,1],[0.,u.m_mu**2.]])

    else :
        def int2(m12_2):
            partial_func = lambda c_omega: phase_space_integrand(c_omega, m12_2)
            integral, err = integ.quadrature(partial_func, -1., 1., vec_func=False, miniter=100, maxiter=500)
            return integral

        integral, error = integ.quadrature(int2, 0., u.m_mu**2., vec_func=False, miniter=100, maxiter=500)
    
    return integral


## the following code was copied from Toby's worksheet
## pauli_blocking.ipynb on 17.04.2020)
## an error was discovered in February 2022 - use the above, corrected, code instead
##mu_width_SM = u.GF**2.*u.m_mu**5./(192.*np.pi**3.)
#
## ---------------------------------------------------------------------------
#def mu_width(p_mu, mu, T, use_quad=True) : 
#    '''muon decay width in the presence of Pauli blocking'''
#    Emu = np.sqrt(u.m_mu**2. + p_mu**2.)
#    
#    def diff_gamma(c_th, m12_2):
#        res=u.GF**2./(128.*u.m_mu**3.*np.pi**3.)*(1.+c_th) \
#                    *(m12_2-u.m_mu**2.)**2.*((1.+c_th)*m12_2+(1.-c_th)*u.m_mu**2.)
#        return res
#
#    def pe_2_boosted(c_th, m12_2, p_mu) :
#        res = (-4*(-1 + c_th**2)*m12_2*u.m_mu**4 \
#              + ((-1 + c_th)*Emu*m12_2 + (1 + c_th)*Emu*u.m_mu**2)**2 \
#              + 2*Emu*(-((-1 + c_th)**2*m12_2**2.) + (1 + c_th)**2*u.m_mu**4)*p_mu \
#              + ((-1 + c_th)*m12_2 - (1 + c_th)*u.m_mu**2)**2*p_mu**2) \
#              / (16.*u.m_mu**4)
#        return res
#
#    def pauli_blocking(pe_2, mu, T) :
#        pe = np.sqrt(pe_2)
#        return (1.-1./(1.+np.exp((pe-mu)/T)))
#
#    def phase_space_integrand(c_th, m12_2) :
#        pe2 = pe_2_boosted(c_th, m12_2, p_mu)
#        pblock = pauli_blocking(pe2, mu, T)
#        gamma = diff_gamma(c_th, m12_2)
#
#        return u.m_mu/Emu*pblock*gamma
#    
#    # Two different methods:
#    #    -- quad is fast but with some bumps appearing (see below)
#    #    -- quadrature is slow but reproduces exactly the mathematica result
#    
#    if use_quad == True :
#        integral, err = integ.nquad(phase_space_integrand, [[-1,1],[0.,u.m_mu**2.]])
#
#    else :
#        def int2(m12_2):
#            partial_func = lambda c_th: phase_space_integrand(c_th, m12_2)
#            integral, err = integ.quadrature(partial_func, -1., 1.,
#                                             vec_func=False, miniter=100, maxiter=500)
#            return integral
#
#        integral, error = integ.quadrature(int2, 0., u.m_mu**2.,
#                                           vec_func=False, miniter=100, maxiter=500)
#    
#    return integral


# ---------------------------------------------------------------------------
#      M U O N   D E C A Y   E V E N T   G E N E R A T I O N
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def dGamma_mu_dE_nu(E_nu):
    '''neutrino spectrum from muon decay (see PDG review on muon decay parameters'''
    y = 2 * E_nu / u.m_mu
    return np.where((y<0) | (y>1), 0.,
                    2. * u.m_mu**4 * u.GF**2/(16*np.pi**3)
                       * np.maximum(0., y**2 * (1 - y)))

# ---------------------------------------------------------------------------
def dGamma_mu_dE_e(E_e):
    '''electron energy from muon decay (same as neutrino energy)'''
    x = 2 * E_e / u.m_mu
    return np.where((x<0) | (x>1), 0.,
                    2. * u.m_mu**4 * u.GF**2/(96*np.pi**3)
                       * np.maximum(0., 3*x**2 - 2*x**3))

# ---------------------------------------------------------------------------
def pdf_cos_alpha(m12, cos_alpha):
    '''angle of each neutrino from muon decay relative to (12) system
       using eq. 4.13 from https://is.cuni.cz/webapps/zzp/download/130133258
       for the angular distribution of neutrinos for given m12'''
    norm = 4/3 * (2*m12**6 - 3*m12**4 * u.m_mu**2 + u.m_mu**6)
    return np.where( (cos_alpha<-1) | (cos_alpha>1), 0.,
             1/norm * (1 - cos_alpha) * (u.m_mu**2 - m12**2)**2 \
                    * (u.m_mu**2 + m12**2 + (u.m_mu**2 - m12**2) * cos_alpha) )


# ---------------------------------------------------------------------------
def lorentz_boost(gamma, n):
    '''Lorentz boost matrix in 4-vector space for a boost
       in direction n (a 3-vector) with boost factor gamma'''
    beta = np.sqrt(1 - 1/gamma**2)
    return np.array([[gamma,             gamma*beta*n[0],       gamma*beta*n[1],
                                                                  gamma*beta*n[2]],
                     [gamma*beta*n[0], 1+(gamma-1)*n[0]**2,     (gamma-1)*n[0]*n[1],
                                                                  (gamma-1)*n[0]*n[2]],
                     [gamma*beta*n[1],   (gamma-1)*n[0]*n[1], 1+(gamma-1)*n[1]**2,
                                                                  (gamma-1)*n[1]*n[2]],
                     [gamma*beta*n[2],   (gamma-1)*n[0]*n[2],   (gamma-1)*n[1]*n[2],
                                                                  1+(gamma-1)*n[2]**2]
                    ])

# ---------------------------------------------------------------------------
def process_mu_decay_at_rest():
    '''Generate a spectrum of neutrino energies from free muon decay at rest'''
    E_mu = np.full(10000, u.m_mu)
    
    # pick electron energy (in muon rest frame)
    umax = np.sqrt(u.tau_mu*dGamma_mu_dE_e(0.5*u.m_mu)) 
                                    # sqrt(dGamma/dx) has max. at x=1
    vmax = 0.5*u.m_mu * np.sqrt(u.tau_mu*dGamma_mu_dE_e(0.5*u.m_mu))
                                    # y*sqrt(dGamma/dy) has max. a x=1 (within 0<x<1)
    E_e  = stats.rvs_ratio_uniforms(lambda E_e: u.tau_mu * dGamma_mu_dE_e(E_e),
                                    umax=umax, vmin=0, vmax=vmax, size=E_mu.shape)

    E_12_0    = u.m_mu - E_e                # energy of nu system in muon rest frame
    m_12_0    = np.sqrt(E_12_0**2 - E_e**2) # mass of nu system in muon rest frame
    
    sqrt_norm = np.sqrt( 4/3 * (2*m_12_0**6 - 3*m_12_0**4 * u.m_mu**2 + u.m_mu**6) )
    def uu(cos_alpha):
        return np.sqrt(pdf_cos_alpha(m_12_0, cos_alpha))
    def vv(cos_alpha):
        return cos_alpha * np.sqrt(pdf_cos_alpha(m_12_0, cos_alpha))
    u_table   = np.array([ uu(-1),
                           uu(m_12_0**2 / (m_12_0**2 - u.m_mu**2)),
                           uu(1) ])
    v_table   = np.array([ vv(-1),
                           vv(1),
                           vv((3*m_12_0**2 - np.sqrt(m_12_0**4 + 8*u.m_mu**4))
                                / (4*m_12_0**2 - 4*u.m_mu**2)),
                           vv((3*m_12_0**2 + np.sqrt(m_12_0**4 + 8*u.m_mu**4))
                                / (4*m_12_0**2 - 4*u.m_mu**2)) ])
    umax      = np.amax(u_table, axis=0)
    vmin      = np.amin(v_table, axis=0)
    vmax      = np.amax(v_table, axis=0)
    cos_alpha = np.array([ stats.rvs_ratio_uniforms(lambda x: pdf_cos_alpha(m12, x),
                             umax=u, vmin=v1, vmax=v2)[0]
                                for m12, u, v1, v2 in zip(m_12_0, umax, vmin, vmax) ])
    
    gamma_12  = E_12_0 / m_12_0                 # Lorentz boost of neutrino (12) system
    beta_12   = np.sqrt(1 - 1/gamma_12**2)      # velocity of neutrino (12) system
    E_nu_1    = 0.5*m_12_0 * gamma_12 * (1 - beta_12*cos_alpha)
    E_nu_2    = 0.5*m_12_0 * gamma_12 * (1 + beta_12*cos_alpha)
                    # neutrino energies boosted from m12 rest frame to muon rest frame
                
    return np.ravel(np.column_stack((E_nu_1, E_nu_2)))


def process_mu_decay(E_mu, kfe, decay_modes=None):
    '''Generate neutrino energies from the decay of muons with energies E_mu
       in the presence of an electron chemical potential (electron Fermi energy) kfe.
       The function works in the T=0 limit. For combinations of E_mu and kfe for which
       no solution exists in this limit, neutrino energies are set to zero, which
       should be interpreted as a close-to-thermal spectrum at a low temperature.

       Arguments:
           E_mu:        table of muon energies
           kfe:         table of electron Fermi momenta
           decay_modes: a table indicating for each muon whether it decayed through
                        free ('f') or assisted ('a') muon decay. If not given, all
                        muons are assumed to have decayed via free muon decay.'''

    # consider first only for muons that decay freely (kfe < 0.5*E_mu)
    if decay_modes is None:
        i_free = np.arange(len(E_mu))
    else:
        i_free = np.where(decay_modes == 'f')[0]
    gamma_mu  = E_mu[i_free] / u.m_mu           # Lorentz boost of muon
    beta_mu   = np.sqrt(1 - 1/gamma_mu**2)      # velocity of muon

    # pick electron energy, requiring it to be above the Fermi energy.
    i_list    = np.arange(len(i_free))
    Lambda_mu = np.full((4,4,*i_list.shape), 0.)
    p_e       = np.full((4,*i_list.shape), 0.)
    E_e0      = np.full(i_list.shape, 0.)
    E_e       = np.full(i_list.shape, 0.)
    while len(i_list) > 0:
        # direction of electron relative to muon
        cos_theta             = rnd.uniform(-1, 1, size=i_list.shape)
        sin_theta             = np.sqrt(1 - cos_theta**2)
        phi                   = rnd.uniform(0, 2*np.pi, size=i_list.shape)
        cos_phi               = np.cos(phi)
        sin_phi               = np.sin(phi)
        Lambda_mu[:,:,i_list] = lorentz_boost(gamma_mu[i_list],
                       np.array([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta]))

        # determine minimum electron energy needed (in the muon rest frame) to
        # overcome Pauli blocking. To do so, we transform k_{Fe} into the muon rest frame.
        # There is a chance that the decays of all remaining muons are forbidden
        # because we work in the T=0 limit here, while the routines that select
        # muons for decay take into account the thermal distributions. We ignore these
        # muons for the moment - they will be dealt with at the end of the function.
        E_e_min       = max(0, np.amin( gamma_mu[i_list] * (1 - beta_mu[i_list]) * kfe[i_list] ))
        Gamma_mu_max  = integ.quad(dGamma_mu_dE_e, E_e_min, 0.5*u.m_mu)[0]
        if Gamma_mu_max < 1e-50*u.MeV:
            ii        = np.setdiff1d(np.arange(len(i_free)), i_list)
            E_e0      = E_e0[ii]
            E_e       = E_e[ii]
            Lambda_mu = Lambda_mu[:,:,ii]
            i_free    = np.setdiff1d(i_free, i_list)
            break

        # generate electron energies in muon rest frame
        tau_mu_min    = 1. / Gamma_mu_max   # minimal muon lifetime
        umax          = np.sqrt(tau_mu_min*dGamma_mu_dE_e(0.5*u.m_mu))
                                # sqrt(dGamma/dx) has max. at x=1
        vmax          = 0.5*u.m_mu * np.sqrt(tau_mu_min*dGamma_mu_dE_e(0.5*u.m_mu))
                                # y*sqrt(dGamma/dy) has max. a x=1 (within 0<x<1)
        try:
            E_e0[i_list]  = stats.rvs_ratio_uniforms(
                                lambda E_e: np.where(E_e>E_e_min, tau_mu_min*dGamma_mu_dE_e(E_e), 0),
                            umax=umax, vmin=0, vmax=vmax, size=i_list.shape)
        # an exception can happen if rvs_ratio_uniforms fails to find points
        # where the PDF is non-zero. In this case, we ignore the remaining muons
        except RuntimeError:
            ii        = np.setdiff1d(np.arange(len(i_free)), i_list)
            E_e0      = E_e0[ii]
            E_e       = E_e[ii]
            Lambda_mu = Lambda_mu[:,:,ii]
            i_free    = np.setdiff1d(i_free, i_list)
            break

        # electron energy in NS frame
        p_e[:,i_list] = np.einsum('ijn,jn->in', Lambda_mu[:,:,i_list],
                                  E_e0[None,i_list] * np.array([1,0,0,-1])[:,None])
        E_e[i_list]   = p_e[0,i_list]

        # determine which configurations violate the Pauli principle
        # they will be replaced in the next iteration
        i_list        = np.nonzero(E_e <= kfe)[0]

#    umax      = np.sqrt(u.tau_mu*dGamma_mu_dE_e(0.5*u.m_mu))
#                                # sqrt(dGamma/dx) has max. at x=1
#    vmax      = 0.5*u.m_mu * np.sqrt(u.tau_mu*dGamma_mu_dE_e(0.5*u.m_mu))
#                                # y*sqrt(dGamma/dy) has max. a x=1 (within 0<x<1)
#    while len(i_list) > 0:
#        # direction of electron relative to muon
#        cos_theta             = rnd.uniform(-1, 1, size=i_list.shape)
#        sin_theta             = np.sqrt(1 - cos_theta**2)
#        phi                   = rnd.uniform(0, 2*np.pi, size=i_list.shape)
#        cos_phi               = np.cos(phi)
#        sin_phi               = np.sin(phi)
#        Lambda_mu[:,:,i_list] = lorentz_boost(gamma_mu[i_list],
#                       np.array([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta]))
#
#        # electron energy in muon rest frame
#        E_e0[i_list]  = stats.rvs_ratio_uniforms(lambda E_e: u.tau_mu*dGamma_mu_dE_e(E_e),
#                                         umax=umax, vmin=0, vmax=vmax, size=i_list.shape)
#
#        # electron energy in NS frame
#        p_e[:,i_list] = np.einsum('ijn,jn->in', Lambda_mu[:,:,i_list],
#                                  E_e0[None,i_list] * np.array([1,0,0,-1])[:,None])
#        E_e[i_list]   = p_e[0,i_list]
#        
#        # determine which configurations violate the Pauli principle
#        # they will be replaced in the next iteration
#        i_list        = np.nonzero(E_e <= kfe[i_free])[0]
   
    # energy of neutrino (12) system in muon rest frame
    E_12_0    = u.m_mu - E_e0                   # energy of nu system in muon rest frame
    m_12_0    = np.sqrt(E_12_0**2 - E_e0**2)    # mass of nu system in muon rest frame

    # pick angle of first neutrino relative to (12) system
    sqrt_norm = np.sqrt( 4/3 * (2*m_12_0**6 - 3*m_12_0**4 * u.m_mu**2 + u.m_mu**6) )
    def uu(cos_alpha):
        return np.sqrt(pdf_cos_alpha(m_12_0, cos_alpha))
    def vv(cos_alpha):
        return cos_alpha * np.sqrt(pdf_cos_alpha(m_12_0, cos_alpha))
    u_table   = np.array([ uu(-1),
                           uu(m_12_0**2 / (m_12_0**2 - u.m_mu**2)),
                           uu(1) ])
    v_table   = np.array([ vv(-1),
                           vv(1),
                           vv((3*m_12_0**2 - np.sqrt(m_12_0**4 + 8*u.m_mu**4))
                                 / (4*m_12_0**2 - 4*u.m_mu**2)),
                           vv((3*m_12_0**2 + np.sqrt(m_12_0**4 + 8*u.m_mu**4))
                                 / (4*m_12_0**2 - 4*u.m_mu**2)) ])
    umax      = np.amax(u_table, axis=0)
    vmin      = np.amin(v_table, axis=0)
    vmax      = np.amax(v_table, axis=0)
    cos_alpha = np.array([ stats.rvs_ratio_uniforms(lambda x: pdf_cos_alpha(m12, x),
                             umax=u, vmin=v1, vmax=v2)[0]
                                for m12, u, v1, v2 in zip(m_12_0, umax, vmin, vmax) ])
    sin_alpha = np.sqrt(1 - cos_alpha**2)
    
    # compute neutrino momenta in muon rest frame
    gamma_12  = E_12_0 / m_12_0                 # Lorentz boost of neutrino (12) system
    beta_12   = np.sqrt(1 - 1/gamma_12**2)      # velocity of neutrino (12) system
    Lambda_12 = lorentz_boost(gamma_12, np.array([0, 0, 1]))
    p_nu_1_0  = np.einsum('ijn,jn->in', Lambda_12, 0.5 * m_12_0[None,:]
      * np.array([np.full(i_free.shape,1), -sin_alpha, np.full(i_free.shape,0), -cos_alpha]))
    p_nu_2_0  = np.einsum('ijn,jn->in', Lambda_12, 0.5 * m_12_0[None,:]
      * np.array([np.full(i_free.shape,1),  sin_alpha, np.full(i_free.shape,0),  cos_alpha]))
    E_nu_1_0  = p_nu_1_0[0]
    E_nu_2_0  = p_nu_2_0[0]
    
    # boost neutrinos into the lab frame
    p_nu_1         = np.einsum('ijn,jn->in', Lambda_mu, p_nu_1_0)
    p_nu_2         = np.einsum('ijn,jn->in', Lambda_mu, p_nu_2_0)
    E_nu_1         = np.zeros(E_mu.shape)
    E_nu_2         = np.zeros(E_mu.shape)
    E_nu_1[i_free] = p_nu_1[0]
    E_nu_2[i_free] = p_nu_2[0]

    # finally, take care of muons decaying via assisted muon decay,
    # as well as those whose decays are forbidden in the T=0 limit (see above)
    # In this case, we randomly distribute the available energy among
    # the two neutrinos (they are undetectable anyway, so we don't aim
    # to be super-precise here)
    i_amd = np.setdiff1d(np.arange(len(E_mu)), i_free)
    E_nu_1[i_amd] = 0.
    E_nu_2[i_amd] = 0.
#    E_nu_1[i_amd] = np.fmax(0., rnd.uniform(0., E_mu[i_amd] - kfe[i_amd]))
#    E_nu_2[i_amd] = E_mu[i_amd] - E_nu_1[i_amd]
    
    return np.ravel(np.column_stack((E_nu_1, E_nu_2)))


# ---------------------------------------------------------------------------
#         S I M U L A T I N G   M U O N   P R O P A G A T I O N
# ---------------------------------------------------------------------------

class muon_propagator:

    # -----------------------------------------------------------------------
    def __init__(self, nscool_dir, mu_width_file=None, tmin=0., tmax=np.inf):
        '''Initialize muon propagator object by reading NSCool output
           from the given data directory.

           Parameters:
               nscool_dir:    NSCool output directory to read in
               mu_width_file: if given, tabulated muon decay rates are read
                              from this file. (Otherwise, such a table is
                              generated and written to mu-width-table.dat
                              in the current working directory.)
               tmin:          ignore NSCool time steps earlier than this value
               tmax:          ignore NSCool time steps later than this value'''

        # Read muon widths
        # ----------------
#        if mu_width_file is None:
#            print("tabulating muon widths ...")
#            p_mu_table      = np.linspace(0*u.MeV,  1000*u.MeV, 101)
#            kfe_table       = np.linspace(0*u.MeV,   500*u.MeV, 101)
#            T_table         = np.logspace(np.log10(1e-6*u.MeV), np.log10(10*u.MeV), 71)
#            mu_width_points = np.array([ [p_mu, kfe, T]
#                   for p_mu in p_mu_table for kfe in kfe_table for T in T_table ])
#            with multiprocess.pool.Pool() as pool:
#                self.mu_width_table = np.array(pool.starmap(
#                        lambda p_mu, kfe, T: mu_width(p_mu, kfe, T, use_quad=True),
#                        mu_width_points))
#            self.mu_width_table = self.mu_width_table.reshape(len(p_mu_table),
#                                                              len(kfe_table),
#                                                              len(T_table))
##            self.mu_width_table = np.array([ [ [ mu_width(p_mu, kfe, T, use_quad=True)
##                   for T in T_table ] for kfe in kfe_table ] for p_mu in p_mu_table ])
#            with open('mu-width-table.dat', 'wb') as f:
#                pickle.dump(self.mu_width_table, f)
#        else:
#            with open(mu_width_file, 'rb') as f:
#                self.mu_width_table = pickle.load(f)
#            p_mu_table = np.linspace(0*u.MeV,  1000*u.MeV, self.mu_width_table.shape[0])
#            kfe_table  = np.linspace(0*u.MeV,   500*u.MeV, self.mu_width_table.shape[1])
#            T_table    = np.logspace(np.log10(1e-6*u.MeV), np.log10(10*u.MeV), self.mu_width_table.shape[2])
#        self.mu_width_interp = interp.RegularGridInterpolator((p_mu_table,
#                                       kfe_table, T_table), self.mu_width_table,
#                                       bounds_error=False, fill_value=np.nan)

        if mu_width_file is None:
            print("tabulating muon widths ...")
            kfe_table       = np.linspace(0*u.MeV, 500*u.MeV, 101)  # electron Fermi momentum
            r_table         = np.linspace(-5, 5, 101) # (E_mu - E_{F,mu}) / T
            T_table         = np.logspace(np.log10(1e-6*u.MeV),np.log10(10*u.MeV),71) # temperature
            mu_width_points = np.array([ [kfe, r, T]
                   for kfe in kfe_table for r in r_table for T in T_table ])
            def p_mu(kfe, r, T):
                """compute muon momentum assuming beta equilibrium (mu_e = mu_mu)
                   everywhere except where mu_e < mu_mu"""
                return np.sqrt(np.fmax(0., (np.fmax(u.m_mu, np.sqrt(kfe**2 + u.m_e**2)) + r*T)**2 - u.m_mu**2))
            with multiprocess.pool.Pool() as pool:
                self.mu_width_table = np.array(pool.starmap(
                    lambda kfe, r, T: mu_width(p_mu(kfe,r,T), kfe, T, use_quad=True),
                    mu_width_points))

            self.mu_width_table = self.mu_width_table.reshape(len(kfe_table),
                                                              len(r_table),
                                                              len(T_table))
            with open('mu-width-table.dat', 'wb') as f:
                pickle.dump(self.mu_width_table, f)
        else:
            with open(mu_width_file, 'rb') as f:
                self.mu_width_table = pickle.load(f)
            kfe_table  = np.linspace(0*u.MeV,   500*u.MeV, self.mu_width_table.shape[0])
            r_table    = np.linspace(-5, 5, self.mu_width_table.shape[1])
            T_table    = np.logspace(np.log10(1e-6*u.MeV), np.log10(10*u.MeV), self.mu_width_table.shape[2])
        self.mu_width_interp = interp.RegularGridInterpolator((kfe_table, r_table, T_table),
                                                              self.mu_width_table,
                                       bounds_error=False, fill_value=np.nan)

        # Read NSCool files
        # -----------------

        # equation of state
        self.eos_file = re.match(" *'(.*)'", np.loadtxt(nscool_dir + 'Cool_Try.in',
                                                 dtype=str, delimiter=';')[3]).group(1)
        self.eos_file = re.match("(.*/packages/nscool/).*", nscool_dir).group(1) \
                            + self.eos_file
        self.eos_metadata = np.loadtxt(self.eos_file, max_rows=1, usecols=(0,1,2), dtype=int)
        self.eos_metadata = { 'itext': self.eos_metadata[0],
                              'imax':  self.eos_metadata[1],
                              'icore': self.eos_metadata[2] }
        self.eos_data_core = np.loadtxt(self.eos_file, skiprows=self.eos_metadata['itext']+1,
                                                       max_rows=self.eos_metadata['icore']-1)
        self.eos_data_core[:,l_rho]   *= u.grams/u.cm**3
        self.eos_data_core[:,l_p]     *= u.dyne/u.cm**2
        self.eos_data_core[:,l_nbar]  *= 1/u.fm**3
        self.eos_data_core_interp = np.array([
            interp.interp1d(self.eos_data_core[:,l_rho], self.eos_data_core[:,j],
                            bounds_error=False, fill_value=0.)
                            for j in range(self.eos_data_core.shape[1]) ])

        # time-independent data
        self.star_file = nscool_dir + '/Star_Try.dat'
        self.star_data = np.loadtxt(self.star_file, skiprows=6, usecols=range(k_mstp+1))
        self.star_data[:,k_r]    *= u.meter             # radius
        self.star_data[:,k_emas] *= u.Msun              # enclosed mass
        self.star_data[:,k_rho]  *= u.grams/u.cm**3     # mass density
        self.star_data[:,k_kfe]  *= 1. / u.fm           # electron Fermi momentum
        self.star_data[:,k_kfmu] *= 1. / u.fm           # muon Fermi momentum
        self.star_data[:,k_kfp]  *= 1. / u.fm           # proton Fermi momentum
        self.star_data[:,k_kfn]  *= 1. / u.fm           # neutron Fermi momentum
        self.star_data[:,k_kfla] *= 1. / u.fm           # hyperon Fermi momentum
        self.star_data[:,k_kfSm] *= 1. / u.fm
        self.star_data[:,k_kfS0] *= 1. / u.fm
        self.star_data[:,k_kfSp] *= 1. / u.fm
        self.star_data[:,k_mstp]  = np.where(self.star_data[:,k_mstp] > 0,
                                             self.star_data[:,k_mstp]*u.m_p, u.m_p)
                                                        # proton effective mass
        self.star_data[:,k_mstn]  = np.where(self.star_data[:,k_mstn] > 0,
                                             self.star_data[:,k_mstn]*u.m_n, u.m_n)
                                                        # neutron effective mass
        self.kfe_interp =interp.interp1d(self.star_data[:,k_r], self.star_data[:,k_kfe],
            bounds_error=False, fill_value=(self.star_data[-1,k_kfe], 1e-6*u.eV))
                  # use kfe=1e-6 eV in regions without electrons as fallback so we don't
                  # need special treatment for particles in kfe=0 regions.
        self.kfmu_interp=interp.interp1d(self.star_data[:,k_r], self.star_data[:,k_kfmu],
            bounds_error=False, fill_value=(self.star_data[-1,k_kfmu], 0.))
        self.kfn_interp=interp.interp1d(self.star_data[:,k_r], self.star_data[:,k_kfn],
            bounds_error=False, fill_value=(self.star_data[-1,k_kfn], 0.))
        self.kfp_interp=interp.interp1d(self.star_data[:,k_r], self.star_data[:,k_kfp],
            bounds_error=False, fill_value=(self.star_data[-1,k_kfp], 0.))
        self.mneff_interp=interp.interp1d(self.star_data[:,k_r], self.star_data[:,k_mstn],
            bounds_error=False, fill_value=(self.star_data[-1,k_mstn], 0.))
        self.mpeff_interp=interp.interp1d(self.star_data[:,k_r], self.star_data[:,k_mstp],
            bounds_error=False, fill_value=(self.star_data[-1,k_mstp], 0.))
                  # JK changed on 06.10.2023 to account for regions where mstp, mstn=0 in the data files
        self.star_data_interp = np.array([ interp.interp1d(self.star_data[:,k_r],
                            self.star_data[:,j], bounds_error=False, fill_value=0.)
                            for j in range(self.star_data.shape[1]) ])

        # impose exact beta equilibrium; NSCool output has tiny numerical inaccuracies
        #   but the Urca rates are very sensitive to these
        self.star_data[:,k_kfmu] = np.sqrt(np.fmax(0, self.star_data[:,k_kfe]**2 + u.m_e**2 - u.m_mu**2 ))

        # extract mass and radius of neutron star
        self.M_NS = self.star_data[-1,k_emas]
        self.R_NS = self.star_data[-1,k_r]

        # read  location of core-crust boundary from the star file
        self.r_CC_boundary = self.star_data[:,k_r][np.loadtxt(self.star_file, max_rows=1, dtype=int)[2]]

        # time-dependent data
        self.temp_file = nscool_dir + '/Temp_Try.dat'
        self.temp_data = np.loadtxt(self.temp_file)
        self.temp_data[:,j_r]        *= u.meter         # radius
        self.temp_data[:,j_rho]      *= u.grams/u.cm**3 # mass density
        self.temp_data[:,j_T]        *= u.Kelvin        # temperature
        self.temp_data[:,j_QmUrca]   *= u.erg/(u.cm**3 * u.sec) # modified Urca luminosity
        self.temp_data[:,j_QdUrca]   *= u.erg/(u.cm**3 * u.sec) # direct Urca luminosity
        self.temp_data[:,j_lambdaE]  *= u.erg/(u.cm*u.sec*u.Kelvin)
                                                 # thermal conductivity due to electrons
        self.temp_data[:,j_lambdaMu] *= u.erg/(u.cm*u.sec*u.Kelvin)
                                                 # thermal conductivity due to muons

        # read time stamps and effective temperatures corresponding to the blocks
        # in the above data
        self.t_table    = []
        self.Teff_table = []
        with open(self.temp_file) as f:
            for l in f:
                match = re.match('.*Time= *([0-9eE\-\+\.]+).*Te_inf *([0-9eE\-\+\.]+)', l)
                if match:
                    self.t_table.append(float(match.group(1)))
                    self.Teff_table.append(float(match.group(2)))
        self.t_table    = np.array(self.t_table)    * u.yrs
        self.Teff_table = np.array(self.Teff_table) * u.Kelvin

        # reshape time-dependent data into an array of dimension (n_t, n_r, n_properties)
        self.temp_data = np.array(np.split(self.temp_data,
               len(self.temp_data) / len(np.unique(self.temp_data[:,j_zone]))))

        # reverse ordering in the r direction. In NSCool output, larger radii come
        # first, which causes problems for instance woth interp.RegularGridInterpolatpor
        # which expects its input arrays to be sorted
        self.temp_data = np.flip(self.temp_data, axis=1)

        # discard time bins that are too early or too late
        t_mask = ((self.t_table >= tmin) & (self.t_table <= tmax))
        self.t_table    = self.t_table[t_mask].copy()
        self.Teff_table = self.Teff_table[t_mask].copy()
        self.temp_data  = self.temp_data[t_mask,:,:].copy()

        # compute rate of modified Urca processes
        # (from the corresponding neutrino luminosity QmUrca)
        # the conversion factor is based on re-evaluating the integral I from
        # Yakovlev/Levenfish
        # (https://ui.adsabs.harvard.edu/abs/1995A%26A...297..717Y/abstract), see also
        # Teukolsky/Shapiro (https://books.google.fr/books?id=d1CRQIcP1zoC, eq. F33),
        # with one factor x_\nu removed
        self.temp_data = np.append(self.temp_data,    # GammaMUrca_E
          (self.temp_data[:,:,j_QmUrca]*0.212138/self.temp_data[:,:,j_T])[:,:,None], axis=2)

        # compute muon Urca rate (NSCool gives only the electron Urca rate)
        # note that muons are *not* included by NSCool
        self.temp_data = np.append(self.temp_data,    # GammaMUrca_Mu
            np.where(self.kfmu_interp(self.temp_data[:,:,j_r]) <= 0.,
                     0.,
                     self.temp_data[:,:,j_GammaMUrcaE]
                         * self.kfmu_interp(self.temp_data[:,:,j_r])
                         / np.sqrt(self.kfmu_interp(self.temp_data[:,:,j_r])**2 + u.m_mu**2)
            )[:,:,None], axis=2)
                #JK

        # now repeat the same for direct Urca processes
        self.temp_data = np.append(self.temp_data,   # GammaDUrca_E
          (self.temp_data[:,:,j_QdUrca]*0.235889/self.temp_data[:,:,j_T])[:,:,None], axis=2)
        self.temp_data = np.append(self.temp_data,   # GammaDUrca_Mu
                          np.where(self.kfmu_interp(self.temp_data[:,:,j_r]) <= 0.,
                              0., self.temp_data[:,:,j_GammaDUrcaE])[:,:,None], axis=2)

        # interpolate time-dependent data
        self.r_table = self.temp_data[0,:,j_r]
        self.temp_data_interp1d = np.array([ [ interp.interp1d(self.temp_data[it,:,j_r],
               self.temp_data[it,:,j], bounds_error=False, fill_value=0.) 
                                          for j in range(self.temp_data.shape[2]) ]
                                            for it in range(self.temp_data.shape[0]) ])
        self.temp_data_interp2d = np.array([
              interp.RegularGridInterpolator((self.t_table, self.temp_data[0,:,j_r]),
                                              self.temp_data[:,:,j],
                                              bounds_error=False, fill_value=0.) 
                                        for j in range(self.temp_data.shape[2]) ])

        # compute mass enclosed within given radius
        # (may be needed for calculation of gravitational potential V(r))
#        r_range          = self.temp_data[0][:,j_r]
#        self.rho_interp  = interp.interp1d(self.star_data[:,k_r],
#                                           self.star_data[:,k_rho])
#        self.M_in_table  = np.array([ integ.quad(lambda r: 4*np.pi*r**2
#                                                 * self.rho_interp(r), 0., R)[0]
#                                                                 for R in r_range ])
#        self.M_in_interp = interp.interp1d(r_range, self.M_in_table,
#                             bounds_error=False, fill_value=(0.,self.M_in_table[-1]))

        # Tabulate rescaling factor for absorption rates
        # ----------------------------------------------
        # rescale mUrca rate to account for deviation from equilibrium, see urca.nb 
        a            = np.concatenate((-np.logspace(10,-10,101), np.logspace(-10,10,101)))
        polylog_a3   = np.array([ np.real(my_polylog_exp_3(x)) for x in a ])
        polylog_a4   = np.array([ np.real(my_polylog_exp_4(x)) for x in a ])
        polylog_a5   = np.array([ np.real(my_polylog_exp_5(x)) for x in a ])
        polylog_a6   = np.array([ np.real(my_polylog_exp_6(x)) for x in a ])
        self.s_mUrca = ( (-a**3 - 4*np.pi**2 * a)   * polylog_a3
                       + (9*a**2 + 12*np.pi**2)     * polylog_a4
                       - 36 * a                     * polylog_a5
                       + 60                         * polylog_a6 ) \
                     * 2/(3*np.pi**7) * 30240./11513. * 2*np.pi * 0.5  # cf. A71
                          # extra factor 0.5 for initial state electron spin average
        self.a_table        = a
        self.s_mUrca_interp = interp.interp1d(self.a_table, self.s_mUrca,
                                  bounds_error=False, fill_value=(self.s_mUrca[0],self.s_mUrca[-1]))

        # rescaling for the dUrca rate
        self.s_dUrca = 2 * ( -a * polylog_a3 + 3 * polylog_a4 ) \
                         * 2520./(457.*np.pi**4) * 0.5  # cf. A60
        self.s_dUrca_interp = interp.interp1d(self.a_table, self.s_dUrca,
                                  bounds_error=False, fill_value=(self.s_dUrca[0],self.s_dUrca[-1]))


    # ---------- Muon Production via Direct Urca Processes ----------
    def f_DU(self, t, r, f_lost):
        """rescaling factor for the direct Urca production rate in the case of
           deviations from equilibrium see eq. A47 in the paper
           
           Parameters:
               t:      age of the neutron star
               r:      radial position
               f_lost: fraction of muons that have been lost
                       (parameterizing the deviation from equilibrium)"""
        
        T      = self.temp_data_interp2d[j_T]((t,r))
        if T <= 0:
            return 0.
        QdUrca = self.temp_data_interp2d[j_QdUrca]((t,r))
        kfn    = self.kfn_interp(r) 
        kfp    = self.kfp_interp(r) 
        kfe    = self.kfe_interp(r) 
        kfmu   = self.kfmu_interp(r) * max(0, (1 - f_lost)**(1/3))
                                               # rescale to account for lost muons
        mue    = np.sqrt(u.m_e**2  + kfe**2)
        mumu   = np.sqrt(u.m_mu**2 + kfmu**2)
        B      = 1/T * (mumu - mue)      # use mue = mun - mup as we can't compute mun, mup
        if kfn - kfp + kfmu > 0:
            return QdUrca * 2520/T / (457*np.pi**6) * I_DU_prime(B)
        else:
            return 0
            # FIXME there should be no QdUrca here, right?

    # ---------- Muon Production via Modified Urca Processes ----------
    def f_MU(self, t, r, f_lost):
        """rescaling factor for the modified Urca production rate in the case
           of deviations from equilibrium see eq. A48 in the paper
           
           Parameters:
               t:      age of the neutron star
               r:      radial position
               f_lost: fraction of muons that have been lost
                       (parameterizing the deviation from equilibrium)"""
        
        T      = self.temp_data_interp2d[j_T]((t,r))
        if T <= 0:
            return 0.
        QmUrca = self.temp_data_interp2d[j_QmUrca]((t,r))
        kfe    = self.kfe_interp(r) 
        kfmu   = self.kfmu_interp(r) * max(0, (1 - f_lost)**(1/3))
                                               # rescale to account for lost muons
        mue    = np.sqrt(u.m_e**2  + kfe**2)
        mumu   = np.sqrt(u.m_mu**2 + kfmu**2)
        B      = 1/T * (mumu - mue)      # use mue = mun - mup as we can't compute mun, mup
        return QmUrca * 60480/T / (11513*np.pi**8) * I_MU_prime(B) * kfmu/mumu   # A48
            # the factor 60480/(11513*np.pi**8) * I_MU_prime(0) ~= 0.5 * 0.212138 is the conversion
            # factor between QmUrca and GammaUrca used in muon_propagator.py. The factor 0.5 is for
            # the fact that here we're only interested in muon production, whereas QmUrca counts
            # both absorption ad production
            # the factor kfmu/mumu corrects for the fact that in the derivation of I_MU_prime, the
            # ultrarelativistic approximation was made when dealing with the momentum and energy
            # prefactors inside the phase space integrals

            # FIXME there should be no QmUrca here, right?


    # -----------------------------------------------------------------------
    def simulate_muons_vectorized(self, N_samples, t_bin_edges, Z=0,
                                  dt0=100*u.sec, dr_min=10*u.m, dr_max=100*u.m,
                                  tolerance=0., include_amd=False, output_file=None, log_file=None,
                                  use_gpu=False, verbosity=0, debug_options=[]):
        '''Simulate an ensemble of N_samples muons, which are equally distributed
           over the given time bins.

           Parameters:
               N_samples:     the number of muons to generate
               t_bin_edges:   the edges of the time bins; in each bin, an equal number of
                              muons will be simulated, and their weights will indicate
                              their total number.
                              This can be either a 1d array listing the edges of continuous
                              bins, or an n x 2 array, which each entry giving the edges
                              of a bin in a possibly discontinuous set of bins
               Z:             positive electric charge of the core
               dt0:           starting time step
               dr_min:        if dr < dr_min for any muon, the time step will be doubled
                              for that muon
               dr_max:        if dr > dr_max for any muon, the time step will be halved
                              for that muon, and the current step will be repeated
               tolerance:     the evolution of muons is stopped once less than this
                              fraction is left. This can significantly speed up runs by
                              discarding muons that keep bouncing around the core, without
                              ever being absorbed or decaying
               include_amd:   include assisted muon decay (subdominant and slow to compute)
                              default: False
                              'partial': include amd only above the muon sphere
               output_file:   if given, dump snapshots of self to this file
               log_file:      file to store diagnostic messages (e.g. how many
                              muons are left)
               verbosity:     the amount of debug info that is printed out'''

        if use_gpu:
            import cupy as cp
            import cupy.random as rnd
            xp = cp
        else:
            import numpy.random as rnd
            xp = np
        def nparray(a):
            if use_gpu and type(a)==cp.ndarray:
                return cp.asnumpy(a)
            else:
                return a

        timing = ('timing' in debug_options)
        nodiff = ('nodiff' in debug_options)
        self.N_samples = N_samples

        # make sure t_bin_edges is a 2D array
        if len(t_bin_edges.shape) == 1:
            self.t_bin_edges = np.array([t_bin_edges[:-1], t_bin_edges[1:]]).T
        elif len(t_bin_edges.shape) == 2:
            if t_bin_edges.shape[1] != 2:
                raise ValueError('axis 1 of t_bin_edges has wrong size (must be 2).')
            else:
                self.t_bin_edges = t_bin_edges
        else:
            raise ValueError('t_bin_edges has invalid shape.')

        # remove or truncate bins that lie beyond the range of the NSCool simulation
        ii = np.where((self.t_bin_edges[:,0] < max(self.t_table))
                    & (self.t_bin_edges[:,1] > min(self.t_table)))[0]
        t_bin_edges_orig = self.t_bin_edges.copy()
        self.t_bin_edges = self.t_bin_edges[ii]
        self.t_bin_edges[self.t_bin_edges > max(self.t_table)] = max(self.t_table)
        self.t_bin_edges[self.t_bin_edges < min(self.t_table)] = min(self.t_table)
        if not np.array_equal(self.t_bin_edges, t_bin_edges_orig):
            print(f'WARNING: truncating time bins '
                + f'(directory {os.path.dirname(self.temp_file)})')
            print('Original bins [yrs]:')
            print(t_bin_edges_orig/u.yrs)
            print('Truncated bins [yrs]:')
            print(self.t_bin_edges/u.yrs)

        self.t_min         = self.t_bin_edges[0,0]
        self.t_max         = self.t_bin_edges[-1,1]
        self.t_bin_widths  = np.diff(self.t_bin_edges, axis=-1).flatten()
        self.t_bin_centers = 0.5 * (self.t_bin_edges[:,0] + self.t_bin_edges[:,1])
        self.n_t_bins      = len(self.t_bin_centers)
        self.t0_table      = np.full((N_samples//self.n_t_bins, self.n_t_bins),
                                     self.t_bin_edges[:,0]).T # launch muons at left bin edge
        self.idx_t0_table  = np.full((N_samples//self.n_t_bins, self.n_t_bins),
                                     np.arange(self.n_t_bins)).T

        # compute total number of muons in the star
        self.n_total_0 = integ.simps(4*np.pi * self.star_data[:,k_r]**2
                           * self.star_data[:,k_kfmu]**3/(3*np.pi**2), x=self.star_data[:,k_r])

        # generate muon starting radii.
        # normalization is the number of muons in an interval of +/- T
        # around the Fermi surface (factor two for the two spin states).
        # The distribution in energy follow df_{FD}/dE ~ f_{DF}*(1 - f_{FD})
        # see also urca.nb, section on "Fermi-Dirac Gymnastics)
        kfmu = self.star_data_interp[k_kfmu](self.temp_data[:,:,j_r])
        mumu = np.sqrt(kfmu**2 + u.m_mu**2)
        T    = self.temp_data[:,:,j_T]
        self.pdf_r_norm_interp = interp.interp1d(self.t_table, integ.simps(
              4*np.pi * self.temp_data[:,:,j_r]**2
            * kfmu**2 * T / ( 2*np.pi**2 * (1 + np.exp(-mumu/T))**2 ),
              x=self.temp_data[:,:,j_r], axis=1))
        self.pdf_r_norm = self.pdf_r_norm_interp(self.t_bin_edges[:,0])

        self.r0_table = np.zeros_like(self.t0_table)
        for j, t0 in enumerate(self.t_bin_centers):
            def pdf_r(r):
                t_r  = np.array([np.full_like(r, t0), r]).T
                kfmu = self.star_data_interp[k_kfmu](r)
                mumu = np.sqrt(kfmu**2 + u.m_mu**2)
                T    = self.temp_data_interp2d[j_T](t_r).flatten()
                return 4*np.pi * r**2 / self.pdf_r_norm[j] \
                     * kfmu**2 * T / ( 2*np.pi**2 * (1 + np.exp(-mumu/T))**2 )
            umax = np.amax(np.sqrt(pdf_r(self.temp_data[0,:,j_r])))
            vmax = np.amax(np.array( self.temp_data[0,:,j_r]
                                  * np.sqrt(pdf_r(self.temp_data[0,:,j_r])) ))
            self.r0_table[j] = stats.rvs_ratio_uniforms(pdf_r, umax=umax,
                             vmin=0, vmax=vmax, size=N_samples//self.n_t_bins)

        if verbosity > 1:
            print("simulate_muons_vectorized: done computing rates in each time bin")

        # prepare output tables
        self.t0_decay_table        = [] # production times of decaying muons
        self.idx_t0_decay_table    = [] # time bin index for decaying muons
        self.id_decay_table        = [] # unique ID for decaying muons
        self.r0_decay_table        = [] # radius where decaying muons originate
        self.t_decay_table         = [] # time of muon decay
        self.r_decay_table         = [] # radius at which muons decay
        self.rate_decay_table      = [] # production rate (per unit time,
                                        #   integrated over radius) for decaying muons
        self.E_mu_decay_table      = [] # parent muon energies at time of decay
        self.kfe_decay_table       = [] # e- Fermi momenta at loc. of muon decay
        self.decay_mode_table      = [] # decay mode for each muon ('f'ree vs. 'a'ssisted)
                                   
        self.t0_loss_table         = [] # production time of re-absorbed muons
        self.idx_t0_loss_table     = [] # time bin index for re-absorbed muons
        self.id_loss_table         = [] # unique ID for re-absorbed muons
        self.r0_loss_table         = [] # starting radius of re-absorbed muons
        self.t_loss_table          = [] # time at which muons are reabsorbed
        self.r_loss_table          = [] # radius at which muons are reabsorbed
        self.rate_loss_table       = [] # production rate (at the respective t0,
                                        #   integrated over radius) for reabsorbed muons
        self.Gamma_loss_table      = [] # absorption rate where muon is lost
                                   
        self.t0_survivor_table     = [] # production time of muons that survive
        self.idx_t0_survivor_table = [] # time bin index for surviving muons
        self.id_survivor_table     = [] # unique ID for surviving muons
        self.r0_survivor_table     = [] # radius where surviving muons originate
        self.t_survivor_table      = [] # times when mu are declared survivors
        self.r_survivor_table      = [] # radii where muo are declared survivors
        self.rate_survivor_table   = [] # production rate (at the respective t0,
                                        #   integrated over radius) for surviving muons

        # create copies of important array for faster access during interpolation
        # (in GPU mode, these arrays will be CuPy arrays)
        t_table           = xp.array(self.t_table)
        r_table_star      = xp.array(self.star_data[:,k_r].copy())
        kfp_data          = xp.array(self.star_data[:,k_kfp].copy())
        kfn_data          = xp.array(self.star_data[:,k_kfn].copy())
        kfmu_data         = xp.array(self.star_data[:,k_kfmu].copy())
        kfe_data          = xp.array(self.star_data[:,k_kfe].copy())
        T_data            = xp.array(self.temp_data[:,:,j_T].copy())
        kappa_e_data      = xp.array(self.temp_data[:,:,j_lambdaE].copy())
        kappa_mu_data     = xp.array(self.temp_data[:,:,j_lambdaMu].copy())
        QdUrca_data       = xp.array(self.temp_data[:,:,j_QdUrca].copy())
        QmUrca_data       = xp.array(self.temp_data[:,:,j_QmUrca].copy())
        a_table           = xp.array(self.a_table)
        s_mUrca           = xp.array(self.s_mUrca)
        s_dUrca           = xp.array(self.s_dUrca)
        pdf_r_norm        = xp.array(self.pdf_r_norm)
        dUrca_active      = (np.count_nonzero(QdUrca_data) > 0.)
                               # a flag that determines whether we need to worry
                               # about direct Urca processes at all

        # initialize arrays that hold muon coordinates during the simulation 
        r0     = xp.array(self.r0_table.flatten())     # original radius for each muon
        r      = xp.array(self.r0_table.flatten())     # current radius for each muon
        t0     = xp.array(self.t0_table.flatten())     # starting time for each muon
        t      = xp.array(self.t0_table.flatten())     # current time for each muon
        dt     = xp.full_like(r, dt0)                  # initial time step
        idx_t0 = xp.array(self.idx_t0_table.flatten()) # t bin index of each muon
        it     = xp.searchsorted(t_table, t)           # each muon's NSCool time bin
        id     = xp.arange(len(r))                     # unique ID for each muon

        # weight factors for each bin, corresponding to the total number of muons
        # represented by each simulated muon
        w = xp.array([ xp.full(self.r0_table.shape[1], p*self.n_t_bins/N_samples)
                                             for p in pdf_r_norm ]).flatten()

        # loop that runs until all muon have been absorbed or decayed
        n_iter = 0
        print("starting main loop ..", time.asctime(time.localtime()))
        while len(r) > 0:
            n_iter = n_iter + 1

            # get interpolated neutron star properties at the current muon coordinates.
            # as interpolation is slow, we simply pick the values from the closest
            # r and/or t bin. At small r, where we don't have data from NSCool, this
            # approach automatically uses the innermost NSCool bin
            time0=time.process_time()
            if timing: print("A", )
            ir                  = xp.searchsorted(r_table_star, r)
            kfp                 = kfp_data[ir]
            kfn                 = kfn_data[ir]
            kfmu                = kfmu_data[ir]
            kfe                 = kfe_data[ir]
            mumu                = xp.sqrt(u.m_mu**2 + kfmu**2) # muon chemical potential
            mue                 = xp.sqrt(u.m_e**2  + kfe**2)  # electron chemical potential
            kfmu_gtr_0          = (kfmu > 0.)

            if timing: print("B", time.process_time() - time0)
            ir                = ir // 2    # exploit the fact that NSCool's Temp data
                                           # skips every other radial step
            T                 = T_data[it,ir]
            kappa_e           = kappa_e_data[it,ir]
            kappa_mu          = kappa_mu_data[it,ir]
            QmUrca            = QmUrca_data[it,ir]
            if dUrca_active:
                QdUrca        = QdUrca_data[it,ir]
            if timing: print("C", time.process_time() - time0)

            # compute derived quantities
            n_mu    = kfmu**3 / (3*np.pi**2)            # muon number density
                                   # https://en.wikipedia.org/wiki/Fermi_energy
            n_e     = kfe**3 / (3*np.pi**2)             # electron number density

            v_th_mu = xp.sqrt(8*T/(np.pi*u.m_mu))       # thermal velocity
                                   # (Maxwell-Boltzmann, no chemical potential)
            Fg      = - u.GN * u.m_mu * r * self.M_NS / self.R_NS**3 \
                      + 0.5 * u.m_mu * u.GN**2 * (r**2 + 3*self.R_NS**2) \
                                     * r * self.M_NS**2 / self.R_NS**6
                                   # approximate gravitational force from interior
                                   # Schwarzschild metric (see urca.nb)

            # Add electric force to simulate a charge imbalance in the NS
            # we model the charge distributrion as a homogeneous sphere, with the
            # radius given by the core-crust boundary
            if Z != 0.:
                Fg -= Z * u.alpha_em * xp.where(r > self.r_CC_boundary,
                                                1/r**2,
                                                r/self.r_CC_boundary**3)
       
            # compute *electron* scattering time scale, which we use as a fallback
            # in regions where we can't estimate the muon mean free path from
            # the thermal conductivity
            old_err = np.seterr(divide='ignore', invalid='ignore')
            tau_e   = kappa_e * 3*kfe / (np.pi**2 * T * n_e)
            if timing: print("C1a", time.process_time() - time0)

            # in regions where there are ambient muons, compute diffusion
            # speed according to Fick's law
            vfmu     = kfmu / mumu  # muon Fermi velocity
            vfe      = 1.           # electron Fermi velocity
            ii       = kfmu_gtr_0  &  (kappa_mu < 1e10*u.erg/u.cm/u.sec/u.Kelvin)
            if xp.count_nonzero(ii) > 0:
                if verbosity >= 2:
                    print(("WARNING: low kappa_mu for {:d}/{:d} muons ".format(len(ii), len(r)) +
                           "at t0={:g} yrs").format(t0/u.yrs))
                    print("  r [km]     = ", r[ii]/u.km)
                    print("  kfmu [MeV] = ", kfmu[ii]/u.MeV)
                    print("  kappa_mu   = ", kappa_mu[ii]/(u.erg/u.cm/u.sec/u.Kelvin))
                kappa_mu[ii] = xp.fmax(1e21*u.erg/u.cm/u.sec/u.Kelvin, kappa_mu[ii])
            tau_mu   = kappa_mu * 3*mumu / (np.pi**2 * T * n_mu)
                         # relaxation time, https://arxiv.org/abs/0705.1963, eq. (4)
                         # this expression can be understood as follows:
                         #   random walk travel distance: dx = \lambda \sqrt{N}
                         #     (\lambda=mfp, N=# of interactions)
                         #   -> dx = \lambda \sqrt{dt / \tau} = v \sqrt{\tau*dt}
                         #   Heat flux: T*\phi = T*n*dx/dt
                         #                     = T * n * v^2 * \tau * dt / (dx*dt)
                         #                     = T * n * T/m * \tau / dx
                         #   from the definition T*\phi = \kappa * dT/dx
                         #   we obtain \kappa = T * n * \tau / m
            mfp_mu   = xp.where(kfmu_gtr_0, tau_mu*vfmu, tau_e*vfe)
                         # muon mean free path; use mfp for e- as a fallback
            if timing: print("C1b", time.process_time() - time0)

            # Distance travelled
            t_coll     = xp.where(kfmu_gtr_0, tau_mu, tau_e/v_th_mu) # collision time, e- as fallback
                             # JK changed fallback from tau_e to tau_e/v_th_mu on 06.10.2023
            sigma_diff = xp.sqrt(dt/t_coll) * mfp_mu
            dr_diff    = xp.fmin(dt * xp.fmax(v_th_mu, vfmu),      # fallback: free propagation
                                 rnd.normal(scale=sigma_diff, size=len(r))) \
                           * xp.cos( rnd.uniform(0, 0.5*np.pi, size=len(r)) ) # pick random direction

            # include gravity
            # (displacement towards r=0 between collisions, times no. of collisions)
            dr_grav  = 0.5 * (Fg/u.m_mu) * dt * xp.fmin(t_coll, dt)
            dr       = dr_diff + dr_grav

            # allow switching off diffusion for debugging purposes
            if nodiff:
                dr *= 0.

            if timing: print("D", time.process_time() - time0)

            # step size control 1: determine muons whose radial step was too large;
            # for them, this time step will be repeated with smaller step size;
            # also, choose smaller step size near the core-crust boundary
            this_dr_min   = np.where((r > self.r_CC_boundary-0.5*u.km),
                                     min(dr_min, 15*u.m),  dr_min)
            this_dr_max   = np.where((r > self.r_CC_boundary-0.5*u.km),
                                     min(dr_max, 40*u.m), dr_max)

            repeat_flag   = ( (xp.abs(sigma_diff) + xp.abs(dr_grav) > this_dr_max) )
            continue_flag = xp.logical_not(repeat_flag)
            if timing: print("D1", time.process_time() - time0)

            # Muon absorption
            # ---------------
            # determine which muons get absorbed in this step
            old_err = np.seterr(divide='ignore')

            # assign energy to each muon, including only energies close to the
            # Fermi surface
            def pdf_x(x):  # PDF follows f_{FD} * (1 - f_{FD})
                return np.exp(x) / (1 + np.exp(x))**2
            rr = xp.array(stats.rvs_ratio_uniforms(pdf_x, umax=2., vmin=-0.223872, 
                                                   vmax=0.223872, size=T.shape))
            E_mu = rr*T + mumu
            a = (E_mu - mue) / T   # we use mue as proxy for mun - mup

            # rescale Urca rate to account for deviation from equilibirum, see urca.nb
            # in the following, we use mue in place of mun - mup (which we can't
            #   easily compute from the information NSCool gives us), assuming
            #   beta equilibrium everywhere
            if timing: print("D1b", time.process_time() - time0)
            ia = xp.searchsorted(a_table, a)
            ia[ia >= len(self.a_table)] = len(self.a_table) - 1
            sm = s_mUrca[ia] * 1/(a + mue/T)**2 / T**4
            if dUrca_active:
                sd = s_dUrca[ia] * 1/(mue * xp.fmax(kfmu, xp.sqrt(2.*u.m_mu*T))) / T**2
                Gamma_abs_table = QmUrca * sm + QdUrca * sd
                ii = (n_mu <= 0)
                Gamma_abs_table[ii] = Gamma_abs_MU(E_mu=E_mu[ii], T=T[ii],
                        m_p_eff=self.mpeff_interp(r[ii]), m_n_eff=self.mneff_interp(r[ii]),
                        pF_p=kfp[ii], pF_n=kfn[ii], pF_e=kfe[ii], pF_mu=kfmu[ii]) + \
                    Gamma_abs_DU(E_mu=E_mu[ii], T=T[ii],
                        m_p_eff=self.mpeff_interp(r[ii]), m_n_eff=self.mneff_interp(r[ii]),
                        pF_p=kfp[ii], pF_n=kfn[ii], pF_e=kfe[ii], pF_mu=kfmu[ii])
            else:
                Gamma_abs_table = QmUrca * sm
                ii = (n_mu <= 0)
                Gamma_abs_table[ii] = Gamma_abs_MU(E_mu=E_mu[ii], T=T[ii],
                        m_p_eff=self.mpeff_interp(r[ii]), m_n_eff=self.mneff_interp(r[ii]),
                        pF_p=kfp[ii], pF_n=kfn[ii], pF_e=kfe[ii], pF_mu=kfmu[ii])

            if timing: print("D1c", time.process_time() - time0)
            Gamma_abs_table[T<=0.] = 0.
                # outside the neutron star (T=0), no absorption is possible
            np.seterr(**old_err)
            Gamma_abs_table[(n_mu<=0.) & (n_e<=0.)] = 0.
            if timing: print("D1d", time.process_time() - time0)

            my_rnd         = rnd.uniform(size=len(r))
            absorbed_muons = ( (my_rnd < Gamma_abs_table*dt)
                             & continue_flag )

            if verbosity >= 3:
                print("  ABS: t        = ", t/u.yrs)
                print("       dt       = ", dt/u.yrs)
                print("       r        = ", r/u.km)
                print("       Gamma*dt = ", Gamma_abs_table*dt)
            if xp.count_nonzero(absorbed_muons) > 0:
                self.t0_loss_table.extend(nparray(t0[absorbed_muons]))
                self.idx_t0_loss_table.extend(nparray(idx_t0[absorbed_muons]))
                self.id_loss_table.extend(nparray(id[absorbed_muons]))
                self.r0_loss_table.extend(nparray(r0[absorbed_muons]))
                self.t_loss_table.extend(nparray(t[absorbed_muons]))
                self.r_loss_table.extend(nparray(r[absorbed_muons]))
                self.rate_loss_table.extend(nparray(w[absorbed_muons]))
                self.Gamma_loss_table.extend(nparray(Gamma_abs_table[absorbed_muons]))
            if verbosity >= 2:
                for k in xp.where(absorbed_muons)[0]:
                    print("  ABS t={:10.5g} dt={:10.5g} Gamma*dt={:10.5g}"
                          .format(t[k]/u.yrs, dt[k]/u.yrs, Gamma_abs_table[k]*dt[k]))
            if timing: print("D2", time.process_time() - time0)

            # Muon decay
            # ----------
            # determine which muons decay in this time step?
            ii   = ( continue_flag
                   & xp.logical_not(absorbed_muons) )
            Gamma_dec_table = np.zeros(T.shape)
            Gamma_amd_table = np.zeros(T.shape)
            if use_gpu:
                Gamma_dec_table[ii] = xp.array( self.mu_width_interp(
                                          np.array([kfe[ii].get(),
                                                   ((E_mu[ii] - mumu[ii]) / T[ii]).get(),
                                                   np.fmax(T[ii].get(), 1e-4*u.MeV)]).T) )
            else:
                Gamma_dec_table[ii] = self.mu_width_interp(xp.array([kfe[ii],
                                                           (E_mu[ii] - mumu[ii]) / T[ii],
                                                           xp.fmax(T[ii], 1e-4*u.MeV)]).T)

            # compute assisted muon decay rate following eq. A86 in the paper
            if include_amd == True:
                Gamma_amd_table[ii] = Gamma_amd(E_mu=E_mu[ii], T=T[ii],
                                          m_p_eff=self.mpeff_interp(r[ii]),
                                          pF_p=kfp[ii], pF_n=kfn[ii], pF_e=kfe[ii], pF_mu=kfmu[ii])
            elif include_amd == 'partial':         # include amd only above the muon sphere, where
                jj = (ii & (kfmu<=0) & (kfe>0))  # it is more important than below
                Gamma_amd_table[jj] = Gamma_amd(E_mu=E_mu[jj], T=T[jj],
                                          m_p_eff=self.mpeff_interp(r[jj]),
                                          pF_p=kfp[jj], pF_n=kfn[jj], pF_e=kfe[jj], pF_mu=kfmu[jj])
            else:
                Gamma_amd_table[ii] = 0.
            if verbosity >= 3:
                print("  DEC: t        = ", t/u.yrs)
                print("       dt       = ", dt/u.yrs)
                print("       r        = ", r/u.km)
                print("       kfe      = ", kfe/u.MeV)
                print("       E_mu     = ", E_mu/u.MeV)
                print("       Gamma*dt = ", Gamma_dec_table*dt)

            decayed_muons     = ii.copy()
            decayed_muons[ii] = my_rnd[ii] < (Gamma_abs_table[ii]
                                            + Gamma_dec_table[ii]
                                            + Gamma_amd_table[ii]) * dt[ii]
            decay_modes       = np.full(decayed_muons.shape, '-')
            decay_modes[ii]   = np.where(my_rnd[ii] < (Gamma_abs_table[ii]
                                                     + Gamma_dec_table[ii]) * dt[ii],
                                         'f', 'a')

            # save muon energy and electron Fermi momentum. They are needed later
            # for the (vectorized) calculation of neutrino energies
            if xp.count_nonzero(decayed_muons) > 0:
                self.t0_decay_table.extend(nparray(t0[decayed_muons]))
                self.idx_t0_decay_table.extend(nparray(idx_t0[decayed_muons]))
                self.id_decay_table.extend(nparray(id[decayed_muons]))
                self.r0_decay_table.extend(nparray(r0[decayed_muons]))
                self.t_decay_table.extend(nparray(t[decayed_muons]))
                self.r_decay_table.extend(nparray(r[decayed_muons]))
                self.rate_decay_table.extend(nparray(w[decayed_muons]))
                self.E_mu_decay_table.extend(nparray(E_mu[decayed_muons]))
                self.kfe_decay_table.extend(nparray(kfe[decayed_muons]))
                self.decay_mode_table.extend(nparray(decay_modes[decayed_muons]))
            for k in xp.where(decayed_muons)[0]:
                if verbosity >= 2:
                    print("  DEC t={:10.5g} dt={:10.5g} Gamma*dt={:10.5g}"
                              .format(t[k]/u.yrs, dt[k]/u.yrs, (Gamma_dec_table * dt)[k]))
            if timing: print("D3", time.process_time() - time0)
            
            # Surviving muons
            # ---------------
            # declare muons that reach the end of their time bin as "survivors"
            surviving_muons = ((t >= xp.array(self.t_bin_edges)[idx_t0,1])
                              & xp.logical_not(decayed_muons)
                              & xp.logical_not(absorbed_muons)
                              & continue_flag)
            if xp.count_nonzero(surviving_muons) > 0:
                self.t0_survivor_table.extend(nparray(t0[surviving_muons]))
                self.idx_t0_survivor_table.extend(nparray(idx_t0[surviving_muons]))
                self.id_survivor_table.extend(nparray(id[surviving_muons]))
                self.r0_survivor_table.extend(nparray(r0[surviving_muons]))
                self.t_survivor_table.extend(nparray(t[surviving_muons]))
                self.r_survivor_table.extend(nparray(r[surviving_muons]))
                self.rate_survivor_table.extend(nparray(w[surviving_muons]))
            if timing: print("E", time.process_time() - time0)

            # step size control 2: actually adjust time steps
            dt[repeat_flag] *= 0.5
            dt[xp.abs(sigma_diff)+xp.abs(dr_grav) < this_dr_min] *= 2.

            # prepare for next step
            t[continue_flag] += dt[continue_flag]
            r[continue_flag]  = xp.abs(r[continue_flag] + dr[continue_flag])
                                    # ensure r>0 (for muons that have crossed the origin)
            if timing: print("F", time.process_time() - time0)

            # remove absorbed and decayed muons, as well as those that
            # have fallen back onto the NS from our lists
            ii     = xp.logical_not(absorbed_muons |
                                    decayed_muons |
                                    surviving_muons)
            r0     = r0[ii]
            r      = xp.fmin(r[ii], 0.999*self.R_NS) # prevent muons from leaving the
                                                     # NS to avoid out-of-bounds checks
            t0     = t0[ii]
            t      = t[ii]
            dt     = dt[ii]
            w      = w[ii]
            idx_t0 = idx_t0[ii]
            it     = it[ii]
            it[t>t_table[it]] += 1 # advance to next NSCool time bin where necessary
            id     = id[ii]
            if timing: print("G", time.process_time() - time0)

            # if only few muons are left (bouncing around the core without being absorbed
            # or decaying), stop
            if len(r) / N_samples < tolerance:
                break

            # write log
            log_string = "iter {:d}: {:d} left, {:d} decayed, {:d} absorbed, {:d} surviving ..." \
                           .format(n_iter,
                                   len(xp.nonzero(ii)[0]), len(self.t0_decay_table),
                                   len(self.t0_loss_table), len(self.t0_survivor_table))
            if log_file is None:
                print(log_string, end='         \r')
            else:
                with open(log_file, 'w') as f:
                    f.write(log_string + '\n')

            # sanity check - emergency stop
            ii = xp.where(r > 3*self.R_NS)[0]
            if len(ii) > 0:
                print(("{:d} muon(s) did not decay until r = {:g} km. " +
                       "This should not have happened.").format(ii.size, (self.R_NS)/u.km))
                break
        # end while len(r) > 0

        # write log
        log_string = "iter {:d}: {:d} left, {:d} decayed, {:d} absorbed, {:d} surviving." \
                       .format(n_iter,
                               len(xp.nonzero(ii)[0]), len(self.t0_decay_table),
                               len(self.t0_loss_table), len(self.t0_survivor_table))
        if log_file is None:
            print(log_string)
        else:
            with open(log_file, 'w') as f:
                f.write(log_string + '\n')

        # Turn output arrays into NumPy arrays
        self.t0_decay_table        = np.array(self.t0_decay_table).flatten()
        self.idx_t0_decay_table    = np.array(self.idx_t0_decay_table).flatten()
        self.id_decay_table        = np.array(self.id_decay_table).flatten()
        self.r0_decay_table        = np.array(self.r0_decay_table).flatten()
        self.t_decay_table         = np.array(self.t_decay_table).flatten()
        self.r_decay_table         = np.array(self.r_decay_table).flatten()
        self.rate_decay_table      = np.array(self.rate_decay_table).flatten()
        self.E_mu_decay_table      = np.array(self.E_mu_decay_table).flatten()
        self.kfe_decay_table       = np.array(self.kfe_decay_table).flatten()
        self.decay_mode_table      = np.array(self.decay_mode_table).flatten()

        self.t0_loss_table         = np.array(self.t0_loss_table).flatten()
        self.idx_t0_loss_table     = np.array(self.idx_t0_loss_table).flatten()
        self.id_loss_table         = np.array(self.id_loss_table).flatten()
        self.r0_loss_table         = np.array(self.r0_loss_table).flatten()
        self.t_loss_table          = np.array(self.t_loss_table).flatten()
        self.r_loss_table          = np.array(self.r_loss_table).flatten()
        self.rate_loss_table       = np.array(self.rate_loss_table).flatten()
        self.Gamma_loss_table      = np.array(self.Gamma_loss_table).flatten()

        self.t0_survivor_table     = np.array(self.t0_survivor_table).flatten()
        self.idx_t0_survivor_table = np.array(self.idx_t0_survivor_table).flatten()
        self.id_survivor_table     = np.array(self.id_survivor_table).flatten()
        self.r0_survivor_table     = np.array(self.r0_survivor_table).flatten()
        self.t_survivor_table      = np.array(self.t_survivor_table).flatten()
        self.r_survivor_table      = np.array(self.r_survivor_table).flatten()
        self.rate_survivor_table   = np.array(self.rate_survivor_table).flatten()

        # decay muons
        # remember that even indices in E_nu_table correspond to \nu_\mu
        # and odd ones to \bar\nu_e -- make sure this is taken into
        # account when analyzing the results 
        print("starting decay ..", time.asctime(time.localtime()))
        self.E_nu_table = process_mu_decay(self.E_mu_decay_table, self.kfe_decay_table)
                                   
        # include gravitational redshift for neutrinos
        self.redshift_table = np.sqrt(1 - 2*u.GN*self.M_NS/self.r_decay_table)
        self.E_nu_table_rs  = self.E_nu_table * np.repeat(self.redshift_table, 2)

        # include neutrino oscillations - vacuum oscillations
        theta12 = 33.82 * np.pi/180.      # NuFit 4.1
        self.rate_decay_table_e      = self.rate_decay_table * 0.25*np.sin(2*theta12)**2
        self.rate_decay_table_mu     = self.rate_decay_table * (0.5-0.25*np.sin(2*theta12)**2)
        self.rate_decay_table_ebar   = self.rate_decay_table * (1-0.5*np.sin(2*theta12)**2)
        self.rate_decay_table_mubar  = self.rate_decay_table * 0.25*np.sin(2*theta12)**2

        # write output by dumping self
        if output_file is not None:
            with open(output_file, 'wb') as f:
                pickle.dump(self, f) 
        print("done.", time.asctime(time.localtime()))


    # -----------------------------------------------------------------------
    def simulate_electrons(self, N_samples, N_steps=1000000, r0=11*u.km, t0=1000*u.yrs,
                           T_enhancement=1., mfp_enhancement=1., Z=1, verbosity=0):
        '''Runs a toy Monte Carlo studying the dynamics of electrons diffusing in
           an electrostatic potential, and against Fermi pressur.

           Parameters:
               N_samples:       the number of muons to generate
               N_steps:         number of random walk steps
               r0:              starting point of electrons (radial coordinate)
               t0:              starting time of electrons
               t_enhancement:   factor by which temperature are artificially enhanced
               mfp_enhancement: factor by which mean free paths are artificially enhanced
               Z:               positive electrostatic charge of NS core
               verbosity:       the amount of debug info that is printed out'''

        r = np.full((N_samples), r0)

        # loop over time steps
        for s in range(N_steps):
            t_r     = np.array([np.full_like(r, t0), r]).T
            kfe     = self.kfe_interp(r)                 # electron Fermi momentum
            mue     = np.sqrt(u.m_e**2 + kfe**2)         # electron chemical potential
            n_e     = kfe**3 / (3*np.pi**2)              # electron number density
            kappa_e = self.temp_data_interp2d[j_lambdaE](t_r)
            T       = self.temp_data_interp2d[j_T](t_r)

            vfe     = kfe / mue                          # electron velocity
            tau_e   = kappa_e * 3*kfe / (np.pi**2 * T * n_e)  # scattering time scale
            mfp_e   = tau_e * vfe                        # mean free path

            # Artificially inflate mean free path and temperature
            mfp_e  *= mfp_enhancement
            T      *= T_enhancement

            # pick length of random walk step
            dr_diff = rnd.exponential(scale=mfp_e, size=len(r)) \
                          * np.cos( rnd.uniform(0, np.pi, size=len(r)) )

            # add electrostatic potential
            F_el    = -Z * u.alpha_em / r**2
            dr_el   = 0.5 * (F_el/u.m_e) * (mfp_e/vfe)**2
                          # displacement towards r=0 between collisions

            # assign random energy around the Fermi surface to each electron
            rr      = rnd.normal(size=T.shape)
            Ee      = np.sqrt((kfe + rr*T)**2 + u.m_e**2)

            # determine Fermi surface at target location
            dr      = dr_diff + dr_el
            kfe2    = self.kfe_interp(r + dr)
            mue2    = np.sqrt(u.m_e**2 + kfe2**2)
            
            # take step only if not forbidden by Pauli blocking
            # (take into account electrostatic energy gain, though)
            r      += np.where(Ee - Z*u.alpha_em*(1/r - 1/(r+dr)) > mue2, dr, 0.)

        # return resulting radial distribution
        return r


