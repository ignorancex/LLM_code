#####################################################################################
#            CLASSICS - CalcuLAtionS of Self Interaction Cross Sections             #
# by Brian Colquhoun, Saniya Heeba, Felix Kahlhoefer, Laura Sagunski and Sean Tulin #
#####################################################################################
# Requirements: python3, numpy, scipy
#
# This code provides the following functions:
#
# sigma(kappa, beta, mode, sign):
#   Returns approximate analytical cross sections for the classical regime (kappa > 1) for given arguments
#   kappa:   Dimensionless momentum in the centre-of-mass frame, given by kappa = m_\chi v / (2 m_\phi)
#   beta:    Rescaled strength of the Yukawa potential, given by beta = 2 \alpha_\chi m_\phi / (m_\chi v^2)
#   mode:    Can take one of the following values: 
#            'T': Returns the momentum transfer cross section for distinguishable particles
#            'V': Returns the viscosity cross section for distinguishable particles
#            'even': Returns the viscosity cross section for identical particles with even spatial wave function
#            'odd': Returns the viscosity cross section for identical particles with odd spatial wave function
#            'scalar: Returns the viscosity cross section for identical scalar particles
#            'fermion': Returns the viscosity cross section for identical fermions (averaged over initial spins)
#            'vector': Returns the viscosity cross section for identical vector particles (averaged over initial spins)
#            If no mode is specified, the default option is 'T'
#   sign:    Can take one of the following values:
#            'attractive': Attractive Yukawa potential
#            'repulsive': Repulsive Yukawa potential
#            If no sign is specified, the default option is 'attractive'
#
# sigma_Hulthen(kappa, beta, mode, sign, eps):
#   Returns approximate analytical cross sections for the quantum regime (kappa < 1) for S-wave-only scattering under the Hulthen approximation, following Tulin, Yu & Zurek (arXiv:1302.3898)
#   The arguments are the same as above, with the addition of
#   eps:     Numerical constant with default value 1.6.
#
# sigma_combined(kappa, beta, mode, sign):
#   Returns the appropriate cross section depending on kappa, i.e. sigma for kappa > 1 and sigma_Hulthen for kappa < 0.4.
#   To ensure continuity, the code linearly interpolates between the two different regimes between kappa = 0.4 and kappa = 1.
#   The arguments are the same as above.
#
# averagedsigma(kappa0, beta0, mode, sign):
#   Returns the averaged cross section for a Maxwell-Boltzmann distribution with velocity dispersion v0 based on pre-calculated tables.
#   The arguments are the same as above with kappa0 = kappa(v = v0) and beta0 = beta(v = v0).
#
# IMPORTANT: The return values of all functions are dimensionless and need to be multiplied with (pi / m_phi^2) in order to obtain actual cross sections.
# 
# Note: The option "approximate_eta" below determines whether the code should use approximate asymptotic expressions of the modified Bessel functions for large argument 
#   approximate_eta = True   is slightly faster but inaccurate for small kappa
#   approximate_eta = False  is slightly slower but gives the best accuracy

import numpy as np
from numpy import sqrt, pi, sin, cos, log, exp, euler_gamma
from scipy.special import kn, gamma, loggamma
from scipy.interpolate import RectBivariateSpline

approximate_eta = False

# Definition of auxillary functions

lmin = lambda beta, kappa: max(1./2.,beta*kappa)
lminp = lambda beta, kappa: max(1.,2.*beta*kappa)
turn = lambda beta, betalow, a: exp(-(max(beta, betalow) - betalow)*a)

if approximate_eta:
    eta = lambda x: -2.*log(x/2.)-1-2.*euler_gamma+(1-euler_gamma-log(x/2.))*x**2.
else:
    eta = lambda x: x**2 * (- kn(1,x)**2 + kn(2,x)*kn(0,x))

zeta = lambda kappa, beta, lmin: (max(lmin, beta*kappa)**2 - lmin**2)/(2*kappa**2*beta**2) + eta(max(lmin, beta*kappa)/kappa)

lambdaT = (1.+cos(2.)+2*sin(2.))/2.
lambdaV = (9.-cos(4.)-4.*sin(4.))/16.

sigmaT_smallbeta = lambda beta, kappa: 2. * beta**2. * zeta(kappa, beta, 0.5)

sigmaV_smallbeta = lambda beta, kappa, lmin: 4. * beta**2. * zeta(kappa, 2.*beta, lmin)

def sigmaTatt(beta, kappa):
  if beta < 1: return sigmaT_smallbeta(beta,kappa)*turn(beta,0.2,-0.64)
  elif beta > 50: return 2. * log(beta) * (log(log(beta)) + 1)
  else: return 4.7*log(beta + 0.82)

def sigmaTrep(beta, kappa):
  if beta <1: return sigmaT_smallbeta(beta,kappa)*turn(beta,0.2,0.53)
  elif beta > 50: return lambdaT * (log(2.*beta)-log(log(2.*beta)))**2.
  else: return 2.9*log(beta + 0.47)

def sigmaVatt(beta, kappa, lmin):
  if beta < 0.5: return sigmaV_smallbeta(beta,kappa,lmin)*turn(beta,0.1,-0.67)
  elif beta > 25: return (1 + log(beta)- 1/(2.*log(beta)))**2/2.
  else: return 2.5*log(beta + 1.05)

def sigmaVrep(beta, kappa, lmin):
  if beta < 0.5: return sigmaV_smallbeta(beta,kappa,lmin)*turn(beta,0.1,0.370562)
  elif beta > 25: return  log(2. * beta) * (lambdaV * log(2. * beta) - (2.*lambdaV - 1) * log(log(2.*beta)))
  else: return 2.8*log(beta + 0.80)

# Reading tabulated grids

modes = ['T','V','even','odd','scalar','fermion','vector']
signs = ['attractive','repulsive']
mode_factor = {'T': 1, 'V': 2/3., 'even': 4/3., 'odd': 0, 'scalar': 4/3., 'fermion': 1/3., 'vector': 8/9.}

beta0grid = np.logspace(-5,5, 101, endpoint=True)
kappa0grid = np.logspace(-3,3, 61, endpoint=True)

averagedsigmainterdict = {}
#averagedsigmadict = {}

for mode in modes:
  for sign in signs:

    outputname_data = 'sigma'+mode+'list_'+sign+'.txt'

    averagedsigmagrid = np.loadtxt(outputname_data)
    averagedsigmaarray = np.array(averagedsigmagrid)[:,2].reshape((len(kappa0grid),len(beta0grid))) + 1e-100

    averagedsigmainterdict[mode+sign] = RectBivariateSpline(np.log10(kappa0grid), np.log10(beta0grid), np.log10(averagedsigmaarray))
#    averagedsigmadict[mode+sign] = lambda x, y: 10**averagedsigmainterdict[mode+sign](np.log10(x),np.log10(y))[0,0]

# Definition of cross section functions

def sigma(kappa, beta, mode = 'T', sign = 'attractive'):
  if not(sign == 'attractive' or sign == 'repulsive'):
    print('Sign not recognized in function sigma()') 
    exit()
  if kappa < 1:
    print('Warning: kappa outside of range of validity in function sigma()')
    return 0.
  if mode == 'T':
    if sign == 'attractive': return sigmaTatt(beta, kappa)
    else: return sigmaTrep(beta, kappa)
  elif mode == 'V':
    if sign == 'attractive': return sigmaVatt(beta, kappa, 1.)
    else: return sigmaVrep(beta, kappa, 1.)
  elif mode == 'even':
    if sign == 'attractive': return sigmaVatt(beta, kappa, 0.5)
    else: return sigmaVrep(beta, kappa, 0.5)
  elif mode == 'odd':
    if sign == 'attractive': return sigmaVatt(beta, kappa, 1.5)
    else: return sigmaVrep(beta, kappa, 1.5)
  elif mode == 'scalar':
    return sigma(kappa, beta, mode = 'even', sign = sign)
  elif mode == 'fermion':
    return 0.75*sigma(kappa, beta, mode = 'odd', sign = sign) + 0.25*sigma(kappa, beta, mode = 'even', sign = sign)
  elif mode == 'vector':
    return 1/3.*sigma(kappa, beta, mode = 'odd', sign = sign) + 2/3.*sigma(kappa, beta, mode = 'even', sign = sign)
  else:
    print('Mode not recognized in function sigma()')
    exit()

def sigma_Hulthen(kappa, beta, mode = 'T', sign = 'attractive', eps=1.6):
    
    if kappa > 1:
      print('Warning: kappa outside of range of validity in function sigma_Hulthen()')
      return 0

    if beta > 1e6:
      print('Warning: numerical instability possible for beta > 10^6 in function sigma_Hulthen()')

    if not(mode in modes):
      print('Mode not recognized in function sigma_Hulthen()')
      exit()

    i = 1j
    unity = 1+0j
    
    if sign == 'attractive':
        beta_signed = -1*beta
    elif sign == 'repulsive':
        beta_signed = beta
    else:
        print('Sign not recognized in function sigma_Hulthen()')
        exit()
    
    lam_p = 1 + i*kappa/eps * (1 + np.sqrt( 1 + 2*beta_signed*eps*unity ) )
    lam_m = 1 + i*kappa/eps * (1 - np.sqrt( 1 + 2*beta_signed*eps*unity ) )
    
    arg = i*gamma(lam_p+lam_m-2)/exp(loggamma(lam_p)+loggamma(lam_m))
    delta_0 = np.angle(arg)
   
    sigma_s_wave = 4*np.pi/kappa**2 * np.sin(delta_0)**2 / np.pi

    return sigma_s_wave * mode_factor[mode]

def sigma_combined(kappa,beta,mode = 'T', sign = 'attractive'):
    if kappa > 1:
      return sigma(kappa,beta,mode,sign)
    elif kappa < 0.4:
      return sigma_Hulthen(kappa,min(beta,1e6),mode,sign)
    else:
      return (1-kappa)/0.6*sigma_Hulthen(0.4,min(beta,1e6),mode,sign) + (kappa-0.4)/0.6*sigma(1,beta,mode,sign)

def averagedsigma(kappa0, beta0, mode = 'T', sign = 'attractive'):
  if kappa0 < kappa0grid[0] or kappa0 > kappa0grid[-1]:
    print('Warning! kappa0 outside of tabulated range in function averagedsigma()')
    kappa0 = np.clip(kappa0, kappa0grid[0], kappa0grid[-1])
  if beta0 < beta0grid[0] or beta0 > beta0grid[-1]:
    print('Warning! beta0 outside of tabulated range in function averagedsigma()')
    beta0 = np.clip(beta0, beta0grid[0], beta0grid[-1])

  if not(sign == 'attractive' or sign == 'repulsive'):
    print('Sign not recognized in function averagedsigma()') 
    exit()
  if mode in modes:
    return 10**averagedsigmainterdict[mode+sign](np.log10(kappa0), np.log10(beta0))[0,0]
  else:
    print('Mode not recognized in function averagedsigma()')
    exit()

