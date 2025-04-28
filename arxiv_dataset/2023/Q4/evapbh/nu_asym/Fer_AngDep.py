##################################################################################
#                                                                                #
#                       Primordial Black Hole Evaporation.                       #
#                 Neutrino-Antineutrino Asymmetry from Kerr PBHs                 #
#                                                                                #
#                         Author: Yuber F. Perez-Gonzalez                        #
#                           Based on: arXiv:2307.14408                           #
#                                                                                #
##################################################################################

import numpy as np
from odeintw import odeintw
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
from scipy.integrate import quad, quad_vec, ode, solve_ivp, solve_bvp
from scipy.optimize import root, toms748
from scipy.special import zeta, kn, factorial, factorial2, eval_legendre, legendre, lpmn, lpmv, hyp2f1
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import minimize, rosen, rosen_der

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, real, imag
from numpy import sin, cos, tan, arcsin, arccos, arctan
from numpy import absolute, angle, array

from pathos.multiprocessing import ProcessingPool as Pool

from termcolor import colored
from tqdm import tqdm

from progressbar import ProgressBar
pbar = ProgressBar()

from collections import OrderedDict
olderr = seterr(all='ignore')

import time

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------- Main Parameters ---------------------------------------------------- #
#
#          - 'c_th' : Cosine @ Theta angle wrt principal BH axis                                                           #
#
#          - 'ast'  : Primordial BH initial angular momentum a*                                                            # 
#
#          - 'w'    : Dimensionless energy parameter, w = GME                                                              #
#
#          - 's'    : Particle spin, code valid only for s = 0 or s = 1                                                    #
#
#------------------------------------------------------------------------------------------------------------------------- #

#-------------------------------------   Credits  ------------------------------------#
#
#      If using this code, please cite:                                               #
#
#      - arXiv:2307.14408                                                             #
#
#-------------------------------------------------------------------------------------#

n = 20

def rplus(GM, a, Q): return GM*(1. + np.sqrt(1 - a**2 - Q**2))
def rmins(GM, a, Q): return GM*(1. - np.sqrt(1 - a**2 - Q**2))

def Omega(GM, a, Q): return a/(GM*(2. - Q**2 + 2.*sqrt(1 - a**2 - Q**2)))
def Phi(GM, a, Q):   return Q*(1. + sqrt(1 - a**2 - Q**2))/(2. - Q**2 + 2.*sqrt(1 - a**2 - Q**2))

def TBH(GM, a, Q): 
    
    kappa_inv = 2.*GM*(1. + (1. - Q**2/2)/sqrt(1 - a**2 - Q**2))
    
    return 1./(2*pi*kappa_inv)

def K0(GM, w, a, Q, m): 
    
    rp = rplus(GM, a, Q)
    rm = rmins(GM, a, Q)
    
    return ((rp**2 + a**2*GM**2) * w - GM * Q * rp - a * GM * m)/(rp - rm)

def K1(GM, w, a, Q):
    rp = rplus(GM, a, Q)
    return 2.*rp*w - Q * GM

def K2(GM, w, a, Q):
    rp = rplus(GM, a, Q)
    rm = rmins(GM, a, Q)
    return (rp - rm)*w

def M0(GM, a, Q, mu):
    rp = rplus(GM, a, Q)
    return rp*mu

def M1(GM, a, Q, mu):
    rp = rplus(GM, a, Q)
    rm = rmins(GM, a, Q)
    return (rp - rm)*mu

def a_ang(i, pars):
    
    w, lm, a, mu, j, m, P = pars
    
    k = abs(m) + i

    ek = (-1)**(j - k) * P
    
    return (a*mu + ek*a*w) * sqrt((k+1)*(k+1) - m*m)/(2*(k + 1)) 


def b_ang(i, pars):
    
    w, lm, a, mu, j, m, P = pars
    
    k = abs(m) + i

    ek = (-1)**(j - k) * P
    
    return ek * (k + 1/2) * (1 - a*w*m/(k*(k+1))) + a*mu*m/(2*k*(k+1)) - lm


def g_ang(i, pars):
    
    w, lm, a, mu, j, m, P = pars
    
    k = abs(m) + i

    ek = (-1)**(j - k) * P
    
    return (a*mu - ek*a*w) * sqrt(k*k - m*m)/(2*k)


def cond_ang(lm, w, a, mu, j, m, P, n):
    
    if not np.isscalar(lm): lm = lm[0]

    ct = [w, lm, a, mu, j, m, P]
    
    f  = b_ang(0, ct)
    
    if f == 0.: f = 1.e-15
            
    C = f
    D = 0
    i = 1
    Del = 2
            
    while abs(Del-1)>1.e-20:
        
        D = b_ang(i, ct) - a_ang(i-1, ct)*g_ang(i, ct)*D
        if D == 0.: D = 1.e-15

        C = b_ang(i, ct) - a_ang(i-1, ct)*g_ang(i, ct)/C
        if C == 0.: C = 1.e-15

        D = 1./D
        Del = C*D
        f *= Del
        i += 1
    
    return f


def exp_lam(w, a, mu, j, m, P):
    
    def H(k): return (k*k - m*m)/(8*k*k*k)

    def K0p(k): return 1./k + 1/(k+1) 
    def K0m(k): return 1./k - 1/(k+1) 

    def K1p(k): return 1./((k+1)*(k-1)) + 1./(k*k)
    def K1m(k): return 1./((k+1)*(k-1)) - 1./(k*k)

    A00 =  P*(j+1/2)
    A10 = -0.5*P*m*K0p(j)
    A01 =  0.5*m*K0m(j)
    A20 =  P*(H(j)+ H(j+1))
    A02 =  A20
    A30 =  0.5*P*m*(K1p(j)*H(j) + K1p(j+1)*H(j+1))
    A03 =  0.5*m*(K1m(j)*H(j) - K1m(j+1)*H(j+1))
    A11 =  2*(H(j+1) - H(j))
    A21 =  0.5*m*((2*K1p(j+1)-K1m(j+1))*H(j+1) - (2*K1p(j)-K1m(j))*H(j))
    A12 =  0.5*m*P*((K1p(j+1)-2*K1m(j+1))*H(j+1) + (K1p(j)-2*K1m(j))*H(j))
    
    f = (A00          + A01*(a*mu)          + A02*(a*mu)**2       + A03*(a*mu)**3 +
         A10*(a*w)    + A11*(a*w)*(a*mu)    + A12*(a*w)*(a*mu)**2 +
         A20*(a*w)**2 + A21*(a*w)**2*(a*mu) +
         A30*(a*w)**3)

    return f

def lambda_lm(w, ast, mu, j, m, P): 
    
    l = j + P/2
    
    if ast*w == 0.:
        
        lm = P*(j + 1/2)
        
    elif w == mu:
        
        lm = -1/2 + P*sqrt((l + 1/2)**2 - 2*m*ast*w + ast*ast*w*w)
        
    elif ast*w < 1.e-2 or (j >= 7/2 and j - m >= 2):
        
        lm = exp_lam(w, ast, mu, j, m, P)
        
    else:
        
        Alm0 = exp_lam(w, ast, mu, j, m, P)

        lm = root(cond_ang, Alm0, args=(w, ast, mu, j, m, P, 100), method='lm').x[0]
    
    return lm

def RadEqs(y, v, GM, w, a, Q, mu, m, lm):

    G = v[0] # 
    F = v[1] # 

    x = log(1. + exp(y))
    
    k0 = K0(GM, w, a, Q, m)
    k1 = K1(GM, w, a, Q)
    k2 = K2(GM, w, a, Q)
    m0 = M0(GM, a, Q, mu)
    m1 = M1(GM, a, Q, mu)
    
    coeff1 = lm*(1. - exp(-x))/(sqrt(x)*sqrt(1.+x))
    
    coeff2 = (k0 + k1*x + k2*x*x)/(x*(1. + x))
    
    coeff3 = (m0 + m1*x)/(sqrt(x)*sqrt(1.+x))
    
    dGdy =  coeff1 * G + (coeff3 + coeff2) * (1. - exp(-x)) * F
    dFdy = -coeff1 * F + (coeff3 - coeff2) * (1. - exp(-x)) * G
    
    return [dGdy, dFdy]

def con_ic (K0, K1, K2, M0, M1, lm):
    
    a1 = (2*(-1j*lm + M0))/(-1j + 4*K0)
    
    b1 = ((2j*(K0 + K2)*(1j + 4*K0 - 4*K1 + 8*K2)*lm**3 + 2j*K0*M0**3 + 8*K0**2*M0**3 - 8*K0*K1*M0**3 + 2j*K2*M0**3 + 24*K0*K2*M0**3 
          - 8*K1*K2*M0**3 + 16*K2**2*M0**3 - 5*M0**2*M1 + 20j*K0*M0**2*M1 - 22j*K1*M0**2*M1 - 8*K0*K1*M0**2*M1 + 8*K1**2*M0**2*M1 
          + 44j*K2*M0**2*M1 + 16*K0*K2*M0**2*M1 - 32*K1*K2*M0**2*M1 + 32*K2**2*M0**2*M1 - 5*M0*M1**2 - 16j*K0*M0*M1**2 - 16*K0**2*M0*M1**2 
          - 24j*K1*M0*M1**2 + 16*K1**2*M0*M1**2 + 8j*K2*M0*M1**2 - 32*K0*K2*M0*M1**2 - 32*K1*K2*M0*M1**2 + 10*M1**3 + 32j*K0*M1**3 
          + 32*K0**2*M1**3 + 8j*K1*M1**3 - 32*K0*K1*M1**3 - 16j*K2*M1**3 + 64*K0*K2*M1**3 
          - (1j + 4*K0 - 4*K1 + 8*K2)*lm**2*(6*(K0 + K2)*M0 + (5j - 2*K1 + 4*K2)*M1) 
          - 1j*lm*(6*(K0 + K2)*(1j + 4*K0 - 4*K1 + 8*K2)*M0**2 + 2*(5j - 2*K1 + 4*K2)*(1j + 4*K0 - 4*K1 + 8*K2)*M0*M1 
                      - (-1j + 4*K0 + 4*K1)*(5j + 4*K0 - 4*K1 + 8*K2)*M1**2))/(
         (-1j + 4*K0)*(-2*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm**2 - 4j*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm*M0 
                         + 6j*K0*M0**2 + 8*K0**2*M0**2 - 8*K0*K1*M0**2 + 6j*K2*M0**2 + 24*K0*K2*M0**2 - 8*K1*K2*M0**2 
                         + 16*K2**2*M0**2 + 1j*(15 + 26j*K1 - 32j*K2 + 8*(2*K0 + K1)*(K0 - K1 + 2*K2))*lm*M1 - 15*M0*M1 
                         - 16*K0**2*M0*M1 - 26j*K1*M0*M1 + 8*K0*K1*M0*M1 + 8*K1**2*M0*M1 + 32j*K2*M0*M1 - 32*K0*K2*M0*M1 
                         - 16*K1*K2*M0*M1 + 15*M1**2 + 8j*K0*M1**2 + 16*K0**2*M1**2 + 12j*K1*M1**2 - 16*K0*K1*M1**2 
                         - 24j*K2*M1**2 + 32*K0*K2*M1**2)))
    
    b2 = ((32*K0**3*(-1j*lm + M0 - (1+1j)*M1)*(-1j*lm + M0 - (1-1j)*M1) 
          + (-5j + 2*K1)*M1*((3j + 2*(-7 - 4j*K1)*K1)*lm + (-3 + 2*K1*(-7j + 4*K1))*M0 
                                + (3 + 4j*K1)*M1) 
           + 2*K0*((-3 + 8*K1**2 + 16*(1j - 2*K2)*K2 - 2*K1*(7j + 4*K2))*lm**2 
                   + 2j*(-3 + 8*K1**2 + 16*(1j - 2*K2)*K2 - 2*K1*(7j + 4*K2))*lm*M0 
                    + (3 + 14j*K1 - 8*K1**2 + 8*(-2j + K1)*K2 + 32*K2**2)*M0**2 + 8*(1j + K1*(-6 - 3j*K1 + 8j*K2) + 9*K2)*lm*M1 
                   + 8*(-1 + K1*(-6j + 3*K1 - 8*K2) + 9j*K2)*M0*M1 + (30 - 80j*K2 + 8*K1*(7j - 2*K1 + 4*K2))*M1**2) + 32*K2**2*(-1j*lm + M0)*((-1j + K1)*M0 
                   - 1j*((-1j + K1)*lm + M1)) 
           + 2*K2*((-3 + 2*K1*(-7j + 4*K1))*lm**2 + (3 + 2*(7j - 4*K1)*K1)*M0**2 + 2*(-19j + 4*(7 + 2j*K1)*K1)*lm*M1 
                   - 8j*(-2j + K1)*M1**2 + 2*M0*((3 + 2j*K1)*(-1j + 4*K1)*lm + (19 + 4*(7j - 2*K1)*K1)*M1)) - 16*K0**2*(1j*M1*(2j*lm - 2*M0 + M1) 
                   + K2*(6*lm**2 + 12j*lm*M0 - 6*M0**2 - 8j*lm*M1 + 8*M0*M1 - 8*M1**2) + K1*((-1j*lm + M0)**2 + 2*M1**2)))/
        ((-1j + 4*K0)*(-2*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm**2 - 4j*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm*M0 + 6j*K0*M0**2 + 8*K0**2*M0**2 - 8*K0*K1*M0**2 
                      + 6j*K2*M0**2 + 24*K0*K2*M0**2 - 8*K1*K2*M0**2 + 16*K2**2*M0**2 + 1j*(15 + 26j*K1 - 32j*K2 + 8*(2*K0 + K1)*(K0 - K1 + 2*K2))*lm*M1 - 15*M0*M1 
                      - 16*K0**2*M0*M1 - 26j*K1*M0*M1 + 8*K0*K1*M0*M1 + 8*K1**2*M0*M1 + 32j*K2*M0*M1 - 32*K0*K2*M0*M1 
                      - 16*K1*K2*M0*M1 + 15*M1**2 + 8j*K0*M1**2 + 16*K0**2*M1**2 + 12j*K1*M1**2 - 16*K0*K1*M1**2 - 24j*K2*M1**2 + 32*K0*K2*M1**2)))
    
    c1 = ((M1*(16*K2**2*(-1j*lm + M0)**2 + 8*K0**2*(-1j*lm + M0 - (1+1j)*M1)*(-1j*lm + M0 - (1-1j)*M1) 
               + M1*((3j + 2*(-7 - 4j*K1)*K1)*lm + (-3 + 2*K1*(-7j + 4*K1))*M0 + (3 + 4j*K1)*M1) 
               + 2*K2*((-1j + 4*K1)*lm**2 + (1j - 4*K1)*M0**2 + 8*(lm + 1j*K1*lm)*M1 
                       - 4j*M1**2 + 2*M0*(lm + 4j*K1*lm - 4*(-1j + K1)*M1))
               + 2*K0*((-1j + 4*K1 - 12*K2)*lm**2 + (1j - 4*K1 + 12*K2)*M0**2 - 4j*(K1 - 4*K2)*lm*M1 
                       + (4j - 8*K1 + 16*K2)*M1**2 + 2*M0*(lm + 4j*(K1 - 3*K2)*lm + 2*(K1 - 4*K2)*M1))))/
          ((-1j + 4*K0)*(-2*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm**2 - 4j*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm*M0 
                            + 6j*K0*M0**2 + 8*K0**2*M0**2 - 8*K0*K1*M0**2 + 6j*K2*M0**2 + 24*K0*K2*M0**2 
                            - 8*K1*K2*M0**2 + 16*K2**2*M0**2 + 1j*(15 + 26j*K1 - 32j*K2 + 8*(2*K0 + K1)*(K0 - K1 + 2*K2))*lm*M1 
                            - 15*M0*M1 - 16*K0**2*M0*M1 - 26j*K1*M0*M1 + 8*K0*K1*M0*M1 + 8*K1**2*M0*M1 + 32j*K2*M0*M1 - 32*K0*K2*M0*M1 
                            - 16*K1*K2*M0*M1 + 15*M1**2 + 8j*K0*M1**2 + 16*K0**2*M1**2 + 12j*K1*M1**2 - 16*K0*K1*M1**2 - 24j*K2*M1**2 
                            + 32*K0*K2*M1**2)))
    
    c2 = ((2*(K0 + K2)*(16*K2**2*(-1j*lm + M0)**2 + 8*K0**2*(-1j*lm + M0 - (1+1j)*M1)*(-1j*lm + M0 - (1-1j)*M1) 
                        + M1*((3j + 2*(-7 - 4j*K1)*K1)*lm + (-3 + 2*K1*(-7j + 4*K1))*M0 
                              + (3 + 4j*K1)*M1) 
                        + 2*K2*((-1j + 4*K1)*lm**2 + (1j - 4*K1)*M0**2 + 8*(lm + 1j*K1*lm)*M1 - 4j*M1**2 + 2*M0*(lm + 4j*K1*lm - 4*(-1j + K1)*M1)) 
                        + 2*K0*((-1j + 4*K1 - 12*K2)*lm**2 + (1j - 4*K1 + 12*K2)*M0**2 - 4j*(K1 - 4*K2)*lm*M1 + (4j - 8*K1 + 16*K2)*M1**2 
                                + 2*M0*(lm + 4j*(K1 - 3*K2)*lm + 2*(K1 - 4*K2)*M1))))/
          ((-1j + 4*K0)*(-2*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm**2 - 4j*(K0 + K2)*(3j + 4*K0 - 4*K1 + 8*K2)*lm*M0 
                            + 6j*K0*M0**2 + 8*K0**2*M0**2 - 8*K0*K1*M0**2 + 6j*K2*M0**2 + 24*K0*K2*M0**2 - 8*K1*K2*M0**2 + 16*K2**2*M0**2 
                            + 1j*(15 + 26j*K1 - 32j*K2 + 8*(2*K0 + K1)*(K0 - K1 + 2*K2))*lm*M1 - 15*M0*M1 - 16*K0**2*M0*M1 
                            - 26j*K1*M0*M1 + 8*K0*K1*M0*M1 + 8*K1**2*M0*M1 + 32j*K2*M0*M1 - 32*K0*K2*M0*M1 - 16*K1*K2*M0*M1 + 15*M1**2 
                            + 8j*K0*M1**2 + 16*K0**2*M1**2 + 12j*K1*M1**2 - 16*K0*K1*M1**2 - 24j*K2*M1**2 + 32*K0*K2*M1**2)))
    

        
    return [a1, b1, c1, b2, c2]

# Initial condition

def int_cond(x, GM, w, a, Q, mu, m, lm):
    
    k0 = K0(GM, w, a, Q, m)
    k1 = K1(GM, w, a, Q)
    k2 = K2(GM, w, a, Q)
    m0 = M0(GM, a, Q, mu)
    m1 = M1(GM, a, Q, mu)
    
    a1, b1, c1, b2, c2 = con_ic (k0, k1, k2, m0, m1, lm)
    
    y = log(exp(x) - 1)

    R1 = exp(1j * k0 * y) * sqrt(x) * (a1 + b1 * x + c1 * x * x)
    R2 = exp(1j * k0 * y) * (1. + b2 * x + c2 * x * x)
    
    G =  0.5  * (R1 + R2)
    F = -0.5j * (R1 - R2)
    
    return [G, F]

def con_far (x, K0, K1, K2, M0, M1, lm):
    
    k = sqrt(K2**2 - M1**2)
    
    c = (2.*K1*K2 - 2.*K2**2 - 2.*M0*M1 + M1**2)/(2.*k)
    
    v = (2.*K1*M1 - K2*M1 - 2.*K2*M0)/(4.*k*k)
    
    kappa = (K2 + K1/x + K0/(x*x))/(1. + 1/x)
    
    M = (M1 + M0/x)/sqrt(1. + 1/x)
    
    alph = 0.25*log((kappa + M)/(kappa - M))
    
    beta = lm*(lm - 1.)/(4.*k*k*x*x)
    
    delt = lm*(lm + 1.)/(4.*k*k*x*x)
    
    gamm = lm*(lm - 1.)/(2.*k*x) - ((lm - 1.)*(k*lm + c*lm + 2.*k*v) - lm*c)/(4.*k*k*x*x)
    
    epsl = lm*(lm + 1.)/(2.*k*x) - ((lm + 1.)*(k*lm + c*lm + 2.*k*v) + lm*c)/(4.*k*k*x*x)
    
    return [alph, beta, delt, gamm, epsl]

def Gamma_l(w, j, m, P, GM, ast, Q, mu):
    
    g = 1.
    
    A   = 4.*pi*GM*GM*(2. - Q**2 + 2.*sqrt(1 - ast**2 - Q**2))
    kap = 2.*pi*TBH(GM, ast, Q)
    
    if GM*w < 2.5e-2:
        
        f = 1
        
        for n in range(1, int(j+1/2)+1): f *= (1. + ((w - m*Omega(GM, ast, Q))/(n*kap - 0.5 * kap))**2) 
        
        jp = int(j + 1/2)
        jm = int(j - 1/2)
        j2 = int(2*j)
        j1 = int(2* j + 1)
    
        Gaml = (factorial(jm)*factorial(jp)/(factorial(j2) * factorial2(j1)))**2 * f * (A*kap*w/(2.*pi))**(2*j+1)
    
    else:
    
        k0 = K0(GM, w, ast, Q, m)
        k1 = K1(GM, w, ast, Q)
        k2 = K2(GM, w, ast, Q)
        m0 = M0(GM, ast, Q, mu)
        m1 = M1(GM, ast, Q, mu)
    
        y0 = -23. #log(x0)

        x0 = log(1. + exp(y0))
    
        lm = lambda_lm(w, ast, mu, j, m, P)
    
        v0 = int_cond(x0, GM, w, ast, Q, mu, m, lm) # Initial condition
            
        # Solving Equations
        solREqs = solve_ivp(lambda t, z: RadEqs(t, z, GM, w, ast, Q, mu, m, lm), [y0, 750.], v0, tol=1.e-7, atol=1.e-10)
    
        Gsol = solREqs.y[0,-1]
        Fsol = solREqs.y[1,-1]
    
        x_inf = solREqs.t[-1]
    
        alph, beta, delt, gamm, epsl = con_far (x_inf, k0, k1, k2, m0, m1, lm)
    
        Ef_num = (0.5*exp(-2.*alph + 2.*delt) * abs(2.*Gsol)**2 + 0.5*exp(2.*(alph + beta)) * abs(2.*Fsol)**2 
                  - tan(gamm - epsl) * real(2.*np.conjugate(Gsol)*2*Fsol))
    
        Ef_den = imag(2.*np.conjugate(Gsol)*2*Fsol)
    
        Ef = Ef_num/Ef_den
    
        Gaml = 2./(Ef + 1.)
    
    return Gaml

def d2Ndedt(w, j, m, P, GM, ast, Q, mu):
    
    g = 1.
    
    k0 = K0(GM, w, ast, Q, m)
    k1 = K1(GM, w, ast, Q)
    k2 = K2(GM, w, ast, Q)
    m0 = M0(GM, ast, Q, mu)
    m1 = M1(GM, ast, Q, mu)
    
    x0 = 1.e-10

    y0 = log(exp(x0) - 1)
    
    lm = lambda_lm(w, ast, mu, j, m, P)
    
    v0 = int_cond(x0, GM, w, ast, Q, mu, m, lm) # Initial condition
            
    # Solving Equations
    solREqs = solve_ivp(lambda t, z: RadEqs(t, z, GM, w, ast, Q, mu, m, lm), [y0, 700.], v0, rtol=1.e-7, atol=1.e-10)
    
    Gsol = solREqs.y[0,-1]
    Fsol = solREqs.y[1,-1]
    
    x_inf = solREqs.t[-1]
    
    alph, beta, delt, gamm, epsl = con_far (x_inf, k0, k1, k2, m0, m1, lm)
    
    Ef_num = (0.5*exp(-2.*alph + 2.*delt) * abs(2.*Gsol)**2 + 0.5*exp(2.*(alph + beta)) * abs(2.*Fsol)**2 
              - tan(gamm - epsl) * real(2.*np.conjugate(Gsol)*2*Fsol))
    
    Ef_den = imag(2.*np.conjugate(Gsol)*2*Fsol)
    
    Ef = Ef_num/Ef_den
    
    Gaml = 2./(Ef + 1.)
    
    FD = 1./(exp((w - m*Omega(GM, ast, Q) - g*Phi(GM, ast, Q))/TBH(GM, ast, Q)) + 1)
    
    return Gaml*FD/(2.*pi)


def sol_0(th, j, m, P):
    
    l = j + P/2
    
    cjmP = P*(l + 1/2) - m
    
    x = cos(th)
    
    if abs(m + 1/2) <= l:
        PLm_p = lpmv(m + 1/2, l, x)
    else:
        PLm_p = 0.
    
    if abs(m - 1/2) <= l:
        PLm_m = lpmv(m - 1/2, l, x)
    else:
        PLm_m = 0.
    
    S1 = sqrt(factorial(j-m)/(2.*np.pi*factorial(j+m))) * ( cos(th/2) * PLm_p + sin(th/2) * cjmP * PLm_m)
    S2 = sqrt(factorial(j-m)/(2.*np.pi*factorial(j+m))) * (-sin(th/2) * PLm_p + cos(th/2) * cjmP * PLm_m)
    
    return np.array([S1, S2])

def b_k(n, pars):
    
    w, a, mu, j, m, P = pars
    
    lm = lambda_lm(w, a, mu, j, m, P)
    
    abg_k = array([abg_ang(i, [w, lm, a, mu, j, m, P]) for i in range(3*n+1)]) 
    
    alp_k = abg_k[:,0]
    bet_k = abg_k[:,1]
    gam_k = abg_k[:,2]
    
    bk = zeros(n)
        
    bk[0] = 1.0
    bk[1] = -(bet_k[0]/alp_k[0])*bk[0]
    
    for i in range(2, n):
    
        f = 1.e-30
        
        C   = f
        D   = 0.
        Del = 0.
    
        for k in range(i, 3*n): 
        
            D = bet_k[k] - alp_k[k-1] * gam_k[k]*D
    
            if D == 0.: D += 1.e-30

            C = bet_k[k] - alp_k[k-1] * gam_k[k]/C
    
            if C == 0.: C += 1.e-30
    
            D = 1./D
    
            Del = C*D
        
            f *= Del
        
            if abs(Del - 1) < 1.e-10*bet_k[0]: break
            
        bk[i] = bk[i-1] * f/alp_k[i-1]

    mn = np.min(bk)
    mx = np.max(bk)

    if mx > abs(mn):
        bk = bk/mx
    else:
        bk = bk/mn
        
    return bk

def Spm(th, w, a, mu, j, m, P, n):
    
    S1 = 0.
    S2 = 0.
    
    if a*w < 1.e-2:
        
        S1, S2 = sol_0(th, j, m, P)
        
    else:
    
        coeff = b_k(n, [w, a, mu, j, m, P])
                       
        for i in range(n):
        
            k = abs(m) + i
        
            S1k, S2k = sol_0(th, k, m, P)
        
            S1 += coeff[i] * S1k * (-1)**(j - k)
            S2 += coeff[i] * S2k
    
    return np.array([S1, S2])

def Spm_n(th, w, a, mu, j, m, P, n):
    
    S1 = 0.
    
    if a*w < 1.e-2:
        
        S1, S2 = sol_0(th, j, m, P)
        
    else:
        
        coeff = b_k(n, [w, a, mu, j, m, P])

        for i in range(n):

            k = abs(m) + i

            S1k, S2k = sol_0(th, k, m, P)

            S1 += coeff[i] * S1k * (-1)**(j - k)
            
    return 2.*pi * sin(th) * S1*S1

#----------------------------------------------------#
#       Integrand for DM particle production Eq      #
#----------------------------------------------------#

def d3A_dEdtdOm (w, th, ast, pars):

    j, m, P, Q, mu = pars
    
    g = 1.

    GM = 1.
    
    Gamma_lm = Gamma_l(w, j, m, P, GM, ast, Q, mu)
    S1, S2   = Spm(th, w, ast, mu, j, m, P, n)

    Norm = quad(Spm_n, 0, pi, args=(w, ast, mu, j, m, P, n))[0]

    pr = sqrt(w*w - mu*mu) 
    
    FD = 1./(exp((w - m*Omega(GM, ast, Q) - g*Phi(GM, ast, Q))/TBH(GM, ast, Q)) + 1) 
    
    Assm = -(0.5/pi) *  (w/pr) * (Gamma_lm*FD) * (S2*S2 - S1*S1)/abs(Norm)
    
    return Assm

def d3N_dEdtdOm (w, th, ast, pars):

    j, m, P, Q, mu = pars
    
    g = 1.
    
    GM = 1.

    Gamma_lm = Gamma_l(w, j, m, P, GM, ast, Q, mu)
    S1, S2   = Spm(th, w, ast, mu, j, m, P, n)

    Norm = quad(Spm_n, 0, pi, args=(w, ast, mu, j, m, P, n))[0]

    pr = sqrt(w*w - mu*mu) 
    
    FD = 1./(exp((w - m*Omega(GM, ast, Q) - g*Phi(GM, ast, Q))/TBH(GM, ast, Q)) + 1) 
    
    d3N = (0.5/pi) * (Gamma_lm*FD) * (S1*S1 + S2*S2)/Norm 
    
    return d3N

def d3AN_dEdtdOm (w, th, ast, pars):

    j, m, P, Q, mu = pars
    
    g = 1.

    GM = 1.
    
    Gamma_lm = Gamma_l(w, j, m, P, GM, ast, Q, mu)
    S1, S2   = Spm(th, w, ast, mu, j, m, P, n)

    Norm = quad(Spm_n, 0, pi, args=(w, ast, mu, j, m, P, n))[0]

    pr = sqrt(w*w - mu*mu) 
    
    FD = 1./(exp((w - m*Omega(GM, ast, Q) - g*Phi(GM, ast, Q))/TBH(GM, ast, Q)) + 1) 

    Assm = -(0.5/pi) *  (w/pr) * (Gamma_lm*FD) * (S2*S2 - S1*S1)/abs(Norm)
    d3N  =  (0.5/pi) * (Gamma_lm*FD) * (S1*S1 + S2*S2)/abs(Norm)
    
    return np.array([Assm, d3N])

#-----------------------------------------------------------------------------------------------------------------#
#                                               d^3N/dEdtdOmega                                                   #
#-----------------------------------------------------------------------------------------------------------------#

class d3F_dEdtdOm:
    '''
    Code to compute the full asymmetry rate for neutrinos, d^3A/dEdtdOmega,
    and the total fermion emission d^3N/dEdtdOmega.
    '''

    def __init__(self, c_th, ast, w, Q, mf):

        self.c_th = c_th  # Cosine @ Theta angle wrt principal BH axis
        self.ast  = ast   # PBH spin parameter a_*
        self.w    = w     # Dimensionless energy parameter w = GME
        self.Q    = Q     # PBH charge
        self.mf   = mf    # Fermion mass
        
        self.jmax = 19    # Maximum value of angular momenta

        self.P = -1       # Parity
        
   #----------------------------------------------------------------------------------------------------------------------------------#
   #                                                         Main functions                                                           #
   #----------------------------------------------------------------------------------------------------------------------------------#
    
    def Asym(self):
        '''
        Hawking ratefor neutrino - antineutrino as function of time, energy and angle
        in units of (nu - antinu)/(GeV*s*sr)
        '''
        
        print(colored("Hawking rate for energy, angle and time...", 'blue'))
        
        d3A = 0.

        UNjt = 1.51927e24 # Conversion factor for rate in GeV^-1 s^-1 sr^-1

        def d3Adhdedt(j, m): return UNjt*d3A_dEdtdOm(self.w, arccos(self.c_th), self.ast, [j/2, m/2, self.P, self.Q, self.mf])
                    
        arr_vals = [(j, m) for j in range(1, self.jmax+2, 2) for m in range(-j, j+1,2)]

        with Pool(12) as pool:
            d2adwdt = np.array(pool.map(lambda x: d3Adhdedt(*x), arr_vals))

        for t in range(int((1/2 + self.jmax/2)*(3/2 + self.jmax/2))):
            d3A += d2adwdt[t]
                    
        return d3A

    def Ntot(self):
        '''
        Hawking rate for neutrino +  antineutrino, as function of time, energy and angle
        in units of (nu + antinu)/(GeV*s*sr)
        '''
        
        print(colored("Hawking rate for energy, angle and time...", 'blue'))
        
        d3N = 0.

        UNjt = 1.51927e24 # Conversion factor for rate in GeV^-1 s^-1 sr^-1

        def d3Ndhdedt(j, m): return UNjt*d3N_dEdtdOm(self.w, arccos(self.c_th), self.ast, [j/2, m/2, self.P, self.Q, self.mf])

        arr_vals = [(j, m) for j in range(1, self.jmax+2, 2) for m in range(-j, j+1,2)]

        with Pool(12) as pool:
            d2ndwdt = np.array(pool.map(lambda x: d3Ndhdedt(*x), arr_vals))

        for t in range(int((1/2 + self.jmax/2)*(3/2 + self.jmax/2))):
            d3N += d2ndwdt[t]
                    
        return d3N

    def A_et_N(self):
        '''
        Hawking rate for Asymmetry and total rate, as function of time, energy and angle
        in units of (nu +- antinu)/(GeV*s*sr)
        '''
        
        print(colored("Total and asymmetry Hawking rate for energy, angle and time...", 'blue'))
        
        d3N = np.array([0., 0.])

        UNjt = 1.51927e24 # Conversion factor for rate in GeV^-1 s^-1 sr^-1

        def d3Ndhdedt(j, m): return UNjt*d3AN_dEdtdOm(self.w, arccos(self.c_th), self.ast, [j/2, m/2, self.P, self.Q, self.mf])

        arr_vals = [(j, m) for j in range(1, self.jmax+2, 2) for m in range(-j, j+1,2)]

        with Pool(12) as pool:
            d2ndwdt = np.array(pool.map(lambda x: d3Ndhdedt(*x), arr_vals))

        for t in range(int((1/2 + self.jmax/2)*(3/2 + self.jmax/2))):
            d3N += d2ndwdt[t]
                    
        return d3N

    
