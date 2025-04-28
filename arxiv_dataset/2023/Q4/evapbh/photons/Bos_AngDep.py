##################################################################################
#                                                                                #
#                       Primordial Black Hole Evaporation.                       #
#               Photon and Scalar Angular Dependence from Kerr PBHs              #
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

from Integrator import Simp1D, Simp2D, Simp2D_varlims, Trap2D_varlims

from lms import lambdalms # Angular eigenvalues

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

def Delta(r, GM, a): return r**2 - 2*GM*r + a**2*GM**2

def rplus(GM, a): return GM*(1. + np.sqrt(1 - a**2))

def rmins(GM, a): return GM*(1. - np.sqrt(1 - a**2))

def rstar(r, GM, a, w, m):

    if a == 0.:
        return r + 2.*GM*np.log(r/(2.*GM) - 1.)
    else:
        return (r + ((2.*GM*rplus(GM,a) + a*GM*m/w)/(rplus(GM,a) - rmins(GM,a)))*np.log(r/rplus(GM,a) - 1.)
                  - ((2.*GM*rmins(GM,a) + a*GM*m/w)/(rplus(GM,a) - rmins(GM,a)))*np.log(r/rmins(GM,a) - 1.))

def Omega(GM, a): return a/(2.*GM*(1. + sqrt(1 - a**2)))

def TBH(GM, a): 
    
    kappa_inv = 2.*GM*(1. + 1./sqrt(1 - a**2))
    
    return 1./(2*pi*kappa_inv)

def abg_ang(p, pars):
    
    sAlm, w, a, l, m, s = pars
    
    kp = abs(m + s)/2
    km = abs(m - s)/2
    
    c = a*w
    
    ap = -2*(p + 1)*(p + 2*km + 1) 
    bp = p*(p - 1) + 2*p*(km + kp + 1 - 2*c) - (2*c*(2*km + s + 1) - (km + kp)*(km + kp + 1)) - (c*c + s*(s + 1) + sAlm)
    gp = 2*c*(p + km + kp + s)
    
    abg = np.array([ap, bp, gp])
    
    return abg


def b_k(n, pars):
    
    w, a, l, m, s = pars

    sAlm = lambdalms(l, m, s, a*w).llms()
    
    abg_k = array([abg_ang(i, [sAlm, w, a, l, m, s]) for i in range(3*n+1)]) 
    
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

def Spm(th, w, a, l, m, s, n):

    c = a*w
    x = cos(th)
    
    kp = abs(m+s)/2
    km = abs(m-s)/2
    
    S = 0.
    
    coeff = b_k(n, [w, a, l, m, s])
        
    for p in range(n): S += coeff[p] * (1 + x)**p
            
    S *= exp(c*x) * (1 + x)**km * (1 - x)**kp 
                
    return S

def Spm_n(th, w, a, l, m, s, n):
    
    c = a*w
    x = cos(th)
    
    kp = abs(m+s)/2
    km = abs(m-s)/2
    
    S = 0.
    
    coeff = b_k(n, [w, a, l, m, s])
        
    for p in range(n): S += coeff[p] * (1 + x)**p
            
    S *= exp(c*x) * (1 + x)**km * (1 - x)**kp                  
    
    return 2 * pi * sin(th) * S*S

def alpha2(GM, a, w, m): return a**2*GM**2 + a*GM*m/w

def alpha(GM, a, w, m):

    if alpha2(GM, a, w, m) >= 0:
        return np.sqrt(alpha2(GM, a, w, m))

    else:
        return np.sqrt(-alpha2(GM, a, w, m))


def r2(r, GM, a, w, m): return r**2 + a**2*GM**2 + a*GM*m/w

def req(rs, v, GM, a, w, m):
                
    r = v[0]
                
    drdrs = (r**2 - 2*GM*r + a**2*GM**2)/(r**2 + a**2*GM**2 + a*GM*m/w)
                
    return [drdrs]

def V(r, GM, a, w, sAlm, m, s):
    
    D = Delta(r, GM, a)
    
    rho_2 = r**2 + alpha2(GM, a, w, m)

    rho_4 = rho_2**2
    
    alp_2 = alpha2(GM, a, w, m)
    
    Vs = 0

    if abs(s) == 0.:

        Vs = (D/rho_4)*(sAlm + (D + 2.*r*(r - GM))/rho_2 - (3.*r**2*D/rho_4))

    elif abs(s) == 1.:

        if alp_2 >= 0.:
        
            Vs = (D/rho_4) * (sAlm - alp_2*D/rho_4  - 1j * sqrt(alp_2)*(2.*(r - GM)/rho_2 - 4.*r*D/rho_4))
                         
        else:
            
            alp_2 = -alp_2
            
            Vs =  (D/rho_4) * (sAlm + (D/rho_4) * alp_2 - 4*r*(D/rho_4)*sqrt(alp_2) + 2.*sqrt(alp_2)*(r - GM)/rho_2)

    return Vs

def CDEqs(rs, v, GM, a, w, sAlm, m, s, tor_f):
    
    psi, dpsi = v
                
    ddpsi= (V(tor_f(rs), GM, a, w, sAlm, m, s) - w**2)*psi
                
    return [dpsi, ddpsi]

def Gamma_l(w, ast, l, m, s):
    
    GM = 1.

    if GM*w > 2.5e-1: eps = 1.e-4
    else: eps = 1.e-3
    
    inf = 1000
    
    if w > 1.: inf /= w
    
    rsminf = rstar(rplus(GM, ast) + eps, GM, ast, w, m)
    rspinf = rstar(rplus(GM, ast) + inf, GM, ast, w, m)
    
    c = ast*GM*w
    
    sAlm = lambdalms(l, -m, -s, c).llms() + c*c + 2*m*c 
    
    if w > -ast*m/(2.*rplus(GM, ast)):
            
        soltor = solve_ivp(lambda t, z: req(t, z, GM, ast, w, m), [rsminf, rspinf], [rplus(GM, ast) + eps], rtol=1.e-10, atol=1.e-10) 

        tor = interp1d(soltor.t, soltor.y[0], bounds_error=False, fill_value=(soltor.y[0,0], soltor.y[0,-1]))

        v0_C = [np.exp(-1j * w * rsminf), -1j * w * np.exp(-1j * w * rsminf)]

        solpsi = solve_ivp(lambda t, z: CDEqs(t, z, GM, ast, w, sAlm, m, s, tor), [rsminf, rspinf], v0_C, method='BDF', rtol=1.e-7, atol=1.e-10)
        
        psi  = solpsi.y[0,-1]
        dpsi = solpsi.y[1,-1]/(1j*w)
        
        Aout = (psi - dpsi)/(2.*np.exp(-1j*w*rspinf))
        
        Glms = 1./abs(Aout)**2
        
    else:
                
        #########################################
        #
        #      Solution in first branch
        #
        #########################################
            
        rdiv = alpha(GM, ast, w, m)
        rsdivm = rstar(rdiv - eps, GM, ast, w, m)
            
        v0_l = [rplus(GM, ast) + eps]

        soltorl = solve_ivp(lambda t, z: req(t, z, GM, ast, w, m), [rsminf, rsdivm], v0_l, method='Radau', rtol=1.e-15, atol=1.e-20)

        torl = interp1d(soltorl.t, soltorl.y[0], bounds_error=False, fill_value=(soltorl.y[0,0], soltorl.y[0,-1]))

        v0CDE_l = [np.exp(1j * w * rsminf), 1j * w * np.exp(1j * w * rsminf)]
            
        solpsil = solve_ivp(lambda t, z: CDEqs(t, z, GM, ast, w, sAlm, m, s, torl), [rsminf, rsdivm], v0CDE_l, 
                            method='BDF', rtol=1.e-7, atol=1.e-10)
        
        #########################################
        #
        #      solution in second branch
        #
        #########################################

        rsdivp = rstar(rdiv + eps, GM, ast, w, m)
        
        v0_r = [np.sqrt(-alpha2(GM, ast, w, m)) + eps]

        soltorr = solve_ivp(lambda t, z: req(t, z, GM, ast, w, m), [rsdivp, rspinf], v0_r, method='Radau', rtol=1.e-15, atol=1.e-20)

        torr = interp1d(soltorr.t, soltorr.y[0], bounds_error=False, fill_value=(soltorr.y[0,0], soltorr.y[0,-1]))
        
        rspinf = soltorr.t[-1]

        v0CDE_r = [solpsil.y[0,-1], 0.5*solpsil.y[0,-1]*(Delta(rdiv + eps, GM, ast)/r2(rdiv + eps, GM, ast, w, m))*(1./eps)]

        solpsir = solve_ivp(lambda t, z: CDEqs(t, z, GM, ast, w, sAlm, m, s, torr), [rsdivp, rspinf], v0CDE_r, 
                            method='BDF', rtol=1.e-7, atol=1.e-10)
        
        psi = solpsir.y[0,-1]
        dpsi = solpsir.y[1,-1]/(1j*w)

        Aout = (psi - dpsi)/(2.*np.exp(-1j*w*rspinf))

        Glms = -(-1.)**(2.*s)/abs(Aout)**2

    return Glms

#----------------------------------------------------#
#       Integrand for DM particle production Eq      #
#----------------------------------------------------#

def d3N_dEdtdOm (w, th, ast, pars):

    l, m, s = pars
    
    GM = 1.

    n = 50

    Glms = Gamma_l(w, ast, l, m, s)
    
    Sp = Spm(th, w, ast, l, -m, -s, n)
    Sm = (-1)**(-l + m) * Spm(pi - th, w, ast, l, -m, -s, n)

    n_0 = quad(Spm_n, 0, pi, args=(w, ast, l, -m, -s, n))[0]
    
    DF = 1./(exp((w + m*Omega(GM, ast))/TBH(GM, ast)) - (-1)**(2*s))
    
    d3N = (Glms*DF/(2.*pi))*(Sp*Sp + Sm*Sm)/n_0

    return d3N


#--------------------------------------------------------------------------------------------------------------------------------#
#                                               Integration of d^3N/dEdtdOmega                                                   #
#--------------------------------------------------------------------------------------------------------------------------------#

class d3B_dEdtdOm:
    '''
    Code to compute the total particle rate for scalar (s=0) and vector (s=1), d^3N/dEdtdOmega.
    '''

    def __init__(self, c_th, ast, w, s):

        self.c_th = c_th  # Cosine @ Theta angle wrt principal BH axis
        self.ast  = ast   # PBH spin parameter a_*
        self.w    = w     # Dimensionless energy parameter w = GME
        self.s    = s      # Particle spin

        assert s == 0 or s == 1, print(colored("Code valid only for spin s = 0 or s = 1", 'red'))
        
        self.jmax = 8  # Maximum of total angular momenta to be computed

   #----------------------------------------------------------------------------------------------------------------------------------#
   #                                                        Main functions                                                            #
   #----------------------------------------------------------------------------------------------------------------------------------#
    
    def Ntot(self):
        '''
        Hawking rate for photon angular distribution as function of time, energy and angle
        in units of (gamma)/(GeV*s*sr)
        '''
        
        d3N = 0.

        UNjt = 1.51927e24 # Conversion factor for rate in GeV^-1 s^-1 sr^-1
        
        def d3Ndhdedt(l, m): return UNjt*d3N_dEdtdOm(self.w, arccos(self.c_th), self.ast, [l, m, self.s])
        
        arr_vals = [(l, m) for l in range(self.s, self.jmax+1) for m in range(-l, l+1)]

        with Pool(12) as pool:
            d3ndwdt = np.array(pool.map(lambda x: d3Ndhdedt(*x), arr_vals))
        
        for t in range((1 + self.jmax - self.s)*(1 + self.jmax + self.s)):
            d3N += d3ndwdt[t]
                            
        return d3N
