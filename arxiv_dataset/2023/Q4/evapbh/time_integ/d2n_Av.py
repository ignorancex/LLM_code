##################################################################################
#                                                                                #
#                        Primordial Black Hole Evaporation.                      #
#                     Particle Angular Dependence from Kerr PBHs                 #
#       Code to integrate on energy and angle d2N/dEdOm to compute secondaries   #
#                                                                                #
#                         Author: Yuber F. Perez-Gonzalez                        #
#                           Based on: arXiv:2307.14408                           #
#                                                                                #
##################################################################################

import numpy as np
from odeintw import odeintw
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, quad_vec, ode, solve_ivp, solve_bvp
from scipy.optimize import root, toms748
from scipy.special import zeta, kn, factorial, factorial2, eval_legendre, legendre, lpmn, lpmv, hyp2f1
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator
from scipy.optimize import minimize, rosen, rosen_der

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, real, imag
from numpy import sin, cos, tan, arcsin, arccos, arctan
from numpy import absolute, angle, array

from pathos.multiprocessing import ProcessingPool as Pool

from tqdm import tqdm

import time

class Simp2D:

    def __init__(self, f, pars, lims):

        self.f    = f
        self.pars = pars
        self.lims = lims

    def weight(self):
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        wx = np.ones(2*Nx+1)
        wx[0]  = 1.
        wx[2*Nx] = 1.

        for i in range(1, 2*Nx):
            if i % 2 == 0:
                wx[i] = 2.
            else:
                wx[i] = 4.
        
        
        wy = np.ones(2*Ny+1)
        wy[0]  = 1.
        wy[2*Ny] = 1.

        for i in range(1, 2*Ny):
            if i % 2 == 0:
                wy[i] = 2.
            else:
                wy[i] = 4
        
        w = np.kron(wy, wx)
        
        return w
        
    def integrand(self, i, j):
        
        pars = self.pars
        
        xi, xf, yi, yf, Nx, Ny = self.lims
        
        dx = (xf - xi)/(2*Nx)
        dy = (yf - yi)/(2*Ny)
        
        #print(i, j, i + j*(2*Nx+1))
                
        return self.f(xi + i*dx, yi + j*dy, pars) 
    

    def integral(self):
        
        pars = self.pars
        
        xi, xf, yi, yf, Nx, Ny = self.lims
        
        dx = (xf - xi)/(2*Nx)
        dy = (yf - yi)/(2*Ny)

        arr_vals = [(i, j) for j in range(2*Ny+1) for i in range(2*Nx+1)]

        with Pool(8) as pool:
            grid = np.array(pool.map(lambda x: self.integrand(*x), arr_vals))
        
        w = self.weight()

        res = 0.

        for j in range(2*Nx+1):
            for k in range(2*Ny+1):
                res += w[(2*Ny+1)*j + k] * grid[(2*Ny+1)*j + k]
        
        return res*dx*dy/9.

#Integrating d2n/dwdO in w and cos_th to compute secondaries

tau_s = 1.0

d2N_tab=np.loadtxt("./Data/d2N_dEdOm_t="+str(np.round(tau_s,3))+".txt")

UNjt = 1.51927e24

Nas = 50
Nws = 50
Nhs = 50

as_arr  = np.linspace(0., 0.9999, Nas)
ws_arr  = np.linspace(2, 6., Nws)
ths_arr = np.linspace(-1, 1, Nhs)

d2Nnu_int = zeros(((Nas,Nws,Nhs)))
d2Nanu_int = zeros(((Nas,Nws,Nhs)))
d2Nf_int = zeros(((Nas,Nws,Nhs)))
d2Nv_int = zeros(((Nas,Nws,Nhs)))
d2Ns_int = zeros(((Nas,Nws,Nhs)))

#d2A_int.shape

for i in tqdm(range(Nas)):
    for j in range(Nws):
        for k in range(Nhs):
            d2Nnu_int[i,j,k]  = d2N_tab[i*Nws*Nhs + j*Nhs + k, 0]
            d2Nanu_int[i,j,k] = d2N_tab[i*Nws*Nhs + j*Nhs + k, 1]
            d2Nf_int[i,j,k]   = d2N_tab[i*Nws*Nhs + j*Nhs + k, 0] + d2N_tab[i*Nws*Nhs + j*Nhs + k, 1]
            d2Nv_int[i,j,k]   = d2N_tab[i*Nws*Nhs + j*Nhs + k, 2]
            d2Ns_int[i,j,k]   = d2N_tab[i*Nws*Nhs + j*Nhs + k, 3]

d2Nnu_I  = RegularGridInterpolator((as_arr, ws_arr, ths_arr), d2Nnu_int, bounds_error=False, fill_value = None)
d2Nanu_I = RegularGridInterpolator((as_arr, ws_arr, ths_arr), d2Nanu_int, bounds_error=False, fill_value = None)

d2Nf_I = RegularGridInterpolator((as_arr, ws_arr, ths_arr), d2Nf_int, bounds_error=False, fill_value = None)
d2Nv_I = RegularGridInterpolator((as_arr, ws_arr, ths_arr), d2Nv_int, bounds_error=False, fill_value = None)
d2Ns_I = RegularGridInterpolator((as_arr, ws_arr, ths_arr), d2Ns_int, bounds_error=False, fill_value = None)

def d2n_i(log_w, cos_th, pars):

    w = 10.**log_w
    
    ast = pars[0]
    
    d2nf = d2Nf_I([ast, log_w, cos_th])[0]
    d2nv = d2Nv_I([ast, log_w, cos_th])[0]
    d2ns = d2Ns_I([ast, log_w, cos_th])[0]
    
    return np.array([d2nf, d2nv, d2ns])*log(10.)*w

Na      = 50 
Nep_bin = 50
Nhp_bin = 50

ep_bin   = np.linspace(2., 6., Nep_bin + 1)
cthp_bin = np.linspace(-1, 1, Nhp_bin + 1)
a_arr    = np.linspace(0.0, 0.9999, Na)

Njk_bin = np.zeros((Na*Nep_bin*Nhp_bin, 3))

start = time.time()

for k in tqdm(range(Na)):
    for j in range(Nhp_bin):
        for i in range(Nep_bin):
            Njk_bin[k*Nhp_bin*Nep_bin + j*Nep_bin + i] = Simp2D(d2n_i, [a_arr[k]],
                                                                [ep_bin[i], ep_bin[i+1], cthp_bin[j], cthp_bin[j+1], 10, 10]).integral()
        
end = time.time()
print(f"Monochromatic Time {end - start} s")


np.savetxt("./Data/Int_d2N_dEdOm_t="+str(np.round(tau_s,3))+".txt", Njk_bin)
