###################################################################################
#                                                                                 #
#                       Primordial Black Hole Evaporation.                        #
#                    Particle angular dependence from Kerr PBHs                   #
#   Class to determine time integration of Hawking flux for a given time window   #
#                                                                                 #
#                         Author: Yuber F. Perez-Gonzalez                         #
#                           Based on: arXiv:2307.14408                            #
#                                                                                 #
###################################################################################

import numpy as np
from odeintw import odeintw
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
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

import BHProp as bh #Schwarzschild and Kerr BHs library

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
#          - 'tau_s' : Time in s before BH evaporation, current tables are for t = 1.0, 10.0, and 100.0                    #
#
#------------------------------------------------------------------------------------------------------------------------- #

#-------------------------------------   Credits  ------------------------------------#
#
#      If using this code, please cite:                                               #
#
#      - arXiv:2307.14408                                                             #
#
#-------------------------------------------------------------------------------------#

#----------------------------------------------------#
#       Integrand for DM particle production Eq      #
#----------------------------------------------------#

def int_EO(t, p, cos_th, sol, d3A_I, d3N_I, d3Nv_I, d3Ns_I):
    
    Mt, ast = sol(t)
    
    UNjt = 1.51927e24
    
    w    = bh.GCF * (Mt/bh.GeV_in_g) * p  # Dimensionless momentum ->  w = G*MBH*p
    
    Jac = log(10.)*10**t                   

    Assm = d3A_I([cos_th, ast, log10(w)])[0]/UNjt
    Ntot = d3N_I([cos_th, ast, log10(w)])[0]/UNjt

    Nv   = d3Nv_I([cos_th, ast, log10(w)])[0]/UNjt

    Ns   = d3Ns_I([cos_th, ast, log10(w)])[0]/UNjt
    
    nu  = 0.5*(Ntot + Assm)
    anu = 0.5*(Ntot - Assm) 

    Jac = log(10.)*10**t    
    
    return np.array([nu, anu, Nv, Ns]) * Jac

#--------------------------------------------------------------------------------------------------------------------------------#
#                                               Integration of d^3N/dEdtdOmega                                                   #
#--------------------------------------------------------------------------------------------------------------------------------#

class Int_Kerr:
    '''
    Code for integration in time of particle emission from Kerr PBH
    '''

    def __init__(self, tau_s, Nscls):

        self.tau_s = tau_s  # Time in s before BH evaporation
        
        self.Nx = 10
        self.Ny = 10

        #*************************************************************************#
        #       Interpolating d^3 A/dwdtdOm and  d^3 N_{nu+nubar}/dwdtdOm         # 
        #*************************************************************************#

        print(colored("Interpolating d^3 A/dwdtdOm and  d^3 Nv,s/dwdtdOm...", 'blue'))

        start = time.time()

        Naf = 100
        Nwf = 100
        Nhf = 100
        
        af_arr  = np.linspace(0., 0.9999, Naf)
        wf_arr  = np.linspace(-3., log10(3.0), Nwf)
        thf_arr = np.linspace(-1, 1, Nhf)

        d3A_tab = np.loadtxt("./Data/d3A_dEdtdOm.txt")

        d3A_int = zeros(((Nhf,Naf,Nwf)))
        d3N_int = zeros(((Nhf,Naf,Nwf)))
        
        for i in range(Nhf):
            for j in range(Naf):
                for k in range(Nwf):
                    d3A_int[i,j,k] = d3A_tab[i*Naf*Nwf + j*Nwf + k, 0]
                    d3N_int[i,j,k] = d3A_tab[i*Naf*Nwf + j*Nwf + k, 1]


        self.d3A_ = RegularGridInterpolator((thf_arr, af_arr, wf_arr), d3A_int, bounds_error=False, fill_value = None)
        self.d3N_ = RegularGridInterpolator((thf_arr, af_arr, wf_arr), d3N_int, bounds_error=False, fill_value = None)

        Nab = 50
        Nwb = 50
        Nhb = 50
        
        ab_arr  = np.linspace(0., 0.9999, Nab)
        wb_arr  = np.linspace(-3., log10(3.0), Nwb)
        thb_arr = np.linspace(-1, 1, Nhb)

        d3Nv_tab = np.loadtxt("./Data/d3Nv_dEdtdOm.txt")
        d3Ns_tab = np.loadtxt("./Data/d3Ns_dEdtdOm.txt")

        d3Nv_int = zeros(((Nhb,Nab,Nwb)))
        d3Ns_int = zeros(((Nhb,Nab,Nwb)))

        for i in range(Nhb):
            for j in range(Nab):
                for k in range(Nwb):
                    d3Nv_int[i,j,k] = d3Nv_tab[i*Nab*Nwb + j*Nwb + k]
                    d3Ns_int[i,j,k] = d3Ns_tab[i*Nab*Nwb + j*Nwb + k]

        self.d3Nv_ = RegularGridInterpolator((thb_arr, ab_arr, wb_arr), d3Nv_int, bounds_error=False, fill_value = None)
        self.d3Ns_ = RegularGridInterpolator((thb_arr, ab_arr, wb_arr), d3Ns_int, bounds_error=False, fill_value = None)

        end = time.time()
            
        print(colored(f"Time is {end - start} s\n", 'magenta'))

        #**********************************************************************************#
        #       Finding the BH initial mass for varying the a* given an initial tau_s      # 
        #**********************************************************************************#

        print(colored("Finding the BH initial mass for varying the a* given an initial tau_s...", 'blue'))

        start = time.time()
        
        a_ar, Min_PBH = np.loadtxt("./Data/f_Min_tau="+str(np.round(self.tau_s,3))+".txt").T

        self.fMin_ = interp1d(a_ar, Min_PBH)
            
        end = time.time()
            
        print(colored(f"Time is {end - start} s\n", 'magenta'))
        
    #----------------------------------------------------------------------------------------------------------------------------------#
    #                                                       Main functions                                                           #
    #----------------------------------------------------------------------------------------------------------------------------------#
        
    def d2N_dEdOm(self, ain, p, cos_th):
        '''                                                                                                                             
        Hawking rate integrated in time, as function of angle and energy
        '''
        
        inTot = np.array([0., 0., 0., 0.]) 
        
        Mi  = 10.**self.fMin_(ain)
        asi = ain
        
        taut = -80.
        
        def PlanckMass(t, v, Mi):
            
            eps = 0.1
            
            if (eps*Mi > bh.MPL): Mst = eps*Mi
            else: Mst = bh.MPL   
            
            return v[0] - Mst
        
        while Mi >= 100.* bh.MPL: 
            
            MPL = lambda t, x:PlanckMass(t, x, Mi)
            MPL.terminal  = True 
            MPL.direction = -1. 
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauSM(t, y), t_span = [-80., 80.], y0 = [Mi, asi],
                                events=MPL, rtol=1.e-10, atol=1.e-15, dense_output=True)
            
            t_min = tau_sol.t[0] 
            t_max = tau_sol.t[-1] 
            
            if bh.GCF * (Mi/bh.GeV_in_g) * p <= 3.0:

                inTot += quad_vec(int_EO, t_min, t_max, args=(p, cos_th, tau_sol.sol, self.d3A_, self.d3N_, self.d3Nv_, self.d3Ns_),
                                  epsabs=1.e-10, epsrel=1.49e-10)[0]
                
            Mi   = tau_sol.y[0,-1]   
            asi  = tau_sol.y[1,-1]   
 
        return inTot

