######################################################################################
#                                                                                    #
#                          Primordial Black Hole Evaporation.                        #
#                      Particle angular dependence from Kerr PBHs                    #
#   Code to use class for time integration of Hawking flux for a given time window   #
#                                                                                    #
#                            Author: Yuber F. Perez-Gonzalez                         #
#                               Based on: arXiv:2307.14408                           #
#                                                                                    #
######################################################################################


import sys
import numpy as np
from odeintw import odeintw
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta
from scipy.special import kn

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, real, imag
from numpy import sin, cos, tan
from numpy import absolute, angle, array

import BHProp as bh

from Int_full import Int_Kerr

from tqdm.contrib.concurrent import process_map

import time

# Computing time integral of d3N/dwdOmdt

start = time.time()

tau_s = 1. # Time in s before BH evaporation

d2N = Int_Kerr(tau_s)

p   = 10.**4 # Momentum in GeV
ain = 0.999  # Initial a_star
cos_th = 0.5 # Cosine @ angle wrt EPBH rotation axis

print(d2N.d2N_dEdOm(ain, p, cos_th))
