import math
import numpy as np

import constants
import functions


# model parameters
xi = 2.0 # velocity of ejected CSM normalized by progenitor escape velocity 
x_ion_floor = 1e-5 # floor value of ionization

# parameters for compact object accretion 
MCO = 10.*constants.Msun # CO mass
p_BB = 0.5 # power-law index of ADIOS model (Blandford & Begelman 99)
r_disk_in = 6. * constants.G*MCO/constants.c**2 # inner edge of accretion disk


# RSG model
Mstar = 10.*constants.Msun
Rstar = 500.*constants.Rsun
kappa = 0.3
T_ion = 6e3

# HeHighMass model
#Mstar = 5.*constants.Msun
#Rstar = 5.*constants.Rsun
#kappa = 0.1
#T_ion = 1e4

# HeLowMass model
#Mstar = 3.*constants.Msun
#Rstar = 50.*constants.Rsun
#kappa = 0.1
#T_ion = 1e4

# CSM mass and binary separation
MCSM = 0.1 * constants.Msun
a_bin = 5. * Rstar

t_arr, L_arr, Lwind_arr, v_arr, x_arr, t_BH = functions.evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion, xi, x_ion_floor, p_BB, r_disk_in)
np.savetxt('prec_lightcurve.txt', np.c_[t_arr/86400., L_arr, v_arr/1e5], header='time [day], lum [erg/s], vel [km/s]', fmt='%.8g')

# output diagnostics
t_prec, L_prec = functions.calc_LC_Lt(t_arr, L_arr, t_BH)
v_prec = max(v_arr)/1e5
print("time=%.3e day, lum=%.3e erg/s, vel=%.3e km/s" % (t_prec, L_prec, v_prec))
