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


# CSM mass and binary separation grid
MCSM_arr = np.logspace(-1, 0.5, 3) * constants.Msun
a_bin_arr = np.logspace(math.log10(3.), math.log10(30.), 3) * Rstar 

######################################## main ##################################
MCSM_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
abin_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
t_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
L_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))
v_prec = np.zeros((len(MCSM_arr),len(a_bin_arr)))

for i,MCSM in enumerate(MCSM_arr):
	for j,a_bin in enumerate(a_bin_arr):
		MCSM_prec[i,j] = MCSM / constants.Msun
		abin_prec[i,j] = a_bin / Rstar
		t_arr, L_arr, Lwind_arr, v_arr, x_arr, t_BH = functions.evolve_CSM(Mstar, MCO, Rstar, kappa, a_bin, MCSM, T_ion, xi, x_ion_floor, p_BB, r_disk_in)
		t_prec[i,j], L_prec[i,j] = functions.calc_LC_Lt(t_arr, L_arr, t_BH)
		v_prec[i,j] = max(v_arr)/1e5
		print("time=%.3e day, lum=%.3e erg/s, vel=%.3e km/s" % (t_prec[i,j], L_prec[i,j], v_prec[i,j])) 

np.savetxt('prec_grid.txt', np.transpose([MCSM_prec.flatten(), abin_prec.flatten(), t_prec.flatten(), L_prec.flatten(), v_prec.flatten()]), header='MCSM [Msun], abin/R*, time [day], lum [erg/s], vel [km/s]', fmt='%.8g')
