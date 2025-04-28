####################################
#  Physical and other constants    #
####################################

import numpy as np

# Units conversions
eV_per_kg = 1.782622e-36
inv_eVs = 6.582119e-16
mP_in_GeV = 1.220890e19 # GeV
mP_in_eV = 1e9*mP_in_GeV # eV
mPred_in_GeV = mP_in_GeV/np.sqrt(8.0*np.pi) # GeV
mPred_in_eV = 1e9*mPred_in_GeV # eV
GNetwon_in_eV = 1/mP_in_eV**2 # eV^-2

# Astro constants
Msol_in_kg = 1.99841e30 # kg
Msol_in_eV = Msol_in_kg/eV_per_kg # eV
mP2_in_eVMsol = mP_in_eV*(mP_in_eV/Msol_in_eV) # eV Msol
GNewton = 1/mP2_in_eVMsol # eV^-1 Msol^-1
tHubble_in_yr = 1.45e10 # yr
tSalpeter_in_yr = 4.5e7 # yr
tSR_in_yr = tSalpeter_in_yr # yr
yr_in_s = 365.25*24*60*60 # s
inv_eVyr = inv_eVs/yr_in_s
inv_tSalpeter = inv_eVyr/tSalpeter_in_yr # eV
inv_tSR = inv_eVyr/tSR_in_yr # eV

# Bosenova constants
# Empircal numerical factor from simulations; see https://arxiv.org/abs/1411.2263
c0_n_bose = 5.0
