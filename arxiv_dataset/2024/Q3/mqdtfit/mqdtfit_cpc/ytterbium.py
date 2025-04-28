# -*- coding: utf-8 -*-
"""
Singlet and triplet D_2 series of Yb

5-channel MQDT model (eigenchannel formulation)

Calculation of theoretical energies

All parameters of the models (choice of channels, ionization limits,
mass-corrected Rydberg constant, eigen quantum defects, choice of rotations
and rotation angles) and all experimental energies and errors are from
H. Lehec et al, Phys. Rev. A 98, 062506 (2018).

Author: R M Potvliege (Physics Department, Durham University)

"""

import numpy as np
import mqdtfit as mqdt


# Number of channels:
nchann = 5


# Ionization limits and orbital angular momentum quantum number
# of each channel. The channel indexes 1, 2, 3, 4 and 5 refer respectively
# to the 6snd singlet D_2, 6snd triplet D_2, 4f^{15}5d6snpA, 4f^{15}5d6snpB 
# and 6snp singlet D_2 channels, in terms of LS-coupled channels. Indexes
# 1 and 2 also refer to the 6s_{1/2}nd_{5/2} and 6s{_1/2}nd_{3/2} channels,
# respectively, in terms of jj-coupled channels.
Ilim = np.empty(nchann+1)
lquantumnumber = np.zeros(nchann+1)
Ilim[1], lquantumnumber[1] = 50443.0704, 2
Ilim[2], lquantumnumber[2] = 50443.0704, 2
Ilim[3], lquantumnumber[3] = 83967.7, 1
Ilim[4], lquantumnumber[4] = 83967.7, 1
Ilim[5], lquantumnumber[5] = 79725.35, 1


# First ionization limit and mass-corrected Rydberg constant for Yb174.
Is = 50443.07042
Rtilda = 109736.96959


#Recoupling transformation matrix.
Uialphabarmatrx = np.diag(np.ones(nchann+1))
Uialphabarmatrx[1,1] =  np.sqrt(3./5.)
Uialphabarmatrx[1,2] =  np.sqrt(2./5.)
Uialphabarmatrx[2,1] = -np.sqrt(2./5.)
Uialphabarmatrx[2,2] =  np.sqrt(3./5.)


# Experimental energies and experimental errors are read from the file
# ytterbiumD2data.dat and stored in the array exp_data in the format
# expected by mqdt.initialize_eigenchannel.
with open("ytterbiumD2data.dat","r") as f:
    exp_data = []
    for line in f:
        columns = line.split()
        exp_en  = float(columns[1])     # Second column: experimental energies
        exp_err = float(columns[2])     # Third column: experimental errors 
        exp_data.append([exp_en,exp_err])


# Eigen quantum defects
mu0_arr = np.zeros(nchann+1)
mu1_arr = np.zeros(nchann+1)
mu0_arr[1] =  0.7295231 
mu0_arr[2] =  0.7522912
mu0_arr[3] =  0.1961204
mu0_arr[4] =  0.2336927
mu0_arr[5] =  0.1528761
mu1_arr[1] = -0.022897461 
mu1_arr[2] =  0.09094972
mu1_arr[3] =  0.0
mu1_arr[4] =  0.0
mu1_arr[5] =  0.0


# Booleans variables indicating whether the corresponding
# mu^{(1)}_\alpha parameters are variable (i.e., are fitting parameters,
# corresponding to a value of 1 of True) or static (corresponding to
# a value of 0 or False).
flvarmu1 = np.ones(nchann+1,dtype=bool)
flvarmu1[3] = False
flvarmu1[4] = False
flvarmu1[5] = False


# Number of rotations in the model and parameters of these rotations
nrotations = 5
rotations = [None]*(nchann+1)
theta_arr = np.empty(nchann+1)
rotations[1], theta_arr[1] = (1,3),  0.005224693
rotations[2], theta_arr[2] = (1,4),  0.03972731
rotations[3], theta_arr[3] = (2,4), -0.007083118
rotations[4], theta_arr[4] = (1,5),  0.1049613
rotations[5], theta_arr[5] = (2,5),  0.07219257


# Variable determining how the MQDT parameters vary with energy.
option_Ecorrection = "A"


# Indexes of the j-channel and of the k-channel.
jindx = 3
kindx = 1


# Variable determining how the energies are calculated.
calculation_method = "zerodeterm"


###########################################################################

# MQDT calculations


# The calculation is first initialized by a call to initialise_eigenchannel.
# The search intervals are then defined, and the MQDT parameters (the eigen
# quantum defects and rotation angles here) are stored in an 1D array called
# params.
mqdt.initialize_eigenchannel(nchann, Ilim, lquantumnumber, Is, Rtilda,
                             exp_data, flvarmu1, option_Ecorrection,
                             nrotations, rotations, Uialphabarmatrx, jindx,
                             kindx, calculation_method)
mqdt.set_searchintervals("R")
params = mqdt.mqdtparams_eigenchannel(mu0_arr, mu1_arr, theta_arr)


# Energies obtained with these MQDT parameters and chi2 value measuring the
# agreement of the model with the data for this set of parameters.
mqdt.print_energies(params, exp_data)
mqdt.print_chi2(params, moreinfo=True)


# The index of the j-channel is changed from its previous value to 5, which
# is done by re-initialising the calculation with a new value of jindx, and
# a Lu-Fano plot of nu_k vs nu_j is produced (the plot is saved in a file
# called LuFanoplot.pdf). The k-channel remains channel 1. Since two channels
# have the same ionisation limit as channel 1, the theoretical curves are 
# drawn as sequences of dots rather than solid curves and two values of nu_k
# are indicated for each value of nu_j.
jindx = 5
mqdt.initialize_eigenchannel(nchann, Ilim, lquantumnumber, Is, Rtilda,
                             exp_data, flvarmu1, option_Ecorrection,
                             nrotations, rotations, Uialphabarmatrx, jindx,
                             kindx, calculation_method)
mqdt.LuFano_plot(params, 1.80, 1.935, 0.00005, 0., 1., file="LuFanoplot.pdf")
