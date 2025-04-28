# -*- coding: utf-8 -*-
"""
Singlet P_1 series of Sr

2-channel MQDT model (K-matrix formulation)

Calculation of theoretical energies and channel fractions

Author: R M Potvliege (Physics Department, Durham University)

"""

import numpy as np
import mqdtfit as mqdt


# Number of channels:
nchann = 2


# Ionization limits and orbital angular momentum quantum number
# of each channel. Here indexes 1 and 2 refer respectively
# to the 5snp singlet P_1 channel and the 4dnp singlet P_1 channel.
Ilim = np.empty(nchann+1)
lquantumnumber = np.zeros(nchann+1)
Ilim[1], lquantumnumber[1] = 45932.2002, 1
Ilim[2], lquantumnumber[2] = 60628.26, 1   


# First ionization threshold and mass-corrected Rydberg constant for Sr88.
# Reference: L. Couturier et al, Phys. Rev. A 99, 022503 (2019)
Is = 45932.2002
Rtilda = 109736.631


# Recoupling transformation matrix. Here this is the unit matrix
# as the dissociation channels are already LS-coupled.
Uialphabarmatrx = np.diag(np.ones(nchann+1))


# Experimental energies and experimental errors.
# References: J. R. Rubbmark and S. A. Borgstrom, Phys. Scr. 27,
#                 172 (1983)
#             J. E. Sansonetti and G. Nave, J. Phys. Chem. Ref.
#                 Data 39, 033103 (2010)
singletP_1states = [
    [ 34098.404 , 0.006 ],  # n =  6  Sansonetti2010
    [ 38906.858 , 0.010 ],  #      7  Sansonetti2010
    [ 42462.136 , 0.014 ],  #      8  Sansonetti2010
    [ 43328.04  , 0.07  ],  #      9  Sansonetti2010
    [ 43936.32  , 0.15  ],  #     10  Sansonetti2010
    [ 44366.42  , 0.03  ],  #     11  Sansonetti2010
    [ 44675.737 , 0.029 ],  #     12  Sansonetti2010
    [ 44903.50  , 0.03  ],  #     13  Sansonetti2010
    [ 45075.29  , 0.03  ],  #     14  Sansonetti2010
    [ 45207.83  , 0.04  ],  #     15  Sansonetti2010
    [ 45311.99  , 0.04  ],  #     16  Sansonetti2010
    [ 45395.34  , 0.04  ],  #     17  Sansonetti2010
    [ 45463.02  , 0.05  ],  #     18  Sansonetti2010
    [ 45518.64  , 0.03  ],  #     19  Sansonetti2010
    [ 45565.00  , 0.03  ],  #     20  Sansonetti2010
    [ 45603.98  , 0.15  ],  #     21  Rubbmark1978
    [ 45637.10  , 0.15  ],  #     22  Rubbmark1978
    [ 45665.43  , 0.15  ],  #     23  Rubbmark1978
    [ 45689.86  , 0.15  ],  #     24  Rubbmark1978
    [ 45711.09  , 0.15  ],  #     25  Rubbmark1978
    [ 45729.67  , 0.15  ],  #     26  Rubbmark1978
    [ 45746.00  , 0.15  ],  #     27  Rubbmark1978
    [ 45760.44  , 0.15  ],  #     28  Rubbmark1978
    [ 45773.14  , 0.15  ]]  #     29  Rubbmark1978
perturber = [
    [ 41172.054 , 0.014 ]]  # 4d5p singlet P_1 state (Sansonetti2010)
exp_data = singletP_1states + perturber


# Initial K-matrix
Kmatrx0 = np.zeros((nchann+1,nchann+1))
Kmatrx1 = np.zeros(nchann+1)
Kmatrx0[1, 1] =   1.12e+01
Kmatrx0[1, 2] =   1.62e+01
Kmatrx0[2, 2] =   2.24e+01
Kmatrx1[1]    =  -9.10e-01
Kmatrx1[2]    =   4.27e+00


# Booleans variables indicating whether the corresponding
# MQDT parameters are variable (i.e., are fitting parameters,
# corresponding to a value of 1) or static (corresponding to
# a value of 0).
flvarK0 = np.ones((nchann+1,nchann+1),dtype=bool)
flvarK1 = np.ones((nchann+1),dtype=bool)


# Variable determining how the MQDT parameters vary with energy.
option_Ecorrection = "E"


# Indexes of the j-channel and of the k-channel.
jindx = 2
kindx = 1


# Variable determining how the energies are calculated.
calculation_method = "Xi2min"


###########################################################################

# MQDT calculations


# The calculation is first initialized by a call to initialize_Kmatrix.
# The search intervals are then defined, and the initial values of
# of the MQDT parameters (the elements of the K-matrix here) are stored
# in an 1D array called params.
mqdt.initialize_Kmatrix(nchann, Ilim, lquantumnumber, Is, Rtilda, exp_data,
                        flvarK0, flvarK1, option_Ecorrection, 
                        Uialphabarmatrx, jindx, kindx, calculation_method)
mqdt.set_searchintervals("A")
params = mqdt.mqdtparams_Kmatrix(Kmatrx0,Kmatrx1)


# Chi2 value measuring the agreement of the model with the data for the
# initial parameters
print("With the initial, unoptimized parameters:\n")
mqdt.print_chi2(params, moreinfo=True)


# Optimization of the model. The optimized parameters are stored in
# an 1D array called optparams. The initial parameters and the optimized
# are then printed out side by side.
print("\n"+"Optimization...\n")
optparams = mqdt.optimizeparams(params, nonzdelt=0.0001)
mqdt.print_mqdtparams_Kmatrix(params, optparams)


# Chi2 value for the optimized parameters, table of mixing coefficients and
# of channel fractions, and plot of channel fractions (the plot is saved in
# a filed called channelfractions.pdf, with channel 2 represented by open
# circles and channel 1 by filled cicles).
print("\n"+"With the optimized parameters:\n")
mqdt.print_chi2(optparams, moreinfo=True)
mqdt.list_Zcoeffsandchannelfractions(optparams, exp_data, with_dbydE=True)
mqdt.plot_channelfractions(optparams, 38000, 46000, 1.0, exp_data, 2, 1,
                           with_dbydE=True, file="channelfractions.pdf")

