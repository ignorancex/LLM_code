# -*- coding: utf-8 -*-
"""
Triplet P_1 series of Sr

2-channel MQDT model (K-matrix formulation)

Calculation of theoretical energies and channel fractions

"""

import numpy as np
import mqdtfit as mqdt

nchann = 2

Ilim = np.empty(nchann+1)
lquantumnumber = np.zeros(nchann+1)
Kmatrx0 = np.zeros((nchann+1,nchann+1))
Kmatrx1 = np.zeros(nchann+1)
flvarK0 = np.ones((nchann+1,nchann+1),dtype=bool)
flvarK1 = np.ones((nchann+1),dtype=bool)
option_Ecorrection = ""
Uialphabarmatrx = np.diag(np.ones(nchann+1))

Ilim[1], lquantumnumber[1] = 45932.2002, 1
Ilim[2], lquantumnumber[2] = 60628.26, 1   

Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:

tripletP1_states = [
    [ 33868.317 , 0.006 ],  # n =  6  Sansonetti2010
    [ 39426.442 , 0.008 ],  #      7  Sansonetti2010
    [ 41719.71  , 0.05  ],  #      8  Armstrong1979
    [ 42990.26  , 0.07  ],  #      9  Armstrong1979
    [ 43761.47  , 0.07  ],  #     10  Armstrong1979
    [ 44264.52  , 0.07  ],  #     11  Armstrong1979
    [ 44610.85  , 0.04  ],  #     12  Armstrong1979
 ## [ 44859.32  , 0.03  ],  #     13  (not included in the calculation)
    [ 45043.89  , 0.02  ],  #     14  Armstrong1979
    [ 45184.54  , 0.02  ]]  #     15  Armstrong1979

perturber = [ 
    [ 37302.731 , 0.006 ]]  # 4d5p triplet P_1 state (Sansonetti2010)

exp_data = tripletP1_states + perturber


Kmatrx0[1, 1] =  -4.19906665e-01
Kmatrx0[1, 2] =  -2.29230418e-01
Kmatrx0[2, 2] =  -3.52617910e-01
Kmatrx1[1]    =   1.08261507e+00
Kmatrx1[2]    =  -1.30477942e+00
option_Ecorrection = "E"

jindx = 2
kindx = 1

calculation_method = "Xi2min"

###########################################################################


mqdt.initialize_Kmatrix(nchann, Ilim, lquantumnumber, Is, Rtilda, exp_data,
                        flvarK0, flvarK1, option_Ecorrection,
                        Uialphabarmatrx, jindx, kindx, calculation_method)

mqdt.set_searchintervals("R")
params = mqdt.mqdtparams_Kmatrix(Kmatrx0,Kmatrx1)

mqdt.print_channelparams(file="tables_3P1.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3P1.txt")
mqdt.print_energies(params, exp_data, file="tables_3P1.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3P1.txt")
mqdt.plot_channelfractions(params, 33800, 45200, 1.0, exp_data, 2,
                           with_dbydE=True, file="channelfractions_3P1.pdf")
