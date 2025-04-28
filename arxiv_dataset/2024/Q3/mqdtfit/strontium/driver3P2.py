# -*- coding: utf-8 -*-
"""
Triplet P_2 series of Sr

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

tripletP2_states = [
    [ 33973.065 , 0.004 ],  # n =  6  Sansonetti2010
    [ 39457.383 , 0.008 ],  #      7  Sansonetti2010
    [ 41735.98  , 0.05  ],  #      8  Armstrong1979
    [ 42999.79  , 0.07  ],  #      9  Armstrong1979
    [ 43767.58  , 0.07  ],  #     10  Armstrong1979
    [ 44268.72  , 0.07  ],  #     11  Armstrong1979
    [ 44613.78  , 0.04  ],  #     12  Armstrong1979
 ## [ 44861.46  , 0.03  ],  #     13 (not included in the calculation)
    [ 45045.54  , 0.02  ],  #     14  Armstrong1979
    [ 45185.79  , 0.02  ]]  #     15  Armstrong1979

perturber = [
    [ 37336.591 , 0.004 ]]  # 4d5p triplet P_2 state (Sansonetti2010)

exp_data = tripletP2_states + perturber

Kmatrx0[1, 1] =  -4.53113343e-01
Kmatrx0[1, 2] =  -2.17961927e-01
Kmatrx0[2, 2] =  -5.28510182e-01
Kmatrx1[1]    =   1.05086557e+00
Kmatrx1[2]    =  -4.05119877e-01

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

mqdt.print_channelparams(file="tables_3P2.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3P2.txt")
mqdt.print_energies(params, exp_data, file="tables_3P2.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3P2.txt")
mqdt.plot_channelfractions(params, 33800, 45200, 1.0, exp_data, 2,
                           with_dbydE=True, file="channelfractions_3P2.pdf")
