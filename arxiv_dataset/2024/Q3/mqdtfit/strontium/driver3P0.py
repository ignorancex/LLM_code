# -*- coding: utf-8 -*-
"""
Triplet P_0 series of Sr

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

tripletP0_states = [
    [ 33853.490 , 0.006 ],  # n =  6  Sansonetti2010
    [ 39411.669 , 0.008 ],  #      7  Sansonetti2010
    [ 41712.05  , 0.05  ],  #      8  Escherick1977b
    [ 42985.86  , 0.07  ],  #      9  Escherick1977b
    [ 43758.65  , 0.07  ],  #     10  Escherick1977b
    [ 44262.70  , 0.07  ],  #     11  Escherick1977b
    [ 44609.51  , 0.04  ],  #     12  Escherick1977b
 ## [ 44858.33  , 0.03  ],  #     13  (not included in the calculation)
    [ 45043.18  , 0.02  ],  #     14  Escherick1977b
    [ 45183.93  , 0.02  ]]  #     15  Escherick1977b

perturber = [
    [ 37292.074 , 0.007 ]] # 4d5p triplet P_0 state (Sansonetti2010)

exp_data = tripletP0_states + perturber

Kmatrx0[1, 1] =  -4.00956527e-01
Kmatrx0[1, 2] =  -2.22056880e-01
Kmatrx0[2, 2] =  -4.02518026e-01
Kmatrx1[1]    =   1.03992333e+00
Kmatrx1[2]    =  -1.02169594e+00

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

mqdt.print_channelparams(file="tables_3P0.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3P0.txt")
mqdt.print_energies(params, exp_data, file="tables_3P0.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3P0.txt")
mqdt.plot_channelfractions(params, 33800, 45200, 1.0, exp_data, 2,
                           with_dbydE=True, file="channelfractions_3P0.pdf")
