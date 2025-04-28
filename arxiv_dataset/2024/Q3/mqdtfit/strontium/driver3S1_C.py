# -*- coding: utf-8 -*-
"""
Triplet S_1 series of Sr

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

Ilim[1], lquantumnumber[1] = 45932.2002, 0
Ilim[2], lquantumnumber[2] = 70048.11, 1   

Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:

tripletS1_states = [
    [ 37424.675    , 0.005  ],  # n =  7  Sansonetti2010
    [ 40761.372    , 0.020  ],  #      8  Sansonetti2010
    [ 42451.16     , 0.35   ],  #      9  Sansonetti2010
    [ 43427.44     , 0.19   ],  #     10  Sansonetti2010
    [ 44043.35     , 0.20   ],  #     11  Sansonetti2010
    [ 44456.25     , 0.21   ],  #     12  Sansonetti2010
    [ 44747.64060  , 0.001  ],  #     13  Couturier2019, error set to 0.001
    [ 44960.24136  , 0.001  ],  #     14  Couturier2019, error set to 0.001
    [ 45120.34233  , 0.001  ],  #     15  Couturier2019, error set to 0.001
    [ 45243.89988  , 0.001  ],  #     16  Couturier2019, error set to 0.001
    [ 45341.24791  , 0.001  ],  #     17  Couturier2019, error set to 0.001
    [ 45419.30972  , 0.001  ],  #     18  Couturier2019, error set to 0.001
    [ 45482.86375  , 0.001  ],  #     19  Couturier2019, error set to 0.001
    [ 45535.29540  , 0.001  ],  #     20  Couturier2019, error set to 0.001
    [ 45579.05661  , 0.001  ],  #     21  Couturier2019, error set to 0.001
    [ 45615.95924  , 0.001  ],  #     22  Couturier2019, error set to 0.001
    [ 45647.36527  , 0.001  ]]  #     23  Couturier2019, error set to 0.001

exp_data = tripletS1_states

Kmatrx0[1, 1] =  -1.03924403e+02
Kmatrx0[1, 2] =  -1.33451654e+02
Kmatrx0[2, 2] =  -1.68045201e+02
Kmatrx1[1]    =  -2.76691239e+01
Kmatrx1[2]    =   5.51718438e+01

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

mqdt.print_channelparams(file="tables_3S1_C.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3S1_C.txt")
mqdt.print_energies(params, exp_data, file="tables_3S1_C.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3S1_C.txt")
mqdt.plot_channelfractions(params, 37000, 46000, 0.012, exp_data, 2,
                           with_dbydE=True, file="channelfractions_3S1_C.pdf")
