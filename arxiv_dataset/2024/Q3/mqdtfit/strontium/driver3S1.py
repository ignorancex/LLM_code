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
    [ 37424.675 , 0.005 ],  # n =  7  Sansonetti2010
    [ 40761.372 , 0.020 ],  #      8  Sansonetti2010
    [ 42451.16  , 0.35  ],  #      9  Sansonetti2010
    [ 43427.44  , 0.19  ],  #     10  Sansonetti2010
    [ 44043.35  , 0.20  ],  #     11  Sansonetti2010
    [ 44456.25  , 0.21  ],  #     12  Sansonetti2010
    [ 44747.65  , 0.15  ],  #     13  Beigang1982
    [ 44960.22  , 0.15  ],  #     14  Beigang1982
    [ 45120.41  , 0.15  ],  #     15  Beigang1982
    [ 45243.88  , 0.15  ],  #     16  Beigang1982
    [ 45341.28  , 0.15  ],  #     17  Beigang1982
    [ 45419.29  , 0.15  ],  #     18  Beigang1982
    [ 45482.89  , 0.01  ],  #     19  Kunze1993
    [ 45535.32  , 0.01  ],  #     20  Kunze1993
    [ 45579.08  , 0.01  ],  #     21  Kunze1993
    [ 45615.99  , 0.01  ],  #     22  Kunze1993
    [ 45647.40  , 0.01  ]]  #     23  Kunze1993

exp_data = tripletS1_states 

Kmatrx0[1, 1] =  -3.48196996e+01
Kmatrx0[1, 2] =  -1.54124102e+02
Kmatrx0[2, 2] =  -6.40168881e+02
Kmatrx1[1]    =  -1.50980912e+01
Kmatrx1[2]    =   3.31280553e+02

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

mqdt.print_channelparams(file="tables_3S1.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3S1.txt")
mqdt.print_energies(params, exp_data, file="tables_3S1.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3S1.txt")
mqdt.plot_channelfractions(params, 37000, 46000, 0.012, exp_data, 2,
                           with_dbydE=True, file="channelfractions_3S1.pdf")
