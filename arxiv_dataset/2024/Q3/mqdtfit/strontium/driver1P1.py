# -*- coding: utf-8 -*-
"""
Singlet P_1 series of Sr

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

Kmatrx0[1, 1] =   1.11680885e+01
Kmatrx0[1, 2] =   1.61693288e+01
Kmatrx0[2, 2] =   2.23961659e+01
Kmatrx1[1]    =  -9.09786173e-01
Kmatrx1[2]    =   4.27262604e+00

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

mqdt.print_channelparams(file="tables_1P1.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_1P1.txt")
mqdt.print_energies(params, exp_data, file="tables_1P1.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_1P1.txt")
mqdt.plot_channelfractions(params, 38000, 46000, 0.2, exp_data, 2,
                           with_dbydE=True, file="channelfractions_1P1.pdf")
