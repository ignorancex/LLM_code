# -*- coding: utf-8 -*-
"""
Singlet F_3 series of Sr

2-channel MQDT model (K-matrix matrix formulation)

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

Ilim[1], lquantumnumber[1] = 45932.2002, 3
Ilim[2], lquantumnumber[2] = 60628.26, 1   

Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:

singletF_3states = [
    [ 39539.013 , 0.007 ],  # n =  4  Sansonetti2010
    [ 41519.04  , 0.19  ],  #      5  Sansonetti2010
    [ 42839.589 , 0.026 ],  #      6  Sansonetti2010
    [ 43656.219 , 0.028 ],  #      7  Sansonetti2010
    [ 44189.889 , 0.029 ],  #      8  Sansonetti2010
    [ 44556.48  , 0.03  ],  #      9  Sansonetti2010
    [ 44818.77  , 0.03  ],  #     10  Sansonetti2010
    [ 45012.82  , 0.03  ],  #     11  Sansonetti2010
    [ 45160.29  , 0.03  ],  #     12  Sansonetti2010
    [ 45274.97  , 0.03  ],  #     13  Sansonetti2010
    [ 45365.90  , 0.03  ],  #     14  Sansonetti2010
    [ 45439.16  , 0.03  ],  #     15  Sansonetti2010
    [ 45499.11  , 0.03  ],  #     16  Sansonetti2010
    [ 45548.76  , 0.03  ],  #     17  Sansonetti2010
    [ 45590.32  , 0.03  ],  #     18  Sansonetti2010
    [ 45625.48  , 0.03  ],  #     19  Sansonetti2010
    [ 45655.49  , 0.03  ],  #     20  Sansonetti2010
    [ 45681.31  , 0.15  ],  #     21  Rubbmark1978
    [ 45703.66  , 0.15  ],  #     22  Rubbmark1978
    [ 45723.16  , 0.15  ],  #     23  Rubbmark1978
    [ 45740.26  , 0.15  ],  #     24  Rubbmark1978
    [ 45755.38  , 0.15  ],  #     25  Rubbmark1978
    [ 45768.66  , 0.15  ],  #     26  Rubbmark1978
    [ 45780.80  , 0.15  ],  #     27  Rubbmark1978
    [ 45791.31  , 0.15  ],  #     28  Rubbmark1978
    [ 45801.03  , 0.15  ]]  #     29  Rubbmark1978

perturber = [ 
    [ 38007.742 , 0.016 ]] # 4d5p singlet F_3 state (Sansonetti2010)

exp_data = singletF_3states + perturber

Kmatrx0[1, 1] =   1.71163074e-01
Kmatrx0[1, 2] =   4.50595126e-01
Kmatrx0[2, 2] =  -6.97829350e-01
Kmatrx1[1]    =  -3.53036781e-01
Kmatrx1[2]    =  -1.31850526e+00

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

mqdt.print_channelparams(file="tables_1F3.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_1F3.txt")
mqdt.print_energies(params, exp_data, file="tables_1F3.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_1F3.txt")
mqdt.plot_channelfractions(params, 37900, 46000, 0.5, exp_data, 2,
                           with_dbydE=True, file="channelfractions_1F3.pdf")
