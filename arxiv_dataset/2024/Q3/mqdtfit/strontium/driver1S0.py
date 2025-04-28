# -*- coding: utf-8 -*-
"""
Singlet S_0 series of Sr

3-channel MQDT model (K-matrix formulation)

Calculation of theoretical energies and channel fractions

"""

import numpy as np
import mqdtfit as mqdt

nchann = 3

Ilim = np.empty(nchann+1)
lquantumnumber = np.zeros(nchann+1)
Kmatrx0 = np.zeros((nchann+1,nchann+1))
Kmatrx1 = np.zeros(nchann+1)
flvarK0 = np.ones((nchann+1,nchann+1),dtype=bool)
flvarK1 = np.ones((nchann+1),dtype=bool)
option_Ecorrection = ""

Ilim[1], lquantumnumber[1] = 45932.2002, 0
Ilim[2], lquantumnumber[2] = 60768.43, 2
Ilim[3], lquantumnumber[3] = 60488.09, 2

# U_i,alphabar matrix:
# i = 1: 5s_1/2 ns_1/2     alphabar = 1: 5sns singlet S_0
# i = 2: 4d_5/2 nd_5/2     alphabar = 2: 4dnd singlet S_0
# i = 3: 4d_3/2 nd_3/2     alphabar = 3: 4dnd triplet P_0 even
Uialphabarmatrx = np.zeros((nchann+1,nchann+1))
Uialphabarmatrx[1,1] =  1.
Uialphabarmatrx[2,2] =  np.sqrt(3./5.)
Uialphabarmatrx[2,3] = -np.sqrt(2./5.)
Uialphabarmatrx[3,2] =  np.sqrt(2./5.)
Uialphabarmatrx[3,3] =  np.sqrt(3./5.)

Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:
    
singletS0_states = [
    [ 38444.013  , 0.007  ], # n=7  Sansonetti2010
    [ 41052.324  , 0.019  ], #   8  Sansonetti2010
    [ 42596.572  , 0.022  ], #   9  Sansonetti2010
    [ 43512.1658 , 0.0010 ], #  10  Beigang1982
    [ 44097.1224 , 0.0010 ], #  11  Beigang1982
    [ 44492.8348 , 0.0010 ], #  12  Beigang1982
    [ 44773.6707 , 0.0010 ], #  13  Beigang1982
    [ 44979.454  , 0.0010 ], #  14  Beigang1982
    [ 45134.9242 , 0.0010 ], #  15  Beigang1982
    [ 45255.2295 , 0.0010 ], #  16  Beigang1982
    [ 45350.2296 , 0.0010 ], #  17  Beigang1982
    [ 45426.5505 , 0.0010 ], #  18  Beigang1982
    [ 45488.7868 , 0.0010 ], #  19  Beigang1982
    [ 45540.2024 , 0.0010 ], #  20  Beigang1982
    [ 45583.1688 , 0.0010 ], #  21  Beigang1982
    [ 45619.4391 , 0.0010 ], #  22  Beigang1982
    [ 45650.3365 , 0.0010 ], #  23  Beigang1982
    [ 45676.8704 , 0.0010 ], #  24  Beigang1982
    [ 45699.82   , 0.15   ], #  25  Rubbmark1978
    [ 45719.8235 , 0.0010 ], #  26  Beigang1982
    [ 45737.35   , 0.15   ], #  27  Rubbmark1978
    [ 45752.7832 , 0.0010 ], #  28  Beigang1982
    [ 45766.48   , 0.15   ], #  29  Rubbmark1978
    [ 45778.6257 , 0.0010 ]] #  30  Beigang1982

perturber = [
    [ 44525.838  , 0.010  ]] # 4d2 triplet P0 even state (Sansonetti2010)
        
exp_data = singletS0_states + perturber

Kmatrx0[1, 1] =   1.05126086e+00
Kmatrx0[1, 2] =   3.75986417e-01
Kmatrx0[1, 3] =  -2.36548468e-02
Kmatrx0[2, 2] =  -6.40092484e-01
Kmatrx0[2, 3] =  -2.06382548e-04
Kmatrx0[3, 3] =   3.00908733e+00
Kmatrx1[1]    =   8.76391110e-01
Kmatrx1[2]    =   4.04258438e-01
Kmatrx1[3]    =  -1.72263065e+01

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

mqdt.print_channelparams(printUialphabarmatrx=False, file="tables_1S0.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_1S0.txt")
mqdt.print_energies(params, exp_data, file="tables_1S0.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_1S0.txt")
mqdt.plot_channelfractions(params, 37000, 46000, 1.00, exp_data, 2, 3,
                           recouple = True, with_dbydE=True,
                           file="channelfractions_1S0_LScoupling.pdf")
mqdt.plot_channelfractions(params, 37000, 46000, 1.00, exp_data, 3, 2,
                           recouple = False, with_dbydE=True,
                           file="channelfractions_1S0_jjcoupling.pdf")
