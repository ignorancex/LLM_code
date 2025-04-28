# -*- coding: utf-8 -*-
"""
Triplet D_3 series of Sr

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
Uialphabarmatrx = np.diag(np.ones(nchann+1))

Ilim[1], lquantumnumber[1] = 45932.2002, 2
Ilim[2], lquantumnumber[2] = 60628.26, 0   
Ilim[3], lquantumnumber[3] = 60628.26, 2   

Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:

tripletD3_states = [
    [ 39703.109 , 0.009 ], # n =  6  Sansonetti2010
    [ 41874.859 , 0.01  ], #      7  Sansonetti2010
    [ 43074.728 , 0.01  ], #      8  Sansonetti2010
    [ 44865.22  , 0.15  ], #     12  Beigang1983
    [ 45043.79  , 0.15  ], #     13  Beigang1983
    [ 45180.44  , 0.15  ], #     14  Beigang1983
    [ 45286.53  , 0.15  ], #     15  Beigang1983
    [ 45370.76  , 0.15  ], #     16  Beigang1983
    [ 45439.08  , 0.15  ], #     17  Beigang1983
    [ 45495.02  , 0.15  ], #     18  Beigang1983
    [ 45542.23  , 0.15  ], #     19  Beigang1983
    [ 45582.38  , 0.15  ], #     20  Beigang1983
    [ 45616.80  , 0.15  ], #     21  Beigang1983
 ## [ 45647.54  , 0.15  ], #     22  (not included in the calculation)
    [ 45673.10  , 0.15  ], #     23  Beigang1983
    [ 45695.94  , 0.15  ], #     24  Beigang1983
    [ 45715.80  , 0.15  ], #     25  Beigang1983
    [ 45733.69  , 0.15  ], #     26  Beigang1983
    [ 45749.16  , 0.15  ], #     27  Beigang1983
    [ 45763.12  , 0.15  ], #     28  Beigang1983
    [ 45775.60  , 0.15  ]] #     29  Beigang1983

exp_data = tripletD3_states

Kmatrx0[1, 1] =  -7.79385678e-01
Kmatrx0[1, 2] =   4.36019812e-01
Kmatrx0[1, 3] =   2.22978797e-01
Kmatrx0[2, 2] =   1.21231421e+00
Kmatrx0[2, 3] =  -1.68322545e-04
Kmatrx0[3, 3] =  -2.23826516e-01
Kmatrx1[1]    =    1.07199739e+00
Kmatrx1[2]    =    8.51416131e+00
Kmatrx1[3]    =    5.54442589e+00

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

mqdt.print_channelparams(file="tables_3D3.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3D3.txt")
mqdt.print_energies(params, exp_data, file="tables_3D3.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3D3.txt")
mqdt.plot_channelfractions(params, 44800, 45800, 0.10, exp_data, 2, 3,
                           with_dbydE=True, file="channelfractions_3D3.pdf")
