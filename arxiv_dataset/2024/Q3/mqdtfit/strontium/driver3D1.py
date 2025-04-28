# -*- coding: utf-8 -*-
"""
Triplet D_1 series of Sr

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

Ilim[1], lquantumnumber[1] = 45932.2002, 2 
Ilim[2], lquantumnumber[2] = 60628.26, 2 


Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:

tripletD1_states = [
    [ 44853.97363 , 1E-03], # n = 12  Couturier2019, error set to 0.001
    [ 45028.66517 , 1E-03], #     13  Couturier2019, error set to 0.001
    [ 45159.60495 , 1E-03], #     14  Couturier2019, error set to 0.001
    [ 45260.88015 , 1E-03], #     15  Couturier2019, error set to 0.001
 ## [ 45341.24788 , 1E-03], #     16  (not included in the calculation)
    [ 45414.53693 , 1E-03], #     17  Couturier2019, error set to 0.001
    [ 45475.23238 , 1E-03], #     18  Couturier2019, error set to 0.001
    [ 45526.94713 , 1E-03], #     19  Couturier2019, error set to 0.001
    [ 45570.91486 , 1E-03], #     20  Couturier2019, error set to 0.001
    [ 45608.38582 , 1E-03], #     21  Couturier2019, error set to 0.001
    [ 45640.47207 , 1E-03], #     22  Couturier2019, error set to 0.001
    [ 45668.10428 , 1E-03], #     23  Couturier2019, error set to 0.001
    [ 45692.04200 , 1E-03], #     24  Couturier2019, error set to 0.001
    [ 45712.89973 , 1E-03], #     25  Couturier2019, error set to 0.001
    [ 45731.17496 , 1E-03], #     26  Couturier2019, error set to 0.001
    [ 45747.27159 , 1E-03], #     27  Couturier2019, error set to 0.001
    [ 45761.51925 , 1E-03], #     28  Couturier2019, error set to 0.001
    [ 45774.18829 , 1E-03], #     29  Couturier2019, error set to 0.001
    [ 45785.50224 , 1E-03], #     30  Couturier2019, error set to 0.001
    [ 45795.64659 , 1E-03], #     31  Couturier2019, error set to 0.001
    [ 45804.77660 , 1E-03], #     32  Couturier2019, error set to 0.001
    [ 45813.02249 , 1E-03], #     33  Couturier2019, error set to 0.001
    [ 45820.49472 , 1E-03], #     34  Couturier2019, error set to 0.001
    [ 45827.28672 , 1E-03], #     35  Couturier2019, error set to 0.001
    [ 45833.47836 , 1E-03], #     36  Couturier2019, error set to 0.001
    [ 45839.13841 , 1E-03], #     37  Couturier2019, error set to 0.001
    [ 45844.32579 , 1E-03], #     38  Couturier2019, error set to 0.001
    [ 45849.09172 , 1E-03], #     39  Couturier2019, error set to 0.001
    [ 45853.48049 , 1E-03], #     40  Couturier2019, error set to 0.001
    [ 45857.53091 , 1E-03], #     41  Couturier2019, error set to 0.001
    [ 45861.27679 , 1E-03], #     42  Couturier2019, error set to 0.001
    [ 45864.74790 , 1E-03], #     43  Couturier2019, error set to 0.001
    [ 45867.97038 , 1E-03], #     44  Couturier2019, error set to 0.001
    [ 45870.96757 , 1E-03], #     45  Couturier2019, error set to 0.001
    [ 45873.75983 , 1E-03], #     46  Couturier2019, error set to 0.001
    [ 45876.36537 , 1E-03], #     47  Couturier2019, error set to 0.001
    [ 45878.80076 , 1E-03], #     48  Couturier2019, error set to 0.001
    [ 45881.08023 , 1E-03], #     49  Couturier2019, error set to 0.001
    [ 45883.21692 , 1E-03]] #     50  Couturier2019, error set to 0.001

exp_data = tripletD1_states

Kmatrx0[1, 1] =  -7.40335913e-01
Kmatrx0[1, 2] =   5.50457207e-01
Kmatrx0[2, 2] =   1.46140049e+00
Kmatrx1[1]    =    9.68468100e-01
Kmatrx1[2]    =    2.77735257e-01

option_Ecorrection = "E"

jindx = 2
kindx = 1

calculation_method = "Xi2min"

###########################################################################

mqdt.initialize_Kmatrix(nchann, Ilim, lquantumnumber, Is, Rtilda, exp_data,
                        flvarK0, flvarK1, option_Ecorrection,
                        Uialphabarmatrx, jindx, kindx, calculation_method)

mqdt.set_searchintervals("A", deltanuj = 0.00015)
params = mqdt.mqdtparams_Kmatrix(Kmatrx0,Kmatrx1)

mqdt.print_channelparams(file="tables_3D1.txt")
mqdt.print_mqdtparams_Kmatrix(params, file="tables_3D1.txt")
mqdt.print_energies(params, exp_data, file="tables_3D1.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_3D1.txt")
mqdt.plot_channelfractions(params, 44800, 45900, 0.20, exp_data, 2,
                           with_dbydE=True, file="channelfractions_3D1.pdf")
