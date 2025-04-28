# -*- coding: utf-8 -*-
"""
Singlet and triplet D_2 series of Sr

6-channel MQDT model (K-matrix formulation)

Calculation of theoretical energies and channel fractions

"""

import numpy as np
import mqdtfit as mqdt

nchann = 6

Ilim = np.empty(nchann+1)
lquantumnumber = np.zeros(nchann+1)
Kmatrx0 = np.zeros((nchann+1,nchann+1))
Kmatrx1 = np.zeros(nchann+1)
flvarK0 = np.ones((nchann+1,nchann+1),dtype=bool)
flvarK1 = np.ones((nchann+1),dtype=bool)
option_Ecorrection = ""

Ilim[1], lquantumnumber[1] = 45932.2002, 2
Ilim[2], lquantumnumber[2] = 45932.2002, 2
Ilim[3], lquantumnumber[3] = 60768.43, 0
Ilim[4], lquantumnumber[4] = 60488.09, 0
Ilim[5], lquantumnumber[5] = 70048.11, 1
Ilim[6], lquantumnumber[6] = 60628.26, 2


# U_i,alphabar matrix:
# i = 1: 5s_1/2 nd_5/2     alphabar = 1: 5snd singlet D_2
# i = 2: 5s_1/2 nd_3/2     alphabar = 2: 5snd triplet D_2
# i = 3: 4d_5/2 ns_1/2     alphabar = 3: 4dns singlet D_2
# i = 4: 4d_3/2 ns_1/2     alphabar = 4: 4dns triplet D_2
# i = 5: 5pnp singlet D_2  alphabar = 5: 5pnp singlet D_2
# i = 6: 4dnd triplet P_2  alphabar = 6: 4dnd triplet P_2
Uialphabarmatrx = np.zeros((nchann+1,nchann+1))
Uialphabarmatrx[1,1] =  np.sqrt(3./5.)
Uialphabarmatrx[1,2] = -np.sqrt(2./5.)
Uialphabarmatrx[2,1] =  np.sqrt(2./5.)
Uialphabarmatrx[2,2] =  np.sqrt(3./5.)
Uialphabarmatrx[3,3] =  np.sqrt(3./5.)
Uialphabarmatrx[3,4] =  np.sqrt(2./5.)
Uialphabarmatrx[4,3] = -np.sqrt(2./5.)
Uialphabarmatrx[4,4] =  np.sqrt(3./5.)
Uialphabarmatrx[5,5] =  1.
Uialphabarmatrx[6,6] =  1.

Is = 45932.2002
Rtilda = 109736.631

# Experimental energies considered in the present calculation and in
# the optimization of the MQDT parameters:

singletD2_states = [
    [ 43021.058   , 0.023 ], # n =  8  Sansonetti2010
    [ 43755.755   , 0.025 ], #      9  Sansonetti2010
    [ 44241.70    , 0.11  ], #     10  Escherick1977
    [ 44578.689   , 0.001 ], #     11  Beigang1982
    [ 44829.6648  , 0.001 ], #     12  Beigang1982
    [ 45012.0249  , 0.001 ], #     13  Beigang1982
    [ 45153.28988 , 0.001 ], #     14  Couturier2019, error set to 0.001
    [ 45263.63120 , 0.001 ], #     15  Couturier2019, error set to 0.001
    [ 45362.14128 , 0.001 ], #     16  Couturier2019, error set to 0.001
    [ 45433.2717  , 0.001 ], #     17  Beigang1982
    [ 45492.6101  , 0.001 ], #     18  Beigang1982
    [ 45542.2955  , 0.001 ], #     19  Beigang1982
    [ 45584.1831  , 0.001 ], #     20  Beigang1982
    [ 45619.7872  , 0.001 ], #     21  Beigang1982
    [ 45650.2617  , 0.001 ], #     22  Beigang1982
    [ 45676.5335  , 0.001 ], #     23  Beigang1982
    [ 45699.3308  , 0.001 ], #     24  Beigang1982
    [ 45719.2336  , 0.001 ], #     25  Beigang1982
    [ 45736.7165  , 0.001 ], #     26  Beigang1982
    [ 45752.22    , 0.06  ], #     27  Escherick1977
    [ 45765.8106  , 0.001 ], #     28  Beigang1982
    [ 45777.9907  , 0.001 ], #     29  Beigang1982
    [ 45788.8895  , 0.001 ]] #     30  Beigang1982

tripletD2_states = [
    [ 43070.268   , 0.009 ], # n =  8  Sansonetti2010
    [ 43804.890   , 0.025 ], #      9  Sansonetti2010
    [ 44287.05    , 0.20  ], #     10  Escherick1977
    [ 44620.08    , 0.16  ], #     11  Escherick1977
    [ 44860.06382 , 0.001 ], #     12  Couturier2019, error set to 0.001
    [ 45036.96042 , 0.001 ], #     13  Couturier2019, error set to 0.001
    [ 45171.49569 , 0.001 ], #     14  Couturier2019, error set to 0.001
    [ 45276.66050 , 0.001 ], #     15  Couturier2019, error set to 0.001
    [ 45350.52392 , 0.001 ], #     16  Couturier2019, error set to 0.001
    [ 45420.84812 , 0.001 ], #     17  Couturier2019, error set to 0.001
    [ 45479.87947 , 0.001 ], #     18  Couturier2019, error set to 0.001
    [ 45530.18514 , 0.001 ], #     19  Couturier2019, error set to 0.001
    [ 45573.19050 , 0.001 ], #     20  Couturier2019, error set to 0.001
    [ 45610.02969 , 0.001 ], #     21  Couturier2019, error set to 0.001
    [ 45641.69594 , 0.001 ], #     22  Couturier2019, error set to 0.001
    [ 45669.04105 , 0.001 ], #     23  Couturier2019, error set to 0.001
    [ 45692.77654 , 0.001 ], #     24  Couturier2019, error set to 0.001
    [ 45713.48769 , 0.001 ], #     25  Couturier2019, error set to 0.001
    [ 45731.65408 , 0.001 ], #     26  Couturier2019, error set to 0.001
    [ 45747.66799 , 0.001 ], #     27  Couturier2019, error set to 0.001
    [ 45761.85143 , 0.001 ], #     28  Couturier2019, error set to 0.001
    [ 45774.46993 , 0.001 ], #     29  Couturier2019, error set to 0.001
    [ 45785.74337 , 0.001 ]] #     30  Couturier2019, error set to 0.001

perturber = [
    [ 44729.56  , 0.11  ]] # 4  4d^2 triplet P_2 state (Esherick1977)

exp_data = singletD2_states + tripletD2_states + perturber

Kmatrx0[1, 1] =  -3.85388310e-01
Kmatrx0[1, 2] =   2.30810347e-01
Kmatrx0[1, 3] =  -2.99689799e-01
Kmatrx0[1, 4] =   6.24839129e-01
Kmatrx0[1, 5] =  -2.38162135e-01
Kmatrx0[1, 6] =  -8.94462406e-02
Kmatrx0[2, 2] =  -4.88187724e-01
Kmatrx0[2, 3] =  -6.41169781e-01
Kmatrx0[2, 4] =   8.10126226e-06
Kmatrx0[2, 5] =  -4.84958202e-01
Kmatrx0[2, 6] =   2.42734982e-03
Kmatrx0[3, 3] =   1.13622499e+00
Kmatrx0[3, 4] =   2.07880487e-01
Kmatrx0[3, 5] =   0.00000000e+00
Kmatrx0[3, 6] =   0.00000000e+00
Kmatrx0[4, 4] =   1.12383070e+00
Kmatrx0[4, 5] =   0.00000000e+00
Kmatrx0[4, 6] =   0.00000000e+00
Kmatrx0[5, 5] =   6.11787736e-01
Kmatrx0[5, 6] =   0.00000000e+00
Kmatrx0[6, 6] =   2.20539967e+00
Kmatrx1[1]    =  -1.77532611e+00
Kmatrx1[2]    =   2.05255442e+00
Kmatrx1[3]    =   4.73380410e+00
Kmatrx1[4]    =   3.98916238e+00
Kmatrx1[5]    =   5.29286844e+00
Kmatrx1[6]    =   6.07956188e+00

option_Ecorrection = "E"

flvarK0[3,5] = False
flvarK0[3,6] = False
flvarK0[4,5] = False
flvarK0[4,6] = False
flvarK0[5,6] = False

jindx = 3

kindx = 1

calculation_method = "Xi2min"

###########################################################################

mqdt.initialize_Kmatrix(nchann, Ilim, lquantumnumber, Is, Rtilda, exp_data,
                          flvarK0, flvarK1, option_Ecorrection,
                          Uialphabarmatrx, jindx, kindx, calculation_method)

mqdt.set_searchintervals("A", deltanuj = 0.000001)
params = mqdt.mqdtparams_Kmatrix(Kmatrx0,Kmatrx1)

mqdt.print_channelparams(printUialphabarmatrx=False, file="tables_1and3D2.txt")
mqdt.print_mqdtparams_Kmatrix(params,file="tables_1and3D2.txt")
mqdt.print_energies(params, exp_data, file="tables_1and3D2.txt")
mqdt.print_chi2(params, moreinfo=True, file="tables_1and3D2.txt")

mqdt.plot_channelfractions(params, 44000, 46000, 1.0, singletD2_states, 2, 1,
                           recouple=True, with_dbydE=True,
                           file="channelfractions_1_1and3D2.pdf")
mqdt.plot_channelfractions(params, 44000, 46000, 0.1, singletD2_states, 4, 3,
                           6, recouple=True, with_dbydE=True,
                           file="channelfractions_2_1and3D2.pdf")
mqdt.plot_channelfractions(params, 44000, 46000, 0.01, singletD2_states, 5,
                           recouple=True, with_dbydE=True,
                           file="channelfractions_3_1and3D2.pdf")
mqdt.plot_channelfractions(params, 44000, 46000, 1.0, tripletD2_states, 2, 1,
                           recouple=True, with_dbydE=True,
                           file="channelfractions_4_1and3D2.pdf")
mqdt.plot_channelfractions(params, 44000, 46000, 0.1, tripletD2_states, 4, 3,
                           6, recouple=True, with_dbydE=True,
                           file="channelfractions_5_1and3D2.pdf")
mqdt.plot_channelfractions(params, 44000, 46000, 0.01, tripletD2_states, 5,
                           recouple=True, with_dbydE=True,
                           file="channelfractions_6_1and3D2.pdf")
