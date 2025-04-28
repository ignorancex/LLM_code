# -*- coding: utf-8 -*-
"""
A suite of programs for empirical MQDT calculations

Author: R M Potvliege (Physics Department, Durham University)

June - September 2023

The following functions are defined in this module:

    initialize_Kmatrix
    initialize_eigenchannel
    sort_exp_data
    set_searchintervals
    set_searchintervals_variable
    set_searchintervals_fixed
    reset_calculation_method
    mqdtparams_Kmatrix
    mqdtparams_eigenchannel
    Umatrxfun
    eigendeterm
    Kmatrixdeterm
    Ffun
    Gkfun
    Gkfun_2chann
    Gkfun_3chann
    Gkfun_mltple
    Xi2
    chi2fun
    th_energies_fun
    prep_initial_simplex
    optimizeparams
    mqdtmatrx_Kmatrix_s
    Kmatrix_fromUandmu
    Kmatrix_Kmatrixformalism
    mqdtmatrices_and_mu1_eigenchannel_s
    chi2andredchi2fun
    print_chi2
    print_energies
    print_channelparams
    print_mqdtparams
    print_mqdtparams_Kmatrix
    print_mqdtparams_eigenchannel
    LuFano_plot
    mixing_coeffs
    mixing_coeffs_eigenchannel
    dKbydE
    mixing_coeffs_Kmatrix
    plot_channelfractions
    list_Zcoeffsandchannelfractions

These programs are distributed under the BSD 3-Clause License.

Copyright 2023 R M Potvliege

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin, minimize_scalar, brentq

# Global variables:
calculation_method = ""
displaychi2 = False
exp_energies = [0]
exp_errors = [0]
flvarK0 = [0]
flvarK1 = [0]
flvarmu1 = [0]
Ilim = [0]
Is = 0.0
jindx = 0
kindx = 0
klist = []
lquantumnumber = [0]
nchann = 0
nenergies = 0
nkchannels = 0
nrotations = 0
ntheta_start = 0
nujleft =[0]
nujright =[0]
nvarparams = 0
option_description = ""
option_Ecorrection = ""
option_searchintervals = ""
rotations = []
Rtilda = 0.0
statparams = [0]
Uialphabarmatrx_s = []
Umatrx = []
varparams = [0]

###############################################################################

def initialize_Kmatrix(nchann_in, Ilim_in, lquantumnumber_in, Is_in, 
                         Rtilda_in, exp_data_in, flvarK0_in, flvarK1_in,
                         option_Ecorrection_in, Uialphabarmatrx_in, jindx_in,
                         kindx_in, calculation_method_in):

    """
    Initializes an MQDT calculation in the K-matrix formulation.

    Args:
        nchann_in (int): The number of channels.
        Ilim_in (1D array[float]): The ionization thresholds of the respective
            channels. At entrey, Ilim_in[i] should be the ionization threshold
            of channel i, i = 1,...,nchann_in, in inverse centimeters.
        lquantumnumber_in (1D array[int]): The orbital angular momentum quantum
            numbers of the respective channels. At entry, lquantumnumber_in[i]
            should be l value of channel i, i = 1,...,nchann_in
        Is_in (float): The ionization threshold, in inverse centimeters.
        Rtilda_in (float): The Rydberg constant for the species considered,
            in inverse centimeters.
        exp_data_in (1D array([float], [float])): The experimental data to be
            used, by default, in the course of the calculations. At entry,
            exp_data_in[i][0] and exp_data_in[i][1], i = 0,..., should be,
            respectively, the i-th measured energy level and the corresponding
            experimental error, in inverse centimeters.
        flvarK0_in (2D array[bool]): Flags indicating whether the respective
            elements of the K^(0) matrix are variable fitting parameters rather
            than set once and for all. At entry, flvarK0_in[i,j] should be True 
            if the (i,j) element of the K^(0) matrix (i, j = 1,...,nchann) is
            variable.
        flvarK1_in (1D array[bool]): Flags indicating whether the respective
            elements of the K^(1) matrix are variable fitting parameters rather
            than set once and for all. At entry, flvarK1_in[i] should be True 
            if the (i,i) element of the K^(1) matrix (i = 1,...,nchann) is
            variable. 
        option_Ecorrection_in (str): A character string indicating the way
            in which the energy correction should be calculated. At entry,
            option_Ecorrection_in should be "A", "E" or "" (equivalent to
            "A"), as is explained in the code.
        Uialphabarmatrx_in (2D array[float]): The U_{i,alphabar} matrix.
            At entry, Uialphabarmatrx_in[i,alphabar] should be the 
            (i,alphabar) element of this matrix (i, alphabar = 1,...,nchann).
        jindx_in (int): The index of the j-channel (1 <= jindx_in <= nchann).
        kindx_in (int): The index of the k-channel (1 <= kindx_in <= nchann).
        calculation_method_in (str): A character string indicating whether
            the predicted energies should be calculated by minimizing the
            Xi^2 function (str = "Xi2min") or by finding the zeros of the
            relevant determinant (str = "zerodeterm").

    This function resets the global variables calculation_method, exp_energies,
    exp_errors, flvarK0, flvarK1, Ilim, Is, jindx, kindx, klist,
    lquantumnumber, nchann, nenergies, nkchannels, nvarparams,
    option_description, option_Ecorrection, Rtilda and Uialphabarmatrx_s.

    Returns:
        None
    """

    global nchann, Ilim, lquantumnumber, Is, Rtilda, flvarK0, flvarK1, \
           Uialphabarmatrx_s, jindx, kindx, calculation_method, nenergies, \
           exp_energies, exp_errors, nvarparams, option_Ecorrection, \
           option_description, klist, nkchannels
    nchann = nchann_in
    Ilim = Ilim_in
    lquantumnumber = lquantumnumber_in
    Is = Is_in
    Rtilda = Rtilda_in
    flvarK0 = flvarK0_in
    flvarK1 = flvarK1_in
    jindx = jindx_in
    kindx = kindx_in
    calculation_method = calculation_method_in
#
#  Determine whether the mu^(1) terms should be multiplied by 1/nu_s^2 ("A"), 
#  where nu_s is the effective principal quantum number measured with respect
#  to the ionization threshold Is, or by (Rtilda/Is)/nu_j^2 ("E"). The default
#  is "A".
    if option_Ecorrection_in == "" or option_Ecorrection_in == "A":
        option_Ecorrection = "A"
    elif option_Ecorrection_in == "E":
        option_Ecorrection = "E"
    else:
        print("Illegal value of option_Ecorrection")
        sys.exit()
#
#  Store the Uialphabar matrix with the indices shifted by one unit
    Uialphabarmatrx_s = np.zeros((nchann, nchann))
    for i in range(1, nchann+1):
        for j in range(1, nchann+1):
            Uialphabarmatrx_s[i-1, j-1] = Uialphabarmatrx_in[i, j]
#
# Experimental energies
    nenergies = len(exp_data_in)
    exp_energies = np.empty((nenergies))
    exp_errors = np.empty((nenergies))
    for i in range(0, nenergies):
        exp_energies[i] = exp_data_in[i][0]
        exp_errors[i] = exp_data_in[i][1]
#
# Number of non-static MQDT parameters
    nvarparams = 0
    for i in range (1, nchann+1):
        for j in range (i, nchann+1):
            if flvarK0[i, j] == True: nvarparams = nvarparams + 1
        if flvarK1[i] == True: nvarparams = nvarparams + 1
#
# Check on the suitability of the choice of jindx and kindx
    if Ilim[jindx] == Ilim[kindx]:
        print("Taking jindx and kindx such that Ilim[jindx] = Ilim[kindx]")
        print("is unsafe.")
        sys.exit("Unsuitable choice of jindx and kindx")
#
# List of the dissociation channels degenerate with channel k:
    klist = []
    nkchannels = 0
    for i in range(1, nchann+1):
        if Ilim[i] == Ilim[kindx]:
            klist.append(i)
            nkchannels = nkchannels + 1
#
# Set the parameter option_description to "K", which signals 
# that the calculation is based on the K-matrix formalism.
    option_description = "K"

###############################################################################

def initialize_eigenchannel(nchann_in, Ilim_in, lquantumnumber_in, Is_in, 
                            Rtilda_in, exp_data_in, flvarmu1_in, 
                            option_Ecorrection_in, nrotations_in, rotations_in, 
                            Uialphabarmatrx_in, jindx_in, kindx_in,
                            calculation_method_in):

    """
    Initializes an MQDT calculation in the eigenchannel formulation.

    Args:
        nchann_in (int): The number of channels.
        Ilim_in (1D array[float]): The ionization thresholds of the respective
            channels. At entrey, Ilim_in[i] should be the ionization threshold
            of channel i, i = 1,...,nchann_in, in inverse centimeters.
        lquantumnumber_in (1D array[int]): The orbital angular momentum quantum
            numbers of the respective channels. At entry, lquantumnumber_in[i]
            should be l value of channel i, i = 1,...,nchann_in
        Is_in (float): The ionization threshold, in inverse centimeters.
        Rtilda_in (float): The Rydberg constant for the species considered,
            in inverse centimeters.
        exp_data_in (1D array[[float, float]]): The experimental data to be
            used, by default, in the course of the calculations. At entry,
            exp_data_in[i][0] and exp_data_Yin[i][1], i = 0,..., should be,
            respectively, the i-th measured energy level and the corresponding
            experimental error, in inverse centimeters.
        flvarmu1_in (1D array[bool]): Flags indicating whether the respective
            values of mu^(1) are variable fitting parameters rather than
            set once and for all. At entry, flvarmu1_in[i] should be True 
            if mu^(1)_i (i = 1,...,nchann) is variable. (No corresponding
            boolean variable is defined for mu^(0) as it is assumed that
            mu^(0) is a variable fitting parameter for every channel.)
        option_Ecorrection_in (str): A character string indicating the way
            in which the energy correction should be calculated. At entry,
            option_Ecorrection_in should be "A", "E" or "" (equivalent to
            "A"), as is explained in the code.
        nrotations_in (int): The number of rotations defining the V-matrix.
        rotations_in (1D array[[int, int]]): The pairs of channel numbers for
            the respective rotation. At entry, rotations_in[k][0] and 
            rotations_in[k][1] should be, respectively, the i and j channel
            indexes of the k-th rotation (i, j = 1,...,nchann,
            k = 0,...,nrotations_in-1).
        Uialphabarmatrx_in (2D array[float]): The U_{i,alphabar} matrix.
            At entry, Uialphabarmatrx_in[i,alphabar] should be the 
            (i,alphabar) element of this matrix (i, alphabar = 1,...,nchann).
        jindx_in (int): The index of the j-channel (1 <= jindx_in <= nchann).
        kindx_in (int): The index of the k-channel (1 <= kindx_in <= nchann).
        calculation_method_in (str): A character string indicating whether
            the predicted energies should be calculated by minimizing the
            Xi^2 function (str = "Xi2min") or by finding the zeros of the
            relevant determinant (str = "zerodeterm").

    This function resets the global variables calculation_method, exp_energies,
    exp_errors, flvarmu1, Ilim, Is, jindx, kindx, klist, lquantumnumber,
    nchann, nkchannels, nenergies, nrotations, ntheta_start, nvarparams,
    option_description, option_Ecorrection, rotations, Rtilda and
    Uialphabarmatrx_s.

    Returns:
        None
    """

    global nchann, Ilim, lquantumnumber, Is, Rtilda, flvarmu1, nrotations, \
           rotations, Uialphabarmatrx_s, jindx, kindx, calculation_method, \
           nenergies, exp_energies, exp_errors, nvarparams, ntheta_start, \
           option_Ecorrection, option_description, klist, nkchannels
#
    nchann = nchann_in
    Ilim = Ilim_in
    lquantumnumber = lquantumnumber_in
    Is = Is_in
    Rtilda = Rtilda_in
    flvarmu1 = flvarmu1_in
    jindx = jindx_in
    kindx = kindx_in
    calculation_method = calculation_method_in
#
# Determine whether the mu^(1) terms should be multiplied by 1/nu_s^2 ("A"), 
# where nu_s is the effective principal quantum number measured with respect
# to the ionization threshold Is, or by (Rtilda/Is)/nu_j^2 ("E"). The default
# is "A".
    if option_Ecorrection_in == "" or option_Ecorrection_in == "A":
        option_Ecorrection = "A"
    elif option_Ecorrection_in == "E":
        option_Ecorrection = "E"
    else:
        print("Illegal value of option_Ecorrection")
        sys.exit()
#
# Check that the number of rotations makes sense, and store the rotation
# parameters in reverse order:
    nrotations = nrotations_in
    if nrotations < 1 or nrotations > round((nchann*(nchann-1))/2, 1):
        sys.exit("Incorrect number of rotations")
    for k in range(1, nrotations+1):
        rotations.append(rotations_in[nrotations+1-k])
#
# Store the Uialphabar matrix with the indices shifted by one unit
    Uialphabarmatrx_s = np.zeros((nchann, nchann))
    for i in range(1, nchann+1):
        for j in range(1, nchann+1):
            Uialphabarmatrx_s[i-1, j-1] = Uialphabarmatrx_in[i, j]
#
# Experimental energies
    nenergies = len(exp_data_in)
    exp_energies = np.empty((nenergies))
    exp_errors = np.empty((nenergies))
    for i in range(0, nenergies):
        exp_energies[i] = exp_data_in[i][0]
        exp_errors[i] = exp_data_in[i][1]
#
# Number of non-static MQDT parameters 
    nvarparams = 0
    for i in range (1, nchann+1):
        nvarparams = nvarparams + 1
    for i in range (1, nchann+1):
        if flvarmu1[i] == True: nvarparams = nvarparams + 1
    ntheta_start = nvarparams
    for i in range (1, nrotations+1):
        nvarparams = nvarparams + 1
#
# Check on the suitability of the choice of jindx and kindx
    if Ilim[jindx] == Ilim[kindx]:
        print("Taking jindx and kindx such that Ilim[jindx] = Ilim[kindx]")
        print("is unsafe.")
        sys.exit("Unsuitable choice of jindx and kindx")
#
# List of the dissociation channels degenerate with channel k:
    klist = []
    nkchannels = 0
    for i in range(1, nchann+1):
        if Ilim[i] == Ilim[kindx]:
            klist.append(i)
            nkchannels = nkchannels + 1
#
# Set the parameter option_description to "E", which signals
# that the calculation is based on the eigenchannel formalism.
    option_description = "E"

###############################################################################

def sort_exp_data(exp_data_in):

    """
    Sorts the content of exp_data_in by order of increasing energies.

    Args:
        exp_data_in (1D array[[float, float]]): The experimental data to be
            sorted. At entry, exp_data_in[i][0] and exp_data_Yin[i][1],
            i = 0,..., should be, respectively, the i-th measured energy
            level and the corresponding experimental error.

    This function does not reset global variables.

    Returns:
        nen (int): The number of energies in exp_data_in.
        exp_en (1D array[float]): The experimental energies, sorted by
            increasing values.
        exp_err (1D array[float]): The corresponding experimental errors.
    """

    exp_data_sorted = copy.deepcopy(exp_data_in)
    exp_data_sorted.sort(key=lambda pair: pair[0])
    nen = len(exp_data_sorted)
    exp_en = np.empty((nen))
    exp_err = np.empty((nen))
    for i in range(0, nen):
        exp_en[i] = exp_data_sorted[i][0]
        exp_err[i] = exp_data_sorted[i][1]
    return(nen, exp_en, exp_err)

###############################################################################

def set_searchintervals(option, deltanuj=None, factor=None):

    """
    Sets the search intervals used in the calculation of energies.

    Args:
        option (str): A single character determining whether the intervals
            should be in proportion of the separation between the experimental
            values of nu_j ("R") or should have a fixed length, the same for
            all the experimental energies ("A").
        deltanuj (optional, float): If specified and option == "A", the value
            of delta nu_j. If not specified and option == "A", delta nu_j is
            taken to be 0.01. Not used if option == "R".
        factor (optional, float): If specified and option == "R", the value
            of the magnification factor. If not specified and option == "R",
            the magnification factor is set to 1/3. Not used if option == "A".

    This function resets the global variable option_searchintervals.

    Returns:
        None
    """

    global option_searchintervals
    option_searchintervals = option
    if option == "R":
        if factor == None:
            fctr = 1./3.  # Default value of the magnification factor
        else:
            fctr = factor
        set_searchintervals_variable(fctr)
    elif option == "A":
        if deltanuj == None:
            dlt = 0.01    # Default value of delta nu_j
        else:
            dlt = deltanuj
        set_searchintervals_fixed(dlt)
    else:
        sys.exit("set_searchintervals: Illegal option.")

###############################################################################

def set_searchintervals_variable(factor):

    """
    Sets the search intervals for the scheme where the length of these
    varies from interval to interval and is calculated in proportion of
    the distance between adjacent experimental values of nu_j.

    Args:
        factor (float): The magnification factor to be used in the
            calculation.

    This function resets the global variables nujleft and nujright.

    Returns:
        None
    """

    global nujleft, nujright
    nujleft = [0.]*nenergies
    nujright = [0.]*nenergies
    nujexp = []
    if nenergies < 2:
        sys.exit("Insufficient number of experimental energies.")
#
# Create a list of experimental nu_j's, then sort it, keeping track of
# the original positions of the entries.
    for i in range (0, nenergies):
        nujexp.append((i, np.sqrt(Rtilda/(Ilim[jindx]-exp_energies[i]))))
    nujexp.sort(key=lambda pair: pair[1])
#
# Pass through the sorted list, define the left and right bounds, and put 
# them at the right places in the arrays nujleft and nujright.
    for i in range (0, nenergies):
        if i == 0:
           deltanuright = nujexp[i+1][1] - nujexp[i][1]
           deltanuleft = deltanuright
        elif i == nenergies-1:
           deltanuleft = nujexp[i][1] - nujexp[i-1][1]
           deltanuright = deltanuleft
        else:
           deltanuright = nujexp[i+1][1] - nujexp[i][1]
           deltanuleft = nujexp[i][1] - nujexp[i-1][1]
        ioriginal = nujexp[i][0]
        nujleft[ioriginal] = nujexp[i][1] - factor*deltanuleft
        nujright[ioriginal] = nujexp[i][1] + factor*deltanuright

###############################################################################

def set_searchintervals_fixed(deltanuj):

    """
    Sets the search intervals for the scheme where the length of these
    intervals is a constant 2 delta nu_j for all intervals.

    Args:
        deltanuj (float): The value of delta nu_j to be used in the
            calculation.

    This function resets the global variables nujleft and nujright.

    Returns:
        None
    """

    global nujleft, nujright
    nujleft = [0.]*nenergies
    nujright = [0.]*nenergies
    for i in range (0, nenergies):
        nujexp = np.sqrt(Rtilda/(Ilim[jindx]-exp_energies[i]))
        nujleft[i] = nujexp - deltanuj
        nujright[i] = nujexp + deltanuj

###############################################################################

def reset_calculation_method(calculation_method_in):

    """
    Resets the global variable calculation_method determining whether
    the energies should be calculated by minimization of the Xi^2 function
    of by finding the zeros of the relevant determinant.

    Args:
        calculation_method_in (str): A character string indicating whether
            the predictred energies should be calculated by minimizing the
            Xi^2 function (str = "Xi2min") or by finding the zeros of the
            relevant determinant (str = "zerodeterm").

    This function resets the global variable calculation_method.

    Returns:
        None
    """

    global calculation_method
    calculation_method = calculation_method_in

###############################################################################

def mqdtparams_Kmatrix(K0matrx, K1matrx):

    """
    Receives user-defined K^(0) and K^(1) matrices and stores them in a 
    1D array.

    Args: 
       K0matrx (2D array[float]): The K^(0) matrix passed to the function.
           At entry, K0matrx[i,j] should be the (i,j) element of that matrix
           (i =  1,...,nchann, j = i,...,nchann). Only the diagonal and upper
           triangle of K0matrx are used; the content of the lower triangle is
           ignored. 
       K1matrx (1D array[float]): The K^(1) matrix passed to the function.
           It is assumed that this matrix is diagonal. At entry, K1matrx[i]
           should be the (i,i) element of that matrix (i =  1,...,nchann).

    This function resets the global variables varparams and statparams.

    Returns:
        params (1D array[float]): A 1D array containing the elements of
            K0matrx and K1matrx stored in the format expected by other
            functions of this library.
    """

    global varparams, statparams
    if option_description != "K":
        sys.exit("mqdtparams_Kmatrix is invoked out of context.")
    varparams = np.array([])
    statparams = np.array([])
    for i in range (1, nchann+1):
        for j in range (i, nchann+1):
            if flvarK0[i, j] == True: 
                varparams = np.append(varparams, K0matrx[i, j])
            else:
                statparams = np.append(statparams, K0matrx[i, j])
        if flvarK1[i] == True:
            varparams = np.append(varparams, K1matrx[i])
        else:
            statparams = np.append(statparams, K1matrx[i])
    if nvarparams != len(varparams):
        print("Inconsistency in mqdtparams_Kmatrix: ", nvarparams, 
              len(varparams))
        sys.exit() 
    if len(statparams) == 0:
        params = varparams
    elif nvarparams == 0:
        params = statparams
    else:
        params = np.append(varparams, statparams)
    return(params)

###############################################################################

def mqdtparams_eigenchannel(mu0_arr, mu1_arr, theta_arr):

    """
    Receives user-defined values of mu^(0), mu^(1) and of the theta angles
    and stores them in a 1D array.

    Args:
       mu0_arr (1D array[float]): The values of mu^(0) passed to this function.
           At entry, mu0_arr[i] should be mu^(0)_i (i = 1,...,nchann).
       mu1_arr (1D array[float]): The values of mu^(1) passed to this function.
           At entry, mu1_arr[i] should be mu^(1)_i (i = 1,...,nchann).
       theta_arr (1D array[float]): The theta angles passed to this function.
           At entry, theta_arr[k] should be theta_{i,j}, where i and j are
           the channel indexes for the k-th rotation (k = 1,...).

    This function resets the global variables varparams and statparams.

    Returns:
        params (1D array[float]): A 1D array containing the elements of
            mu0_arr, mu1_arr and theta_arr stored in the format expected by
            other functions of this library.
    """

    global varparams, statparams
    if option_description != "E":
        sys.exit("mqdtparams_eigenchannel is invoked out of context.")
    varparams = np.array([])
    statparams = np.array([])
    for i in range (1, nchann+1):
        varparams = np.append(varparams, mu0_arr[i])
    for i in range (1, nchann+1):
        if flvarmu1[i] == True:
            varparams = np.append(varparams, mu1_arr[i])
        else:
            statparams = np.append(statparams, mu1_arr[i])
# Store the angles in reverse order
    for i in range (1, nrotations+1):
        varparams = np.append(varparams, theta_arr[nrotations+1-i])
    if nvarparams != len(varparams):
        print("Inconsistency in mqdtparams_eigenchannel: ", nvarparams, 
              len(varparams))
        sys.exit() 
    if len(statparams) == 0:
        params = varparams
    else:
        params = np.append(varparams, statparams)
    return(params)

###############################################################################

def Umatrxfun():

    """
    Forms and returns the U-matrix used in eigenchannel MQDT calculations.

    Args:
        None

    This function resets the global variable Umatrx, the U-matrix.
        The elements U_{i,alpha} (i, alpha = 1,...,nchann) of this matrix are
        store in Umatrx[i-1,alpha-1].

    Returns:
        None
    """
    global Umatrx
    Vmatrx = np.diag(np.ones(nchann))
    for k in range (0, nrotations):
        theta = varparams[ntheta_start+k]
        rotmatrx = np.diag(np.ones(nchann))
# Recall that the rotations are stored in the order of their application, 
# in the arrays varparams and rotations...
        i = rotations[k][0] - 1
        j = rotations[k][1] - 1
        rotmatrx[i, i] = np.cos(theta)
        rotmatrx[j, j] = rotmatrx[i, i]
        rotmatrx[i, j] = -np.sin(theta)
        rotmatrx[j, i] = -rotmatrx[i, j]
        Vmatrx = np.matmul(rotmatrx, Vmatrx)
    Umatrx = np.matmul(Uialphabarmatrx_s, Vmatrx)
    return()

###############################################################################

def eigendeterm(nu_j):

    """
    Calculates the determinant whose zeros determine the theoretical energies
    in eigenchannel MQDT calculations and returns its value.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        det (float): The value of the determinant.
    """

    if option_Ecorrection == "A":
        x = (Is - Ilim[jindx])/Rtilda + 1/nu_j**2
    else:
        E = Ilim[jindx] - Rtilda/nu_j**2
        x = (Is - E)/Is
    mu_s = np.zeros(nchann)
    kvar = -1
    kstat = -1
    for alpha in range (1, nchann+1):
        kvar = kvar + 1
        mu_s[alpha-1] = varparams[kvar]
    for alpha in range (1, nchann+1):
        if flvarmu1[alpha] == True:
            kvar = kvar + 1
            mu_s[alpha-1] = mu_s[alpha-1] + x*varparams[kvar]
        else:
            kstat = kstat + 1
            mu_s[alpha-1] = mu_s[alpha-1] + x*statparams[kstat]
    matrx = copy.deepcopy(Umatrx)
    for i in range (0, nchann):
        nu_i = Ffun(i+1, nu_j)
        for alpha_s in range (0, nchann):
            matrx[i, alpha_s] = matrx[i, alpha_s] * \
                                           np.sin(np.pi*(nu_i + mu_s[alpha_s]))
    det = np.linalg.det(matrx)
    return(det)

###############################################################################

def Kmatrixdeterm(nu_j, nu_k=None):

    """
    Calculates the determinant whose zeros determine the theoretical energies
    in the K-matrix formulation of MQDT, and returns its value.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.
        nu_k (optional, float): The value of nu_k to be used in the
            calculation. If not specified, nu_k is calculated from the value
            of nu_j.

    This function does not reset global variables.

    Returns:
        det (float): The value of the determinant.
    """

    if nu_k == None: nu_k = Ffun(kindx, nu_j)
    amatr = mqdtmatrx_Kmatrix_s(nu_j, nu_k, fl_no_tan_nu_k=False)
    det = np.linalg.det(amatr)
    return(det)

###############################################################################

def Ffun(i, nu_j):

    """
    Calculates and returns nu_i, given nu_j.

    Args:
        i (int): The channel for which the value of nu is sought.
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        nu_i (float): The value of nu_i.
    """

    if i < 1 or i > nchann: print('wrong i: ',i)
    if jindx < 1 or jindx > nchann: print('wrong jindx: ',jindx)
    if ((Ilim[i]-Ilim[jindx])/Rtilda + 1.0/nu_j**2) < 0:
        print("Negative argument ",i,jindx,nu_j)
        sys.exit()
    nu_i = 1.0/np.sqrt((Ilim[i]-Ilim[jindx])/Rtilda + 1.0/nu_j**2)
    return(nu_i)

###############################################################################

def Gkfun_mltple(nu_j):

    """
    Given nu_j, calculates and returns all the possible values of G_k(nu_j).

    This function is intended for calculations in which other channels have
    the same ionisation limit as channel k.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        nuks (1D array[float]): A 1D array containing the possible values
        of nu_k modulo 1, assuming that nu_k = G_k(nu_j).
    """
    
# Find the coefficients of the polynomial from the value of the determinant
# calculated at several values of tan(pi nu_k)
    x = np.empty((nkchannels+1))
    y = np.empty((nkchannels+1))
    for k in range(0,nkchannels+1):
        nu_k = (k+0.3)/(nkchannels+1)  # There is not strong reason for an
        x[k] = np.tan(np.pi*nu_k) 
        y[k] = Kmatrixdeterm(nu_j, nu_k)
    a = np.polynomial.polynomial.polyfit(x,y,nkchannels)
#
# Find the possible values of nu_k from the roots of this polynomial.
    roots = np.polynomial.polynomial.polyroots(a)
    nuks = np.empty((nkchannels))
    for k in range(0,nkchannels):
        nuks[k] = (np.arctan(roots[k])/np.pi)%1
        if nuks[k] < 0: nuks[k] = nuks[k] + 1
#
    return(nuks)

###############################################################################

def Gkfun(nu_j):

    """
    Given nu_j, calculates and returns a single value of G_k(nu_j) (any number
    of channels).

    This function is intended for calculations in which no channel has the same
    ionisation limit as channel k.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        nu_k_mod1 (float): The value of nu_k, modulo 1, assuming that
            nu_k = G_k.
    """

    nu_k = Ffun(kindx, nu_j)
    amatr = mqdtmatrx_Kmatrix_s(nu_j, nu_k, fl_no_tan_nu_k=True)
    km1 = kindx - 1
    akk = amatr[km1, km1]
    amatr[km1, km1] = 0.0
    det1 = np.linalg.det(amatr)
    amatr[km1] = 0.0   # zeros the row corresponding to the k-channel
    amatr[km1, km1] = 1.0
    det2 = np.linalg.det(amatr)
    res = -det1/det2
    nu_k =  np.arctan(res-akk)/np.pi 
    nu_k_mod1 = nu_k%1
    if nu_k_mod1 < 0: nu_k_mod1 = nu_k_mod1 + 1
    return(nu_k_mod1)
     
###############################################################################

def Gkfun_2chann(nu_j):

    """
    Given nu_j, calculates and returns a single value of G_k(nu_j) for a
    2-channel calculation.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        nu_k_mod1 (float): The value of nu_k, modulo 1, assuming that
            nu_k = G_k.
    """

    nu_k = 0.  # Bogus value, as nu_k won't be used in this context.
    Kmat = mqdtmatrx_Kmatrix_s(nu_j, nu_k, fl_no_tan_nu_k=True)
    if jindx == 2:
        a22 = Kmat[1, 1]
        res = Kmat[0, 1]**2/a22
        nu_k = np.arctan(res-Kmat[0, 0])/np.pi
    else:
        a11 = Kmat[0, 0]
        res = Kmat[0, 1]**2/a11
        nu_k = np.arctan(res-Kmat[1, 1])/np.pi    
    nu_k_mod1 = nu_k%1
    if nu_k_mod1 < 0: nu_k_mod1 = nu_k_mod1 + 1
    return(nu_k_mod1)       

###############################################################################

def Gkfun_3chann(nu_j):

    """
    Given nu_j, calculates and returns a single value of G_k(nu_j) for a
    3-channel calculation.

    This function is intended for calculations in which no channel has the same
    ionisation limit as channel k.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        nu_k_mod1 (float): The value of nu_k, modulo 1, assuming that
            nu_k = G_k.
    """

    nu_k = Ffun(kindx, nu_j)
    Kmat = mqdtmatrx_Kmatrix_s(nu_j, nu_k, fl_no_tan_nu_k=True)
    if kindx == 1:
        a22 = Kmat[1, 1]
        a33 = Kmat[2, 2]
        res = a33*Kmat[0, 1]**2 + a22*Kmat[0, 2]**2 \
                  - 2.0*Kmat[0, 1]*Kmat[0, 2]*Kmat[1, 2]
        res = res/(a22*a33-Kmat[1, 2]**2)
        nu_k = np.arctan(res-Kmat[0, 0])/np.pi
    elif kindx == 2:
        a11 = Kmat[0, 0]
        a33 = Kmat[2, 2]
        res = a33*Kmat[0, 1]**2 + a11*Kmat[1, 2]**2 \
                  - 2.0*Kmat[0, 1]*Kmat[0, 2]*Kmat[1, 2]
        res = res/(a11*a33-Kmat[0, 2]**2)
        nu_k = np.arctan(res-Kmat[1, 1])/np.pi
    else:
        a11 = Kmat[0, 0]
        a22 = Kmat[1, 1]
        res = a22*Kmat[0, 2]**2 + a11*Kmat[1, 2]**2 \
                  - 2.0*Kmat[0, 1]*Kmat[0, 2]*Kmat[1, 2]
        res = res/(a11*a22-Kmat[0, 1]**2)
        nu_k = np.arctan(res-Kmat[2, 2])/np.pi    
    nu_k_mod1 = nu_k%1
    if nu_k_mod1 < 0: nu_k_mod1 = nu_k_mod1 + 1
    return(nu_k_mod1)               

###############################################################################

def Xi2(nu_j):

    """
   Given nu_j, calculates and returns the square of the function Xi.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function does not reset global variables.

    Returns:
        res (float): The value of [Xi(nu_j)]^2.
    """
    
    if nkchannels == 1:
        if nchann == 2:
            res = (Gkfun_2chann(nu_j) - Ffun(kindx, nu_j)%1)**2
        elif nchann == 3:
            res = (Gkfun_3chann(nu_j) - Ffun(kindx, nu_j)%1)**2
        else:
            res = (Gkfun(nu_j) - Ffun(kindx, nu_j)%1)**2
    else:
        F = Ffun(kindx, nu_j)%1
        G = Gkfun_mltpl(nu_j)
        res = (G[0] - F)**2
        for k in range(1, nkchannels):
            res = min(res, (G[k] - F)**2)
    return(res)

###############################################################################

def chi2fun(varparams_in):

    """
    Calculates, returns and optionally writes out the chi2 statistics for the
    MQDT model specified in the list of arguments, given the experimental data
    provided in the most recent call to either initialize_Kmatrix or
    initialize_eigenchannel.

    Args:
        varparams_in (1D array[float]): The variable parameters of the model in
            the 1D format used, e.g., by the function mqdtparams_Kmatrix or
            the function mqdtparams_eigenchannel.

    This function resets the global variables varparams and (only if
    option_description == "E") Umatrx.

    If the global variable displaychi2 has been set to True prior to the
    call to this function, the value of chi2 and the largest actual value taken
    by the determinant (if calculation_method == zerodeterm) or by Xi^2(nu_j)
    (if calculation_method == "Xi2min") over all the solutions found in the
    calculation are written out on the standard output stream.

    Returns:
        The value of chi2.
    """

    global varparams
    varparams = varparams_in
# Stores the U-matrix in the global variable Umatrx for calculations in the
# eigenchannel formalism:
    if option_description == "E": Umatrxfun()
    chi2 = 0.0
    if displaychi2 == True: maxvalue = 0.0
#
    for i in range(0, nenergies):
#
        if calculation_method == "Xi2min":
            nujexp= np.sqrt(Rtilda/(Ilim[jindx]-exp_energies[i]))
            nujl = nujleft[i]
            nujr = nujright[i]
            Xi2l = Xi2(nujl)
            Xi2e = Xi2(nujexp)
            Xi2r = Xi2(nujr)
            if Xi2e < Xi2l and Xi2e < Xi2r:
                bracket = (nujl, nujexp, nujr)
            else:
# A search of a bracketing interval is carried out if possible.
                if option_searchintervals != "A":
                    print(exp_energies[i], Xi2l, Xi2e, Xi2r)
                    sys.exit("Minimum of Xi2 cannot be bracketed")
                deltanu = (nujr - nujl)/2.
                nujl0 = nujexp
                nujr0 = nujexp
                Xi2l0 = Xi2e
                Xi2r0 = Xi2e
                bracket = None
                for k in range(0, 5000):
                    nujll = nujl - deltanu
                    nujrr = nujr + deltanu
                    Xi2ll = Xi2(nujll)
                    Xi2rr = Xi2(nujrr)
                    if Xi2ll > Xi2l and Xi2l0 > Xi2l:
                        bracket = (nujll, nujl, nujl0)
                        break
                    elif Xi2r0 > Xi2r and Xi2rr > Xi2r:
                        bracket = (nujr0, nujr, nujrr)
                        break
                    else:
                        nujl0 = nujl
                        nujl = nujll
                        nujr0 = nujr
                        nujr = nujrr
                        Xi2l0 = Xi2l
                        Xi2l = Xi2ll
                        Xi2r0 = Xi2r
                        Xi2r = Xi2rr
                if bracket == None:
                    print(exp_energies[i])
                    sys.exit("Minimum of Xi2 cannot be bracketed")
# At this stage, the minimum of Xi-squared is meant to be bracketed...
# The function minimize_scalar is imported from scipy.optimize.
            res = minimize_scalar(Xi2, bracket=bracket, method="Brent",
                                  tol=1e-12)
            nujcalc = res.x
            if displaychi2 == True:
                Xi2calc = Xi2(nujcalc)
                if Xi2calc > maxvalue: maxvalue = Xi2calc
#        
        elif calculation_method == "zerodeterm":
            nujl = nujleft[i]
            nujr = nujright[i]
            if option_description == "K":
                dl = Kmatrixdeterm(nujl)
                dr = Kmatrixdeterm(nujr)
            else:
                dl = eigendeterm(nujl)
                dr = eigendeterm(nujr)
            if dl*dr <= 0:
                bracket = (nujl, nujr)
            else:
# A search of a bracketing interval is carried out if possible.
                if option_searchintervals != "A":
                    print(exp_energies[i], dl, dr)
                    sys.exit("The zero of the determinant cannot be bracketed")
                deltanu = (nujr - nujl)/2.
                bracket = None
                for k in range(0, 5000):
                    nujll = nujl - deltanu
                    nujrr = nujr + deltanu
                    if option_description == "K":
                        dll = Kmatrixdeterm(nujll)
                        drr = Kmatrixdeterm(nujrr)
                    else:
                        dll = eigendeterm(nujll)
                        drr = eigendeterm(nujrr)
                    if dll*dl <= 0:
                       bracket = (nujll, nujl)
                       break
                    elif dr*drr <= 0:
                       bracket = (nujr, nujrr)
                       break
                    else:
                       nujl = nujll
                       nujr = nujrr
                       dl = dll
                       dr = drr
                if bracket == None:
                    print(exp_energies[i])
                    sys.exit("The zero of the determinant cannot be bracketed")
# At this stage, the required zero is meant to be between bracket[0] and
# bracket[1]... The function brentq is imported from scipy.optimize.
            if option_description == "K":
                nujcalc = brentq(Kmatrixdeterm, bracket[0], bracket[1])
            else:
                nujcalc = brentq(eigendeterm, bracket[0], bracket[1])
            if displaychi2 == True:
                if option_description == "K":
                    absdet = abs(Kmatrixdeterm(nujcalc))
                else:
                    absdet = abs(eigendeterm(nujcalc))
                if absdet > maxvalue: maxvalue = absdet
# 
        else:
            sys.exit("Illegal calculation method")
#
        th_energy = Ilim[jindx] - Rtilda/nujcalc**2
        chi2 = chi2 + ((exp_energies[i]-th_energy)/exp_errors[i])**2
#   
    if displaychi2 == True: 
        print("chi2, largest departure from 0: ", chi2, maxvalue)
    return(chi2)

###############################################################################

def th_energies_fun(params_in, exp_energies_in):

    """
    Calculates and returns the theoretical energies matching a list of
    experimental energies.

    Args:
        params_in (1D array[float]): The parameters of the model in the 1D
            format used, e.g., by the function mqdtparams_Kmatrix or the
            function mqdtparams_eigenchannel.
        exp_energies_in (1D array[float]): A set of experimental energies, in
            inverse centimeters. The energies in this set must be included in
            the experimental energies passed to initialize.Kmatrix or
            initialize.eigenchannel through the array exp_data_in.

    This function resets the global variables varparams, statparams and, if
    option_description == "E", Umatrx.

    Returns:
        A 1D array (float) containing the calculated energies matching
            the experimental energies passed to this function through
            the array exp_energies_in.
    """

    global varparams, statparams
    varparams = params_in[0:nvarparams]
    statparams = params_in[nvarparams:]
# Stores the U-matrix in the global variable Umatrx for calculations in the
# eigenchannel formalism:
    if option_description == "E": Umatrxfun()
# Match the content of exp_energies_in to the content of exp_energies:
    indxs = abs(exp_energies_in[:, None] - exp_energies).argmin(axis=1)
    th_en = np.array([])
#
    for i in range(0, len(indxs)):
        ii = indxs[i]
#
        if calculation_method == "Xi2min":
            nujexp= np.sqrt(Rtilda/(Ilim[jindx]-exp_energies[ii]))
            nujl = nujleft[ii]
            nujr = nujright[ii]
            Xi2l = Xi2(nujl)
            Xi2e = Xi2(nujexp)
            Xi2r = Xi2(nujr)
            if Xi2e < Xi2l and Xi2e < Xi2r:
                bracket = (nujl, nujexp, nujr)
            else:
# A search of a bracketing interval is carried out if possible.
                if option_searchintervals != "A":
                    print(exp_energies[ii], Xi2l, Xi2e, Xi2r)
                    sys.exit("Minimum of Xi2 cannot be bracketed")
                deltanu = (nujr - nujl)/2.
                nujl0 = nujexp
                nujr0 = nujexp
                Xi2l0 = Xi2e
                Xi2r0 = Xi2e
                bracket = None
                for k in range(0, 5000):
                    nujll = nujl - deltanu
                    nujrr = nujr + deltanu
                    Xi2ll = Xi2(nujll)
                    Xi2rr = Xi2(nujrr)
                    if Xi2ll > Xi2l and Xi2l0 > Xi2l:
                        bracket = (nujll, nujl, nujl0)
                        break
                    elif Xi2r0 > Xi2r and Xi2rr > Xi2r:
                        bracket = (nujr0, nujr, nujrr)
                        break
                    else:
                        nujl0 = nujl
                        nujl = nujll
                        nujr0 = nujr
                        nujr = nujrr
                        Xi2l0 = Xi2l
                        Xi2l = Xi2ll
                        Xi2r0 = Xi2r
                        Xi2r = Xi2rr
                if bracket == None:
                    print(exp_energies[ii])
                    sys.exit("Minimum of Xi2 cannot be bracketed")
# The function minimize_scalar is imported from scipy.optimize.
            res = minimize_scalar(Xi2, bracket=bracket, method="Brent",
                                  tol=1e-12)
            nujcalc = res.x
# 
        elif calculation_method == "zerodeterm":
            nujl = nujleft[ii]
            nujr = nujright[ii]
            if option_description == "K":
                dl = Kmatrixdeterm(nujl)
                dr = Kmatrixdeterm(nujr)
            else:
                dl = eigendeterm(nujl)
                dr = eigendeterm(nujr)
            if dl*dr <= 0:
                bracket = (nujl, nujr)
            else:
# A search of a bracketing interval is carried out if possible.
                if option_searchintervals != "A":
                    print(exp_energies[i], dl, dr)
                    sys.exit("The zero of the determinant cannot be bracketed")
                deltanu = (nujr - nujl)/2.
                bracket = None
                for k in range(0, 5000):
                    nujll = nujl - deltanu
                    nujrr = nujr + deltanu
                    if option_description == "K":
                        dll = Kmatrixdeterm(nujll)
                        drr = Kmatrixdeterm(nujrr)
                    else:
                        dll = eigendeterm(nujll)
                        drr = eigendeterm(nujrr)
                    if dll*dl <= 0:
                       bracket = (nujll,nujl)
                       break
                    elif dr*drr <= 0:
                       bracket = (nujr,nujrr)
                       break
                    else:
                       nujl = nujll
                       nujr = nujrr
                       dl = dll
                       dr = drr
                if bracket == None:
                    print(exp_energies[ii])
                    sys.exit("The zero of the determinant cannot be bracketed")
# At this stage, the required zero is meant to be between bracket[0] and
# bracket[1]... The function brentq is imported from scipy.optimize.
            if option_description == "K":
                nujcalc = brentq(Kmatrixdeterm, bracket[0], bracket[1])
            else:
                nujcalc = brentq(eigendeterm, bracket[0], bracket[1])
#
        else:
            sys.exit("Illegal calculation method")
#
        th_en = np.append(th_en, Ilim[jindx] - Rtilda/nujcalc**2)
#    
    return(th_en)

###############################################################################

def prep_initial_simplex(varparams_in, nonzdelt=None, zdelt=None):

    """
    Constructs the initial simplex required for an optimization of the
    MQDT parameters.

    The construction is identical to that done in the _minimize_neldermead
    function of the scipy.optimize library.

    Args:
        varparams_in (1D array[float]): The variable parameters of the model in
            the 1D format used, e.g., by the function mqdtparams_Kmatrix or
            the function mqdtparams_eigenchannel.
        nonzdelt (optional, float): The parameter of this name used in the
            construction of the simplex. Its default value is the same as
            in the _minimize_neldermead function, i.e., 0.05.
        zdelt (optional, float): The parameter of this name used in the
            construction of the simplex. Its default value is the same as
            in the _minimize_neldermead function, i.e., 0.00025.

    This function does not resets global variables.

    Returns:
        simplex (2D array[float]): The initial simplex, in the format required
            by the scipy.optimize function fmin.
    """

    if nonzdelt == None: nonzdelt = 0.05
    if zdelt == None: zdelt = 0.00025
    n = nvarparams
    simplex = np.empty((n + 1, n))
    simplex[0] = varparams_in
    for k in range(n):
        y = np.array(varparams_in, copy=True)             
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt
        simplex[k + 1] = y
    return(simplex)

###############################################################################

def optimizeparams(initialparams, monitor=None, nonzdelt=None, zdelt=None,
                   maxfun=None, maxiter=None):

    """
    Optimizes the variable parameters of an MQDT model by chi2-fitting.

    The calculation is based on the fmin function of the scipy.optimize
    library.

    Args:
        initialparams (1D array[float]): The parameters of the MQDT model to
            be used as a starting point of the optimization in the 1D format
            used, e.g., by the function mqdtparams_Kmatrix or the function
            mqdtparams_eigenchannel.
        monitor (optional, bool): The global variable displaychi2 is set to
            False if monitor = False or the value of monitor is not specified,
            and is set to True if monitor == True (which triggers the output
            of additional information by the functions chi2fun_eigenchannel
            and chi2fun_Kmatrix in the course of the calculation).
        nonzdelt (optional, float): The parameter of this name used in the
            construction of the simplex by the function prep_initial_simplex.
            Its default value is the same as in the _minimize_neldermead
            function of the scipy.optimize library, i.e., 0.05.
        zdelt (optional, float): The parameter of this name used in the
            construction of the simplex by the function prep_initial_simplex.
            Its default value is the same as in the _minimize_neldermead
            function of the scipy.optimize library, i.e., 0.00025.
        maxfun (optional, int): The maximum number of function evaluations by
            fmin. The default value of this optional parameter is 3,000.
        maxiter (optional, int): The maximum number of iterations by fmin.
            The default value of this optional parameter is 3,000.

    This function resets the global variables varparams, statparams and
    displaychi2.

    Returns:
        optparams (1D array[float]): The parameters of the optimized model.
    """

    global varparams, statparams, displaychi2
    if nvarparams == 0:
        print("error: optimizeparams is invoked, but there are no")
        print("parameters to optimize (nvarparams = 0)...")
        sys.exit()
    if monitor == None or monitor == False:
        displaychi2 = False
    elif monitor == True:
        displaychi2 = True
    else:
        sys.exit("Illegal value of monitor.")
    if maxfun == None:
        mxf = maxfun
    else:
        mxf = 3e3
    if maxiter == None:
        mit = maxiter
    else:
        mit = 3e3
#       
    varparams = initialparams[0:nvarparams]
    statparams = initialparams[nvarparams:]
    initial_simplex = prep_initial_simplex(varparams, nonzdelt, zdelt)
# The function fmin is imported from scipy.optimize.
    optvarparams= fmin(chi2fun, initialparams[0:nvarparams], maxfun=mxf, 
                       maxiter=mit, initial_simplex=initial_simplex)
# 
    displaychi2 = False  # Return displaychi2 to its default value

    if len(statparams) == 0:
        optparams = optvarparams
    else:
        optparams = np.append(optvarparams, statparams)
    return(optparams)

###############################################################################

def mqdtmatrx_Kmatrix_s(nu_j, nu_k, fl_no_tan_nu_k):

    """
    Constructs the matrix used in the K-matrix formulation. The actual work
        is done either by Kmatrix_fromUandmu or by Kmatrix_Kmatrixformalism.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.
        nu_k (float): The value of nu_k to be used in the calculation.
        fl_no_tan_nu_k (bool): A flag determining whether the tan(pi nu_k) term
           should (fl_no_tan_nu_k == False) or should not (fl_no_tan_nu_k ==
           True) be added to the corresponding diagonal element of the matrix.

    This function does not reset global variables.

    Returns:
        matrx_s (2D array[float]): The matrix built by this function.
            The element M_{i,j} (i, j = 1,...,nchann) of this matrix is
            returned through matrx_s[i-1,j-1].
    """
    
    if option_description == "E":
        matrx_s = Kmatrix_fromUandmu(nu_j, nu_k, fl_no_tan_nu_k)
    elif option_description == "K":
        matrx_s = Kmatrix_Kmatrixformalism(nu_j, nu_k, fl_no_tan_nu_k)
    else:
        sys.exit("mqdtmatrx_Kmatrix_s: illegal value of option_description")

    return(matrx_s)

###############################################################################

def Kmatrix_fromUandmu(nu_j, nu_k, fl_no_tan_nu_k):

    """
    Constructs the matrix K + tan pi nu_i from the eigen channel defects and
        the U-matrix.

    This function does not reset global variables.

    Arguments and returns are as for mqdtmatrx_Kmatrix_s.

    """

    if option_Ecorrection == "A":
        x = (Is - Ilim[jindx])/Rtilda + 1/nu_j**2
    else:
        E = Ilim[jindx] - Rtilda/nu_j**2
        x = (Is - E)/Is
    mu_s = np.zeros(nchann)
    kvar = -1
    kstat = -1
    for alpha in range (1, nchann+1):
        kvar = kvar + 1
        mu_s[alpha-1] = varparams[kvar]
    for alpha in range (1, nchann+1):
        if flvarmu1[alpha] == True:
            kvar = kvar + 1
            mu_s[alpha-1] = mu_s[alpha-1] + x*varparams[kvar]
        else:
            kstat = kstat + 1
            mu_s[alpha-1] = mu_s[alpha-1] + x*statparams[kstat]
    matrx_s = np.zeros((nchann, nchann))
    for alpha_s in range (0, nchann):
        matrx_s[alpha_s] = np.tan(np.pi*mu_s[alpha_s])*Umatrx[:,alpha_s]
    matrx_s = Umatrx.dot(matrx_s)
#
# Add the tan(pi nu_i) term to the corresponding diagonal element unless,
# for i = k, if fl_no_tan_nu_k is True.
    for i in range (1, nchann+1):
        im1 = i - 1
        if fl_no_tan_nu_k == True and i == kindx: continue
        if i in klist:
            matrx_s[im1, im1] = matrx_s[im1, im1] + np.tan(np.pi*nu_k)
        else:
            matrx_s[im1, im1] = matrx_s[im1, im1] + np.tan(np.pi*Ffun(i, nu_j))

    return(matrx_s)


###############################################################################

def Kmatrix_Kmatrixformalism(nu_j, nu_k, fl_no_tan_nu_k):

    """
    Constructs the matrix K + tan pi nu_i from the parameters of the
        K-matrix stored in varparams and statparams.

    This function does not reset global variables.

    Arguments and returns are as for mqdtmatrx_Kmatrix_s.

    """

    if option_Ecorrection == "A":
        x = (Is - Ilim[jindx])/Rtilda + 1/nu_j**2
    else:
        E = Ilim[jindx] - Rtilda/nu_j**2
        x = (Is - E)/Is
    matrx_s = np.zeros((nchann, nchann))
    kvar = -1
    kstat = -1
    for i in range (1, nchann+1):
        im1 = i - 1
        for j in range (i, nchann+1):
            jm1 = j - 1
            if flvarK0[i,j] == True:
                kvar = kvar + 1
                matrx_s[im1,jm1] = varparams[kvar]
            else:
                kstat = kstat + 1
                matrx_s[im1,jm1] = statparams[kstat]
# lower triangle:
            if im1 != jm1: matrx_s[jm1, im1] = matrx_s[im1, jm1]
        if flvarK1[i] == True:
            kvar = kvar + 1
            matrx_s[im1, im1] = matrx_s[im1, im1] + x*varparams[kvar]
        else:
            kstat = kstat + 1
            matrx_s[im1, im1] = matrx_s[im1, im1] + x*statparams[kstat]
# Do not add the tan(pi nu_k) term to the corresponding diagonal element if
# fl_no_tan_nu_k is True.
        if fl_no_tan_nu_k == True and i == kindx: continue
        if i in klist:
            matrx_s[im1, im1] = matrx_s[im1, im1] + np.tan(np.pi*nu_k)
        else:
            matrx_s[im1, im1] = matrx_s[im1, im1] + np.tan(np.pi*Ffun(i, nu_j))
    return(matrx_s)

###############################################################################

def mqdtmatrices_and_mu1_eigenchannel_s(nu_j):

    """
    Constructs and returns arrays used by mixing_coeffs_eigenchannel.

    Args:
        nu_j (float): The value of nu_j to be used in the calculation.

    This function resets the global variable Umatrx.

    Returns:
        matrxsin_s (2D array[float]): The sine-function matrix. The element
            S_{i,j} (i, j = 1,...,nchann) of this matrix is returned through
            matrxsin_s[i-1,j-1].
        matrxcos_s (2D array[float]): The cosine-function matrix. The element
            C_{i,j} (i, j = 1,...,nchann) of this matrix is returned through
            matrxcos_s[i-1,j-1].
        mu1_arr (1D array[float]): The mu^(1) MQDT parameters.
    """

    Umatrxfun()  # Stores the U-matrix in the global variable Umatrx.
    if option_Ecorrection == "A":
        x = (Is - Ilim[jindx])/Rtilda + 1/nu_j**2
    else:
        E = Ilim[jindx] - Rtilda/nu_j**2
        x = (Is - E)/Is
    mu_s = np.zeros(nchann)
    mu1_arr = np.zeros(nchann+1)
    kvar = -1
    kstat = -1
    for i in range (1, nchann+1):
        kvar = kvar + 1
        mu_s[i-1] = varparams[kvar]
    for i in range (1, nchann+1):
        if flvarmu1[i] == True:
            kvar = kvar + 1
            mu1_arr[i] = varparams[kvar]
            mu_s[i-1] = mu_s[i-1] + x*varparams[kvar]
        else:
            kstat = kstat + 1
            mu1_arr[i] = statparams[kstat]
            mu_s[i-1] = mu_s[i-1] + x*statparams[kstat]
#
    matrxsin_s = copy.deepcopy(Umatrx)
    matrxcos_s = copy.deepcopy(Umatrx)
    for i in range (0, nchann):
        nu_i = Ffun(i+1, nu_j)
        for j in range (0, nchann):
            matrxsin_s[i, j] = matrxsin_s[i, j] \
                                               * np.sin(np.pi*(nu_i + mu_s[j]))
            matrxcos_s[i, j] = matrxcos_s[i, j] \
                                               * np.cos(np.pi*(nu_i + mu_s[j]))
#
    return(matrxsin_s, matrxcos_s, mu1_arr)

###############################################################################

def chi2andredchi2fun(params_in):

    """
    Returns the chi2 and reduced chi2 statistics (interface with chi2fun).

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.

    This function resets the global variables varparams and statparams.

    Returns:
        Two floating point numbers, respectively the chi2 and reduced chi2
        statistics.
    """

    global varparams, statparams
    varparams = params_in[0:nvarparams]
    statparams = params_in[nvarparams:]
    chi2 = chi2fun(varparams)
    return(chi2, chi2/(nenergies-nvarparams))

###############################################################################

def print_chi2(params_in, moreinfo=None, file=None):

    """
    Prints out the chi2 and reduced chi2 statistics, and optionally the number
    of degrees of freedom.

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.
        moreinfo (optional, bool): The number of data points, the number of
            fitting parameters and the number of degrees of freedom are also
            written out if moreinfo == True. This additional information is not
            written out if moreinfo == False or the value of moreinfo is not
            specified.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is 
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

    chi2, redchi2 = chi2andredchi2fun(params_in)
    if moreinfo == True:
        addline1 = "\n" + "Number of data points: {0:n}".format(nenergies)
        addline2 = "\n" + "Number of fitted MQDT parameters: {0:n}".\
                                                        format(nvarparams)
        addline3 = "\n" + "Number of degrees of freedom: {0:n}".\
                                              format(nenergies-nvarparams)
        print(addline1 + addline2 + addline3)
    line = "\n" + "chi2: {0:.3e}  Reduced chi2: {1:.3e}".\
                                                     format(chi2, redchi2)
    print(line)
#
    if file != None:
        with open(file, "a") as f:
            if moreinfo == True: f.write(addline1 + addline2 + addline3\
                                                                   + "\n")
            f.write(line + "\n")
            f.close()

###############################################################################

def print_energies(params_in, exp_data_in, file=None):

    """
    Prints out experimental energies and the corresponding calculated energies
    for a given MQDT model.

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.
        exp_data_in (1D array([float], [float])): The experimental data to be
            included in the printout. This array is expected to be identical
            or to be a subset of the array exp_data_in previously passed to the
            function initialize_Kmatrix or initialize_eigenchannel.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

    print("\n"+"Energies\n"+"--------")
    print("(a): Experimental energies, in inverse centimeters.")
    print("(b): Experimental errors, in inverse centimeters.")
    print("(c): Calculated energies, in inverse centimeters.")
    print("(d): Difference (a) - (b), in inverse centimeters.")
    print("(e): Difference (a) - (b), in MHz.")
    print("(f): Contribution to the total chi2.")
    nen, exp_en, exp_err = sort_exp_data(exp_data_in)
    th_en = th_energies_fun(params_in, exp_en)
    headings = "    (a)       "
    headings = headings + "  (b)  "
    headings = headings + "       (c)    "
    headings = headings + "      (d)  "
    headings = headings + "       (e)  "
    headings = headings + "      (f)  "
    print(headings)
    line = []
    for i in range(0, len(th_en)):
        chi2term = (th_en[i] - exp_en[i])**2/exp_err[i]**2
        line.append("{0:.6f}  {1:.6f}  {2:.6f}  {3: .6f}  {4: .3e}  {5:.3e}"
                    .format(exp_en[i], exp_err[i], th_en[i],
                            exp_en[i]-th_en[i], 
                            (exp_en[i]-th_en[i])*29979.2458, 
                            chi2term))
        print(line[i])
#
    if file != None:
        with open(file, "a") as f: 
            f.write("\n"+"Energies\n"+"--------\n")
            f.write("(a): Experimental energies, in inverse centimeters.\n")
            f.write("(b): Experimental errors, in inverse centimeters.\n")
            f.write("(c): Calculated energies, in inverse centimeters.\n")
            f.write("(d): Difference (a) - (b), in inverse centimeters.\n")
            f.write("(e): Difference (a) - (b), in MHz.\n")
            f.write("(f): Contribution to the total chi2.\n")
            f.write(headings + "\n")
            for i in range(0, len(th_en)):
                f.write(line[i] + "\n")
            f.close()
    
###############################################################################

def print_channelparams(printlvalues=None, printUialphabarmatrx=None,
                        file=None):

    """
    Prints out the details of the channels included in the model.

    Args:
        printlvalues (optional, bool): The l value of each channel is written 
            out if printlvalues == True. This information is not written out
            if printlvalues == False or the value of this variable is not
            specified.
        printUialphabarmatrx (optional, bool): The U_{i,alphabar} matrix is
            written out if printUialphabarmatrx == True. This information is
            not written out if printUialphabarmatrx == False or the value of
            this variable is not specified.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

    print("\n"+"Channel parameters\n"+"------------------")
    line = []
    string = "Is = {0:.4f}".format(Is)
    print(string)
    line.append(string)
    string = "Rtilda = {0:.3f}".format(Rtilda)
    print(string)
    line.append(string)
    m = 2
    for i in range(1, nchann+1):
        m = m + 1
        if printlvalues == True:
            string = "Ilim[{0:n}] = {1:.4f}  l[{2:n}] = {3:n}".format(i, \
                                                 Ilim[i], i, lquantumnumber[i])
        else:
            string = "Ilim[{0:n}] = {1:.4f}".format(i,Ilim[i])
        print(string)
        line.append(string)
    if printUialphabarmatrx == True:
        for i in range(1, nchann+1):
            for j in range(1, nchann+1):
                m = m + 1
                string = "Uialphabarmatrx[{0:n},{1:n}] = {2: .8f}".format(i, \
                                                j, Uialphabarmatrx_s[i-1, j-1])
                print(string)
                line.append(string)
#
    if file != None:
        with open(file, "a") as f: 
            f.write("\n"+"Channel parameters\n"+"------------------\n")
            for i in range (0, m):
                f.write(line[i]+"\n")
            f.write("\n")
        f.close()
        
###############################################################################

def print_mqdtparams(*params_in, file=None):

    """
    Prints out the elements of the K^(0) and K^(1) matrices (for calculations
    in the K-matrix formalism) or the eigen quantum defects and rotation
    angles (for calculations in the eigenchannel formalism) corresponding to
    a given list of MQDT parameters.

    The actual work is done either by print_mqdtparams_Kmatrix or by
    print_mqdtparams_eigenchannel.

    Args:
        params_in (one or several 1D arrays[float]): The parameters of the
            models considered, in the 1D format used, e.g., by the
            functions mqdtparams_Kmatrix or mqdtparams_eigenchannel.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

    if option_description == "E":
        print_mqdtparams_eigenchannel(*params_in, file=None)
    elif option_description == "K":
        print_mqdtparams_Kmatrix(*params_in, file=None)
    else:
        sys.exit("print_mqdtparams: illegal value of option_description")

        
###############################################################################

def print_mqdtparams_Kmatrix(*params_in, file=None):

    """
    Prints out the elements of the K^(0) and K^(1) matrices corresponding to
    a given list of MQDT parameters.

    Args:
        params_in (one or several 1D arrays[float]): The parameters of the
            models considered, in the 1D format used, e.g., by the
            functions mqdtparams_Kmatrix.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

    if option_description != "K":
        sys.exit("print_mqdtparams_Kmatrix is invoked out of context.")
    nsets = len(params_in)
    K0matrices = np.zeros((nchann+1, nchann+1, nsets))
    K1matrices = np.zeros((nchann+1, nsets))
    kvar = -1
    kstat = nvarparams-1
#
    for i in range (1, nchann+1):
        for j in range (i, nchann+1):
            if flvarK0[i,j] == True:
                kvar = kvar + 1
                for k in range(nsets):
                    K0matrices[i,j,k] = params_in[k][kvar]
            else:
                kstat = kstat + 1
                for k in range(nsets):
                    K0matrices[i,j,k] = params_in[k][kstat]
        if flvarK1[i] == True:
            kvar = kvar + 1
            for k in range(nsets):
                K1matrices[i, k] = params_in[k][kvar]
        else:
            kstat = kstat + 1
            for k in range(nsets):
                K1matrices[i, k] = params_in[k][kstat]
#
    print("\n"+"MQDT parameters\n"+"---------------")
    line = []
    m = 0
    for i in range (1, nchann+1):
        for j in range (i, nchann+1):
            m = m + 1
            string = "K0matrx[{0:n}, {1:n}] =".format(i, j)
            for k in range(nsets):
                string = string + "  {0: .8e}".format(K0matrices[i, j, k]) 
            print(string)
            line.append(string)
    for i in range (1, nchann+1):
        m = m + 1
        string = "K1matrx[{0:n}]    =".format(i)
        for k in range(nsets):
            string = string + "  {0: .8e}".format(K1matrices[i, k]) 
        print(string)
        line.append(string)
#
    if file != None:
        with open(file, "a") as f: 
            f.write("\n"+"MQDT parameters\n"+"---------------\n")
            for i in range (0, m):
                f.write(line[i]+"\n")
            f.write("\n")
        f.close()
        
###############################################################################

def print_mqdtparams_eigenchannel(*params_in, file=None):

    """
    Prints out the values of mu^(0), mu^(1) and theta corresponding to
    a given list of MQDT parameters.

    Args:
        params_in (one or several 1D arrays[float]): The parameters of the
            models considered, in the 1D format used, e.g., by the
            functions mqdtparams_eigenchannel.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

    if option_description != "E":
        sys.exit("print_mqdtparams_eigenchannel is invoked out of context.")
    nsets = len(params_in)
    mu0s = np.zeros((nchann+1, nsets))
    mu1s = np.zeros((nchann+1, nsets))
    thetas = np.zeros((nrotations+1, nsets))
    rot = np.zeros((nrotations+1, 2))
    kvar = -1
    kstat = nvarparams-1
#
    for i in range (1, nchann+1):
        kvar = kvar + 1
        for k in range(nsets):
            mu0s[i, k] = params_in[k][kvar]
    for i in range (1, nchann+1):
        if flvarmu1[i] == True:
            kvar = kvar + 1
            for k in range(nsets):
                mu1s[i, k] = params_in[k][kvar]
        else:
            kstat = kstat + 1
            for k in range(nsets):
                mu1s[i, k] = params_in[k][kstat]
    for i in range (1, nrotations+1):
        kvar = kvar + 1
# Store the angles in reverse order:
        for k in range(nsets):
            thetas[nrotations+1-i, k] = params_in[k][kvar] 
        rot[nrotations+1-i][0] = rotations[i-1][0]
        rot[nrotations+1-i][1] = rotations[i-1][1]
#
    print("\n"+"MQDT parameters\n"+"---------------")
    line = []
    m = 0
    for i in range (1, nchann+1):
        m = m + 1
        string = "mu0[{0:n}] =".format(i)
        for k in range(nsets):
            string = string + "  {0: .8e}".format(mu0s[i, k])
        print(string)
        line.append(string)
    for i in range (1, nchann+1):
        m = m + 1
        string = "mu1[{0:n}] =".format(i)
        for k in range(nsets):
            string = string + "  {0: .8e}".format(mu1s[i, k])
        print(string)
        line.append(string)
    for i in range (1, nrotations+1):
        m = m + 1
        string = "rotation {0:n}: ({1:n}, {2:n}), theta =".\
                     format(i, rot[i, 0], rot[i, 1])
        for k in range(nsets):
            string = string + "  {0: .8e}".format(thetas[i, k])
        print(string)
        line.append(string)
#
    if file != None:
        with open(file, "a") as f:
            f.write("\n"+"MQDT parameters\n"+"---------------\n")
            for i in range (0, m):
                f.write(line[i]+"\n")
            f.write("\n")
        f.close()

###############################################################################

def LuFano_plot(params_in, nujmin, nujmax, nujstep, nukmin, nukmax, file=None):

    """
    Draws a Lu-Fano plot.

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.
        nujmin (float): The smallest value of nu_j to be considered.
        nujmax (float): The largest value of nu_j to be considered.
        nujstep (float): The step in nu_j used in the plot.
        nukmin (float): The smallest value of nu_k to be considered.
        nukmax (float): The largest value of nu_k to be considered.
        file (optional, str): The name of the file in to which the plot should
            be copied, if it should be copied to a named file besides being
            displayed. It is only displayed if this file is not specified.

    The function draws a plot of nu_k vs. nu_j, with nu_j on the horizontal
    axis and nu_k on the vertical axis. The experimental data are represented
    as open circles (the experimental energies used to this effect are derived
    from the array exp_data_in previously passed to the function
    initialize_Kmatrix or initialize_eigenchannel).

    This function resets the global variables varparams, statparams and, if
    option_description == "E", Umatrx.

    Returns:
        None
    """

    global varparams, statparams
    varparams = params_in[0:nvarparams]
    statparams = params_in[nvarparams:]
# Store the U-matrix in the global variable Umatrx for calculations in the
# eigenchannel formalism:
    if option_description == "E": Umatrxfun()
    nujarr = np.array([])
    nukarr = np.array([])
    nu_j = nujmin
#
# List of the dissociation channels degenerate with channel j:
    jlist = []
    for i in range(1, nchann+1):
        if Ilim[i] == Ilim[jindx]: jlist.append(i)
#
    while nu_j <= nujmax:
        if nkchannels == 1:
            if nchann == 2:
                nu_k = Gkfun_2chann(nu_j)
            elif nchann == 3:
                nu_k = Gkfun_3chann(nu_j)
            else:
                nu_k = Gkfun(nu_j)
            nujarr = np.append(nujarr, nu_j)
            nukarr = np.append(nukarr, -nu_k)
        else:
            nuks = Gkfun_mltple(nu_j)
            for ks in range(0,nkchannels):
                nujarr = np.append(nujarr, nu_j)
                nukarr = np.append(nukarr, -nuks[ks])
        nu_j = nu_j + nujstep
#
    if nkchannels == 1:
        plt.plot(nujarr, nukarr, linewidth=1.0)
    else:
        s = [0.8 for n in range(len(nujarr))]
        plt.scatter(nujarr, nukarr, s=s)
#
    for i in range(0, nenergies):
        nujexp= np.sqrt(Rtilda/(Ilim[jindx]-exp_energies[i]))
        nukexp= np.sqrt(Rtilda/(Ilim[kindx]-exp_energies[i]))
        plt.plot(nujexp, -(nukexp%1), marker="o", color="k", markersize=6, 
                 markerfacecolor="white", markeredgewidth=1)
#
    plt.xlabel(r"nu_j")
    plt.ylabel(r"-(nu_k mod 1)")
    plt.xlim(nujmin, nujmax)
    plt.ylim(-nukmax, -nukmin)
    if file != None:
       plt.savefig(file, format='PDF')
    plt.show()

###############################################################################

def mixing_coeffs(params_in, E, recpl, with_dbydE):

    """
    Returns the channel fractions and Z-coefficients for a state of given
    energy.  (The actual calculation is done either by the function
    mixing_coeffs_eigenchannel or the function mixing_coeffs_Kmatrix.)

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.
        E (float): The energy of the state, in inverse centimeters.
        recpl (bool): A flag determining whether the coefficients are to be
            transformed from the dissociation channel coupling scheme
            (i-scheme) to the intermediate coupling scheme (alphabar-scheme).
            They are transformed only if recpl == True.
        with_dbydE (bool): A flag determining whether the dependence in energy
            of the MQDT parameters should (True) or should noa (False) be
            taken into account in the normalisation of the Z-coefficients.

    This function does not reset global variables.

    Returns:
        Two 1D arrays of floating points numbers, respectively the channel
            fractions and the Z-coefficients.
    """

    if option_description == "E":
        return(mixing_coeffs_eigenchannel(params_in, E, recpl, with_dbydE))
    elif option_description == "K":
        return(mixing_coeffs_Kmatrix(params_in, E, recpl, with_dbydE))
    else:
        sys.exit("mixing_coeffs: illegal value of option_description")
        
###############################################################################

def mixing_coeffs_eigenchannel(params_in, E, recpl, with_dbydE):

    """
    Calculates and returns the channel fractions and Z-coefficients for a state
    of given energy (only for calculations in the eigenchannel formulation
    of MQDT).

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_eigenchannel.
        E (float): The energy of the state, in inverse centimeters.
        recpl (bool): A flag determining whether the coefficients are to be
            transformed from the dissociation channel coupling scheme
            (i-scheme) to the intermediate coupling scheme (alphabar-scheme).
            They are transformed only if recpl == True.
        with_dbydE (bool): A flag determining whether the dependence in energy
            of the MQDT parameters should (True) or should noa (False) be
            taken into account in the normalisation of the Z-coefficients.

    This function resets the global variables varparams and statparams.

    Returns:
        Two 1D arrays of floating points numbers, respectively the channel
            fractions and the Z-coefficients.
    """

    global varparams, statparams
    varparams = params_in[0:nvarparams]
    statparams = params_in[nvarparams:]
    nu_j = np.sqrt(Rtilda/(Ilim[jindx]-E))
#
# Solve the equation 
#     Sum_alpha [U_i, alpha sin(pi nu_i + pi mu_alpha)]A_alpha = 0.
# This is done by setting A_1 = 0 and finding the other A_i's by solving
# the corresponding system of inhomogeneous equations.
    matrx_whole_sin, matrx_whole_cos, mu1_arr = \
                                      mqdtmatrices_and_mu1_eigenchannel_s(nu_j)
    matrx_part = np.empty((nchann-1, nchann-1))
    a = np.empty((nchann-1))
    for im1 in range(0, nchann-1):
        a[im1] = matrx_whole_sin[im1, 0]
        for jm1 in range(0, nchann-1):
            matrx_part[im1, jm1] = matrx_whole_sin[im1, jm1+1]
    a = -a        
    a_part = np.linalg.solve(matrx_part, a)
    Acoeffs = np.empty(nchann+1)
    for i in range(1, nchann+1):
        if i == 1:
           Acoeffs[i] = 1.
        else:
           Acoeffs[i] = a_part[i-2]
#
# Having the A-coefficients, calculate the C-coefficients. This
# is done by multiplying the part of the Acoeffs array containing
# the A-coefficients by the matrix [U_{i, alpha} cos(pi nu_i + pi mu_alpha)].
    Ccoeffs = np.zeros(nchann+1)
    Ccoeffs[1:nchann+1] = matrx_whole_cos.dot(Acoeffs[1:nchann+1])
#
# Multiply the C-coefficients by nu_i^3/2 and the sign factor, which gives
# the Z-coefficients. Calculate Z^dagger Z (i.e., Z^T Z, since Z is
# a real vector).
    Zcoeffs = np.empty(nchann+1)
    sum = 0.0
    for i in range(1, nchann+1):
        nu_i = Ffun(i, nu_j)
        Zcoeffs[i] = \
                      (-1.0)**(lquantumnumber[i]+1)*np.sqrt(nu_i)**3*Ccoeffs[i]
        sum = sum + Zcoeffs[i]**2
        if with_dbydE == True: 
            if option_Ecorrection == "A":
                sum = sum - 2.0*mu1_arr[i]*Acoeffs[i]**2
            else:
                sum = sum - 2.0*mu1_arr[i]*(Rtilda/Is)*Acoeffs[i]**2
#
# Normalise the Z-coefficients (sum includes the energy correction if
# including this correction is required):
    Zcoeffs = Zcoeffs/np.sqrt(sum)
#
# Transform from the dissociation channel coupling scheme (i-scheme) to
# the intermediate coupling scheme (alphabar-scheme) if required. This is done
# by multiplying the parts of the arrays containing the coefficients by
# the transpose of the nchann by nchann transformation matrix contained
# in Uialphabarmatrx_s.
    if recpl == True:
        Zcoeffs[1:nchann+1] = np.transpose(Uialphabarmatrx_s).\
                                                       dot(Zcoeffs[1:nchann+1])
#
# In our definition, the channel fraction are taken to be given by the
# squares of the elements of Zcoeffs:
    channelfractions = np.empty(nchann+1)
    for i in range(1, nchann+1):
        channelfractions[i] = Zcoeffs[i]**2
#
    return(channelfractions, Zcoeffs)

###############################################################################

def dKbydE():

    """
    Calculates and returns the diagonal matrix dK/dE appearing in the Kmatrix
    calculation of mixing coefficients.

    Args:
        None

    This function does not reset global variables.

    Returns:
        dKbydE_diag (1D array[float]): At return, dKbydE_diag[i] is the (i,i)
            element of dK/dE.
    """

    dKbydE_diag = np.zeros(nchann)
    kvar = -1
    kstat = -1
    for i in range (1, nchann+1):
        for j in range (i, nchann+1):
            if flvarK0[i,j] == True:
                kvar = kvar + 1
            else:
                kstat = kstat + 1
        if flvarK1[i] == True:
            kvar = kvar + 1
            K1 = varparams[kvar]
        else:
            kstat = kstat + 1
            K1 = statparams[kstat]
        if option_Ecorrection == "A":
            dKbydE_diag[i-1] = -K1
        else:
            dKbydE_diag[i-1] = -Rtilda*K1/Is
    return(dKbydE_diag)

###############################################################################

def mixing_coeffs_Kmatrix(params_in, E, recpl, with_dbydE):

    """
    Calculates and returns the channel fractions and Z-coefficients for a state
    of given energy (only for calculations in the K-matrix formulation
    of MQDT).

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix.
        E (float): The energy of the state, in inverse centimeters.
        recpl (bool): A flag determining whether the coefficients are to be
            transformed from the dissociation channel coupling scheme
            (i-scheme) to the intermediate coupling scheme (alphabar-scheme).
            They are transformed only if recpl == True.
        with_dbydE (bool): A flag determining whether the dependence in energy
            of the MQDT parameters should (True) or should noa (False) be
            taken into account in the normalisation of the Z-coefficients.

    This function resets the global variables varparams and statparams.

    Returns:
        Two 1D arrays of floating points numbers, respectively the channel
            fractions and the Z-coefficients.
    """

    global varparams, statparams
    varparams = params_in[0:nvarparams]
    statparams = params_in[nvarparams:]
    nu_j = np.sqrt(Rtilda/(Ilim[jindx]-E))
#
# Solve the equation Sum_i [K_i, alpha + delta_i, alpha tan(pi nu_i)]c_i = 0.
# This is done by setting c_1 = 0 and finding the other c_i's by solving
# the corresponding system of inhomogeneous equations. The results are then
# divided by cos(pi nu_i), which gives the C-coefficients.
    nu_k = Ffun(kindx, nu_j)
    matrx_whole = mqdtmatrx_Kmatrix_s(nu_j, nu_k, fl_no_tan_nu_k=False)
    matrx_part = np.empty((nchann-1, nchann-1))
    c = np.empty((nchann-1))
    for im1 in range(0, nchann-1):
        c[im1] = matrx_whole[im1, 0]
        for jm1 in range(0, nchann-1):
            matrx_part[im1, jm1] = matrx_whole[im1, jm1+1]
    c = -c        
    c_part = np.linalg.solve(matrx_part, c)
    Ccoeffs = np.empty(nchann+1)
    for i in range(1, nchann+1):
        nu_i = Ffun(i, nu_j)
        if i == 1:
            Ccoeffs[i] = 1./np.cos(np.pi*nu_i)
        else:
            Ccoeffs[i] = c_part[i-2]/np.cos(np.pi*nu_i)
#
# Multiply the C-coefficients by nu_i^3/2 and the sign factor, which gives
# the Z-coefficients. Calculate Z^dagger Z (i.e., Z^T Z, since Z is
# a real vector).
    Zcoeffs = np.empty(nchann+1)
    sum = 0.0
    for i in range(1, nchann+1):
        nu_i = Ffun(i, nu_j)
        Zcoeffs[i] = \
                      (-1.0)**(lquantumnumber[i]+1)*np.sqrt(nu_i)**3*Ccoeffs[i]
        sum = sum + Zcoeffs[i]**2
#
# Normalise the Z-coefficients so that their squares sum to 1 after
# normalisation. This is the default normalisation.
    Zcoeffs = Zcoeffs/np.sqrt(sum)
#
# If required, take the dependence of the K-matrix on energy into account in
# the normalisation of the Z-coefficients. The calculation assumes that only
# the diagonal elements of the K-matrix depend on the energy, so that dK/dE
# is a diagonal matrix.
    if with_dbydE == True:
        dKbydE_diag = dKbydE()
        sum = 0.0
        for i in range(1, nchann+1):
            nu_i = Ffun(i, nu_j)
            q_i = -np.sqrt(2./(np.pi*nu_i**3))*np.cos(np.pi*nu_i)
            dKbydE_i = dKbydE_diag[i-1]
            sum = sum + Zcoeffs[i] * (1. + q_i*dKbydE_i*q_i) * Zcoeffs[i]
        Zcoeffs = Zcoeffs/np.sqrt(sum)
#
# Transform from the dissociation channel coupling scheme (i-scheme) to
# the intermediate coupling scheme (alphabar-scheme) if required. This is done
# by multiplying the parts of the arrays containing the coefficients by
# the nchann by nchann transformation matrix contained in Uialphabarmatrx_s.
    if recpl == True:
        Zcoeffs[1:nchann+1] = np.transpose(Uialphabarmatrx_s).\
                                                       dot(Zcoeffs[1:nchann+1])
#
# In our definition, the channel fraction are taken to be given by the
# squares of the elements of Zcoeffs:
    channelfractions = np.empty(nchann+1)
    for i in range(1, nchann+1):
        channelfractions[i] = Zcoeffs[i]**2
#
    return(channelfractions, Zcoeffs)

###############################################################################

def plot_channelfractions(params_in, emin, emax, ymax, exp_data_in, *indx,
                          recouple=None, with_dbydE=None, file=None):
    
    """
    Plots channel fractions.

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.
        emin (float): The smallest energy to be considered.
        emax (float): The largest energy to be considered.
        ymax (float): The upper limit of the range of channel fractions 
            appearing in the plot (the value of ymax does control the
            scale of the vertical axis).
        exp_data_in (1D array([float], [float])): The experimental data from
            which the experimental energies to be included in the plot should
            be extracted. This array is expected to be identical or to be
            a subset of the array exp_data_in previously passed to the
            function initialize_Kmatrix or initialize_eigenchannel.
        indx (one or several integers): The value(s) of i or alphabar for
            which channel fractions should be plotted.
        recouple (optional, bool): A flag determining whether the plotted
            channel fractions are calculated from mixing coefficients
            transformed from the dissociation channel coupling scheme
            (i-scheme) to the intermediate coupling scheme (alphabar-scheme).
            These coefficients are transformed if recouple == True.
            They are not transformed if recouple == False or the value of
            recouple is not specified.
        with_dbydE (optional, bool): A flag determining whether the dependence
            in energy of the MQDT parameters should (True) or should not
            (False) be taken into account in the normalisation of the
            Z-coefficients (hence the channel fractions). The default is False.
        file (optional, str): The name of the file in to which the plot should
            be copied, if it should be copied to a named file besides being
            displayed. It is only displayed if this file is not specified.

    This function draws a plot of channel fractions vs. energy (in inverse
    centimeters). The experimental energies are represented by vertical dashed
    lines. Up to four values of i or alphabar can be specified under indx,
    as a comma-separated list. The respective channel fractions are
    represented, by open circles, filled circles, open squares and filled   
    squares.
    
    This function does not reset global variables.

    Returns:
        None
    """

    nen, exp_en, exp_err = sort_exp_data(exp_data_in)
    nindices = len(indx)
    if nindices > 4:
        print("plot_channelfractions can plot results for at most")
        print("four channels")
        sys.exit()
    if recouple == None or recouple == True:
        recpl = True
    elif recouple == False:
        recpl = False
    else:
        print("Illegal value of recouple")
        sys.exit()
    if with_dbydE == None or with_dbydE == False:
        wdbdE = False
    elif with_dbydE == True:
        wdbdE = True
    else:
        print("Illegal value of with_dbydE")
        sys.exit()
#
# Start by indicating the measured energies:
    for i in range(0, len(exp_en)):
        plt.axvline(x=exp_en[i], color="grey", ls="--")
#
# Then plot the channel fractions:
    th_en = th_energies_fun(params_in, exp_en)
    for i in range(0, len(th_en)):
        channelfractions, Z_coeffs = mixing_coeffs(params_in, th_en[i], recpl,
                                                   wdbdE)
        for k in range(0, nindices):
            if k == 0: plt.plot(th_en[i], channelfractions[indx[k]],
                                marker="o", color="k", markersize=8, 
                                markerfacecolor="white", markeredgewidth=1)
            if k == 1: plt.plot(th_en[i], channelfractions[indx[k]],
                                marker="o", color="k", markersize=8, 
                                markerfacecolor="grey", markeredgewidth=1)
            if k == 2: plt.plot(th_en[i], channelfractions[indx[k]], 
                                marker="s", color="k", markersize=8, 
                                markerfacecolor="white", markeredgewidth=1)
            if k == 3: plt.plot(th_en[i], channelfractions[indx[k]], 
                                marker="s", color="k", markersize=8, 
                                markerfacecolor="grey", markeredgewidth=1)
# 
    plt.xlabel(r"Term energy (cm$^{-1}$)")
    plt.ylabel(r"Channel fraction")
    plt.xlim(emin, emax)
    plt.ylim(0.0, ymax)
    if file != None:
       plt.savefig(file, format='PDF')
    plt.show()

###############################################################################

def list_Zcoeffsandchannelfractions(params_in, exp_data_in, recouple=None,\
                                    with_dbydE=None, file=None):

    """
    Prints out tables of Z-coefficients and channel fractions.

    Args:
        params_in (1D array[float]): The parameters of the model in the
            1D format used, e.g., by the functions mqdtparams_Kmatrix or
            mqdtparams_eigenchannel.
        exp_data_in (1D array([float], [float])): The experimental data from
            which the experimental energies to be considered should
            be extracted. This array is expected to be identical or to be
            a subset of the array exp_data_in previously passed to the
            function initialize_Kmatrix or initialize_eigenchannel.
        recouple (optional, bool): A flag determining whether the mixing
            coefficients should first be transformed from the dissociation
            channel coupling scheme (i-scheme) to the intermediate coupling
            scheme (alphabar-scheme) before being printed out. These
            coefficients are transformed if recouple == True. They are not
            transformed if recouple == False or the value of recouple is not
            specified.
        with_dbydE (optional, bool): A flag determining whether the dependence
            in energy of the MQDT parameters should (True) or should not
            (False) be taken into account in the normalisation of the
            Z-coefficients (hence the channel fractions). The default is False.
        file (optional, str): The name of the file to which the information
            should be outputted, if it should be written to a named file
            besides being written to the standard output stream. It is
            outputted only to the latter if this file is not specified.

    This function does not reset global variables.

    Returns:
        None
    """

#
    if recouple == None or recouple == True:
        recpl = True
    elif recouple == False:
        recpl = False
    else:
        print("Illegal value of recouple")
        sys.exit()
    if with_dbydE == None or with_dbydE == False:
        wdbdE = False
    elif with_dbydE == True:
        wdbdE = True
    else:
        print("Illegal value of with_dbydE")
        sys.exit()
    nen, exp_en, exp_err = sort_exp_data(exp_data_in)
#
    th_en = th_energies_fun(params_in, exp_en)
    nenerg = len(exp_en)
    channelfractions = np.empty((nenerg, nchann+1))
    Zcoeffs = np.empty((nenerg, nchann+1))
    for i in range(0, nenerg):
        channelfractions[i], Zcoeffs[i] = mixing_coeffs(params_in, th_en[i],\
                                                        recpl, wdbdE)
#
    print("\n"+"Z-coefficients\n"+"--------------")
    line = []
    for i in range(0, nenerg):
        string = "{0:.6f} ".format(exp_en[i])
        for k in range(0, nchann):
            string = string + " {0: .4e} ".format(Zcoeffs[i][k+1])
        line.append(string)
        print(line[i])
    if file != None:
        with open(file, "a") as f: 
            f.write("\n"+"Z-coefficients\n"+"--------------\n")
            for i in range(0, nenerg):
                f.write(line[i] + "\n")
            f.write("\n")
        f.close()
#
    print("\n"+"Channel fractions\n"+"-----------------")
    line = []
    for i in range(0, nenerg):
        string = "{0:.6f} ".format(exp_en[i])
        for k in range(0, nchann):
            string = string + " {0: .4e} ".format(channelfractions[i][k+1])
        line.append(string)
        print(line[i])
    if file != None:
        with open(file, "a") as f: 
            f.write("\n"+"Channel fractions\n"+"-----------------\n")
            for i in range(0, nenerg):
                f.write(line[i] + "\n")
            f.write("\n")
        f.close( )
