import matlab.engine
import time
import os
import numpy as np
from math import pi
import h5py
import argparse
from scipy.stats import qmc
import multiprocessing as mp
import sys
import joblib
import scipy.io
import scipy.interpolate
import scipy.io as sio
import re
from itertools import product
from scipy.ndimage import shift

def Interpol_FA_trains_for_opti(FA_values, m):
    """
    Generate FA trains of n_pulse by interpolation on m chosen values specified in list FA_values
    :param FA_values: list of len(m) of desired FA_values on which interpolate
    :return: numpy array of size (nb_pulses, 1) representing the desired FA_train
    """
    assert len(FA_values) == m, "number of chosen FA values should be equal to m"
    x = np.linspace(0, 150, m)
    angle = FA_values
    x.sort()
    cs = scipy.interpolate.CubicSpline(x, angle, bc_type='clamped')
    x_new = np.linspace(0, 150, 150)
    alpha_train = cs(x_new)
    return alpha_train


############################# USEFUL FUNCTIONS ###############################
def normalize_signals_from_matrix(matrix, norm, axis=1):
    if norm == 'L1':
        matrix_normalized = matrix / np.linalg.norm(matrix, 1, axis=axis, keepdims=True)
    elif norm == 'L2':
        matrix_normalized = matrix / np.linalg.norm(matrix, 2, axis=axis, keepdims=True)
    elif norm == 'max':
        matrix_normalized = matrix / np.max(matrix, axis=axis, keepdims=True)
    elif norm == 'minmax':
        matrix_normalized = matrix - np.amin(matrix, axis=axis, keepdims=True)
        matrix_normalized = matrix_normalized / np.amax(matrix_normalized, axis=axis, keepdims=True)
    elif norm == 'stand':
        matrix_normalized = matrix - np.mean(matrix, axis=axis, keepdims=True)
        matrix_normalized = matrix_normalized / np.std(matrix, axis=axis, keepdims=True)
    return matrix_normalized


def lorentzian(x, df, gamma):
    return (2. / (pi * gamma)) / (1. + ((x - df) / (gamma / 2.)) ** 2.)


def lorentzian_matrix(df0, gamma_list, df_list):
    # Put each of these vectors into its own dimension to return an array of size ngamma x ndf0 x ndf_list
    gamma_list = gamma_list.reshape((-1, 1, 1))
    df0 = df0.reshape((1, -1, 1))
    df_list = df_list.reshape((1, 1, -1))
    matrix = lorentzian(df_list, df0, gamma_list)

    return matrix


def calculate_fwhm(values, signal_data):
    half_max = np.max(signal_data) / 2
    # Find indices where signal crosses half-maximum
    above_half_indices = np.where(signal_data >= half_max)[0]
    # Calculate FWHM in terms of the provided 'values'
    fwhm = values[above_half_indices[-1]] - values[above_half_indices[0]]
    #     plt.plot(values, signal_data)
    #     plt.plot(values, np.ones(signal_data.shape) *half_max)
    #     plt.plot(values[above_half_indices[-1]], signal_data[above_half_indices[-1]], 'ro')
    #     plt.plot(values[above_half_indices[0]], signal_data[above_half_indices[0]], 'ro')
    return fwhm


def initseq(T1values, T2values, B1values, dfvalues, FA_train, TR_train, TE_train, phi_train, n_echoes, spoil, invpulse,
            inv_time, grid):
    """
    initialize seq, make parameters as appropriate type for Matlab call
    :param T1values: list of T1 values in seconds
    :param T2values: list of T2 values in seconds
    :param B1values: list of B1 sensitiity values (between 0 and 2)
    :param dfvalues: list of B0 offset values in Hz
    :param FA_train: (1, N) in deg
    :param TR_train: (1, N) in ms
    :param TE_train: (1, N) in ms
    :param phi_train: (1, N) in deg
    :param n_echoes: number of echoes in multiechoes sequence
    :param spoil: spoiling type, 1 for spoiled, 0 for balanced
    :param invpulse: 0 for no inv pulse, 1 for inv pulse
    :return: matlab sequence and model parameters
    """
    if invpulse == 1:
        FA_train = np.insert(FA_train, 0, 0, axis=1)
        phi_train = np.insert(phi_train, 0, 0, axis=1)
        TR_train = np.insert(TR_train, 0, inv_time, axis=1)
        TE_train = np.insert(TE_train, 0, TE_train[0, 0], axis=1)
    N = FA_train.shape[1]  # pulses
    fa_train_rad = FA_train * (pi / 180)
    phi_rad = phi_train * (pi / 180)
    Np = matlab.double([[N]])
    n_echoes = matlab.double([[n_echoes]])
    fa_train_rad = matlab.double(fa_train_rad.tolist())
    phi_rad = matlab.double(phi_rad.tolist())
    TR_train = TR_train * 1e-3  # s
    TR = matlab.double(TR_train.tolist())
    TE_train = TE_train * 1e-3
    TE = matlab.double(TE_train.tolist())

    Tacq = np.concatenate((TE_train[0:1, 0:1], TE_train[0:1, 1:] + np.cumsum(TR_train[0:1])[:-1]), axis=1)

    if grid == 'regular' or grid == 'custom':
        T1listg, T2listg, B1listg, dflistg = np.meshgrid(T1values, T2values, B1values, dfvalues)
        T1list_tmp = T1listg.flatten()
        T2list_tmp = T2listg.flatten()
        B1list_tmp = B1listg.flatten()
        dflist_tmp = dflistg.flatten()

    elif grid == 'sobol' or grid == 'compartments':
        T1list_tmp = T1values
        T2list_tmp = T2values
        B1list_tmp = B1values
        dflist_tmp = dfvalues

    T1list = T1list_tmp[T1list_tmp > T2list_tmp]
    T2list = T2list_tmp[T1list_tmp > T2list_tmp]
    B1list = B1list_tmp[T1list_tmp > T2list_tmp]
    dflist = dflist_tmp[T1list_tmp > T2list_tmp]

    nb_signals = T1list.shape[0]

    spoil = matlab.double([[spoil]])  # 0 for 'bSSFP', 1 for 'FID'

    invPulse = matlab.double([[invpulse]])  # 0 for no inv pulse, 1 for inv pulse

    return FA_train, phi_train, Tacq, TR_train, T1list, T2list, B1list, dflist, N, Np, fa_train_rad, phi_rad, TR, TE, \
        spoil, invPulse, invpulse, nb_signals, n_echoes


def dico_based(T1values, T2values, B1values, dfvalues, FA_train, TR_train, TE_train, phi_train, n_echoes, spoil, vasc,
               invpulse, inv_time, grid, eng):
    """ generated based dico of parameters (n,𝜹f,T1,T2)
    initialize seq, make parameters as appropriate type for Matlab call
    :param T1values: list of T1 values in seconds
    :param T2values: list of T2 values in seconds
    :param B1values: list of B1 sensitiity values (between 0 and 2)
    :param dfvalues: list of B0 offset values in Hz
    :param FA_train: (1, N) in deg
    :param TR_train: (1, N) in ms
    :param TE_train: (1, N) in ms
    :param phi_train: (1, N) in deg
    :param n_echoes: number of echoes in multiechoes sequence
    :param spoil: spoiling type, 1 for spoiled, 0 for balanced
    :param vasc: boolean for vascular parameters or not
    :param invpulse: 0 for no inv pulse, 1 for inv pulse
    :param inv_time: time to wait after inversion
    :return: matlab sequence and model parameters
    """

    FA_train, phi_train, Tacq, TR_train, T1list, T2list, B1list, dflist, N, Np, fa_train_rad, phi_rad, TR, TE, \
        spoil, invPulse, invpulse, nb_signals, N_echoes = initseq(T1values, T2values, B1values, dfvalues,
                                                                  FA_train,
                                                                  TR_train, TE_train, phi_train, n_echoes,
                                                                  spoil, invpulse, inv_time, grid)

    if vasc:
        nb_parameters = 8  # T1, T2, df, gamma, B1, SO2, Vf, R
    elif grid == 'compartments':
        nb_parameters = 4  # T1, T2
    else:
        nb_parameters = 5  # T1, T2, df, gamma, B1

    Param = np.zeros((nb_signals, nb_parameters))

    if grid == 'compartments':
        Param[:, 0] = T1list
        Param[:, 1] = T2list
        Param[:, 2] = dflist
        Param[:, 3] = B1list
    else:
        Param[:, 0] = T1list
        Param[:, 1] = T2list
        Param[:, 2] = dflist
        Param[:, 4] = B1list
    # Param = Param[Param[:,0,0]>Param[:,0,1]]
    Sequence = np.zeros((N, 4))
    print(N, 'pulses')
    Sequence[:, 0] = FA_train
    Sequence[:, 1] = Tacq
    Sequence[:, 2] = TR_train
    Sequence[:, 3] = phi_train
    # 4th dim of Param is reserved for gamma for T2star calculations
    Mag = np.zeros((nb_signals, N), dtype=np.complex_)
    batch = 10000
    for i in range(int(nb_signals / batch)):
        T1list_tmp = T1list[i * batch: (i + 1) * batch]
        T2list_tmp = T2list[i * batch: (i + 1) * batch]
        B1list_tmp = B1list[i * batch: (i + 1) * batch]
        dflist_tmp = dflist[i * batch: (i + 1) * batch]
        T1list_ml = matlab.double(T1list_tmp.flatten().tolist())
        T2list_ml = matlab.double(T2list_tmp.flatten().tolist())
        B1list_ml = matlab.double(B1list_tmp.flatten().tolist())
        dflist_ml = matlab.double(dflist_tmp.flatten().tolist())
        test = np.array((eng.BlochOverlay_python(Np, fa_train_rad, phi_rad, TR, TE, T1list_ml, T2list_ml, B1list_ml,
                                                 dflist_ml, spoil, invPulse, N_echoes)))
        Mag[i * batch: (i + 1) * batch, :] = test
    # simulate for the lasts signals
    lasts = nb_signals % batch
    if lasts != 0:
        T1list_tmp = T1list[nb_signals - lasts: nb_signals]
        T2list_tmp = T2list[nb_signals - lasts: nb_signals]
        B1list_tmp = B1list[nb_signals - lasts: nb_signals]
        dflist_tmp = dflist[nb_signals - lasts: nb_signals]
        T1list_ml = matlab.double(T1list_tmp.flatten().tolist())
        T2list_ml = matlab.double(T2list_tmp.flatten().tolist())
        B1list_ml = matlab.double(B1list_tmp.flatten().tolist())
        dflist_ml = matlab.double(dflist_tmp.flatten().tolist())
        test = np.array((eng.BlochOverlay_python(Np, fa_train_rad, phi_rad, TR, TE, T1list_ml, T2list_ml, B1list_ml,
                                                 dflist_ml, spoil, invPulse, N_echoes)))
        Mag[nb_signals - lasts: nb_signals, :] = test
    # some shape fixing
    if invpulse == 1:
        Mag = Mag[:, 1:, np.newaxis]
    else:
        Mag = Mag[:, :, np.newaxis]
    Data = {'parameters': Param, 'sequence': Sequence}
    return Data, Mag


def generate_gamma_dico_par(Data, Mag, gammalist):
    """ Expand a (n,𝜹f,T1,T2) based dico into a (n,𝜹f,T1,T2,Γ) expanded dico
    :param Data:
    :param Mag:
    :param gammalist:
    :return:
    """
    Data1 = Data.copy()
    Param = Data1['parameters']
    dflist = Param[:, 2]
    # ensure the lorentzian is fully represented for a center d0:
    min_df, max_df, max_gamma = min(dflist), max(dflist), max(gammalist)
    assert max_gamma < abs(max_df - min_df), 'gamma is too large for this df distribution'
    #max_gamma += 10
    mask = (min_df + max_gamma < dflist) & (dflist < max_df - max_gamma)
    dflist_reduced = np.unique(dflist[mask])
    dflist = np.unique(dflist)

    # Compute Lorenzian matrix, for all center dfs (dflist_reduced), and all gamma values.
    matrix = lorentzian_matrix(dflist_reduced, gammalist, dflist)  # size [ngamma, ndf_reduced, ndf]
    matrix = normalize_signals_from_matrix(matrix, 'L1', axis=2)  # normalize weights to sum 1 in frequency direction
    # expand an n dims dictionnary into an n+1 dims where the (N+1)th dimension is the value of Γ such as :
    # (1/T2*) = (1/T2) + Γ*pi
    ndf, ngamma, nte, npar = dflist.size, gammalist.size, Mag.shape[1], Param.shape[1]
    Param = np.tile(Param[mask], (ngamma + 1, 1, 1))  # add a new gamma dimension (plus one for original data)
    for i_g in range(ngamma):  # parcours gamma value that represent FWHM value
        Param[i_g + 1, :, 3] = gammalist[i_g]
    Param = Param.reshape((-1, npar))
    Mag = Mag.reshape((-1, ndf, nte))  # split df dimension from others (T1, T2...) for matrix multiplication
    # convolution by Lorentzian
    new_signal_pond = np.tensordot(matrix, Mag, axes=(2, 1)).swapaxes(1, 2).reshape((-1, nte, 1))
    Mag = Mag.reshape((-1, nte, 1))  # go back to original size
    debug = Mag[mask]
    Mag = np.concatenate((Mag[mask], new_signal_pond))
    Data1['parameters'] = Param
    return Data1, Mag


def apply_noise_aliasing(matrix_of_signals, SNR):
    mu = 0
    sigma = np.sqrt((abs(matrix_of_signals) ** 2) / SNR)
    noise = np.random.normal(mu, sigma, size=(matrix_of_signals.shape[0], matrix_of_signals.shape[1]))
    return matrix_of_signals + noise


def apply_noise2(matrix_of_signals, SNR):
    noise = np.random.randn(matrix_of_signals.shape[0], matrix_of_signals.shape[1])
    noise = noise * (np.mean(abs(matrix_of_signals)) / SNR)
    return matrix_of_signals + noise


def get_T2star(T2, gamma):
    return 1 / ((1 / T2) + pi * gamma)


def do_matching(e, monte_carlo, pred_norm, nb_parameters, param_list, param_values, monte_carlo_index,
                range_param):
    if nb_parameters == 3:
        error = {'T1': np.zeros((50, 2000)), 'T2': np.zeros((50, 2000)), 'B1': np.zeros((50, 2000))}
    elif nb_parameters == 6:
        error = {'gamma': np.zeros((50, 2000)), 'df': np.zeros((50, 2000)), 'T1': np.zeros((50, 2000)),
                 'T2': np.zeros((50, 2000)), 'T2star': np.zeros((50, 2000)), 'B1': np.zeros((50, 2000))}

    if e > 0:
        sys.stdout.write("\r{0}".format(e))
    SNR = 4
    noised_monte_carlo = apply_noise2(monte_carlo, SNR)
    scoring = np.matmul(noised_monte_carlo, pred_norm.T)  # dot product : matrix of size (50,nb_signals)
    index_match = []
    for i in range(noised_monte_carlo.shape[0]):
        index_match.append(np.where(scoring[i, :] == np.amax(scoring[i, :]))[0][0])
    for j in range(nb_parameters):
        error[param_list[j]] = (
                abs(param_values[param_list[j]][monte_carlo_index][:, 0] - param_values[param_list[j]][index_match][
                                                                           :, 0]) /
                (range_param[param_list[j]][1] - range_param[param_list[j]][0]))
    return error


def MonteCarlo(Data, Mag, mode):
    if mode == 'FISP':
        mode = 'classic'
    else:
        mode = 'expanded'
    print("starting MC process...")
    Mag = abs(Mag)
    Param = Data
    nb_pulses = Mag.shape[1]
    prediction = Mag
    param_list = ['T1', 'T2', 'B1']
    T1_param = np.zeros((Param[:, 0].shape[0], nb_pulses))
    T2_param = np.zeros((Param[:, 1].shape[0], nb_pulses))
    B1_param = np.zeros((Param[:, 4].shape[0], nb_pulses))
    if mode == "classic":
        for i in range(nb_pulses):
            T1_param[:, i] = Param[:, 0]
            T2_param[:, i] = Param[:, 1]
            B1_param[:, i] = Param[:, 4]
        param_values = {'T1': T1_param, 'T2': T2_param, 'B1': B1_param}
        range_param = {'T1': [min(T1_param[:, 0]), max(T1_param[:, 0])],
                       'T2': [min(T2_param[:, 0]), max(T2_param[:, 0])],
                       'B1': [min(B1_param[:, 0]), max(B1_param[:, 0])]}
    elif mode == "expanded":
        gamma_param = np.zeros((Param[:, 3].shape[0], nb_pulses))
        df_param = np.zeros((Param[:, 2].shape[0], nb_pulses))
        T2star_param = np.zeros((Param[:, 2].shape[0], nb_pulses))
        for i in range(nb_pulses):
            T1_param[:, i] = Param[:, 0]
            T2_param[:, i] = Param[:, 1]
            gamma_param[:, i] = Param[:, 3]
            df_param[:, i] = Param[:, 2]
            B1_param[:, i] = Param[:, 4]
            T2star_param[:, i] = get_T2star(T2_param[:, i], gamma_param[:, i])
        param_list.append('gamma')
        param_list.append('df')
        param_list.append('T2star')
        param_values = {'gamma': gamma_param, 'df': df_param, 'T1': T1_param, 'T2': T2_param, 'T2star': T2star_param,
                        'B1': B1_param}
        range_param = {'gamma': [min(gamma_param[:, 0]), max(gamma_param[:, 0])],
                       'df': [min(df_param[:, 0]), max(df_param[:, 0])],
                       'T1': [min(T1_param[:, 0]), max(T1_param[:, 0])],
                       'T2': [min(T2_param[:, 0]), max(T2_param[:, 0])],
                       'T2star': [min(T2star_param[:, 0]), max(T2star_param[:, 0])],
                       'B1': [min(B1_param[:, 0]), max(B1_param[:, 0])]}

    pred_norm = normalize_signals_from_matrix(prediction[:, :, 0],
                                              'L2')  # normalized matrix of signal, shape:(nb_signals,nb_pulses)
    nparam = len(param_values
                 )
    # SOBOL DISTRIBUTION FOR MONTE CARLO 50 SIGNAL CHOICE
    print("sobol making...")
    if mode == "classic":
        l_bounds = [range_param['T1'][0], range_param['T2'][0], range_param['B1'][0]]
        u_bounds = [range_param['T1'][1], range_param['T2'][1], range_param['B1'][1]]
    elif mode == 'expanded':
        l_bounds = [range_param['gamma'][0], range_param['df'][0], range_param['T1'][0], range_param['T2'][0],
                    range_param['T2star'][0], range_param['B1'][0]]
        u_bounds = [range_param['gamma'][1], range_param['df'][1], range_param['T1'][1], range_param['T2'][1],
                    range_param['T2star'][1], range_param['B1'][1]]

    sampler = qmc.Sobol(nparam, scramble=False)

    tuplet = []
    print(l_bounds)
    print(u_bounds)
    while len(tuplet) < 50:
        sample = sampler.random()
        point = qmc.scale(sample, l_bounds, u_bounds)
        tuplet.append(point[0])

    monte_carlo_index = []
    print("matching of 50 signals...")
    for m in range(50):
        if mode == "classic":
            T1_compared, T2_compared, B1_compared = tuplet[m]
            index = np.argmin(np.sqrt(
                (T1_param[:, 0] - T1_compared) ** 2 + (T2_param[:, 0] - T2_compared) ** 2 + (
                        B1_param[:, 0] - B1_compared) ** 2))
        elif mode == "expanded":
            gamma_compared, df_compared, T1_compared, T2_compared, T2star_compared, B1_compared = tuplet[m]
            index = np.argmin(np.sqrt(
                (gamma_param[:, 0] - gamma_compared) ** 2 + (df_param[:, 0] - df_compared) ** 2 + (
                        T1_param[:, 0] - T1_compared) ** 2 + (T2_param[:, 0] - T2_compared) ** 2 + (
                        T2star_param[:, 0] - T2star_compared) ** 2 + (B1_param[:, 0] - B1_compared) ** 2))
        monte_carlo_index.append(index)
    monte_carlo_index = np.array(monte_carlo_index)
    monte_carlo = pred_norm[monte_carlo_index]
    print("error computations...")
    errors = joblib.Parallel(n_jobs=mp.cpu_count(), prefer='processes')(joblib.delayed(do_matching) \
                                                                            (e, monte_carlo, pred_norm, nparam,
                                                                             param_list, param_values,
                                                                             monte_carlo_index,
                                                                             range_param) for e in
                                                                        range(2000))

    error = errors
    if mode == "classic":
        moyennes = {'T1': 0, 'T2': 0, 'B1': 0}
        devs = {'T1': 0, 'T2': 0, 'B1': 0}
        errors_values = {'T1': [], 'T2': [], 'B1': []}
    elif mode == "expanded":
        moyennes = {'gamma': 0, 'df': 0, 'T1': 0, 'T2': 0, 'T2star': 0, 'B1': 0}
        devs = {'gamma': 0, 'df': 0, 'T1': 0, 'T2': 0, 'T2star': 0, 'B1': 0}
        errors_values = {'gamma': [], 'df': [], 'T1': [], 'T2': [], 'T2star': [], 'B1': []}

    for e in range(len(error)):
        for j in range(nparam):
            errors_values[param_list[j]].append(error[e][param_list[j]])

    for j in range(nparam):
        moyennes[param_list[j]] = np.mean(errors_values[param_list[j]]) * 100  # percent
        devs[param_list[j]] = np.std(errors_values[param_list[j]], ddof=1) * 100
    print('\n', moyennes, devs)
    return moyennes, devs, errors_values, monte_carlo_index


def generate_vasc_dico_par(Data, Mag, distribution_path, which_distrib):
    """ Expand a (n,𝜹f,T1,T2) based dico into a (n,𝜹f,T1,T2,Γ) expanded dico
    :param Data: array
    :param Mag:
    :param distribution_path: location of the folder containing the saved geometries from which we get frequencies distributions
    :param which_distrib: index into listdir of distribution_path to select a specific nb of distribs
    :return:
    """
    nb_signals_primo = Mag.shape[0];

    distribs_names = os.listdir(distribution_path)
    distribs_names = sorted(distribs_names)[which_distrib]
    Data1 = Data.copy()
    Param = Data1['parameters']
    dflist = Param[:, 2]
    gamma_list = []

    # first distrib to get infos
    distrib = sio.loadmat(distribution_path + distribs_names[0])
    values = np.concatenate(distrib['Histo'][0][0][1].T, axis=0)
    print("Loading distribution from 3D vessels geometries...")
    distrib_matrix = np.zeros((len(distribs_names), values.shape[0]))

    for distrib_name in distribs_names:
        file_name = distribution_path + distrib_name
        distrib = sio.loadmat(file_name)
        binar = np.concatenate(distrib['Histo'][0][0][0].T, axis=0)
        # ensure the lorentzian is fully represented for a center d0:
        gamma_list.append(calculate_fwhm(values, binar))
        distrib_matrix[distribs_names.index(distrib_name)] = binar

    assert max(gamma_list) < abs(
        max(dflist) - min(dflist)), 'FWHM of some distributions are larger than the df distribution'
    #mask = ((-30 < dflist) & (dflist<30))
    #mask = (min(dflist) + max(gamma_list) < dflist) & (dflist < max(dflist) - max(gamma_list))
    dflist_reduced = np.unique(dflist)
    #dflist_reduced = np.unique(dflist[mask])
    dflist = np.unique(dflist)

    # Compute Lorenzian matrix, for all center dfs (dflist_reduced), and all gamma values.
    matrix = np.zeros((len(gamma_list), dflist_reduced.shape[0], dflist.shape[0]))  # size [ngamma, ndf_reduced, ndf]
    for j in range(dflist_reduced.shape[0]):
        for i in range(len(gamma_list)):
            matrix[i, j, :] = np.roll(distrib_matrix[i], int(dflist_reduced[j]))

    matrix = normalize_signals_from_matrix(matrix, 'L1', axis=2)  # normalize weights to sum 1 in frequency direction
    # expand an n dims dictionnary into an n+1 dims where the (N+1)th dimension is the value of Γ such as :
    # (1/T2*) = (1/T2) + Γ*pi
    ndf, ngamma, nte, npar = dflist.size, len(gamma_list), Mag.shape[1], Param.shape[1]
    Param = np.tile(Param, (ngamma + 1, 1, 1))  # add a new gamma dimension (plus one for original data)
    # Param = np.tile(Param[mask], (ngamma + 1, 1, 1))  # add a new gamma dimension (plus one for original data)

    vasc_list = []
    for distrib_name in distribs_names:
        file_name = distribution_path + distrib_name
        SO2 = float(re.search(r'SO2_([0-9.e-]+)', file_name[:-4]).group(1))
        VF = float(re.search(r'VF_([0-9.e-]+)', file_name[:-4]).group(1))
        R = float(re.search(r'R_([0-9.e-]+)(?!\d|\.)', file_name[:-4]).group(1))
        vasc_list.append([SO2, VF, R])
    for i_g in range(ngamma):  # parcours gamma value that represent FWHM value
        Param[i_g + 1, :, 3] = gamma_list[i_g]
        Param[i_g + 1, :, 5:] = vasc_list[i_g]
    Param = Param.reshape((-1, npar))
    Mag = Mag.reshape((-1, ndf, nte))  # split df dimension from others (T1, T2...) for matrix multiplication
    # convolution by Lorentzian
    new_signal_pond = np.tensordot(matrix, Mag, axes=(2, 1)).swapaxes(1, 2).reshape((-1, nte, 1))
    Mag = Mag.reshape((-1, nte, 1))  # go back to original size
    Mag = np.concatenate((Mag, new_signal_pond))

    # Mag = np.concatenate((Mag[mask], new_signal_pond))
    # delete the gamma=0 part of the dico with (SO2, VF, R) = (0,0,0) that we don't want because unrealistic geometry
    Param_copy = np.delete(Param, np.s_[0:nb_signals_primo], 0)
    Mag_copy = np.delete(Mag, np.s_[0:nb_signals_primo], 0)
    dfmask = [-30,30]
    col_mask = 2
    idx_mask = np.where((Param_copy[:,col_mask]>= -30) & (Param_copy[:,col_mask]<= 30))[0]
    print('Removing', (Mag_copy.shape[0] - idx_mask.shape[0]), 'signals to ensure fully convoluted distribution')
    Mag_copy = Mag_copy[idx_mask,:]
    Param_copy = Param_copy[idx_mask,:]
    Data1['parameters'] = Param_copy
    return Data1, Mag_copy


def generate_multi_comp_dico_par(Data, Mag, inside):
    print(Mag.shape)
    print(Data['parameters'].shape)
    n_signals, n_pulses = Mag[:, :, 0].shape
    n_parameters = Data['parameters'].shape[1]
    # Generate combinations of weights (percentages)
    if inside == 'myelin' and n_signals == 3:
        weights_combinations = list(product(range(101), repeat=n_signals))
        # Filter combinations where the sum is 100
        valid_combinations = [w for w in weights_combinations if sum(w) == 100]
    elif inside == 'cbv' and n_signals == 2:
        weights_combinations = list(product(range(1001), repeat=n_signals))
        # Filter combinations where the sum is 100
        valid_combinations = [w for w in weights_combinations if sum(w) == 1000]
        valid_combinations = [[round(0.1 * w, 1) for w in weights] for weights in valid_combinations]
    elif inside == 'cbv' and n_signals == 3:
        weights_combinations = list(product(range(101), repeat=n_signals))
        # Filter combinations where the sum is 100
        valid_combinations = [w for w in weights_combinations if sum(w) == 100]
    else:
        raise ValueError('wrong number of signals or wrong compartments inside type (allowed : "myelin" or "cbv"')

    # Initialize lists to store the result
    combined_mags = np.zeros((len(valid_combinations), n_pulses, 1))
    combined_parameters = np.zeros((len(valid_combinations), n_signals))

    # Iterate through valid combinations
    for weights in valid_combinations:
        # Calculate the linear combination
        combined_mag = np.sum([w / 100 * Mag[i] for i, w in enumerate(weights)], axis=0)

        # Append the combined magnitude and associated parameters
        combined_mags[valid_combinations.index(weights), :, :] = abs(combined_mag)

        combined_parameters[valid_combinations.index(weights), 0] = weights[0]
        combined_parameters[valid_combinations.index(weights), 1] = weights[1]
        if n_signals == 3:
            combined_parameters[valid_combinations.index(weights), 2] = weights[2]
    print(combined_mags.shape)
    print(combined_parameters.shape)

    Data['parameters'] = combined_parameters
    Mag = combined_mags
    return Data, Mag


def generate_vasc_dico_par_2(Data, Mag, distrib):
    """ Expand a (n,𝜹f,T1,T2) based dico into a (n,𝜹f,T1,T2,Γ) expanded dico but load distrib from mat struct precomputed
    :param Data: array
    :param Mag:
    :param distribution_path: location of the precomputed matstruct with the histograms of the distrib
    :return:
    """
    nb_signals_primo = Mag.shape[0];
    big_structure = distrib['bigStructure']
    fn = list(big_structure.keys())

    Data1 = Data.copy()
    Param = Data1['parameters']
    dflist = Param[:, 2]
    gamma_list = []

    # first distrib to get infos
    values = big_structure[fn[0]]['histo']['values']
    print("Distributions have been load... computing....")
    distrib_matrix = np.zeros((len(fn), values.shape[0]))

    for i in range(len(fn)):
        binar = big_structure[fn[i]]['histo']['bin']
        # ensure the lorentzian is fully represented for a center d0:
        # gamma_list.append(calculate_fwhm(values, binar))
        gamma_list.append(5)
        distrib_matrix[i] = binar

    # assert max(gamma_list) < abs(
    #     max(dflist) - min(dflist)), 'FWHM of some distributions are larger than the df distribution'
    dflist_reduced1 = np.unique(dflist)
    dflist_reduced = dflist_reduced1[dflist_reduced1 == 0] #keep only 0Hz for GESFIDE experiments 
    # dflist_reduced = np.unique(dflist[mask])
    dflist = np.unique(dflist)

    # Compute Lorenzian matrix, for all center dfs (dflist_reduced), and all gamma values.
    matrix = np.zeros((len(gamma_list), dflist_reduced.shape[0], dflist.shape[0]))  # size [ngamma, ndf_reduced, ndf]
    for j in range(dflist_reduced.shape[0]):
        for i in range(len(gamma_list)):
            #matrix[i, j, :] = shift(distrib_matrix[i], dflist_reduced[j], mode='wrap')
            integer_shift = int(np.floor(dflist_reduced[j]))
            fraction = dflist_reduced[j] - integer_shift
            # Perform the integer shift
            rolled_distribution = np.roll(distrib_matrix[i], integer_shift)
            matrix[i, j, :] = (1 - fraction) * rolled_distribution + fraction * np.roll(rolled_distribution, -1)
            #matrix[i, j, :] = np.roll(distrib_matrix[i], int(dflist_reduced[j]))

    matrix = normalize_signals_from_matrix(matrix, 'L1', axis=2)  # normalize weights to sum 1 in frequency direction
    # expand an n dims dictionnary into an n+1 dims where the (N+1)th dimension is the value of Γ such as :
    # (1/T2*) = (1/T2) + Γ*pi
    ndf, ngamma, nte, npar = dflist.size, len(gamma_list), Mag.shape[1], Param.shape[1]
    Param = np.tile(Param, (ngamma + 1, 1, 1))  # add a new gamma dimension (plus one for original data)
    #Param = np.tile(Param[mask], (ngamma + 1, 1, 1))  # add a new gamma dimension (plus one for original data)

    vasc_list = []
    voxel_name = []
    for i in range(len(fn)):
        SO2 = big_structure[fn[i]]['SO2']
        VF = big_structure[fn[i]]['VF']
        R = big_structure[fn[i]]['R']
        vasc_list.append([SO2, VF, R])
        voxel_name.append(big_structure[fn[i]]['Voxel name'])
    for i_g in range(ngamma):  # parcours gamma value that represent FWHM value
        Param[i_g + 1, :, 3] = gamma_list[i_g]
        Param[i_g + 1, :, 5:] = vasc_list[i_g]
    Param = Param.reshape((-1, npar))
    Mag = Mag.reshape((-1, ndf, nte))  # split df dimension from others (T1, T2...) for matrix multiplication
    # convolution by Lorentzian
    new_signal_pond = np.tensordot(matrix, Mag, axes=(2, 1)).swapaxes(1, 2).reshape((-1, nte, 1))
    Mag = Mag.reshape((-1, nte, 1))  # go back to original size
    Mag = np.concatenate((Mag, new_signal_pond))

    # Mag = np.concatenate((Mag[mask], new_signal_pond))
    # delete the gamma=0 part of the dico with (SO2, VF, R) = (0,0,0) that we don't want because unrealistic geometry
    Param_copy = np.delete(Param, np.s_[0:nb_signals_primo], 0)
    Mag_copy = np.delete(Mag, np.s_[0:nb_signals_primo], 0)
    # keeping only df=0Hz for GESFIDE
    filtered_indices = Param_copy[:, 2] == 0
    Param_filtered = Param_copy[filtered_indices]
    Mag_filtered = Mag_copy
    Data1['parameters'] = Param_filtered
    Data1['voxellist'] = voxel_name

    return Data1, Mag_filtered

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
