from scipy.interpolate import griddata
import numpy as np

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft
from numpy import meshgrid
from numpy.linalg import norm
import math


sys.path.append('../sPOD/lib/')
from sPOD_tools import shifted_rPCA, shifted_POD, give_interpolation_error, build_all_frames
from transforms import transforms
from farge_colormaps import farge_colormap_multi

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16   # 16
MEDIUM_SIZE = 18   # 18
BIGGER_SIZE = 20   # 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cm = farge_colormap_multi(type='velocity', etalement_du_zero=0.02, limite_faible_fort=0.15)


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.flip(np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8))


def my_interpolated_state(Nmodes, U_list, frame_amplitude_list, mu_points, Nx, Ny, Nt, mu_vec, trafos_test):

    TA_list = []
    qtilde = 0
    nf = 0
    for U, frame_modes, amplitudes in zip(U_list, Nmodes, frame_amplitude_list):

        VT = []
        for k in range(frame_modes):
            a = griddata(mu_points, amplitudes[k].T, mu_vec, method='linear')
            VT.append(np.squeeze(a))
        VT = np.asarray(VT)
        TA_list.append(VT)
        Q = U[:, :frame_modes] @ VT
        qframe = np.reshape(Q, [Nx, Ny, 1, Nt])

        qtilde += trafos_test[nf].apply(qframe)

        nf = nf + 1

    return qtilde, TA_list


def my_interpolated_state_onlyTA(Nmodes, frame_amplitude_list, mu_points, mu_vec):

    TA_list = []
    for frame_modes, amplitudes in zip(Nmodes, frame_amplitude_list):

        VT = []
        for k in range(frame_modes):
            a = griddata(mu_points, amplitudes[k].T, mu_vec, method='linear')
            VT.append(np.squeeze(a))
        VT = np.asarray(VT)

        TA_list.append(VT)

    return TA_list


def my_delta_interpolate(delta_list, mu_points, mu_vec):
    from scipy import interpolate

    shifts_list = []
    for frame in range(len(delta_list)):
        a = griddata(mu_points, delta_list[frame].T, mu_vec, method='linear')
        a = np.asarray(a)
        a = np.reshape(a, newshape=[-1])

        shifts_list.append(a)

    return shifts_list
