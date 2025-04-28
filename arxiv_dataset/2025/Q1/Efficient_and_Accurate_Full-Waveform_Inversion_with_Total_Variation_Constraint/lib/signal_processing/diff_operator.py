import numpy as np
from numpy import typing as npt


def D(x: npt.NDArray):
    diff1 = np.concatenate((x[1:], x[-1:]), axis=0) - x
    diff2 = np.concatenate((x[:, 1:], x[:, -1:]), axis=1) - x
    result = np.stack((diff1, diff2), axis=2)
    return result


def Dt(y: npt.NDArray):
    ret0 = np.concatenate([-y[0:1, :, 0], -y[1:-1, :, 0] + y[:-2, :, 0], y[-2:-1, :, 0]], axis=0)
    ret1 = np.concatenate([-y[:, 0:1, 1], -y[:, 1:-1, 1] + y[:, :-2, 1], y[:, -2:-1, 1]], axis=1)
    return ret0 + ret1
