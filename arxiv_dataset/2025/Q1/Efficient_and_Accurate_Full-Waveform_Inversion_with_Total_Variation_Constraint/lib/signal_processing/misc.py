from typing import Tuple

import numpy as np
from numpy import typing as npt
from scipy.ndimage import gaussian_filter, zoom


def zoom_and_crop(data: npt.NDArray, target_shape: Tuple[int, int]):
    scale = max(target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return zoom(data, scale)[: target_shape[0], : target_shape[1]]


def smoothing_with_gaussian_filter(data: npt.NDArray, n_iter: int, sigma: float):
    ret = data.copy()
    for _ in range(n_iter):
        ret = gaussian_filter(ret, sigma=sigma)
    return ret


def calc_psnr(signal0: npt.NDArray, signal1: npt.NDArray, max_value: float):
    mse = np.mean((signal0.astype(float) - signal1.astype(float)) ** 2)
    return 10 * np.log10((max_value**2) / mse)
