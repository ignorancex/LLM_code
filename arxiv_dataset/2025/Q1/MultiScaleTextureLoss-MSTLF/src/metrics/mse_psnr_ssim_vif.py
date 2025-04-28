import numpy as np
from math import log10
import piq
from scipy import fftpack
from skimage.metrics import structural_similarity
from sewar.full_ref import vifp
import os
from piq import SSIMLoss

def mean_squared_error(x, y):
    # MSE
    return np.square(np.subtract(x, y)).mean()


def peak_signal_to_noise_ratio(x, y):
    mse = mean_squared_error(x, y)

    if mse == 0:  # MSE is zero means no noise is present in the signal. Therefore, PSNR have no importance.
        return 100

    max_pixel = 1

    # PSNR
    return 10 * log10((max_pixel ** 2) / mse)


def structural_similarity_index(x, y):
    # SSIM
    return structural_similarity(x, y, data_range=2)



