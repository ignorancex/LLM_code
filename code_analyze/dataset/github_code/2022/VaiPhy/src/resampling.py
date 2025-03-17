import os
import sys
sys.path.append(os.getcwd())

import numpy as np


def categorical_sampling(p):
    x = np.random.rand(1)
    bins = np.cumsum(p)
    return np.digitize(x, bins)[0]


def multinomial_resampling(ws, size=0):
    if size > 0:
        u = np.random.rand(size)
    else:
        u = np.random.rand(*ws.shape)
    bins = np.cumsum(ws)
    return np.digitize(u, bins)


def stratified_resampling(ws, size=0):
    # Determine number of elements
    if size > 0:
        N = size
    else:
        N = len(ws)
    u = (np.arange(N) + np.random.rand(N)) / N
    bins = np.cumsum(ws)
    return np.digitize(u, bins)


def systematic_resampling(ws, size=0):
    if size > 0:
        N = size
    else:
        N = len(ws)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    ind = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(ws)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            ind[i] = j
            i += 1
        else:
            j += 1
    return ind
