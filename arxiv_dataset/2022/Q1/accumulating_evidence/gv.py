"""
Compute unknown coefficient in Gross-Vitells method from simulations
====================================================================
"""

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from model import Channels
from stats import euler_from_lambda, coeff
from data import n_pseudo, n_channels, resolutions


n_simulations = 1000


if __name__ == "__main__":

    assert n_simulations <= n_pseudo

    masses = np.linspace(140, 155, 30)
    level = 2.

    files = ["data/pseudo_channel_{}.npy".format(c) for c in range(n_channels)]
    observeds = np.array([np.load(f) for f in files])[:, :n_simulations, :]

    def iteration(i):
        channels = Channels(observeds[:, i, :], resolutions)

        test_statistic = channels.scan(masses)
        e = euler_from_lambda(test_statistic, level)
        return e

    crossings = Parallel(n_jobs=-1)(delayed(iteration)(i) for i in tqdm(range(n_simulations)))

    # save results

    N = coeff(np.mean(crossings), level, n_channels)
    print("N = ", N)

    file_name = "data/gross_vitells_coefficient"
    np.save(file_name, N)
