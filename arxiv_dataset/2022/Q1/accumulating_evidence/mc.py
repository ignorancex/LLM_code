"""
MC simulations of test-statistic
================================
"""

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from model import Channels
from data import n_pseudo, n_channels, resolutions


n_simulations = 100000


if __name__ == "__main__":

    assert n_simulations <= n_pseudo

    masses = np.linspace(140, 155, 15)

    files = ["data/pseudo_channel_{}.npy".format(c) for c in range(n_channels)]
    observeds = np.array([np.load(f) for f in files])[:, :n_simulations, :]

    def iteration(i):
        channels = Channels(observeds[:, i, :], resolutions)
        global_ = channels.fit(masses)
        local = channels.fixed_mass_fit(151.)
        return (local, global_)

    pairs = Parallel(n_jobs=-1)(delayed(iteration)(i) for i in tqdm(range(n_simulations)))
    test_statistic_local, test_statistic_global = zip(*pairs)

    # save results

    file_name = "data/test_statistic_global"
    np.save(file_name, test_statistic_global)

    file_name = "data/test_statistic_local"
    np.save(file_name, test_statistic_local)
