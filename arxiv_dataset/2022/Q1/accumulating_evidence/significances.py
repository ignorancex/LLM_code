"""
Compute significances
=====================
"""

import numpy as np

from stats import p_global, p_local, p_mc, z_from_p
from data import n_channels


observed_lambda = 5.1**2


if __name__ == "__main__":

    # Asymptotics

    pl = p_local(observed_lambda, n_channels)
    zl = z_from_p(pl)

    N = np.load("data/gross_vitells_coefficient.npy")

    pg = p_global(observed_lambda, n_channels, N)
    zg = z_from_p(pg)
    tf = pg / pl

    print("Asymptotic local p = ", pl)
    print("Asymptotic global p = ", pg)
    print("Asymptotic local significances = ", zl)
    print("Asymptotic global significance = ", zg)
    print("Asymptotic trials factor = ", tf)

    # Simulations

    test_statistic_global = np.load("data/test_statistic_global.npy")
    test_statistic_local = np.load("data/test_statistic_local.npy")

    pg = p_mc(test_statistic_global, observed_lambda)
    pl = p_mc(test_statistic_local, observed_lambda)

    zg = z_from_p(pg)
    zl = z_from_p(pl)
    tf = pg / pl

    print("MC local p = ", pl)
    print("MC global p = ", pg)
    print("MC local significances = ", zl)
    print("MC global significance = ", zg)
    print("MC trials factor = ", tf)
