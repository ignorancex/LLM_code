"""
Generate pseudo-data
====================
"""

import numpy as np
from scipy.stats import poisson

# number of pseudo-data sets and channels

n_pseudo = 100000
n_channels = 6

# detector resolution per channel

sigma_gg = 1.5
sigma_bb = 14.
resolutions = [sigma_gg] * 5 + [sigma_bb]

# dummy background model

bins = np.linspace(140, 155, 15)
background = np.zeros_like(bins) + 1000


if __name__ == "__main__":

    # generate pseudo-data

    np.random.seed(151)

    for c in range(n_channels):
        file_name = "data/pseudo_channel_{}".format(c)
        d = poisson.rvs([background for _ in range(n_pseudo)])
        np.save(file_name, d)
