# Copyright 2019 by Ben Granett, granett@gmail.com
# All rights reserved.
# This file is part of the VIPERS-SHAM project,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors,cm


def load_sdss():
    """ """
    rmin = 0
    cf_obs = []
    cov = []
    lc = 0
    for line in file("../data/sdss/SDSS_wp_covariance.dat"):
        line = line.strip()
        if line == "": continue
        if line.startswith("#"):
            continue
        w = [float(v) for v in line.split()]
        if len(w)==3:
            cf_obs.append(w)
        elif len(w)>5:
            cov.append(w)

    cov = np.array(cov)
    wp = np.array(cf_obs)
    r = wp[:,0]
    return r, wp[:,1], cov 



snapshots = [
    0.44200,
    0.45050,
    0.47090,
    0.49220,
    0.50000,
    0.52600,  # z=0.9
    0.5300,
    0.54980,
    0.55630,
    0.58760,
    0.58640,
    0.60080,
    0.62230,
    0.62800, # z=0.6
    0.66430,
    0.65650,
    0.71240,
    0.71730,
    0.77240,
    0.81920,
    0.83240,
    0.87550,
    0.90740,
    1.00000,
]


def load_sham(sample='L1', template="../data/sham400/wp_snap{snapshot:7.5f}_{sample}.txt"):
    """ """
    data = {}
    
    for snap in snapshots:
        path = template.format(snapshot=snap, sample=sample)
        if not os.path.exists(path):
            continue
    
        cf = np.loadtxt(path,unpack=True)
        logging.info("loaded %s", path)
        
        data[snap] = cf
    
    return data
