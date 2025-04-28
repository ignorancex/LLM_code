import paths
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import corner
import emcee
from multiprocessing import freeze_support

#import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')
from orbitize import driver

# FIX THE SEED SO THAT RESULTS ARE REPRODUCIBLE

np.random.seed(44)

from orbitize import read_input

# load in astrometry of V773 Tau
orb_file = paths.data / "apj416631t1_ascii_mod_2xerr.txt"

# MCMC parameters
num_temps = 20
num_walkers = 1000
num_threads = 2

plx    = 7.70  # from Torres 2012 page 7 - better than GAIA as explicit solve using VLBI data
plxerr = 0.19  # from Torres 2012 page 7

myDriver = driver.Driver(
    orb_file, # data file
    'MCMC',        # choose from: ['OFTI', 'MCMC']
    1,             # number of planets in system
    5.27,          # total mass [M_sun] Boden 2012 Table 4
    plx,         # system parallax [mas]
    mass_err=0.65, # mass error [M_sun] Boden 2012 Table 4
    plx_err=plxerr, # parallax error [mas]
    mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
)
total_orbits = 1000000 # number of steps x number of walkers (at lowest temperature)
burn_steps = 2000 # steps to burn in per walker
thin = 4 # only save every 2nd step
#total_orbits = 1000 # number of steps x number of walkers (at lowest temperature)
#burn_steps = 20 # steps to burn in per walker

orbits = myDriver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

import orbitize.results

myResults = myDriver.sampler.results
myResults.save_results(paths.data / 'v773_tau_AB.hdf5')
