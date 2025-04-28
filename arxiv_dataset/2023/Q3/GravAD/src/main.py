import jax
from jax import jit, grad, lax, vmap, random
import jax.numpy as jnp
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import math
import time
import pickle
import re
import os
import glob

import helper
import plots
from to_csv import process_files
from constants import *
from calculations import *


def main():
    
    pattern_real = r'(\w+)_(\w+)_.*'
    pattern_simu = r'(\d+)_(\d+)_.*'

    folder_results_real = "../data/results_real"
    folder_results_sim  = "../data/results_simulated"

    folder_max_snr_real = "../data/max_snr_real"
    folder_max_snr_sim  = "../data/max_snr_simulated"


    # Clear folders
    helper.clear_folder("../data")
    helper.clear_folder('../figures')

    # Compile the code --> allows for accurate timings
    jit_compile()

    # Analyse real signals
    if True:
        real_signals()
        plotter(pattern_real, folder_results_real, folder_max_snr_real)
    
    # Analyse simulated signals (WIP)
    if False:
        simulated_signals()
#       plotter(pattern_simu, folder_results_sim, folder_max_snr_sim)

    # Get a compilation of results in csv
    process_files()

if __name__ == "__main__":
    main()

