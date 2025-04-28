# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:41:11 2024

@author: David J. Kedziora
"""

import os

import pandas as pd
import numpy as np
from lmfit import fit_report

from time import time

import load
import calc
import plot

#%% This script loads a set of datasets.
# It converts them into cumulative sums of histograms.

folder_data = "../data/"

# Only full filename prefixes that contain a listed substring will be loaded.
full_filename_requirements = ["SEQUR"]
# full_filename_requirements = ["1p2uW_3000cps"]
# full_filename_requirements = ["10uW_12000cps"]

random_seed = 0

constants = {}
constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.

knowns = {}
knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.

#%% Loop through all the datasets available, as long as they meet requirement.
full_filename_prefixes = load.get_full_filename_prefixes(folder_data)

for full_filename_prefix in full_filename_prefixes:
    
    # Do not load filenames that do not contain a required substring.
    if not ((full_filename_requirements is None) or (len(full_filename_requirements) == 0)):
        is_filename_skippable = True
        
        for requirement in full_filename_requirements:
            if requirement in full_filename_prefix:
                is_filename_skippable = False
                
        if is_filename_skippable:
            continue
    
    # Load the experiment.
    df_events, sr_delays, range_snapshots = load.load_experiment(folder_data, full_filename_prefix, constants)
    
    # For the dataset with a 0.128 ns resolution, combine into 0.256 ns bins.
    # Note: There are an odd number of bins, so the final bin is inaccurate.
    if "1p2uW_3000cps_time bin width 128 ps" in full_filename_prefix:
        full_filename_prefix = full_filename_prefix.replace("_time bin width 128 ps", "")
        plot.plot_event_histogram(df_events.sum(axis=1), sr_delays, constants,
                                  "1p2uW_128ps")
        df_events = df_events + df_events.shift(-1, fill_value=0)
        df_events = df_events.iloc[::2, :]
        df_events.reset_index(drop=True, inplace=True)
        print(sr_delays)
        sr_delays = sr_delays[::2]
        sr_delays.reset_index(drop=True, inplace=True)
        plot.plot_event_histogram(df_events.sum(axis=1), sr_delays, constants,
                                  "1p2uW_256ps")
    
    folder_prefix, filename_prefix = os.path.split(full_filename_prefix)
    
    # Determine fitting parameters of interest.
    param_ids = ["rate_env", "g2_zero", "rate_bg", "decay_peak", "delay_mpe"]
    param_fit_ids = ["value", "stderr"]
    mc_param_fits = dict()
            
    #%% Work out some details about the full histogram.
    sr_best = df_events.sum(axis=1)
    duration_best = len(range_snapshots)*constants["duration_snapshot"]
    sum_per_sec = sr_best.sum()/duration_best
    print("This dataset details %i two-photon events over %i seconds, i.e. %f per second." 
          % (sr_best.sum(), duration_best, sum_per_sec))
    
    # Extract details of the delay domain and create a series of bin edges.
    # Also create a series of delay-bin centres; use this correction for fitting.
    d_delays = sr_delays[1] - sr_delays[0]
    n_delays = len(sr_delays)
    sr_edges = pd.concat([sr_delays, pd.Series([n_delays*d_delays])], ignore_index=True)
    sr_centres = sr_delays + d_delays/2

    for do_normalise in [True, False]:

        df_cumsum = df_events.cumsum(axis=1)
        df_ml = df_cumsum.T

        g2_zero = np.empty(len(range_snapshots), dtype=float)
        rate_bg = np.empty(len(range_snapshots), dtype=float)
        rate_env = np.empty(len(range_snapshots), dtype=float)
        decay_peak = np.empty(len(range_snapshots), dtype=float)
        delay_mpe = np.empty(len(range_snapshots), dtype=float)

        row = 0
        for idx, sr_current in df_ml.iterrows():

            print(idx + constants["duration_snapshot"])
            fit = calc.estimate_g2zero_pulsed(sr_current, sr_centres, knowns,
                                            use_poisson_likelihood = False,
                                            in_duration = idx + constants["duration_snapshot"])
            
            rate_bg[row] = fit.params["rate_bg"].value
            rate_env[row] = fit.params["rate_env"].value
            decay_peak[row] = fit.params["decay_peak"].value
            delay_mpe[row] = fit.params["delay_mpe"].value
            g2_zero[row] = fit.params["g2_zero"].value

            row += 1
                    
        df_ml.columns = [str(int(x)) + " ps" for x in sr_centres*(10**12)]

        sr_events = df_ml.sum(axis=1)

        if do_normalise:
            df_ml = df_ml.div(sr_events, axis=0)

        df_ml.loc[:, "events"] = sr_events
        df_ml.loc[:, "rate_bg"] = rate_bg
        df_ml.loc[:, "rate_env"] = rate_env
        df_ml.loc[:, "decay_peak"] = decay_peak
        df_ml.loc[:, "delay_mpe"] = delay_mpe
        df_ml.loc[:, "g_fit"] = g2_zero
        
        # The last fit gives the 'ground truth' for g.
        df_ml.loc[:, "g_best"] = fit.params["g2_zero"].value

        if do_normalise:
            df_ml.to_csv(folder_data + "ml/sps_cumsum_norm_" + filename_prefix + ".csv", index=False)
        else:
            df_ml.to_csv(folder_data + "ml/sps_cumsum_" + filename_prefix + ".csv", index=False)