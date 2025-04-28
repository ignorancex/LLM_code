# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:15:35 2023

@author: David J. Kedziora
"""

import os

import pandas as pd
import numpy as np
from lmfit import fit_report

from time import time
import multiprocessing as mp
from functools import partial

import load
import calc
import plot

import seaborn as sns
import matplotlib.pyplot as plt

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

random_seed = 0

constants = {}
constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.

knowns = {}
knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.

#%% Examine comparison between normal fit and Poisson-noise fit.

full_filename_prefix = "InGaAs-GaAs QDs\\FI-SEQUR project demonstrator sample\\1p2uW_3000cps_time bin width 128 ps"

df_events, sr_delays, range_snapshots = load.load_experiment(folder_data, full_filename_prefix, constants)
    
# Identify save/load destination for any plots and results.
plot_prefix = folder_plots + full_filename_prefix
save_prefix = folder_saves + full_filename_prefix + "_seed_" + str(random_seed)

# Determine fitting parameters of interest.
fit_prefix_ids = ["fits", "pnoise_fits"]
num_event_ids = ["1000000", "100000", "10000", "1000", "100"]
param_ids = ["amp_env", "amp_ratio", "bg", "decay_peak", "delay_mpe"]
param_fit_ids = ["value", "stderr"]
mc_results = dict()

for fit_prefix in fit_prefix_ids:
    mc_results[fit_prefix] = dict()
    for num_events in num_event_ids:
        mc_results[fit_prefix][num_events] = dict()
        for param_id in param_ids:
            mc_results[fit_prefix][num_events][param_id] = pd.read_pickle(save_prefix + "_mc_" + fit_prefix
                                                                          + "_sample_size_" + str(num_events)
                                                                          + "_" + param_id + ".pkl")
            
# # Clean.
# is_stderr_too_big = mc_results["pnoise_fits"]["100"]["bg"]["stderr"]>1
# for param_id in param_ids:
#     df_temp = mc_results["pnoise_fits"]["100"][param_id]
#     df_temp.drop(df_temp[is_stderr_too_big].index, inplace=True)

# Plot
plotted_num_events = ["1000000", "100000", "10000", "1000"]
plotted_fit_prefix = ["fits"]

for param_id in param_ids:
    df_plot = pd.DataFrame(columns=param_fit_ids)
    for fit_prefix in fit_prefix_ids:
        if fit_prefix in plotted_fit_prefix:
            fit_label = "P"
            if fit_prefix == "fits":
                fit_label = "LS"
                
            # First row is the original best fit.
            df_temp = mc_results[fit_prefix][num_event_ids[0]][param_id]
            df_plot = pd.concat([df_plot,
                                 df_temp.loc[0].to_frame().T.assign(label="Best " + fit_label)], 
                                ignore_index=True)
            
            for num_events in num_event_ids:
                if num_events in plotted_num_events:
                    df_temp = mc_results[fit_prefix][num_events][param_id]
                    df_plot = pd.concat([df_plot,
                                         df_temp.loc[1:].assign(label=fit_label + " " + num_events)],
                                        ignore_index=True)
        
        # dataset = pd.concat([mc_results["pnoise_fits"]["1000"]["amp_ratio"].assign(dataset="set1"), 
        #                      mc_results["fits"]["1000"]["amp_ratio"].assign(dataset="set2")])
        
    g = sns.JointGrid(data=df_plot, x="value", y="stderr", hue="label")
    
    g.plot_joint(sns.scatterplot)
    # g.plot_marginals(sns.stripplot, hue="dataset", dodge=True)
    sns.boxplot(df_plot, x=g.hue, y=g.y, ax=g.ax_marg_y)
    sns.boxplot(df_plot, y=g.hue, x=g.x, ax=g.ax_marg_x)