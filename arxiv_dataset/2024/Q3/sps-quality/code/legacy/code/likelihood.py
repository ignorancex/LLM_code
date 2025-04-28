# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:05:19 2022

@author: David J. Kedziora
"""

import os
import pandas as pd
import numpy as np

import string
from time import time

import calc
import plot

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

full_filename_prefixes = set()
for root, subdirs, files in os.walk(folder_data):
    for filename in files:
        file_path = os.path.join(root, filename).replace(folder_data, "", 1)
        if file_path.endswith(".txt"):
            prefix = rchop(file_path, ".txt")
            prefix = rchop(prefix.rstrip(string.digits).rstrip(" _"), "part").rstrip(" _")
            prefix = rchop(prefix.rstrip(string.digits).rstrip(" _"), "day").rstrip(" _")
            full_filename_prefixes.add(prefix)
prefix_requirements = [
#     "1p2uW_3000cps_time bin width 128 ps",
#     "2p5uW_4000cps",
#     "4uW_4100cps",
#     "8uW_5100cps",
#     "10uW_6000cps",
#     "10uW_12000cps",
#     "20uW_7000cps",
    "30uW_7000cps"
]
random_seed = 0

# The number of traces to generate per datafile.
# Each trace shows how g2(0) estimates change with greater sampling.
num_traces = 1

# Define how far the bound of several windows should be from its centre.
# These are used for the quick methods of deriving g2(0).
# E.g. +-3 ns at 0.256 ns per delay bin means 23 bins are used in the window.
smoothing_bound = 1e-9     # Unit: s.

# Define experimental device parameters, i.e. for the laser and detectors.
# These will not be used in the fitting methods of deriving g2(0).
pulse_freq = 80e6            # Unit: Hz.
delta_zero = 60.7e-9        # Unit: s.

# Define detection-sampling parameters.
snapshot_bin_size = 10

# SI conversion factors when working with data and later plotting.
delay_unit = 1e-9           # 1 nanosecond.


for full_filename_prefix in full_filename_prefixes:
    
    notAccepted = True
    if prefix_requirements is not None and len(prefix_requirements) > 0:
        for prefix_requirement in prefix_requirements:
            if prefix_requirement in full_filename_prefix:
                print("Requirement checked: %s contains %s" % (full_filename_prefix, prefix_requirement))
                notAccepted = False
        if notAccepted:
            continue
    
    folder_prefix, filename_prefix = os.path.split(full_filename_prefix)
    
    # Prepares save destination for any plots and storable results.
    plot_prefix = folder_plots + full_filename_prefix
    save_prefix = folder_saves + full_filename_prefix + "_traces_" + str(num_traces) + "_seed_" + str(random_seed)
    
    for folder_base in [folder_plots, folder_saves]:
        if not os.path.exists(os.path.join(folder_base, folder_prefix)):
            os.makedirs(os.path.join(folder_base, folder_prefix))
    
    # Loads data files into a single dataframe.
    # Establishes delay/snapshot arrays, as well as a detection event matrix.
    # WARNING: There is no bug checking. Ensure files are appropriate to merge.
    df_delays = None
    df_events = None
    for filename_data in os.listdir(folder_data + folder_prefix):
        if filename_data.startswith(filename_prefix):
            df = pd.read_csv(folder_data + os.path.join(folder_prefix, filename_data), sep="\t", header=None)
            if df_events is None:
                print("Loading into dataframe: %s" % folder_data + os.path.join(folder_prefix, filename_data))
                df_delays = df[df.columns[0]]*delay_unit
                df_events = df[df.columns[1:]]
            else:
                print("Merging into dataframe: %s" % folder_data + os.path.join(folder_prefix, filename_data))
                df_events = pd.concat([df_events, df[df.columns[1:]]], axis=1)
    range_snapshots = range(0, df_events.columns.size * snapshot_bin_size, 
                            snapshot_bin_size)
    df_events.columns = range_snapshots
    
    # Calculate helper variables; these may be recalculated inside functions.
    delay_bin_size = df_delays[1] - df_delays[0]
    delay_range = df_delays[df_delays.size-1] - df_delays[0]
    
    # Create a histogram of events across snapshots and plot it.
    plot.plot_event_history(df_events, range_snapshots, plot_prefix)
    
    # Sum all snapshots of the detection events.
    df_sample_full = df_events.sum(axis=1)