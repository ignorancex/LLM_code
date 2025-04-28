# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:31:55 2023

@author: David J. Kedziora
"""

import os
import string

import pandas as pd

def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def filename_neaten(in_filename):
    # Fix certain typos in a raw-data filename.
    # Used prior to generating filename prefixes for loading and saving.
    filename = in_filename.replace("_ ex", "_ex").replace(" ulsed", " pulsed")
    return filename

def get_full_filename_prefixes(in_folder_data):
    # Generate one filename prefix per single experiment.
    # The prefix may need to identify several datasets.
    full_filename_prefixes = set()
    for root, subdirs, files in os.walk(in_folder_data):
        for filename in files:
            file_path = os.path.join(root, filename).replace(in_folder_data, "", 1)
            if file_path.endswith(".txt"):
                prefix = rchop(filename_neaten(file_path), ".txt")
                prefix = rchop(prefix.rstrip(string.digits).rstrip(" _"), "part").rstrip(" _")
                prefix = rchop(prefix.rstrip(string.digits).rstrip(" _"), "day").rstrip(" _")
                prefix = rchop(prefix.rstrip(string.digits).rstrip(" _"), "test").rstrip(" _")
                # Custom file fusions.
                ending_cuts = [["auto_0p22mW_5K_ex1231nm pulsed_1319p4nm-1323p6nm", "_1319p4nm-1323p6nm"],
                               ["auto_0p22mW_5K_ex1231nm pulsed_1319p3nm-1323p55nm", "_1319p3nm-1323p55nm"],
                               ["auto_0p25mW_5K_ex1120nm pulsed_1319p3nm-1323p7nm", "_1319p3nm-1323p7nm"],
                               ["auto_0p25mW_5K_ex1120nm pulsed_1319p4nm-1323p7nm", "_1319p4nm-1323p7nm"]]#,
                               # ["10uW_12000cps", "_12000cps"],
                               # ["10uW_6000cps", "_6000cps"]]
                for ending, cut in ending_cuts:
                    if prefix.endswith(ending): prefix = rchop(prefix, cut)
                full_filename_prefixes.add(prefix)
    return full_filename_prefixes

def load_experiment(in_folder_data, in_full_filename_prefix, in_constants):
    # Loads data files into a single dataframe.
    # Establishes delay/snapshot arrays, as well as a detection event matrix.
    # WARNING: There is no bug checking. Ensure files are appropriate to merge.
    
    duration_snapshot = in_constants["duration_snapshot"]
    unit_delay = in_constants["unit_delay"]
    
    folder_prefix, filename_prefix = os.path.split(in_full_filename_prefix)
    
    df_delays = None
    df_events = None
    for filename_data in os.listdir(in_folder_data + folder_prefix):
        if filename_neaten(filename_data).startswith(filename_prefix):
            df = pd.read_csv(in_folder_data + os.path.join(folder_prefix, filename_data), sep="\t", header=None)
            if df_events is None:
                print("Loading into dataframe: %s" % in_folder_data + os.path.join(folder_prefix, filename_data))
                df_delays = df[df.columns[0]]*unit_delay
                df_events = df[df.columns[1:]]
            else:
                print("Merging into dataframe: %s" % in_folder_data + os.path.join(folder_prefix, filename_data))
                df_events = pd.concat([df_events, df[df.columns[1:]]], axis=1)
    range_snapshots = range(0, df_events.columns.size * duration_snapshot, 
                            duration_snapshot)
    df_events.columns = range_snapshots
    
    return df_events, df_delays, range_snapshots