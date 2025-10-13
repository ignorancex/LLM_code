#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import numpy as np
import pandas as pd

#%%
from utils import *
from module_spectrogram import *	# Low frequency spectrogram functions

#===============================================================
# Assign configuration parameters to variables 
from rfa_single_conf import *

# added params
cut_off_freq = 10
specwindowsecs = 5
specstrides = 50
dct_num = 2

#%%
# File paths
processed_files_txt = <path/to/txt/file/with/list/of/wav/files> 
labels_file_csv = <path/to/csv/file/with/labels> 
merged_csv = <path/to/write/csv/file/with/features> 

#%%
# Read processed file list
processed_files = []
if os.path.exists(processed_files_txt):
    with open(processed_files_txt, "r") as f:
        processed_files = [line.strip() for line in f.readlines()]

# Remove '_combined' and '.wav' from filenames
processed_files_clean = [os.path.basename(f).replace("_combined", "").replace(".wav", "") for f in processed_files]
#%%
# Read labels CSV
df_labels = pd.read_csv(labels_file_csv)

# Merge based on filenames without extensions
df_processed = pd.DataFrame({"file_name": processed_files_clean, "processed_path": processed_files})

#%%
# FM
rf_var_fm, mag_var_fm = list(get_RF_var_FM(processed_files_txt, specwindowsecs, specstrides))
# Convert list to array
var_fm_all = pd.DataFrame(rf_var_fm)
# Assign new column names
var_fm_all.columns = [f'var_fm_{i}' for i in range(var_fm_all.shape[1])]

# AM
rf_var_am, mag_var_am = list(get_RF_var_AM(processed_files_txt, specwindowsecs, specstrides, cut_off_freq))
# Convert list to DataFrame
var_am_all = pd.DataFrame(rf_var_am)
# Assign new column names
var_am_all.columns = [f'var_am_{i}' for i in range(var_am_all.shape[1])]    

# 2ddct-based feats 
# FM
fm_2ddct_all = get_2ddct_AM(processed_files_txt, dct_num, specwindowsecs, specstrides, cut_off_freq)
# Convert list to array
fm_2ddct_all = pd.DataFrame(fm_2ddct_all)
# Assign new column names
fm_2ddct_all.columns = [f'fm_2ddct_{i}' for i in range(fm_2ddct_all.shape[1])]

# AM
am_2ddct_all = get_2ddct_FM(processed_files_txt, dct_num, specwindowsecs, specstrides)
# Convert list to array
am_2ddct_all = pd.DataFrame(am_2ddct_all)
# Assign new column names
am_2ddct_all.columns = [f'am_2ddct_{i}' for i in range(am_2ddct_all.shape[1])]

#%%
# Concatenate the arrays
features_df = pd.concat([var_am_all, var_fm_all, am_2ddct_all, fm_2ddct_all], axis=1)

#%%
# Concatenate row-wise (axis=0)
combined_df = pd.concat([df_processed, features_df], axis=1)

# Display the result
print(combined_df)
#%%
# Merging df_labels and df_processed on different columns
df_merged = pd.merge(df_labels, combined_df, left_on="adressfname", right_on="file_name", how="inner")

# Save merged CSV
df_merged.to_csv(merged_csv, index=False)

print(f"Merged file saved to: {merged_csv}")
#%%
