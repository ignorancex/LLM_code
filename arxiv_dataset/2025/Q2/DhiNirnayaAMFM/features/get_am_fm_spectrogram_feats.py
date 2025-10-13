#!/usr/bin/env python

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from scipy.fftpack import dct   # Discrete Cosine Transform
from scipy.signal import find_peaks
import numpy as np
#===============================================================
# RFA custom module import
from module_spectrogram_v2 import *	# Low frequency spectrogram functions
from rfa_single_conf import *

specwindowsecs = 5  # Window size for spectrogram   # in seconds
specstrides = 100   # Stride for spectrogram
# Number of DCT coefficients to keep for each dimension
# (e.g., 2D DCT will produce a matrix of size dct_num x dct_num)
# This is used to reduce the dimensionality of the spectrogram features.
# For example, if dct_num=2, we keep the top-left 2x2 block of the DCT matrix.
dct_num = 2

def get_am_fm_2ddct_feats(am_spec, fm_spec, dct_num):
    """
    Compute 2ddct of AM and FM spectrograms.
    Returns:
        dct2d_am: 2ddct of AM spectrogram
        dct2d_am: 2ddct of FM spectrogram
    """
    filtered_am_spec = am_spec[~np.all(am_spec == 0, axis=1)]
    # Compute 2D DCT of the AM spectrogram
    # We take the log of the magnitude to reduce dynamic range
    # and then apply DCT to the log-magnitude spectrogram.
    aaa = np.log10(filtered_am_spec[:,1:])
    dct2d = dct(dct(aaa.T).T)
    dct2d = dct2d[0:dct_num, 0:dct_num]
    dct2d_am = dct2d.flatten()

    filtered_fm_spec = fm_spec[~np.all(fm_spec == 0, axis=1)]
    aaa = np.log10(filtered_fm_spec[:,1:])
    dct2d = dct(dct(aaa.T).T)
    dct2d = dct2d[0:dct_num, 0:dct_num]
    dct2d_fm = dct2d.flatten()

    return dct2d_am, dct2d_fm

def get_am_fm_variance_feats(am_spec_mag, am_spec_fre, fm_spec_mag, fm_spec_fre, num_R_form=6):
    """
    Compute variance of AM and FM spectrograms.
    Returns:
        var_am: variance of AM spectrogram
        var_fm: variance of FM spectrogram
    """
    # AM spectrogram processing for variance of RFs
    filtered_am_spec = am_spec_mag[~np.all(am_spec_mag == 0, axis=1)]

    magg = filtered_am_spec[:,1:]
    free = am_spec_fre[:,1:]
    free = free[0,:]
    RF_fre = []
        
    for x in range(len(magg)):
        # Get the slice of the spectrogram      
        slice_spec = magg[x,:]
        # Find peaks in the slice
        # and get the frequencies and magnitudes of the peaks
        # We assume that the first column is time and the rest are frequency bins
        # Find peaks in the slice
        # and get the frequencies and magnitudes of the peaks
        peaks, _ = find_peaks(slice_spec)
        freq1, magg1 = free[peaks], slice_spec[peaks]
        
        ranked = np.argsort(magg1)
            
        if peaks.shape[0] >= num_R_form:
            # Get the top num_R_form (6) peaks
            # ranked[::-1] sorts in descending order
            # [:6] takes the top 6 peaks
            # If there are less than 6 peaks, we will pad with -1000.0 later
            # to ensure the output is always 6 elements long
            top_magg_indx = ranked[::-1][:6]
            magg_top = magg1[top_magg_indx]
            freq_top = freq1[top_magg_indx]
            RF_fre += [freq_top]
    RF_fre1 = np.array(RF_fre)
    # Compute variance
    var_am = np.var(RF_fre1, axis = 0)

    # FM spectrogram processing for variance of RFs
    filtered_fm_spec = fm_spec_mag[~np.all(fm_spec_mag == 0, axis=1)]

    magg = filtered_fm_spec[:,1:]
    free = fm_spec_fre[:,1:]
    free = free[0,:]
    RF_fre = []
        
    for x in range(len(magg)):
        slice_spec = magg[x,:]
        peaks, _ = find_peaks(slice_spec)
        freq1, magg1 = free[peaks], slice_spec[peaks]
        
        ranked = np.argsort(magg1)
            
        if peaks.shape[0] >= num_R_form:
            top_magg_indx = ranked[::-1][:num_R_form]
            magg_top = magg1[top_magg_indx]
            freq_top = freq1[top_magg_indx]
            RF_fre += [freq_top]
    RF_fre1 = np.array(RF_fre)
    # Compute variance
    var_fm = np.var(RF_fre1, axis = 0)

    # Pad with -1000.0 if they are not num_R_form (6) elements each
    # if len(var_am) != 6 or len(var_fm) != 6: pad -1000.0 to make them 6 elements each
    if len(var_am) < num_R_form:
        var_am = np.pad(var_am, (0, num_R_form - len(var_am)), constant_values=-1000.0)
    if len(var_fm) < num_R_form:
        var_fm = np.pad(var_fm, (0, num_R_form - len(var_fm)), constant_values=-1000.0)

    return var_am, var_fm

def extract_features_spectrogram(wav_path: Path, specwindowsecs, specstrides, dct_num, num_R_form=6):
    fm_spec_mag, fm_spec_fre = get_FM_spectrogram(wav_path, specwindowsecs, specstrides)
    am_spec_mag, am_spec_fre = get_AM_spectrogram(wav_path, specwindowsecs, specstrides)

    # Feats: variance of RFs - dims = 6 + 6 (AM+FM)
    var_am, var_fm = get_am_fm_variance_feats(am_spec_mag, am_spec_fre, fm_spec_mag, fm_spec_fre, num_R_form=6)
    
    # Feats: 2ddct of rhythm spectrogram - dims = dct_num + dct_num (AM+FM)
    dct2d_am, dct2d_fm = get_am_fm_2ddct_feats(am_spec_mag, fm_spec_mag, dct_num)

    # appeding all features 
    row = pd.Series(
        np.concatenate(
            [dct2d_am, dct2d_fm, var_am, var_fm],
            axis=0
        ),
        index=[
            f"dct_am_{i+1}" for i in range(dct_num*dct_num)
        ] + [
            f"dct_fm_{i+1}" for i in range(dct_num*dct_num)
        ] + [
            f"var_am_{i+1}" for i in range(num_R_form)
        ] + [
            f"var_fm_{i+1}" for i in range(num_R_form)
        ]
    )
    
    # Ensure all features are float32
    row = row.astype(np.float32)

    # Check if the row has the expected number of features
    expected_len = (dct_num*dct_num) + (dct_num*dct_num) + num_R_form + num_R_form  # dct_am + dct_fm + var_am + var_fm
    if len(row) != expected_len:
        raise ValueError(f"Extracted features length {len(row)} does not match expected {expected_len}")

    # Return the row as a Series            

    return row
