#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 15 22:22:17 2021
@author: Parismita
"""

#======================================================
# Library modules

import numpy as np 
from scipy.signal import medfilt, hilbert
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import librosa
import scipy
from module_F0_v1 import *	# FM demodulation (F0 estimation, 'pitch' tracking)
from rfa_single_conf import *

from pathlib import Path
import numpy as np
import scipy.fft
import time
#======================================================
# Variable definitions

envelopescale = 1.0

specfreqmin = 0.5
specfreqmax = 10
peakthreshold = 0.5

specwindowsecs = 3
specstrides = 250
dcclip = 2
spectrogramheatmap = "YlOrRd"
specgramdotsize = 2

fontsize = 10
rhythmcount = 6

def mag_AM_onewav(wavfilename):
    #fs, signal = wave.read(wavfilename)
    signal, fs = librosa.load(wavfilename, sr=16000)
    signallen = len(signal)
    seconds = signallen / fs
    period = 1/fs
    maxfreq = int(fs/2)

    # Waveform normalisation to -1 ..., 0 ... 1
    signal = signal/np.max(signal)

    #======================================================
    # Envelope demodulation
    # Hilbert transform to get the envelope of the signal

    # Use the Hilbert transform to get the envelope of the signal
    # The envelope is the absolute value of the Hilbert transform
    # and is then smoothed with a median filter
    # The envelope is then normalised to the maximum value
    # The median filter size is set to 501 samples
    # which is approximately 31.3 ms at 16 kHz sampling rate

    if signallen < 501:
        raise ValueError("Signal length must be at least 501 samples for median filtering.")

    #start_time = time.time()
    envelope = np.abs(hilbert(signal))
    #envelope = medfilt(envelope, 501)
    envelope = median_filter(envelope, size=501, mode="nearest")
    envelope = envelope / np.max(envelope)
    #print(time.time() - start_time, "seconds to compute envelope")

    #======================================================
    # Spectral transformation

    mags = np.abs(np.fft.rfft(envelope))
    freqs = np.abs(np.fft.rfftfreq(envelope.size,period))
    samplesperhertz = int(len(mags) / maxfreq)
    minsamples = int(specfreqmin * samplesperhertz)
    maxsamples = int(specfreqmax * samplesperhertz)
    freqsegment = freqs[minsamples:maxsamples]
    magsegment = mags[minsamples:maxsamples]
    magsegment = magsegment / np.max(magsegment)
    fre_mag = [freqsegment, magsegment]
    
    return fre_mag, signal, envelope


def mag_FM_onewav(wavfilename):
    f0array, framerate, frameduration = f0estimate_rapt(wavfilename, 16000)
    f0arraylength = len(f0array)
    
    # for plotting
    signal, fs = librosa.load(wavfilename, sr=16000)
    signal = signal/np.max(signal)
	
    #===============================================================
    # FFT low frequency spectral analysis of F0 estimation track

    fmspecmags = np.abs(np.fft.rfft(f0array))
    fmspecmaglen = len(fmspecmags)
    fmspecfreqs = np.linspace(0,framerate/2,fmspecmaglen)

    # Extraction of low frequency spectral segment, with magnitude filter
    lffmspecmaglen = int(round(fmspecfreqmax * fmspecmaglen / (framerate / 2)))
    lffmspecmags = fmspecmags[1:lffmspecmaglen]
    lffmspecfreqs = fmspecfreqs[1:lffmspecmaglen]
    lffmspecmags = lffmspecmags / np.max(lffmspecmags)
    
    fre_mag = [lffmspecfreqs, lffmspecmags]

    return fre_mag, signal, f0array

def compute_am_fm_spectrum_dct_threshold_based_RFs_feature(
    audiopath: Path, rhythmcount: int = 6, threshold: float = 0.3
) -> np.ndarray:
    
    # compute duration of the audio file
    # If the duration is less than 8 seconds, return a default feature vector
    duration = librosa.get_duration(path=audiopath)
    if duration < 1:
        feature_vector = np.concatenate((-1000, -1000, -1000, -1000, 
                                     [-1000, -1000, -1000], [-1000, -1000, -1000]))
    else:
        mag_fre_am, signal, env = mag_AM_onewav(audiopath)
        mag_fre_fm, signal, f0array = mag_FM_onewav(audiopath)

        fre_am = mag_fre_am[0] 
        mag_am = mag_fre_am[1]
        fre_fm = mag_fre_fm[0] 
        mag_fm = mag_fre_fm[1]

        # compute DCT features for AM LF spectrum and FM LF spectrum
        # DCT is used to extract the low frequency components of the spectrum
        # and to reduce the dimensionality of the feature vector

        # DCT values for AM and FM spectra
        # We take only the first 4 coefficients as features
        dct_values_am = scipy.fft.dct(mag_am, type=2)
        dct_values_am = dct_values_am[:4]

        dct_values_fm = scipy.fft.dct(mag_fm, type=2)
        dct_values_fm = dct_values_fm[:4]

        # compute RFs for AM and FM spectra
        # RFs are the frequencies of the peaks in the spectrum
        # We take the top rhythmcount peaks as features
    
        # For AM spectrum
        peaks, _ = find_peaks(mag_am)
        fre, magg = fre_am[peaks], mag_am[peaks]
        ranked = np.argsort(magg)
        top_magg_indx = ranked[::-1][:rhythmcount]
        freq_top_am = fre[top_magg_indx]

        # For FM spectrum
        peaks, _ = find_peaks(mag_fm)
        fre, magg = fre_fm[peaks], mag_fm[peaks]
        ranked = np.argsort(magg)
        top_magg_indx = ranked[::-1][:rhythmcount]
        freq_top_fm = fre[top_magg_indx]

        # compute threshold based features for AM and FM spectra
        # We compute the number of peaks above a threshold, the mean frequency and the variance of the frequencies above the threshold
        # threshold = 0.3

        # For AM spectrum
        peaks, _ = find_peaks(mag_am)
        fre, magg = fre_am[peaks], mag_am[peaks]
        magR_sci = magg[magg > threshold]
        freR_sci = fre[magg > threshold]
    
        am_len = len(magR_sci)
        am_mean = float("{0:.3f}".format(np.mean(freR_sci)))
        am_var = float("{0:.3f}".format(np.var(freR_sci)))

        # For FM spectrum
        peaks, _ = find_peaks(mag_fm)
        fre, magg = fre_fm[peaks], mag_fm[peaks]
        magR_sci = magg[magg > threshold]
        freR_sci = fre[magg > threshold]
    
        fm_len = len(magR_sci)
        fm_mean = float("{0:.3f}".format(np.mean(freR_sci)))
        fm_var = float("{0:.3f}".format(np.var(freR_sci)))

        # Combine all features into a single feature vector
        feature_vector = np.concatenate((dct_values_am, dct_values_fm, 
                                     [am_len, am_mean, am_var], [fm_len, fm_mean, fm_var]))

    return feature_vector 
    
def extract_feats_with_names(audiopath: Path, rhythmcount: int = 6, threshold: float = 0.3):
    vec = compute_am_fm_spectrum_dct_threshold_based_RFs_feature(audiopath, rhythmcount, threshold)
    # Create feature names based on the number of rhythms and the features extracted
    # 4 DCT coefficients for AM and FM, rhythmcount RFs for AM and FM, and threshold-based features
    # am_peak_count, am_freq_mean, am_freq_var, fm_peak_count, fm_freq_mean, fm_freq_var
    # Total features = 4 (DCT AM) + 4 (DCT FM) + rhythmcount (RF AM) + rhythmcount (RF FM) + 6 (additional features)
    
    # Create names for the features

    names: list[str] = (
        [f"dct_am_{i+1}" for i in range(4)]
        + [f"dct_fm_{i+1}" for i in range(4)]
        + [
            "am_peak_count",
            "am_freq_mean",
            "am_freq_var",
            "fm_peak_count",
            "fm_freq_mean",
            "fm_freq_var",
        ]
    )
    return vec, names
