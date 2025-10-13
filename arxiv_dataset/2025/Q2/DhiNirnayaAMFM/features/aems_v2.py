#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:16:44 2021

@author: administrator
"""

#======================================================
# Library modules

import sys, re 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io.wavfile as wave 
from scipy.signal import medfilt, hilbert
from scipy.signal import find_peaks
import librosa
import scipy
from tqdm import tqdm
from module_F0_v1 import *	# FM demodulation (F0 estimation, 'pitch' tracking)
from rfa_single_conf import *
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

def mag_AM_onewav_all(wavfilename):
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
    # In previous versions, a peak-picking algorithm was used.
    # In the present version, the absolute Hilbert Transform is used.

    envelope = np.abs(hilbert(signal))
    envelope = medfilt(envelope, 501)
    envelope = envelope / np.max(envelope)

    #======================================================
    # Spectral transformation

    mags = np.abs(np.fft.rfft(envelope))
    freqs = np.abs(np.fft.rfftfreq(envelope.size,period))
    samplesperhertz = int(len(mags) / maxfreq)
    minsamples = np.int(specfreqmin * samplesperhertz)
    maxsamples = np.int(specfreqmax * samplesperhertz)
    freqsegment = freqs[minsamples:maxsamples]
    magsegment = mags[minsamples:maxsamples]
    magsegment = magsegment / np.max(magsegment)
    fre_mag = [freqsegment, magsegment]
    
    return envelope, fre_mag


def mag_FM_onewav_all(wavfilename):
    #fs, signal = wave.read(wavfilename)
    signal, fs = librosa.load(wavfilename, sr=16000)
    signallen = len(signal)
    seconds = signallen / fs
    period = 1/fs
    maxfreq = int(fs/2)

    # Waveform normalisation to -1 ..., 0 ... 1
    signal = signal/np.max(signal)

    f0array, framerate, frameduration = f0estimate(signal, fs)
    f0arraylength = len(f0array)

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
    
    return f0array, fre_mag

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
    # In previous versions, a peak-picking algorithm was used.
    # In the present version, the absolute Hilbert Transform is used.

    envelope = np.abs(hilbert(signal))
    envelope = medfilt(envelope, 501)
    envelope = envelope / np.max(envelope)

    #======================================================
    # Spectral transformation

    mags = np.abs(np.fft.rfft(envelope))
    freqs = np.abs(np.fft.rfftfreq(envelope.size,period))
    samplesperhertz = int(len(mags) / maxfreq)
    minsamples = np.int(specfreqmin * samplesperhertz)
    maxsamples = np.int(specfreqmax * samplesperhertz)
    freqsegment = freqs[minsamples:maxsamples]
    magsegment = mags[minsamples:maxsamples]
    magsegment = magsegment / np.max(magsegment)
    fre_mag = [freqsegment, magsegment]
    
    return fre_mag


def mag_FM_onewav(wavfilename):
    #fs, signal = wave.read(wavfilename)
    signal, fs = librosa.load(wavfilename, sr=16000)
    signallen = len(signal)
    seconds = signallen / fs
    period = 1/fs
    maxfreq = int(fs/2)

    # Waveform normalisation to -1 ..., 0 ... 1
    signal = signal/np.max(signal)

    f0array, framerate, frameduration = f0estimate(signal, fs)
    f0arraylength = len(f0array)

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
    
    #======================================================
    # Spectral transformation
    '''
    mags = np.abs(np.fft.rfft(envelope))
    freqs = np.abs(np.fft.rfftfreq(envelope.size,period))
    samplesperhertz = int(len(mags) / maxfreq)
    minsamples = np.int(specfreqmin * samplesperhertz)
    maxsamples = np.int(specfreqmax * samplesperhertz)
    freqsegment = freqs[minsamples:maxsamples]
    magsegment = mags[minsamples:maxsamples]
    magsegment = magsegment / np.max(magsegment)
    fre_mag = [freqsegment, magsegment]
    '''
    return fre_mag

def get_AM_rzf_onewav(wavfilename, rhythmcount, threshold):
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
    # In previous versions, a peak-picking algorithm was used.
    # In the present version, the absolute Hilbert Transform is used.

    envelope = np.abs(hilbert(signal))
    envelope = medfilt(envelope, 501)
    envelope = envelope / np.max(envelope)

    #======================================================
    # Spectral transformation

    mags = np.abs(np.fft.rfft(envelope))
    freqs = np.abs(np.fft.rfftfreq(envelope.size,period))
    samplesperhertz = int(len(mags) / maxfreq)
    minsamples = np.int(specfreqmin * samplesperhertz)
    maxsamples = np.int(specfreqmax * samplesperhertz)
    freqsegment = freqs[minsamples:maxsamples]
    magsegment = mags[minsamples:maxsamples]
    magsegment = magsegment / np.max(magsegment)
    
    freR_gib = []
    magR_gib = []
    for i, (f, m) in enumerate(zip(freqsegment, magsegment)):
        if m > threshold:
            freR_gib.append(f)
            magR_gib.append(m)
              
    peaks, _ = find_peaks(magsegment)
    freq, magg = freqsegment[peaks], magsegment[peaks]
    
    magR_sci = magg[magg > threshold]
    freR_sci = freq[magg > threshold]
    
    ranked = np.argsort(magg)
    top_magg_indx = ranked[::-1][:rhythmcount]
    magg_top = magg[top_magg_indx]
    freq_top = freq[top_magg_indx]
    
    return magR_gib, freR_gib, magR_sci, freR_sci, magg_top, freq_top
    
def get_AM_rzf_all(file_path, rhythmcount, threshold):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    magR_gib_all = []
    freR_gib_all = []
    
    rhytm_mag_th = []
    rhytm_fre_th = []
    rhytm_mag_top = []
    rhytm_fre_top = []
    
    for line in lines:
        line2read = line.strip('\n')
        magR_gib, freR_gib, magg_th, freq_th, magg_top, freq_top = mag_onewav(line2read, rhythmcount, threshold)
        
        magR_gib_all.append(magR_gib)
        freR_gib_all.append(freR_gib)
        
        rhytm_mag_th.append(magg_th)
        rhytm_fre_th.append(freq_th)
        
        rhytm_mag_top.append(magg_top)
        rhytm_fre_top.append(freq_top)
        
    #rhytm_mag_th = np.array(rhytm_mag_th)
    #rhytm_fre_th = np.array(rhytm_fre_th)
    
    #rhytm_mag_top = np.array(rhytm_mag_top)
    #rhytm_fre_top = np.array(rhytm_fre_top)
    
    
    return magR_gib_all, freR_gib_all, rhytm_mag_th, rhytm_fre_th, rhytm_mag_top, rhytm_fre_top



def mag_AM_all(file_path):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    fre_mag_all = []
    
    for line in lines:
        line2read = line.strip('\n')
        fre_mag = mag_AM_onewav(line2read)
        fre_mag_all.append(fre_mag)
    
    return fre_mag_all
        
        
def mag_FM_all(file_path):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    fre_mag_all = []
    
    for line in lines:
        line2read = line.strip('\n')
        fre_mag = mag_FM_onewav(line2read)
        fre_mag_all.append(fre_mag)
    
    return fre_mag_all
    
def mag_FM_onewav_plot(wavfilename, rhythmcount):
    #fs, signal = wave.read(wavfilename)
    signal, fs = librosa.load(wavfilename, sr=16000)
    signallen = len(signal)
    seconds = signallen / fs
    period = 1/fs
    maxfreq = int(fs/2)

    # Waveform normalisation to -1 ..., 0 ... 1
    signal = signal/np.max(signal)

    f0array, framerate, frameduration = f0estimate(signal, fs)
    f0arraylength = len(f0array)

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
              
    peaks, _ = find_peaks(lffmspecmags)
    freq, magg = lffmspecfreqs[peaks], lffmspecmags[peaks]
    
    ranked = np.argsort(magg)
    top_magg_indx = ranked[::-1][:rhythmcount]
    magg_top = magg[top_magg_indx]
    freq_top = freq[top_magg_indx]
    fre_mag_top = [freq_top, magg_top]
    
    return signal, f0array, f0arraylength, fre_mag, fre_mag_top 

def mag_FM__RF_all(file_path):
    with open(file_path, 'r') as fd1:
        lines = fd1.readlines()
    
    RFs_all = []
    
    for line in tqdm(lines, desc="Processing files", unit="file"):
        line2read = line.strip('\n')
        signal, f0array, f0arraylength, fre_mag, fre_mag_top = mag_FM_onewav_plot(line2read, 1)
        RFs_all.append(fre_mag_top[0])
    
    return RFs_all
