# module_spectrogram.py
# D. Gibbon
# Created 2021-08-16
# Modified 2022-04-18
# Spectrogram array module for rfa.py

#===============================================================

import numpy as np
#===============================================================
#===============================================================
# System and library module import
from scipy.signal import find_peaks
import sys, re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
from scipy.signal import butter, lfilter, medfilt
from datetime import datetime
import librosa

# RFA custom module import
from module_F0_v1 import *	# FM demodulation (F0 estimation, 'pitch' tracking)
#from module_spectrogram import *	# Low frequency spectrogram functions
#from module_dendrogram import *	# Spectral dendrogram drawing functions

#===============================================================
#===============================================================
# Assign configuration parameters to variables (see .conf file)
from rfa_single_conf import *
f0medfilter = 3

from scipy.fftpack import dct
#===============================================================

def spectrogramarray(signal, fs, specfreqmin, specfreqmax, specdownsample, spectrumpower, specwindowsecs, specstrides):

	# Brute force downsampling, optional
	#signal = signal[::specdownsample]
	#fs = int(round(fs/specdownsample))
	period = 1/fs
	signallen = len(signal)
	signalsecs = int(round(signallen / fs))

	#============================================
	windowlen = int(round(specwindowsecs * fs))	# window length sec -> sample
	signalleneffective = signallen - windowlen	# first to last stride pos
	stride = int(round(signalleneffective / specstrides))	# time step

	# Moving window
	# Stride start and end counters
	counterstart = np.array(range(0,signalleneffective,stride))	# start & end
	counterend = counterstart + windowlen

	magarray = []
	freqarray = []
	for countstart,countend in zip(counterstart,counterend):
		segment = np.abs(signal[countstart:countend])				# window-length segment
		segment = list(segment)
		segment = segment
		segment = np.array(segment)
		mags = abs(np.fft.rfft(segment))						# FFT magnitudes
		freqs = np.abs(np.fft.rfftfreq(segment.size,period))	# FFT frequencies
#		freqs = np.linspace(0, fs/2, len(mags))
		magarray += [ mags ]								# collect FFTs
		freqarray += [ freqs ]

	#============================================
	# Spectrum properties

	spectrummax = int(round(fs/2))
	rowlen = len(freqarray[0])
	elementsperhertz = int(round( rowlen / spectrummax ))

#	print("Elements per hertz:", elementsperhertz)
	xmin = specfreqmin * elementsperhertz
	xmax = specfreqmax * elementsperhertz
	sfmin = int(np.floor(xmin))
	sfmax = int(np.ceil(xmax))
	magarray = np.array([ x[sfmin:sfmax] for x in magarray ])**spectrumpower
	freqarray = [ x[sfmin:sfmax] for x in freqarray ]

	#Detect maximum vector through spectrogram
	maxmags = np.array([ max(mags[1:]) for mags in magarray ])
#	print(maxmags.shape)

	#============================================
	# Loop to define spectrogram as a spectrum sequence
	maxfreqs = []
	for mags, freqs in zip(magarray, freqarray):
		maxofmags = np.max(mags)
		# An error with a very deep voice (60Hz) threw an error
		if maxofmags == 0.0: maxofmags = 0.0001	# a hack, sorry
		mags = mags/maxofmags
		mags = list(mags[1:])
		maxmag = np.max(mags)
		maxmagpos = mags.index(maxmag)
		maxfreq = freqs[maxmagpos]
		maxmags += [maxmag]
		maxfreqs += [maxfreq]

	return np.array(magarray), np.array(freqarray), maxmags, maxfreqs

#===============================================================

# Rotation of spectrogram array as heatmap

def plotspectrogramheatmap(pltobj, freqarray, magarray, signalsecs, specfreqmin, specfreqmax, specgramdotsize, specheatmaptype, fontsize):

	# y-axis as scale of the range, number of spectra in spectrogram
	y = np.linspace(specfreqmin,specfreqmax,len(magarray[0]))

	# x-axis as signal time range, number of spectra in spectrogram
	x = np.linspace(0,signalsecs,len(magarray))

	# Colormap is derived from magnitudes at each frequency/spectrum
	for i, freqvals, magvals in zip(x, freqarray, magarray):
		freqvals = freqvals[1:]
		magvals = magvals[1:]
		x = [i] * len(freqvals)
		pltobj.scatter(
			x,freqvals, c=magvals, cmap=specheatmaptype,
			marker="s", s=specgramdotsize)

	# Spectrogram properties
	pltobj.set_xlim(0,np.ceil(signalsecs))
	sfmin = np.floor(specfreqmin)
	sfmax = np.ceil(specfreqmax)
	pltobj.set_ylim(sfmin,sfmax)
	pltobj.grid(b=True, which="major", axis="both")
	pltobj.set_xlabel("Time (s)", fontsize=fontsize)
	pltobj.set_ylabel("Freq (Hz)", fontsize=fontsize)

	return

def get_2ddct_AM(file_path, dct_num, specwindowsecs, specstrides, cut_off_freq):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    dct2d_all = []
    
    for line in lines:
        line2read = line.strip('\n')
        line2read = line2read.split(',')[0]
        
        #fs, signal = wave.read(line2read)    # read sampling frequency and signal
        signal, fs = librosa.load(line2read, sr=16000)
        if len(signal.shape) == 2:
            signal = signal[:,1]
      
        signallength = len(signal)        # define numerical signal length
        signalseconds = signallength / fs    # define signal length in seconds
        signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1

        #===============================================================
        # Butterworth low pass filter (5 Hz is a typical upper limit for the LF spectrum)
        b, a = butter(5, cut_off_freq / (0.5 * fs), btype="low")	# define Butterworth filter
        '''
        #===============================================================
        #===============================================================
        # AM demodulation (envelope extraction) by full-wave rectification
        envelope = lfilter(b, a, abs(signal))		# apply filter to create lf envelope
        envelope = envelope-min(envelope) / (max(envelope)-min(envelope))	# scale 0 ... 1
        #===============================================================
        # AM low frequency spectral analysis
        # FFT of complete envelope, output magnitude values
        amspecmags = np.abs(np.fft.rfft(envelope))
        amspecmaglen = len(amspecmags)

        # Extraction of low frequency spectrum segment
        lfamspecmaglen = int(round(amspecfreqmax * amspecmaglen / (fs / 2)))
        lfamspecmags = amspecmags[1:lfamspecmaglen]	# DC cutoff
        lfamspmMin = min(lfamspecmags)
        # Scale to 0...1
        lfamspecmags = (lfamspecmags-lfamspmMin) / (np.max(lfamspecmags)-lfamspmMin)

        # Assign LF spectrum frequencies to magnitude values
        lfamspecfreqs = np.linspace(0,fs/2,amspecmaglen)
        lfamspecfreqs = lfamspecfreqs[1:lfamspecmaglen]

        # Identification of highest magnitude spectral frequencies
        amtopmagscount = magscount
        amtopmags = sorted(lfamspecmags)[-amtopmagscount:]
        amtoppos = [ list(lfamspecmags).index(m) for m in amtopmags ]
        amtopfreqs = [ lfamspecfreqs[p] for p in amtoppos ]

        # Redefinition for column chart display
        amrhythmbars = lfamspecfreqs
        amweightlist = lfamspecmags
        '''
        #===============================================================
        # Create AM spectrogram and max magnitude value trajectory
        ammagarray, amfreqarray, ammaxmags, ammaxfreqs = spectrogramarray(signal, fs,amspecfreqmin, amspecfreqmax, specdownsample, 
                                                                          spectrumpower, specwindowsecs, specstrides)
        aaa = np.log10(ammagarray[:,1:])
        dct2d = dct(dct(aaa.T).T)
        dct2d = dct2d[0:dct_num, 0:dct_num]
        dct2d = dct2d.flatten()
        
        dct2d_all.append(dct2d)
    
    return dct2d_all



def get_RF_var_AM(file_path, specwindowsecs, specstrides, cut_off_freq):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    RF_var_all = []
    mag_var_all = []
    
    for line in lines:
        line2read = line.strip('\n')
        line2read = line2read.split(',')[0]
        print(line2read)
        fs, signal = wave.read(line2read)    # read sampling frequency and signal
        if len(signal.shape) == 2:
            signal = signal[:,1]
      
        signallength = len(signal)        # define numerical signal length
        signalseconds = signallength / fs    # define signal length in seconds
        signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1

        #===============================================================
        # Butterworth low pass filter (5 Hz is a typical upper limit for the LF spectrum)
        b, a = butter(5, cut_off_freq / (0.5 * fs), btype="low")	# define Butterworth filter
       
        #===============================================================
        # Create AM spectrogram and max magnitude value trajectory
        ammagarray, amfreqarray, ammaxmags, ammaxfreqs = spectrogramarray(signal, fs,amspecfreqmin, amspecfreqmax, specdownsample, 
                                                                          spectrumpower, specwindowsecs, specstrides)
        
        magg = ammagarray[:,1:]
        free = amfreqarray[:,1:]
        free = free[0,:]
        RF_mag = []
        RF_fre = []
        
        for x in range(len(magg)):
            slice_spec = magg[x,:]
                  
            peaks, _ = find_peaks(slice_spec)
            
            freq1, magg1 = free[peaks], slice_spec[peaks]
        
            ranked = np.argsort(magg1)
            
            if peaks.shape[0] >= 6:
                top_magg_indx = ranked[::-1][:6]
                magg_top = magg1[top_magg_indx]
                freq_top = freq1[top_magg_indx]
                RF_mag += [magg_top]
                RF_fre += [freq_top]
        
        RF_mag1 = np.array(RF_mag)
        RF_fre1 = np.array(RF_fre)
        
        #print(RF_fre1.shape)
        
        RF_var = np.var(RF_fre1, axis = 0)
        mag_var = np.var(RF_mag1, axis = 0)
        RF_var_all += [RF_var]
        mag_var_all += [mag_var]
        
    return RF_var_all, mag_var_all

def get_RF_var_FM(file_path, specwindowsecs, specstrides):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    RF_var_all = []
    mag_var_all = []
    
    for line in lines:
        line2read = line.strip('\n')
        line2read = line2read.split(',')[0]
        print(line2read)
        '''
        fs, signal = wave.read(line2read)    # read sampling frequency and signal
        if len(signal.shape) == 2:
            signal = signal[:,1]
      
        signallength = len(signal)        # define numerical signal length
        signalseconds = signallength / fs    # define signal length in seconds
        signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1
        
        f0array, framerate, frameduration = f0estimate(signal, fs)
        '''
        f0array, framerate, frameduration = f0estimate_rapt(line2read, fs)

        #===============================================================
        # Create AM spectrogram and max magnitude value trajectory
        '''
        ammagarray, amfreqarray, ammaxmags, ammaxfreqs = spectrogramarray(signal, 
                                                                          fs,
                                                                          amspecfreqmin, 
                                                                          amspecfreqmax, 
                                                                          specdownsample, 
                                                                          spectrumpower, 
                                                                          specwindowsecs, 
                                                                          specstrides)
        '''
        
        #===============================================================
        # Create FM spectrogram and max value trajectory
        #f0array = f0array[f0array != 0]
        fmmagarray, fmfreqarray, fmmaxmags, fmmaxfreqs = spectrogramarray(f0array, 
                                                                          framerate, 
                                                                          fmspecfreqmin, 
                                                                          fmspecfreqmax,
                                                                          specdownsample, 
                                                                          spectrumpower, 
                                                                          specwindowsecs, 
                                                                          specstrides)

        
        
        magg = fmmagarray[:,1:]
        free = fmfreqarray[:,1:]
        free = free[0,:]
        RF_mag = []
        RF_fre = []
        
        for x in range(len(magg)):
            #print(magg.shape)
            slice_spec = magg[x,:]
            peaks, _ = find_peaks(slice_spec)
            #print(peaks.shape)
            freq1, magg1 = free[peaks], slice_spec[peaks]
        
            ranked = np.argsort(magg1)
            
            if peaks.shape[0] >= 6:
                top_magg_indx = ranked[::-1][:6]
                magg_top = magg1[top_magg_indx]
                freq_top = freq1[top_magg_indx]
                RF_mag += [magg_top]
                RF_fre += [freq_top]
        
        RF_mag1 = np.array(RF_mag)
        RF_fre1 = np.array(RF_fre)
        
        #print(RF_fre1.shape)
        
        RF_var = np.var(RF_fre1, axis = 0)
        mag_var = np.var(RF_mag1, axis = 0)
        RF_var_all += [RF_var]
        mag_var_all += [mag_var]
        
    return RF_var_all, mag_var_all

def get_2ddct_FM(file_path, dct_num, specwindowsecs, specstrides):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    dct2d_all = []
    
    for line in lines:
        line2read = line.strip('\n')
        line2read = line2read.split(',')[0]
        print(line2read)
        '''
        #fs, signal = wave.read(line2read)    # read sampling frequency and signal
        signal, fs = librosa.load(line2read, sr=16000)
        if len(signal.shape) == 2:
            signal = signal[:,1]
      
        signallength = len(signal)        # define numerical signal length
        signalseconds = signallength / fs    # define signal length in seconds
        signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1

        f0array, framerate, frameduration = f0estimate(signal, fs)
        '''
        f0array, framerate, frameduration = f0estimate_rapt(line2read, fs)

        #===============================================================
        # Create FM spectrogram and max value trajectory
        #f0array = f0array[f0array != 0]
        fmmagarray, fmfreqarray, fmmaxmags, fmmaxfreqs = spectrogramarray(f0array, 
                                                                          framerate, 
                                                                          fmspecfreqmin, 
                                                                          fmspecfreqmax,
                                                                          specdownsample, 
                                                                          spectrumpower, 
                                                                          specwindowsecs, 
                                                                          specstrides)
        
        filtered_arr = fmmagarray[~np.all(fmmagarray == 0, axis=1)]
        aaa = np.log10(filtered_arr[:,1:])
        dct2d = dct(dct(aaa.T).T)
        dct2d = dct2d[0:dct_num, 0:dct_num]
        dct2d = dct2d.flatten()
        
        dct2d_all.append(dct2d)
    
    return dct2d_all
