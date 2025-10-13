
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
from datetime import datetime
import librosa
from scipy.signal import medfilt, hilbert
from scipy.ndimage import median_filter
# RFA custom module import
from module_F0_v1 import *	# FM demodulation (F0 estimation, 'pitch' tracking)
#===============================================================
# Assign configuration parameters to variables (see .conf file)
from rfa_single_conf import *

def spectrogramarray(signal, fs, specfreqmin, specfreqmax, spectrumpower, specwindowsecs, specstrides):

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

	return np.array(magarray), np.array(freqarray)

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

def get_AM_spectrogram(file_path, specwindowsecs, specstrides):
    fs, signal = wave.read(file_path)    # read sampling frequency and signal
    if len(signal.shape) == 2:
        signal = signal[:,1]
      
    signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1
    signallength = len(signal)        # define numerical signal length
    signalseconds = signallength / fs    # define signal length in seconds
    envelope = np.abs(hilbert(signal))
    envelope = median_filter(envelope, size=501, mode="nearest")
    envelope = envelope / np.max(envelope)

    #===============================================================
    # Create AM spectrogram and max magnitude value trajectory
    ammagarray, amfreqarray = spectrogramarray(envelope,
                                                                      fs,
                                                                      amspecfreqmin,
                                                                      amspecfreqmax,
                                                                      spectrumpower,
                                                                      specwindowsecs,
                                                                      specstrides)
        
    return ammagarray, amfreqarray


def get_FM_spectrogram(file_path, specwindowsecs, specstrides):

    '''
    signal, fs = librosa.load(file_path, sr=16000)
    if len(signal.shape) == 2:
        signal = signal[:,1]

    signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1
    signallength = len(signal)        # define numerical signal length
    signalseconds = signallength / fs    # define signal length in seconds

    f0array, framerate, frameduration = f0estimate(signal, fs)
    '''
    f0array, framerate, frameduration = f0estimate_rapt(file_path, 16000)
    f0array = median_filter(f0array, size=101, mode="nearest")

    #===============================================================
    # Create FM spectrogram and max value trajectory
    fmmagarray, fmfreqarray = spectrogramarray(f0array,
                                                                      framerate,
                                                                      fmspecfreqmin,
                                                                      fmspecfreqmax,
                                                                      spectrumpower,
                                                                      specwindowsecs,
                                                                      specstrides)
    return fmmagarray, fmfreqarray
