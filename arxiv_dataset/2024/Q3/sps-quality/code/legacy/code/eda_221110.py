# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:40:32 2022

@author: David J. Kedziora
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time

data_filename = "2p5uW_4000cps_day1.txt"
data_folder = "../data/"

# Define by how many units of delay a detection may be off.
# This defines (+-) integration limits when comparing 'areas under peaks'.
# E.g. +-3 ns at 0.256 ns per delay bin means 23 bins are used for integration.
delay_error = 1

# Experimentally set parameters.
snapshot_bin_size = 10
snapshot_unit = 1       # 1 second.
delay_bin_size = None   # Gets calculated later.
delay_unit = 1e-9       # 1 nanosecond.

# Load data file.
# Convert data into a matrix of detection events.
# Rows denote detection delays and columns denote sampled 'snapshots'.
# Ensure rows and columns are indexed appropriately.
df = pd.read_csv(data_folder + data_filename, sep="\t", header=None)
df_events = df[df.columns[1:]]
df_delays = df[df.columns[0]]
df_snapshots = range(0, 
                     df_events.columns.size * snapshot_bin_size, 
                     snapshot_bin_size)
df_events.columns = df_snapshots

# Calculate variables and generate histograms.
delay_bin_size = df_delays[1] - df_delays[0]
delay_range = df_delays[df_delays.size-1] - df_delays[0]

df_hist_snapshot = df_events.sum(axis=0)
df_hist_delay = df_events.sum(axis=1)

# Apply rolling average to the summed histogram to smooth out noise.
kernel_size = max(1, int(delay_error*2/delay_bin_size))
kernel = np.ones(kernel_size)/kernel_size
df_hist_delay_smooth = np.convolve(df_hist_delay, kernel, mode="same")

# Plot delay histogram, raw and smoothed.
fig_hist_delay, ax_hist_delay = plt.subplots()
ax_hist_delay.plot(df_delays, df_hist_delay, label="Raw")
ax_hist_delay.plot(df_delays, df_hist_delay_smooth, 
                   label="Rolling Avg. (" + str(kernel_size) + " bins)")
ax_hist_delay.set_xlabel("Delay (ns)")
ax_hist_delay.set_ylabel("Events per bin (" + str(delay_bin_size) + " ns)")
ax_hist_delay.legend()

# Plot snapshot histogram, raw and smoothed.
fig_hist_snapshot, ax_hist_snapshot = plt.subplots()
ax_hist_snapshot.plot(df_snapshots, df_hist_snapshot, label="Raw")
df_hist_snapshot_smooth = np.convolve(df_hist_snapshot, kernel, mode="same")
ax_hist_snapshot.plot(df_snapshots, df_hist_snapshot_smooth, 
                      label="Rolling Avg. (" + str(kernel_size) + " bins)")
ax_hist_snapshot.set_xlabel("Snapshot (s)")
ax_hist_snapshot.set_ylabel("Events per bin (" 
                            + str(snapshot_bin_size) + " s)")
ax_hist_snapshot.legend()

# Calculate FFT of delay histogram.
t = time()
fft_hist_delay = np.fft.fft(df_hist_delay)
fft_freq_hist_delay = np.fft.fftfreq(n=df_hist_delay.size,
                                     d=delay_bin_size)
print("FFT: %f s" % (time() - t))

# Plot FFT of delay histogram.
fig_fft_hist_delay, ax_fft_hist_delay = plt.subplots()
ax_fft_hist_delay.plot(fft_freq_hist_delay, fft_hist_delay)

# Get the nonzero frequency with max amplitude from the first half of the FFT.
# This should be the pulsing laser frequency; convert it to the pulse period.
# This dictates how many peaks should be visible in the delay histogram.
id_freq_max_amp = np.argmax(fft_hist_delay[1:int(fft_hist_delay.size/2)]) + 1
freq_pulse = fft_freq_hist_delay[id_freq_max_amp]
period_pulse = 1/freq_pulse
n_peaks = 1 + int(delay_range/period_pulse)

# Define the core fitting function for the delay histogram.
# x: domain (delays)
# n: number of peaks (including multi-photon-emission) due to laser frequency
# bg: background number of events (noise)
# delta: a shift in delay marking the first peak
# amps: n amplitudes for the n peaks
# taus: n exponential decay rates for the n peaks
def func(x, n, bg, delta, amps, taus):
    fit = bg
    for i in range(n_peaks):
        fit += amps[i]*np.exp(-abs(x-delta-i*period_pulse)/taus[i])
    return fit

# A basic form of the core fitting function with four arguments.
# The third and fourth denote a common peak amplitude and decay rate.
def func_basic(x, n, *args):
    bg, delta, amp_all, tau_all = args[0]
    # print("BG %f. Delta %f. Amp all %f. Tau all %f." % (bg, delta, amp_all, tau_all))
    return func(x, n, bg, delta, [amp_all]*n, [tau_all]*n)

# A fine form of the core fitting function with 2n arguments.
# Assumes the background and delta are already optimised.
# The first n args are amplitudes, and the second n args are decay rates.
def func_fine(x, n, bg, delta, *args):
    return func(x, n, bg, delta, list(args[0][:n]), list(args[0][n:2*n]))

# A fitting function for a single peak with two arguments.
# Index i denotes which peak is being optimised.
# Mutable p_current is surgically edited with a new amplitude and decay rate.
def func_peak(x, n, i, bg, delta, p_current, *args):
    p_current[i] = args[0][0]
    p_current[i+n] = args[0][1]
    return func(x, n, bg, delta, p_current[:n], p_current[n:2*n])

# Run an initial fit, prioritising background and delta optimisation.
p_init = [0, period_pulse/2] + [1, 1]
p_bounds = ([0, 0] + [0, 0],
            [np.inf, period_pulse] + [np.inf, np.inf])

t = time()
p_opt, p_cov = curve_fit(lambda x, *p: func_basic(x, n_peaks, p), xdata=df_delays, ydata=df_hist_delay, p0=p_init, bounds=p_bounds)
print("Basic fit: %f s" % (time() - t))

bg_opt, delta_opt, amp_all, tau_all = p_opt

# p_init = [amp_all]*n_peaks + [tau_all]*n_peaks
# p_bounds = ([0]*2*n_peaks,
#             [np.inf]*2*n_peaks)

# t = time()
# p_opt, p_cov = curve_fit(lambda x, *p: func_fine(x, n_peaks, bg_opt, delta_opt, p), xdata=df_delays, ydata=df_hist_delay, p0=p_init, bounds=p_bounds)
# print("Simultaneous peak fit: %f s" % (time() - t))

# Run an iterative fit to refine each peak.
p_current = [amp_all]*n_peaks + [tau_all]*n_peaks
p_bounds = ([-np.inf, 0], [np.inf]*2)
t = time()
for i_peak in range(n_peaks):
    # print(i_peak)
    p_init = [p_current[i_peak], p_current[i_peak+n_peaks]]
    p_opt, p_cov = curve_fit(lambda x, *p: func_peak(x, n_peaks, i_peak, bg_opt, delta_opt, p_current, p), xdata=df_delays, ydata=df_hist_delay, p0=p_init, bounds=p_bounds)
print("Iterative peak fit: %f s" % (time() - t))

# Calculate the SPS quality as a ratio between the smallest and other peaks.
amps = p_current[:n_peaks]

id_amp_min = np.argmin(amps)
amp_min = amps[id_amp_min]
amp_avg_else = np.mean(amps[:id_amp_min] + amps[id_amp_min+1:])
g2zero = amp_min/amp_avg_else

# Plot delay histogram and fit.
fig_fit, ax_fit = plt.subplots()
ax_fit.plot(df_delays, df_hist_delay, label="Raw")
ax_fit.plot(df_delays, func_fine(df_delays, n_peaks, bg_opt, delta_opt, p_current), label="Fit")
ax_fit.set_xlabel("Delay (ns)")
ax_fit.set_ylabel("Events per bin (" + str(delay_bin_size) + " ns)")
ax_fit.legend()
