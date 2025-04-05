from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
from scipy import signal

fsamp = 250
tsample = 1 / fsamp
f_low = 50
f_high = 1
order = 2
channel_vector = [1,2, 3, 4, 5]
n_ch = len(channel_vector)
df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
df = df[channel_vector].dropna(axis=0)

processed_signal = df.copy()

b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))

b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal.ix[:, i]))

Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
b3, a3 = butter(order, Wn, btype='stop')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal.ix[:, i]))

start = 7000
end = 8000
plt.figure(figsize=(12, 8))
for h in range(0, n_ch):
    plt.subplot(5,1,h+1)
    # f, Pxx_spec = signal.periodogram(processed_signal.ix[:, h][start * fsamp:end * fsamp], fsamp, 'flattop',
    #                                  scaling='spectrum')

    # f, Pxx_spec = signal.welch(processed_signal.ix[:, h][start * fsamp:end * fsamp], fsamp, 'flattop', 128, scaling='spectrum')
    # wavelet = signal.ricker
    # widths = np.arange(1, 11)
    # cwtmatr = signal.cwt(processed_signal.ix[:, h][start * fsamp:end * fsamp], wavelet, widths)
    plt.plot(processed_signal.ix[:, h][start :end])
    # plt.semilogy(fsamp, np.sqrt(Pxx_spec))
    # plt.ylim([1e-4, 1e1])
plt.show()


# plt.figure()
# plt.semilogy(f, np.sqrt(Pxx_spec))
# plt.ylim([1e-4, 1e1])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Linear spectrum [V RMS]')
# plt.show()

print "---"
