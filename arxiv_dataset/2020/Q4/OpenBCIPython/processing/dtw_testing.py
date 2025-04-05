import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from lib.dtw import dtw

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

def nomalize_signal(input_signal):
    mean = np.mean(input_signal, axis=0)
    input_signal -= mean
    return input_signal / np.std(input_signal, axis=0)

processed_signal = nomalize_signal(processed_signal)

pattern=np.array(processed_signal.ix[:, 2][start :end]).reshape(-1,1)
data=np.array(processed_signal.ix[:, 2][7000:8000]).reshape(-1,1)

def my_custom_norm(x, y):
    return (x * x) + (y * y)

dist, cost, acc, path = dtw(pattern, data, dist=my_custom_norm)
print 'Normalized distance between the two sounds:', dist, cost, acc

plt.imshow(acc.T, origin='lower', interpolation='nearest')
plt.plot(cost-acc, 'w')
# plt.xlim((-0.5, acc.shape[0]-0.5))
# plt.ylim((-0.5, acc.shape[1]-0.5))
plt.show()

# x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
# y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)
#
# dist, cost, acc, path = dtw(x, y, dist=my_custom_norm)