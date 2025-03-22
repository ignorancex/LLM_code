# code to get fig4 plot from the paper- orbit and action evolution of single star particle- filtered and unfiltered.
# can get action.h5 file from the following website and change the file paths here accordingly ('stellar-actions-I/data/actions.h5' in following code)
# https://www.mso.anu.edu.au/~arunima/stellar-actions-I-data/

import numpy as np
import h5py
imoprt matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Read HDF5
with h5py.File('stellar-actions-I/data/actions.h5', 'r') as f:
    J = f['actions'][:]
    C = f['coordinates'][:]
    V = f['velocities'][:]
    A = f['age'][:]
    kappa = f['kappa'][:]
    nu = f['nu'][:]
    R_g = f['R_g'][:]
    M = f['mass'][:]
    ID = f['ID'][:]
    

k=1
i=101
# Find the index of the star with ID 22878914
star_index = (ID == 22878914).nonzero()[0][0]  # Get the index of the star

# Get the data for that star using the index
J = J[star_index]
C = C[star_index]
V = V[star_index]
A = A[star_index]
kappa = kappa[star_index]
nu = nu[star_index]
R_g = R_g[star_index]
M = M[star_index]

J=J/M      #getting specific actions
R=C[:,:,1].flatten()
z=C[:,:,2].flatten()
times = np.arange(R.shape[0])

# Butterworth filter setup
fs = 1.0  # Sampling frequency (1/snapshot interval)
cutoff = 1 / 30  # Desired cutoff frequency (inverse of time period in snapshots)
order = 10  # Filter order

# Design the filter
b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)

class Counter:
    def __init__(self):
        self.value = 0  # Initialize the counter

    def increment(self):
        self.value += 1  # Increment the counter


# Define a function to filter a single time series while ignoring NaNs
def filter_time_series(ts, b, a, counter):
    finite_mask = ~np.isnan(ts)  # Mask for valid (non-NaN) values
    filtered = np.full_like(ts, np.nan) # Initialize output with NaNs
    if finite_mask.sum() > max(len(a), len(b)) * 3:  
        filtered[finite_mask] = filtfilt(b, a, ts[finite_mask])  # Apply filter to valid data
        counter.increment()  # Increment the counter
    return filtered

data_JR = J[:,0,0]
counter=Counter()
filtered_data_JR=np.apply_along_axis(filter_time_series, axis=0, arr=data_JR, b=b, a=a, counter=counter)

data_Jphi = J[:,0,1]
counter=Counter()
filtered_data_Jphi=np.apply_along_axis(filter_time_series, axis=0, arr=data_Jphi, b=b, a=a, counter=counter)

data_Jz = J[:,0,2]
counter=Counter()
filtered_data_Jz=np.apply_along_axis(filter_time_series, axis=0, arr=data_Jz, b=b, a=a, counter=counter)


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, axs = plt.subplots(5, 1, figsize=(6,15),sharex='all')

ax1 = axs[0]
ax1.plot(times, R, color="purple")

ax1.set_ylabel("R (pc)")# (kpc km/s)")


ax2=axs[1]
ax2.plot(times, z, color="purple")
ax2.set_ylabel("z(pc)")


axs[2].plot(times, data_JR,label='data',color="grey")
axs[2].plot(times, filtered_data_JR,label='filtered', color="magenta")

axs[2].set_ylabel("$J_R$ (kpc km/s)")
axs[2].legend(loc='upper left')

ax1 = axs[3]
ax1.plot(times, data_Jphi, label='data',color="grey")
ax1.plot(times, filtered_data_Jphi, label='filtered', color="magenta")


ax1.set_ylabel("$J_{\phi}$ (kpc km/s)")
ax1.legend(loc='upper left')


ax2=axs[4]
ax2.plot(times, data_Jz,label='data',color="grey")
ax2.plot(times, filtered_data_Jz, label='filtered', color="magenta")

ax2.set_ylabel("$J_z$ (kpc km/s)")
ax2.set_xlabel("Time (Myr)")
ax2.legend(loc='upper right')

plt.tight_layout()

# plt.savefig('/g/data/jh2/ax8338/action/heatmap/orbit_paper.pdf')
    
