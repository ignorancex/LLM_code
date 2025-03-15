## code for plot 5 of the paper- distributions of actions at the last snapshot
## can get action.h5 file from the following website and change the file paths here accordingly ('stellar-actions-I/data/actions.h5' in following code)
## https://www.mso.anu.edu.au/~arunima/stellar-actions-I-data/

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import h5py


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

with h5py.File('stellar-actions-I/data/actions.h5', 'r') as f:
    J = f['actions'][:]
    C = f['coordinates'][:]
    V = f['velocities'][:]
    A = f['age'][:]
    M = f['mass'][:]

J=J[-1,:,:]/M.T  #specific actions at the last snapshot
R=C[-1,:,1]/1000  #radius in kpc at the last snapshot
A=A.T.flatten()

def distribution(ages, Jz,age_bin_size=50):
    # Bin ages into intervals 
    age_bins = np.arange(0,470, age_bin_size)  # Bin edges
    age_bin_centers = 0.5 * (age_bins[:-1] + age_bins[1:])
    
    # Calculate percentiles for each bin
    percentiles = [2.5, 16, 50, 84, 97.5]  # Define the percentiles
    percentile_data = {p: [] for p in percentiles}
    
    for i in range(len(age_bins) - 1):
        mask = (ages >= age_bins[i]) & (ages < age_bins[i + 1])
        if np.sum(mask) > 0:  # Avoid empty bins
            for p in percentiles:
                percentile_data[p].append(np.percentile(Jz[mask], p))
                # print(mask.sum())
        else:
            for p in percentiles:
                percentile_data[p].append(np.nan)
    return percentiles,age_bin_centers,percentile_data

##garzon2024 plot data:
times_9 =[0.1,0.12555555555555600,0.15555555555555600,0.18,0.20222222222222200,
          0.24333333333333300,0.26,0.29555555555555600,0.32,0.34,0.36888888888888900,
          0.3911111111111110,0.42000000000000000,0.45]
Jz_9 = [0.7864077669902910,0.8058252427184470,0.825242718446602,0.825242718446602,
        0.8640776699029130,0.9029126213592230,0.9223300970873790,0.9805825242718450,
        1.0194174757281600,1.0388349514563100,1.058252427184470,1.0776699029126200,
        1.0970873786407800,1.116504854368930]

times_12= [0.1,0.12222222222222200,0.14222222222222200,0.1788888888888890,
           0.20555555555555600,0.2311111111111110,0.26,0.2966666666666670,
           0.34,0.3788888888888890,0.42,0.45]
Jz_12 = [2.087378640776700,2.2038834951456300,2.2815533980582500,2.456310679611650,
         2.5922330097087400,2.7087378640776700,2.844660194174760,3.0000000000000000,
         3.213592233009710,3.4271844660194200,3.640776699029130,3.776699029126210]

fig, axs = plt.subplots(3, 3, figsize=(10,6),sharex='all',sharey='row')

ages = A[(R<5)&(R>4)]
JR = J[:,0][(R<5)&(R>4)]
percentiles,age_bin_centers,percentile_data = distribution(ages,JR)

axs[0,0].plot(age_bin_centers, percentile_data[50],color='orangered',label='R= 4-5 kpc')
axs[0,0].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='orangered',edgecolor='none',alpha=0.1)


ages = A[(R<10)&(R>9)]
JR = J[:,0][(R<10)&(R>9)]
percentiles,age_bin_centers,percentile_data = distribution(ages,JR)


axs[0,1].plot(age_bin_centers, percentile_data[50],color='blue',label='R= 9-10 kpc')
axs[0,1].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='blue',edgecolor='none',alpha=0.1)

ages = A[(R<13)&(R>12)]
JR = J[:,0][(R<13)&(R>12)]
percentiles,age_bin_centers,percentile_data = distribution(ages,JR)


axs[0,2].plot(age_bin_centers, percentile_data[50],color='green',label='R= 9-10 kpc')
axs[0,2].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='green',edgecolor='none',alpha=0.1)
axs[0,2].set_title('R = 12-13 kpc')

axs[0,0].set_ylabel('$J_R$ (kpc km/s)')
axs[0,0].set_title('R = 4-5 kpc')
axs[0,1].set_title('R = 9-10 kpc')


ages = A[(R<5)&(R>4)]
Jphi = J[:,1][(R<5)&(R>4)]
percentiles,age_bin_centers,percentile_data = distribution(ages,Jphi)


axs[1,0].plot(age_bin_centers, percentile_data[50],color='orangered',label='R= 4-5 kpc')
axs[1,0].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='orangered',edgecolor='none',alpha=0.1)

ages = A[(R<10)&(R>9)]
Jphi = J[:,1][(R<10)&(R>9)]
percentiles,age_bin_centers,percentile_data = distribution(ages,Jphi)


axs[1,1].plot(age_bin_centers, percentile_data[50],color='blue',label='R= 9-10 kpc')
axs[1,1].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='blue',edgecolor='none',alpha=0.1)

ages = A[(R<13)&(R>12)]
Jphi = J[:,1][(R<13)&(R>12)]

percentiles,age_bin_centers,percentile_data = distribution(ages,Jphi)


axs[1,2].plot(age_bin_centers, percentile_data[50],color='green',label='R= 12-13 kpc')
axs[1,2].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='green',edgecolor='none',alpha=0.1)

axs[1,0].set_ylabel('$J_{\phi}$ (kpc km/s)')


ages = A[(R<5)&(R>4)]
Jz = J[:,2][(R<5)&(R>4)]
percentiles,age_bin_centers,percentile_data = distribution(ages,Jz)


axs[2,0].plot(age_bin_centers, percentile_data[50],color='orangered',label='our data')
axs[2,0].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='orangered',edgecolor='none',alpha=0.1)



ages = A[(R<10)&(R>9)]
Jz = J[:,2][(R<10)&(R>9)]
percentiles,age_bin_centers,percentile_data = distribution(ages,Jz)


axs[2,1].plot(age_bin_centers, percentile_data[50],color='blue')#,label='our data')
axs[2,1].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='blue',edgecolor='none',alpha=0.1)
axs[2,1].plot(np.array(times_9)*1000,Jz_9,'r',label='Garzon+ 2024')
axs[2,1].legend()

ages = A[(R<13)&(R>12)]
Jz = J[:,2][(R<13)&(R>12)]
percentiles,age_bin_centers,percentile_data = distribution(ages,Jz)


axs[2,2].plot(age_bin_centers, percentile_data[50],color='green')#,label='our data')
axs[2,2].fill_between(age_bin_centers , percentile_data[16],percentile_data[84],color='green',edgecolor='none',alpha=0.1)
axs[2,2].plot(np.array(times_12)*1000,Jz_12,'r',label='Garzon+ 2024')
axs[2,2].legend()

axs[2,0].set_xlabel("Age (Myr)")
axs[2,1].set_xlabel("Age (Myr)")
axs[2,2].set_xlabel("Age (Myr)")
axs[2,0].set_ylabel('$J_z$ (kpc km/s)')



plt.tight_layout()

# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/R1_new_vz.pdf')
    

    
  
    
