## getting plot 13 of the paper- orbits of typical simulation and initial stars
## data of typical stars orbits (10 of each category) provided in the data directory
#'typical' are the stars which have change in action around the median change in action of the entire sample (49-51 percentile) 

import np
import matlplotlib.pyplot as plt
star = np.load('stellar-actions-I/data/star.npz')
initial_star = np.load('stellar-actions-I/data/initial_star.npz')

z = star['z']
zi=initial_star['z']

vz= star['vz']
vzi = initial_star['vz']

fig, axs = plt.subplots(3, 1, figsize=(5,7),sharex='all')

    
axs[0].plot(z[0,:],vz[0,:],c='c',label='star particle')
axs[0].plot(zi[0,:],vzi[0,:],c='darkblue',label='initial star')
axs[0].set_ylabel(r'$v_z$ (km/s)')

axs[1].plot(z[1,:],vz[1,:],c='c',label='star particle')
axs[1].plot(zi[1,:],vzi[1,:],c='darkblue',label='initial star')
axs[1].set_ylabel(r'$v_z$ (km/s)')

axs[2].plot(z[2,:],vz[2,:],c='c',label='star particle')
axs[2].plot(zi[2,:],vzi[2,:],c='darkblue',label='initial star')
axs[2].set_ylabel(r'$v_z$ (km/s)')

axs[2].set_xlabel('$z$ (pc)')
axs[2].legend()

plt.tight_layout()
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/zvsvz2.pdf')
