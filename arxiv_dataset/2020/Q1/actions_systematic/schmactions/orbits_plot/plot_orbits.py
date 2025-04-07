import sys
sys.path.append('../')
from schmactions import schmactions

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

textwidth = 7.10000594991
columnwidth = 3.35224200913

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

init_pos = [8, 0, 0] * u.kpc

init_vel_list = [[0, -190, 10] * u.km/u.s,
                 [0, -190, 50] * u.km/u.s,
                 [0, -190, 190] * u.km/u.s]

fig, ax = plt.subplots(2, 3, sharey='row', figsize=(textwidth,5.5))

for init_vel, x in zip(init_vel_list, np.transpose(ax)):
    s = schmactions(init_pos, init_vel, save_orbit=True)
    pos = np.transpose(s.orbit.xyz.to_value(u.kpc))[:2000] # only grab the first 2 Gyr

    print(s.res['freqs'])

    R = np.linalg.norm(pos[:,:2], axis=1)
    phi = np.mod(np.arctan2(pos[:,1], pos[:,2]), 2.*np.pi)

    # x vs. y
    x[0].plot(pos[:,0], pos[:,1])
    x[0].set_xlim(-12, 12)
    x[0].set_ylim(-12, 12)

    # R vs. z
    x[1].plot(R, pos[:,2])
    x[1].set_xlim(5, 11)
    x[1].set_ylim(-10, 10)

ax[0][0].set_ylabel(r'$y\,[\,\text{kpc}\,]$')
for x in ax[0]:
    x.set_xlabel(r'$x\,[\,\text{kpc}\,]$')

ax[0][0].set_title(r'\texttt{thin-disk}')
ax[0][1].set_title(r'\texttt{thick-disk}')
ax[0][2].set_title(r'\texttt{halo}')

ax[1][0].set_ylabel(r'$z\,[\,\text{kpc}\,]$')
for x in ax[1]:
    x.set_xlabel(r'$R\,[\,\text{kpc}\,]$')

fig.tight_layout()
fig.savefig('orbits.pdf')
