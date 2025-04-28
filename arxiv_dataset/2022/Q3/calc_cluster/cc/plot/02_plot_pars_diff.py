# plots a histogram of PARS grid magnitude difference percentages
import sys, time, pickle
sys.path.append('..')
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import config as cf

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern"],
    "font.size": 20
})

print('Loading PARS...', end='', flush=True)
start = time.time()
with open('../../data/pars_grid_ZM' + str(cf.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.' + '\n', flush=True)

max_mag = 7
bins = np.linspace(0, max_mag, max_mag + 1)

iAV = np.argmin(np.abs(pars.av - cf.A_V))
aAV = pars.dims.index('av')
Mag = pars.Mag.take([iAV], axis=aAV)
tau_diff = np.abs(np.diff(Mag, axis=pars.dims.index('tau'))).flatten() / cf.std[0]
omega_diff = np.abs(np.diff(Mag, axis=pars.dims.index('omega'))).flatten() / cf.std[0]
inc_diff = np.abs(np.diff(Mag, axis=pars.dims.index('inc'))).flatten() / cf.std[0]
gamma_diff = np.abs(np.diff(Mag, axis=pars.dims.index('gamma'))).flatten() / cf.std[0]

fig, ax = plt.subplots()
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.hist((tau_diff, omega_diff, inc_diff, gamma_diff), bins=bins, alpha=0.8)
ax.legend((r'$\tau$', r'$\omega$', r'$i$', r'$\gamma$'))
ax.set_xlabel(r'$\Delta m / \sigma_m$')
ax.set_xticks(bins)
plt.tight_layout()
plt.savefig('../../data/pars_diff.pdf', dpi=300)
plt.close()