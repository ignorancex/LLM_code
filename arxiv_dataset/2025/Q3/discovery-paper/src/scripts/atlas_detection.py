"""
Created on Wed Jul 8 2025

@author: afeinstein20
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

# Setup the style
params = Table.read('./src/data/rcParams.txt', format='csv')
for i in range(len(params)):
    plt.rcParams[params['name'][i]] = params['value'][i]
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True

dat = np.loadtxt('./src/data/3I_xc.txt', skiprows=1)

fig = plt.figure(figsize=(8,6))
fig.set_facecolor('w')

lw = 3

plt.semilogy(dat[:,0], dat[:,1]/dat[:,3], '#011638', lw=lw,
             label='Volume / Crossing\n'+'time [au$^3$ yr$^{-1}$]')
plt.semilogy(dat[:,0], dat[:,1], '#9055a2', lw=lw,
             label='Volume [au$^3$]')
plt.semilogy(dat[:,0], dat[:,3], '#e8c1c5', lw=lw,
             label='Time [yr]')


leg = plt.legend(facecolor='w', loc='upper right')
for line in leg.get_lines():
    line.set_linewidth(5.0)

plt.grid(alpha=0.3)

plt.xlabel('H [mag]', fontsize=18)
plt.ylabel('Volume and V/dt', fontsize=18)

plt.xlim(10, 23)
plt.ylim(0.001, 10000)

plt.savefig('./src/tex/figures/3I_xc_v2.png',
            bbox_inches='tight', dpi=600)
