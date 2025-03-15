# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:22:58 2021

@author: alankar
"""

import numpy as np
import h5py as hdf
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycse import regress
#from scipy import interpolate
from decimal import Decimal
import statsmodels.api as sm

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

redshift = 0.5
a = 1/(1+redshift)
mp = 1.6726e-24
kB = 1.3807e-16
Msun = 2e33
yr = 365*24*60**2
kpc = 3.086e21
ckpc = kpc*a
h = 0.67
XH = 0.76

data = hdf.File('../data-gasandcloud.h5', 'r')
print(list(data.keys()))

SFR = np.array(data['SFR'])
#exclude ISM --> SFR == 0
condition = SFR==0 #np.logical_or(SFR==0, SFR!=0)
density = np.array(data['density'])[condition]
Pmag = np.array(data['Pmag'])[condition]
temperature = np.array(data['temperature'])[condition]
data.close()

tmp = density.shape[0] #restrict the number of points to analyze for fast prototyping

color_min, color_max = np.min(temperature[:tmp]), np.max(temperature[:tmp])

print('Plotting gamma_m')
fig, ax = plt.subplots(1, 1, figsize=(13,10))

scatter = ax.scatter(np.log10(density/mp)[:tmp], np.log10(Pmag/kB)[:tmp], s=3, 
                      c=temperature[:tmp], cmap = 'viridis', norm=colors.LogNorm(), zorder=-10,
                      vmin=color_min, vmax=color_max)
ax.set_rasterization_zorder(0)
ax.set_xlabel(r'Density [$\rm log\ m_p\ cm^{-3}$]',size=32)
ax.set_ylabel(r'Magnetic Pressure [$\rm log\ k_B\ K\ cm^{-3}$]',size=32)
ax.tick_params(axis='both', which='major', labelsize=28, direction="out", pad=5)
ax.tick_params(axis='both', which='minor', labelsize=28, direction="out", pad=5)
ax.grid()

dens_line = np.linspace(np.min(np.log10(density/mp)[:tmp]), 
                        np.max(np.log10(density/mp)[:tmp]), 50)
pfit, stats = np.polynomial.Polynomial.fit(np.log10(density/mp)[:tmp],
                                           np.log10(Pmag/kB)[:tmp], 1, 
                                           full=True, 
                                           window=(np.min(np.log10(density/mp)[:tmp]), np.max(np.log10(density/mp)[:tmp])),
                                           domain=(np.min(np.log10(density/mp)[:tmp]), np.max(np.log10(density/mp)[:tmp])))
C, gamma_m = pfit
print(pfit)

x = np.log10(density/mp)[:tmp]
y = np.log10(Pmag/kB)[:tmp]
A = np.column_stack([x**0, x])

p, pint, se = regress(A, y, alpha=0.05)

N = 500
B = np.random.normal(p[0], se[0], N)
M = np.random.normal(p[1], se[1], N)
for b,m in zip(B, M):
    ax.plot(dens_line, m*dens_line + b, '-', color='tomato', alpha=0.01)

txt = r"$\rm \gamma _m \approx %.2f \pm %.1f \times 10^{%d}$"%( p[1], fman(se[1].item()), fexp(se[1].item()) )
ax.plot(dens_line, p[0]+p[1]*dens_line, color='tomato', linewidth=5, linestyle='-', 
        label=r'$\rm p_{mag} \propto \rho ^ {\gamma_m}$'+'\n'+txt)


cbaxes = inset_axes(ax, width="45%", height="5%", loc='lower right',
                    bbox_to_anchor=(0.08,0.07,0.9,0.8), bbox_transform=ax.transAxes) 
cb = plt.colorbar(scatter, fraction=0.025, orientation = 'horizontal', cax=cbaxes, #
                  ticks=[1e4,1e5,1e6,1e7,1e8])
cb.set_label(r'Temperature [K]', rotation=0, size=26, labelpad=-86)
cb.ax.tick_params(labelsize=25, pad=5) 

ax.legend(loc='upper left', prop={'size': 26}, ncol=1, shadow=False, fancybox=True, framealpha=0.4)

plt.savefig('pres-mag.png', transparent=True, bbox_inches='tight')
plt.show()
