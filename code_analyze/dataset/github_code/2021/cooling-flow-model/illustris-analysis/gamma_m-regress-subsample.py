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
import random
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

def RegisterCustomColormaps():
    """Adds the 'RdGn' colormap to matplotlib's database
    """
    import matplotlib as mpl 
    LUTSIZE = mpl.rcParams['image.lut']
        
    # RdGn colormap
    RdGn =  [ [0.40392156862745099,  0.0                ,  0.12156862745098039],
              [0.69803921568627447,  0.09411764705882353,  0.16862745098039217],
              [0.83921568627450982,  0.37647058823529411,  0.30196078431372547],
              [0.95686274509803926,  0.6470588235294118 ,  0.50980392156862742],
              [0.99215686274509807,  0.85882352941176465,  0.7803921568627451 ],
              [1.0                ,  1.0                ,  1.0                ],
              #[0.96862745098039216,  0.96862745098039216,  0.96862745098039216],
              [0.85098039215686272,  0.94117647058823528,  0.82745098039215681],
              [0.65098039215686276,  0.85882352941176465,  0.62745098039215685],
              [0.35294117647058826,  0.68235294117647061,  0.38039215686274508],
              [0.10588235294117647,  0.47058823529411764,  0.21568627450980393],
              [0.0                ,  0.26666666666666666,  0.10588235294117647] ]
    
    plt.register_cmap(cmap=colors.LinearSegmentedColormap.from_list('RdGn', RdGn, LUTSIZE) )

RegisterCustomColormaps()

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
condition = np.logical_or(SFR==0, SFR!=0)
density = np.array(data['density'])[condition]
Pmag = np.array(data['Pmag'])[condition]
temperature = np.array(data['temperature'])[condition]
data.close()

tmp = density.shape[0] #restrict the number of points to analyze for fast prototyping

color_min, color_max = 0.3, 1.8 #np.min(temperature[:tmp]), np.max(temperature[:tmp])
bincount = 80
bins = np.log10(np.logspace( np.min(np.log10(density/mp)[:tmp]), np.max(np.log10(density/mp)[:tmp]), 
                            bincount))
print('Width: %.1f dex'%(bins[1]-bins[0]))

print('Plotting gamma_m')
fig, ax = plt.subplots(1, 1, figsize=(13,10))

for i in range(1,len(bins)):
    #print('Bin #: ',i)
    choose = None
    if i != (len(bins)-1):
        choose = np.logical_and(np.log10(density/mp)[:tmp]>=bins[i-1], np.log10(density/mp)[:tmp]<bins[i])
    else:
        choose = np.logical_and(np.log10(density/mp)[:tmp]>=bins[i-1], np.log10(density/mp)[:tmp]<=bins[i])
    
    density_supset     = (np.log10(density/mp)[:tmp])[choose]
    pressure_supset    = (np.log10(Pmag/kB)[:tmp])[choose]
    temperature_supset = (temperature[:tmp])[choose]
       
    #choose at max 1000 points
    points_no = min(1000, pressure_supset.shape[0])
    subsample = np.column_stack((density_supset, pressure_supset, temperature_supset))
    subsample = subsample[np.random.choice(subsample.shape[0], points_no, replace=False), :]
    density_subsample     = subsample[:,0]
    pressure_subsample    = subsample[:,1]
    temperature_subsample = subsample[:,2]
    
    temperature_subsample /= np.percentile(temperature_subsample, 50)
    #smooth the colors
    temperature_subsample = lowess(temperature_subsample, density_subsample , frac=0.2)[:,1]
        
    scat = ax.scatter(density_subsample, pressure_subsample, 
                          c=temperature_subsample, cmap = 'RdGn',
                          norm=colors.Normalize(vmin=color_min, vmax=color_max), zorder=-10)
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
    ax.plot(dens_line, m*dens_line + b, '-', color='gray', alpha=0.01)

txt = r"$\rm \gamma _m \approx %.2f \pm %.1f \times 10^{%d}$"%( p[1], fman(se[1].item()), fexp(se[1].item()) )
ax.plot(dens_line, p[0]+p[1]*dens_line, color='tab:gray', linewidth=5, linestyle='-', 
        label=r'$\rm p_{mag} \propto \rho ^ {\gamma_m}$'+'\n'+txt)

cbaxes = inset_axes(ax, width="45%", height="5%", loc='lower right',
                    bbox_to_anchor=(0.08,0.07,0.9,0.8), bbox_transform=ax.transAxes) 
cb = plt.colorbar(scat, fraction=0.025, orientation = 'horizontal', cax=cbaxes)#, #
                  #ticks=[1e4,1e5,1e6,1e7,1e8])
cb.set_label(r'$\rm Delta$ Temperature', rotation=0, size=26, labelpad=-86)
cb.ax.tick_params(labelsize=25, pad=5) 



ax.legend(loc='upper left', prop={'size': 26}, ncol=1, shadow=False, fancybox=True, framealpha=0.4)

plt.savefig('pres-mag-subsample.png', transparent=True, bbox_inches='tight')
plt.show()
