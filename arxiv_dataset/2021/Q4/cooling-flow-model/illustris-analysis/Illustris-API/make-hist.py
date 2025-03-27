# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:39:35 2021

@author: alankar
"""

import numpy as np
import h5py 
import matplotlib
import matplotlib.pyplot as plt
#import cmasher as cmr
#from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import glob
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from itertools import product

## Plot Styling
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.minor.width'] = 0.45
matplotlib.rcParams['xtick.minor.width'] = 0.45
matplotlib.rcParams['ytick.major.size'] = 2.75
matplotlib.rcParams['xtick.major.size'] = 2.75
matplotlib.rcParams['ytick.minor.size'] = 1.75
matplotlib.rcParams['xtick.minor.size'] = 1.75
matplotlib.rcParams['legend.handlelength'] = 2
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams['axes.axisbelow'] = True
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{cmbright}  \usepackage[T1]{fontenc}')

XH = 0.76
gamma = 5/3.
kB = 1.3807e-16
mp = 1.6726e-24
pc = 3.086e18
yr = 365*24*60**2
Msun = 1.989e33
Myr = 1e6*yr

h = 0.6774

UnitLength = 1e3*pc
UnitTime = 1e9*yr
UnitMass = 1e10*Msun


UnitVelocity = UnitLength/UnitTime
UnitEnergy = UnitMass * UnitLength**2 / UnitTime**2


start = 28
stop  = 99

for snapshot in range(start, stop+1):
    mini = True
    with h5py.File(glob.glob('./hist/snap_%03d_cutout*'%snapshot)[0],'r') as hdf:
        Redshift = np.array(hdf['Redshift'])
        NumberDensity = np.array(hdf['NumberDensity'])
        Temperature = np.array(hdf['Temperature'])
        Volume = np.array(hdf['Volume'])/UnitLength**3
        try:
            Lambda = -np.array(hdf['Lambda'])
            mu     = np.array(hdf['mu'])
            mini = False
            #exclude heating
            include = Lambda>0
            NumberDensity = NumberDensity[include]
            Temperature = Temperature[include]
            Volume = Volume[include]
            Lambda = Lambda[include]
            mu     = mu[include]
        except: pass
        Mass = NumberDensity*Volume
        a = 1/(1+Redshift)
    
    if mini:
      fig = plt.figure()
      plt.plot([-3.5,], [6,], 'X', color='ghostwhite', markersize=5, alpha=0.8, mec='slategrey')
      plt.hist2d(np.log10(NumberDensity), np.log10(Temperature), weights=Volume, bins=(500,500), 
                 density=True, cmap='magma', norm=LogNorm(vmin=10**-8.3, vmax=5) )
      
      #plt.xscale('log')
      #plt.yscale('log')
      cbar = plt.colorbar()
      cbar.ax.set_ylabel('Volume filling fraction', size=9)
      cbar.ax.tick_params(labelsize=8)
      plt.text(-0.5, 8.1, 'z=%.2f'%float(Redshift), size=15)
      plt.xlim(xmin=-6.5, xmax=1.5)
      plt.ylim(ymin=3.8, ymax =8.5)
      plt.grid()
      plt.xlabel(r'Number density $\rm (cm^{-3})\ [log]$', size=12)
      plt.ylabel(r'Temperature $\rm (K)\ [log]$', size=12)
      plt.savefig('./hist/snap_%03d.png'%snapshot, transparent=True)
      plt.show()
      plt.close(fig)
        
    if not(mini):        
        print(snapshot)
        fig = plt.figure()
        
        plt.hist2d(np.log10(NumberDensity), np.log10(Temperature), weights=Volume, bins=(500,500), 
                   density=True, cmap='magma', norm=LogNorm(vmin=10**-8.3, vmax=5) )
        
        #plt.xscale('log')
        #plt.yscale('log')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Volume filling fraction', size=9)
        cbar.ax.tick_params(labelsize=8)
        plt.text(-0.5, 8.1, 'z=%.2f'%float(Redshift), size=15)
        plt.xlim(xmin=-6.5, xmax=1.5)
        plt.ylim(ymin=3.8, ymax =8.5)
        plt.grid()
        plt.xlabel(r'Number density $\rm (cm^{-3})\ [log]$', size=12)
        plt.ylabel(r'Temperature $\rm (K)\ [log]$', size=12)
        
        
        tcool = (1./(gamma-1))*(kB*Temperature)/(NumberDensity*Lambda*(XH*mu)**2)
        tcool = np.log10(tcool/Myr)
        n_min = -6.5 
        n_max = 1.5
        T_min = 3.8
        T_max = 8.5
        N_n, N_T = 100, 101
        T_vals = np.linspace(T_min, T_max, N_T)
        n_vals = np.linspace(n_min, n_max, N_n)
        tcool_vals = np.zeros((N_T-1,N_n-1))
        
        #choice_T = map(lambda r:np.logical_and(np.log10(Temperature  )>=T_vals[r], np.log10(Temperature  )<T_vals[r+1]), np.arange(N_T-1))
        #choice_n = map(lambda c:np.logical_and(np.log10(NumberDensity)>=n_vals[c], np.log10(NumberDensity)<n_vals[c+1]), np.arange(N_n-1))
        Temperature   = np.log10(Temperature)
        NumberDensity = np.log10(NumberDensity)
        def tcool_fill(tup):
            r, c = tup
            choice_T = np.logical_and(Temperature>=T_vals[r],
                                      Temperature<T_vals[r+1])
            choice_n = np.logical_and(NumberDensity>=n_vals[c],
                                      NumberDensity<n_vals[c+1])
            choice = np.logical_and(choice_T, choice_n)
            if tcool[choice].shape[0] == 0 : tcool_vals[r, c] = np.nan
            else: tcool_vals[r, c] = np.percentile(tcool[choice], 50)
            return '%d,%d'%(r,c)
        
        seq = list(map(tcool_fill, product(range(N_T-1), range(N_n-1))))
        #for item in seq: pass
        '''
        for r in range(N_T-1):
            choice_T = np.logical_and(np.log10(Temperature)>=T_vals[r],
                                      np.log10(Temperature)<T_vals[r+1])
            for c in range(N_n-1):
                choice_n = np.logical_and(np.log10(NumberDensity)>=n_vals[c],
                                      np.log10(NumberDensity)<n_vals[c+1])
                choice = np.logical_and(choice_T, choice_n)
                if tcool[choice].shape[0] == 0 : tcool_vals[r, c] = np.nan
                else: tcool_vals[r, c] = np.percentile(tcool[choice], 50)
        '''
        
        plt.plot([-3.5,], [6,], 'X', color='ghostwhite', markersize=5, alpha=0.8, mec='slategrey')
        #plt.hist2d(np.log10(NumberDensity), np.log10(Temperature), weights=np.log10(tcool), bins=(500,500), 
        #       density=True, cmap='magma', norm=LogNorm() )
        #n_interp = np.random.choice(np.log10(NumberDensity),1000)
        #choose = NumberDensity==n_interp
        #T_interp = np.log10(Temperature)[choose]
        #tcool = np.log10(tcool)[choose]
        #tcool = interpolate.interp2d(n_interp, T_interp, tcool, kind='linear')
        #n_vals, T_vals = np.meshgrid(np.linspace(-6.5,1.5,100), np.linspace(3.8,8.5,101))
        #tcool = tcool(n_vals,T_vals)
        CS = plt.contour(n_vals[:-1], T_vals[:-1], tcool_vals,
                         10,colors='black')
        plt.clabel(CS, fontsize=9, inline=1, fmt='%1.1f')
        plt.savefig('./hist/snap_%03d.png'%snapshot, transparent=True)
        #plt.show()
        plt.close(fig)
    '''
    plt.hist(NumberDensity, weights=Volume, bins=20, density=False, log=True)
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel(r'Number density $\rm (cm^{-3})$')
    plt.show()
    
    plt.hist(Temperature, weights=Volume, bins=20, density=False, log=True)
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel(r'Temperature $\rm (K)$')
    plt.show()
    '''