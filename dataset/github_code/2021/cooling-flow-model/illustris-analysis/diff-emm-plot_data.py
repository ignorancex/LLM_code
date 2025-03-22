# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:29:04 2021

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from operator import itemgetter
from decimal import Decimal

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

Msun = 2e33 #g
yr = 365*24*60**2 #s

hdf = h5py.File('../data-gasandcloud.h5', 'r')
cl_r = np.array(hdf['cloud_size'])
hdf.close()

fig = plt.figure(figsize=(13,15))
#0 --> proxy for floating cutoff
cutoffs = [0, 5, 20] #, 20, 50, 100, 200, 300, 499] 
size_sel_min, size_sel_max = np.min(cl_r), np.max(cl_r) #1.0, 1.5 #kpc
floating = False
if 0 in cutoffs: floating = True
Edot_allcl, T_plot_allcl = [], []
showit = [5, 40] #show all cloud total DEM

for cutoff in cutoffs:
    emm = None
    if cutoff!=0: emm = np.load('./diff-emm_all-cloud_%03dkpc.npy'%cutoff)
    else: emm = np.load('./diff-emm_all-cloud_floating.npy') 
    
    
    #Temperature is in log scale
    Temperature, Edot = [], [] #(cloud_no, Temperature) and (cloud_no, Edot) 
    
    #Cloud selection based on cloud size
    for i,radius in enumerate(cl_r):
        if (radius>=size_sel_min and radius<=size_sel_max):
            Temperature.append(emm[i,:,0]) #(cloud_no, Temperature, Edot)
            Edot.append(emm[i,:,1])
            
    Temperature = np.array(Temperature).flatten()
    select = 3.8 #selection based on gas temperature
    cut = Temperature>=select
    Temperature = Temperature[cut]
    Edot = np.array(Edot).flatten()[cut]
    
    #The binning of Edot in Temperature bins 
    #require temperature arranged in ascending order
    sorter = sorted(zip(Temperature,Edot), key=itemgetter(0))
    Temperature, Edot = zip(*sorter)
    Temperature, Edot = np.array(Temperature), np.array(Edot)
    print('Sort completed!')
    
    Tstart, Tstop, Npts = np.min(Temperature), np.max(Temperature), 100
    T_vals = np.linspace(Tstart,Tstop,Npts+1)
    dT = T_vals[1] - T_vals[0]
    Edot_vals, Edot_16p, Edot_84p, T_plot = [], [], [], []
    
    tmp, T_curr = [], Tstart+dT
    for i in range(Temperature.shape[0]):
        if Temperature[i]<=T_curr:
            tmp.append(Edot[i])
        else:
            T_curr += dT
            tmp = np.array(tmp)
            tmp = tmp[np.isfinite(np.log10(tmp))] #no heating terms
            if len(tmp) == 0: 
                tmp = []
                continue
            Edot_vals.append(np.percentile(tmp,50))
            Edot_16p.append(np.percentile(tmp,16))
            Edot_84p.append(np.percentile(tmp,84))
            if floating or (cutoff in showit): 
                Edot_allcl.append(np.sum(tmp))
                T_plot_allcl.append(T_curr-1.5*dT)
            T_plot.append(T_curr-1.5*dT)
            tmp = []        
    
    T_plot, Edot_vals, Edot_16p, Edot_84p = np.array(T_plot), np.array(Edot_vals), np.array(Edot_16p), np.array(Edot_84p)
    if floating or (cutoff in showit):
        T_plot_allcl, Edot_allcl = np.array(T_plot_allcl), np.array(Edot_allcl)
        if floating: floating = False
    
    if cutoff == 5:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:cyan', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        #plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:blue')
    elif cutoff == 10:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:green', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:green')    
    elif cutoff == 40:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:olive', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:olive')
    elif cutoff == 100:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='yellow', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='yellow')
    elif cutoff == 20:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:purple', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:purple')
        #plt.semilogy(T_plot_allcl,Edot_allcl,linewidth=8, color='tab:purple', linestyle='-.',
        #             label=r'All clouds (%d kpc)'%cutoff)
    elif cutoff == 200:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='orange', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='orange')
    elif cutoff == 300:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:brown', 
                     label=r'Median cloud (%d kpc)'%cutoff)
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:brown')
    elif cutoff == 1200:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:gray', 
                     label=r'Median cloud (%d kpc)'%(cutoff+1))
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:gray')
        plt.semilogy(T_plot_allcl,Edot_allcl,linewidth=5, color='tab:gray', linestyle='-.',
                     label=r'All clouds (%d kpc)'%cutoff)
    else:
        plt.semilogy(T_plot,Edot_vals,linewidth=5, color='tab:blue', 
                     label=r'Median cloud ($\rm 3 R_{cloud}$)')
        plt.fill_between(T_plot, Edot_16p, Edot_84p, alpha=0.5, color='tab:blue')
        #plt.semilogy(T_plot_allcl,Edot_allcl,linewidth=5, color='tab:blue', linestyle=':',
        #             label=r'All clouds ($\rm 3 R_{cloud}$)')
    
    T_plot_allcl, Edot_allcl = [], []

#Analytic ODE prediction
data = np.loadtxt('../bernoulli-model(rTBeMdot).txt')
distance, Temperature, Be, Mdot = [data[:,i] for i in range(data.shape[1])]
diff_emm_analytic = np.gradient(Be, Temperature)*Mdot*Temperature*np.log(10)
Mdot = np.average(Mdot)/(Msun/yr)
plt.semilogy(np.log10(Temperature), diff_emm_analytic, color='chocolate', linestyle='--',linewidth=5, 
             label=r' $ \dot{\rm M}_{\rm cool} \rm = %.1f \times 10^{%d}\ M_\odot yr^{-1}$'%(fman(Mdot),fexp(Mdot))
             #label=r'Steady cooling flow model'+'\n'+r'($ \dot{\rm M} \rm = %.1f \times 10^{%d}\ M_\odot yr^{-1}$)'%(fman(Mdot),fexp(Mdot)),
             )
scale = 8e2
diff_emm_analytic *= scale
Mdot *= scale
plt.semilogy(np.log10(Temperature), diff_emm_analytic, color='firebrick', linestyle='--',linewidth=5, 
             label=r' $ \dot{\rm M}_{\rm cool} \rm = %.1f \times 10^{%d}\ M_\odot yr^{-1}$'%(fman(Mdot),fexp(Mdot))
             #label=r'$\times %d$ Steady cooling flow model'%scale+'\n'+r'($ \dot{\rm M} \rm = %.1f \times 10^{%d}\ M_\odot yr^{-1}$)'%(fman(Mdot),fexp(Mdot)),
             )

hdf = h5py.File('../data-gasandcloud.h5', 'r')
SFR = np.array(hdf['SFR'])
#exclude ISM --> SFR == 0
condition = SFR==0
Temperature = np.log10(np.array(hdf['temperature']))[condition]
nH = np.array(hdf['nH'])[condition]
vol = np.array(hdf['volume'])[condition]
Edot = -(np.array(hdf['LAMBDA'])[condition])*nH**2*vol
density = np.array(hdf['density'])[condition]
hdf.close()

#DEM all gas
scale = 1#e7
#Generate DEM data
hist_data, x_edges = np.histogram(Temperature, bins=100, weights=Edot)

dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
diff_emm = hist_data/dT
plt.semilogy(Temperature_bin,diff_emm/scale,linewidth=5, color='black', zorder=3,
             label=r'All halo gas')

data_nocl = np.load('diff-emm-isolated.npy')
plt.semilogy(data_nocl[:,0], data_nocl[:,1], color='tab:gray', linewidth=5, 
             label=r'Non-cloud gas')
'''
data_onlycl = np.load('diff-emm-onlycl.npy')
plt.semilogy(data_onlycl[:,0], data_onlycl[:,1], color='tab:green', linewidth=5, 
             label=r'Cloud gas')
'''
plt.ylabel(r'$\rm \dfrac{d\dot{E}_{cool} [erg\ s^{-1}]}{d log_{10}( T [K])} $',size=28)
plt.xlabel(r'$\rm \log_{10}(T[K])$',size=28)
plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
plt.grid()
plt.ylim(ymin=10**34.2)#, ymax=1e40)
plt.xlim(xmin=np.min(Temperature_bin)-0.05, xmax=8.2)
plt.legend(loc='lower right', prop={'size': 22},framealpha=0.3, bbox_to_anchor=(0.88, 0))
fig.tight_layout()
plt.savefig('./diff-emm-cuts.png', transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

#------------------Other useful diagonistics-------------------------------
fig, ax1 = plt.subplots(figsize=(13,10))

color = 'tab:red'
hist_data, x_edges = np.histogram(Temperature, bins=100, weights=vol, density=True)
dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
vol_temp = hist_data#/dT
ax1.semilogy(Temperature_bin,vol_temp, linewidth=5, color=color) #, 
             #label=r'Volume weighted')
ax1.set_ylabel(r'$\rm \dfrac{dV}{d log_{10}( T [K])} $',size=28, color=color)
ax1.tick_params(axis='y', which='major', labelsize=24, direction="out", pad=5, labelcolor=color)
ax1.tick_params(axis='y', which='minor', labelsize=24, direction="out", pad=5, labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
hist_data, x_edges = np.histogram(Temperature, bins=100, weights=vol*density, density=True)
dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
mass_temp = hist_data#/dT
ax2.semilogy(Temperature_bin,mass_temp, linewidth=5, color=color) #, 
             #label=r'Mass weighted')
ax2.set_ylabel(r'$\rm \dfrac{dM}{d log_{10}( T [K])} $',size=28, color=color)
ax2.tick_params(axis='y', which='major', labelsize=24, direction="out", pad=5, labelcolor=color)
ax2.tick_params(axis='y', which='minor', labelsize=24, direction="out", pad=5, labelcolor=color)

hist_data, x_edges = np.histogram(Temperature, bins=100, weights=Edot, density=True)
dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
diff_emm = hist_data#/dT
plt.semilogy(Temperature_bin,diff_emm,linewidth=5, color='black', linestyle='-.',
             label=r'$\rm \dfrac{d\dot{E}_{cool} [erg\ s^{-1}]}{d log_{10}( T [K])} $')

ax1.set_xlabel(r'$\rm \log_{10}(T[K])$',size=28)
ax1.grid()
ax1.set_xlim(xmin=np.min(Temperature_bin)-0.05, xmax=8.2)
ax1.set_ylim(ymin=1e-6, ymax=10**0.5)
ax2.set_ylim(ymin=1e-6, ymax=10**0.5)
ax1.tick_params(axis='x', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
ax1.tick_params(axis='x', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
plt.legend(loc='lower left', prop={'size': 24},framealpha=0.3, bbox_to_anchor=(0.1, 0))
plt.savefig('./diff-emm-comparepdfs.png', transparent=True, bbox_inches='tight')
fig.tight_layout()
plt.show()
plt.close(fig)
np.savetxt('pdf-data.txt', np.vstack((Temperature_bin, vol_temp, mass_temp, diff_emm)).T, 
        header='log10(T[K])			Vol_PDF		Mass_PDF		Diff_emm')

# simple plot asked by thesis referee
fig, ax1 = plt.subplots(figsize=(13,10))

color = 'tab:red'
hist_data, x_edges = np.histogram(Temperature, bins=100, weights=vol, density=True)
dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
vol_temp = hist_data#/dT
ax1.semilogy(Temperature_bin,vol_temp, linewidth=5, color=color,
             label=r'$\rm \dfrac{dV}{d log_{10}( T [K])} $') 
ax1.set_ylabel('Probability distribution functions (PDFs)', size=28)
# ax1.set_ylabel(r'$\rm \dfrac{dV}{d log_{10}( T [K])} $',size=28, color=color)
ax1.tick_params(axis='y', which='major', labelsize=24, direction="out", pad=5)
ax1.tick_params(axis='y', which='minor', labelsize=24, direction="out", pad=5)

color = 'tab:blue'
hist_data, x_edges = np.histogram(Temperature, bins=100, weights=vol*density, density=True)
dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
mass_temp = hist_data#/dT
ax1.semilogy(Temperature_bin,mass_temp, linewidth=5, color=color, 
             label=r'$\rm \dfrac{dM}{d log_{10}( T [K])} $')

hist_data, x_edges = np.histogram(Temperature, bins=100, weights=Edot, density=True)
dT = x_edges[1]-x_edges[0]
Temperature_bin = x_edges[1:]-dT/2
diff_emm = hist_data#/dT
plt.semilogy(Temperature_bin,diff_emm,linewidth=5, color='black', linestyle='-.',
             label=r'$\rm \dfrac{d\dot{E}_{cool} [erg\ s^{-1}]}{d log_{10}( T [K])} $')

ax1.set_xlabel(r'$\rm \log_{10}(T[K])$',size=28)
ax1.grid()
ax1.set_xlim(xmin=np.min(Temperature_bin)-0.05, xmax=8.2)
ax1.set_ylim(ymin=3e-6, ymax=10**0.5)
ax1.tick_params(axis='x', which='major', labelsize=24, direction="out", pad=5, labelcolor='black')
ax1.tick_params(axis='x', which='minor', labelsize=24, direction="out", pad=5, labelcolor='black')
plt.legend(loc='lower left', prop={'size': 24},framealpha=0.3, bbox_to_anchor=(0.1, 0))
plt.savefig('./diff-emm-comparepdfs-simple_referee.png', transparent=True, bbox_inches='tight')
fig.tight_layout()
plt.show()
plt.close(fig)

individual = False
if individual:
    fig = plt.figure(figsize=(13,10))
    emm = np.load('./diff-emm_all-cloud_floating.npy') 
    for i in range (emm.shape[0]):
        plt.semilogy(emm[i,:,0], emm[i,:,1],linewidth=2)
    plt.ylabel(r'$\rm \dfrac{d\dot{E}_{cool} [erg\ s^{-1}]}{d log_{10}( T [K])} $',size=28)
    plt.xlabel(r'$\rm \log_{10}(T[K])$',size=28)
    plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
    plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
    plt.grid()
    plt.ylim(ymin=10**33.8, ymax=10**45.3)
    plt.xlim(xmin=3.8, xmax=8.2)
    plt.show()
    plt.close(fig)
    
