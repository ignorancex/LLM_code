#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:06:28 2021

@author: alankar
"""

#./output-8192//plots/8192_fields-avg.txt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import sys
from scipy import interpolate
import sys
from decimal import Decimal
from cycler import cycler

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

Msun = 2e33
yr = 365*24*60**2
mp = 1.6726219e-24
kB = 1.380649e-16
pc = 3.086e18
kpc = 1e3*pc

X = 0.7154
Y = 0.2703
Z = 0.0143
mu = 1./(2*X+0.75*Y+0.5625*Z)
mue = 2./(1+X)
mui = 1./(1/mu-1/mue)
Tfloor = 1.e4


res = [128,256,512,1024,2048,4096,8192,16384,32768]

colors = ['tab:blue', 'tab:orange', 'tab:green', 
                                'tab:red', 'tab:olive', 'tab:brown', 
                                'tab:cyan','tab:gray', 'tab:purple']
default_cycler = (cycler(color=colors) )

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=default_cycler)


#--------------------- Comparing ---------------------------

#Plotting model and PDE data together
fig, axs = plt.subplots(2, 3, figsize=(20,11))
gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.32, hspace=0.02) # set the spacing between axes.
axs = np.array([ [plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs1[2])], 
                 [plt.subplot(gs1[3]), plt.subplot(gs1[4]), plt.subplot(gs1[5])] ])

for (index,resolution) in enumerate(res):
    data = np.loadtxt('./output-%d/plots/%d_%s-avg.txt'%(resolution,resolution,sys.argv[1]), 
                      skiprows=2)
    
    #----------------------------------
    radius = data[:,0]
    start = 4
    n_50 = np.log10(data[:,start])
    n_16, n_84 = np.log10(data[:,start+1]),np.log10(data[:,start+2])
    
    
    #----------------------------------
    start = 1
    Mdot_50 = data[:,start]
    Mdot_16, Mdot_84 = data[:,start+1], data[:,start+2]
    
    #----------------------------------
    start = 19
    temp_50 = np.log10(data[:,start])
    temp_16, temp_84 = np.log10(data[:,start+1]), np.log10(data[:,start+2])
    
    #----------------------------------
    start = 7
    vel_50 = data[:,start]
    vel_16, vel_84 = data[:,start+1], data[:,start+2]
    
    
    #----------------------------------
    start = 10
    Ptot_50 = np.log10(data[:,start])
    Ptot_16, Ptot_84 = np.log10(data[:,start+1]), np.log10(data[:,start+2])
    
    
    #----------------------------------
    start = 13
    ent_50 = np.log10(data[:,start])
    ent_16, ent_84 = np.log10(data[:,start+1]), np.log10(data[:,start+2])
    
    if resolution == res[-1]: axs[0, 0].fill_between(radius*pc/kpc, n_16, n_84, alpha=0.3, color=colors[index])
    axs[0, 0].plot(radius*pc/kpc, n_50, label=r'$\rm \Delta x/ pc = %.2f $'%(1500/resolution))

    if index==0: axs[0, 0].grid()
    axs[0, 0].set_ylim(ymax=-1.0)
    axs[0, 0].set_ylabel(r'number density [$\rm cm^{-3}\ (log)$]', size=20 )
    axs[0, 0].get_xaxis().set_ticklabels([])
    axs[0, 0].set_xlim(xmin=0., xmax=1.5)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=18)
    axs[0, 0].tick_params(axis='both', which='minor', labelsize=16)
    axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    
    if resolution == res[-1]: axs[0, 1].fill_between(radius*pc/kpc, Ptot_16, Ptot_84, alpha=0.3, color=colors[index])
    axs[0, 1].plot(radius*pc/kpc, Ptot_50)
    
    if index==0: axs[0, 1].grid()
    axs[0, 1].set_ylim(ymin=2.4, ymax=2.52)
    axs[0, 1].set_xlim(xmin=0., xmax=1.5)
    axs[0, 1].set_ylabel(r'$\rm p_{gas}$ [$\rm \times k_B \ K cm^{-3}\ (log)$]', size=20 )
    axs[0, 1].get_xaxis().set_ticklabels([])
    axs[0, 1].tick_params(axis='both', which='major', labelsize=18)
    axs[0, 1].tick_params(axis='both', which='minor', labelsize=16)
    axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    
    if resolution == res[-1]: axs[0, 2].fill_between(radius*pc/kpc, vel_16, vel_84, alpha=0.3, color=colors[index])
    axs[0, 2].plot(radius*pc/kpc, vel_50)
    
    if index==0: axs[0, 2].grid()
    axs[0, 2].set_ylabel(r'velocity [$\rm km s^{-1}$]', size=20 )
    axs[0, 2].set_xlim(xmin=0., xmax=1.5)
    axs[0, 2].get_xaxis().set_ticklabels([])
    axs[0, 2].tick_params(axis='both', which='major', labelsize=18)
    axs[0, 2].tick_params(axis='both', which='minor', labelsize=16)
    axs[0, 2].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    
    if resolution == res[-1]: axs[1, 0].fill_between(radius*pc/kpc, ent_16, ent_84, alpha=0.3, color=colors[index])    
    axs[1, 0].plot(radius*pc/kpc, ent_50)
    
    if index==0: axs[1, 0].grid()
    axs[1, 0].set_xlabel(r'distance [$\rm kpc$]', size=20 )
    axs[1, 0].set_ylabel(r'$\rm p_{gas}/\rho ^{\gamma}\ [\rm CGS \times 10^{32}\ (log)$]', size=20 )
    axs[1, 0].set_xlim(xmin=0., xmax=1.5)
    #axs[1, 0].set_ylim(ymin=-1.2, ymax=2.2)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=18)
    axs[1, 0].tick_params(axis='both', which='minor', labelsize=16)
    axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    if resolution == res[-1]: axs[1, 1].fill_between(radius*pc/kpc, temp_16, temp_84, alpha=0.3, color=colors[index])        
    axs[1, 1].plot(radius*pc/kpc, temp_50)
    
    if index==0: axs[1, 1].grid()
    axs[1, 1].set_xlim(xmin=0., xmax=1.5)
    #axs[1, 1].set_ylim(ymin=3.5)
    axs[1, 1].set_ylabel(r' temperature [$\rm K\ (log)$]', size=20 )
    axs[1, 1].set_xlabel(r'distance [$\rm kpc$]', size=20 )
    axs[1, 1].tick_params(axis='both', which='major', labelsize=18)
    axs[1, 1].tick_params(axis='both', which='minor', labelsize=16)
    axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(6))
    
    
    if resolution == res[-1]:
        axs[1, 2].fill_between(radius*pc/kpc, Mdot_16/1e-4, Mdot_84/1e-4, alpha=0.3, color=colors[index])       
    axs[1, 2].plot(radius*pc/kpc, Mdot_50/1e-4)
    
    if index==0: axs[1, 2].grid()
    axs[1, 2].set_xlim(xmin=0., xmax=1.5)
    axs[1, 2].set_ylim(ymin=-5, ymax = 7.5)
    axs[1, 2].set_ylabel(r'$\rm \dot{M}$ [$\rm \times 10^{-4}\ M_\odot yr^{-1}$] ', size=20 )
    axs[1, 2].set_xlabel(r'distance [$\rm kpc$]', size=20 )
    axs[1, 2].tick_params(axis='both', which='major', labelsize=18)
    axs[1, 2].tick_params(axis='both', which='minor', labelsize=16)
    axs[1, 2].xaxis.set_major_locator(plt.MaxNLocator(6))
    '''
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    '''
#fig.suptitle('Critical radius is approximately set to %.2f kpc\n'%(R0/kpc*shift)+r'$\dot{\rm M} \rm \approx %.1f \times 10^{%d}\ M_{\odot} \ yr^{-1}$'%( fman(Mdot_avg) ,fexp(Mdot_avg)), size=25)
axs[0, 0].legend(loc='best', fontsize='x-large')
fig.subplots_adjust(top=0.92)
plt.savefig('./%s.png'%('convergence-spread'), transparent=True, bbox_inches='tight')
plt.show()