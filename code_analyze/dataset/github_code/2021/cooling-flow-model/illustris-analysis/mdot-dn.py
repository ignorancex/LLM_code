# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:04:17 2021

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from decimal import Decimal

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

X, Y, Z = 0.7154, 0.2703, 0.0143
Z *= 10**-0.2
X = 1.-(Y+Z)
mu = 1/(2*X+0.75*Y+0.5625*Z)
mue = 2./(1+X)
mui = 1/(1/mu-1/mue)
muH = 1/X

gamma = 5./3
gamma_m = 1.03
sizes = [0.5, 1.0, 0.5, 2.0] #kpc

K = 4*np.pi
q = 2

mdot_50, mdot_16, mdot_84 = [], [], []
Be, dn_temperature = [], []
dn_radius = None
for select in range(len(sizes)):
    r_crit = sizes[select]
    start = select*3+1
    print('Choosing cloud of size %.1f kpc'%sizes[select])
    
    #----------------------------------
    dn_data = np.loadtxt('../dylan-nelson/fig10_TNG50-1_z0.5_h8_nH.txt', skiprows=5)
    dn_radius = dn_data[:,0]
    #print(dn_radius)
    dn_nH_50 = 10**dn_data[:,start]
    dn_nH_16, dn_nH_84 = dn_data[:,start+1],dn_data[:,start+2]
    
    #----------------------------------
    dn_data = np.loadtxt('../dylan-nelson/fig10_TNG50-1_z0.5_h8_vel_rel.txt', skiprows=5)
    dn_radius = dn_data[:,0]
    dn_vel_50 = dn_data[:,start]
    dn_vel_16, dn_vel_84 = dn_data[:,start+1],dn_data[:,start+2]
    
    mdot_16.append( -K*(dn_radius*kpc)**q*(dn_nH_16*mu*mp)*(dn_vel_16*1e5)/(Msun/yr) )
    mdot_50.append( -K*(dn_radius*kpc)**q*(dn_nH_50*mu*mp)*(dn_vel_50*1e5)/(Msun/yr) )
    mdot_84.append( -K*(dn_radius*kpc)**q*(dn_nH_84*mu*mp)*(dn_vel_84*1e5)/(Msun/yr) )


sizes = np.hstack(([0,],sizes))
fig, axs = plt.subplots(2, 1, figsize=(13,10))
gs1 = gridspec.GridSpec(2, 1)
gs1.update(wspace=0.28, hspace=0.008) # set the spacing between axes.
axs = np.array([ [plt.subplot(gs1[0]),], 
                 [plt.subplot(gs1[1]),] ])
axs[0, 0].get_xaxis().set_ticklabels([])
axs[0, 0].set_xlim(xmin=0., xmax=5.0)
axs[1, 0].set_xlim(xmin=0., xmax=5.0)
axs[1, 0].invert_yaxis()

colors = ['tab:red' ,'tab:blue', 'tab:green', 'tab:purple']
for cloud in range(1,len(sizes)): 
    inflow, outflow = np.median(mdot_50[cloud-1][mdot_50[cloud-1]>0]), -np.median(mdot_50[cloud-1][mdot_50[cloud-1]<0])
    #net = inflow-outflow
    net = outflow/inflow
    axs[0, 0].plot(dn_radius, mdot_50[cloud-1], linewidth=5, linestyle='-', color=colors[cloud-1] )
             #label=r'$\rm R_{cl}=%.1f \ kpc\ -\ %.1f \ kpc$ ($\rm %.1f \times 10^{%d}\ M_\odot yr^{-1}$)'%(sizes[cloud-1], 
             #sizes[cloud], fman(net),fexp(net)) )
    if cloud==2: axs[0, 0].fill_between(dn_radius, mdot_16[cloud-1], mdot_84[cloud-1],
                    alpha=0.3, color=colors[cloud-1] )
    
    axs[1, 0].plot(dn_radius, -mdot_50[cloud-1], linewidth=5, linestyle='-', color=colors[cloud-1],
                   label=r'$\rm R_{cl}=%.1f \ kpc\ -\ %.1f \ kpc$'%(sizes[cloud-1], sizes[cloud]))
    if cloud==2: axs[1, 0].fill_between(dn_radius, -mdot_16[cloud-1], -mdot_84[cloud-1],
                    alpha=0.3, color=colors[cloud-1] )
    
    
axs[0, 0].set_yscale('log')
axs[1, 0].set_yscale('log')  
axs[0, 0].set_ylim(ymin=10**-5.5)
axs[1, 0].set_ylim(ymax=10**-5.5)

axs[0,0].text(3.3, 10, r'inflow', fontsize=28, color='black')
axs[1,0].text(3.3, 10, r'outflow', fontsize=28, color='black')
fig.supylabel(r'$\dot{\rm M}$ [$\rm M_\odot yr^{-1}$]',size=28, x=-0.00001)
axs[1, 0].set_xlabel(r'distance [$\rm kpc$]',size=28)
axs[0, 0].tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
axs[1, 0].tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
axs[0, 0].grid()
axs[1, 0].grid()
#axs[0, 0].set_xticks([])
axs[1, 0].legend(loc='lower left', prop={'size': 18},framealpha=0.3, shadow=False, fancybox=True)
plt.savefig('mdot-tng.png', transparent=True, bbox_inches='tight')
plt.show()
