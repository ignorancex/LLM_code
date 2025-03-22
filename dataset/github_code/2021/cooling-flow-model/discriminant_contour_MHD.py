#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:49:10 2020

@author: prateek
contourplot of discriminant in beta_0-Lambda_T plane
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 16})

q=2; g=5/3; gm=4/3; nL=100; nB=150
LT = np.linspace(-1,4,nL) #Lambda_T
b0 = np.logspace(-2,3,nB) #beta_0
#A = np.outer(np.ones(nL), g+1 + gm*(gm-g)/(gm+g*b0))
#B = q*( np.outer((g-1)*LT+4-g, np.ones(nB)) 
#+ gm*np.outer(np.ones(nL), 1/b0 - gm*(2-g/gm+1/b0)/(gm+g*b0)) )
#C = q*( np.outer(q*(LT-2)+1, np.ones(nB)) 
#+ q*gm*( np.outer(LT-1,1/b0) + gm*np.outer(np.ones(nL),(1+1/b0)/(gm+g*b0)) ) )

A = np.outer( np.ones(nL), g+1 + gm*(gm+1)/(g*b0) )

B = q*( np.outer((g-1)*LT+4-g, np.ones(nB)) 
- (gm/g)*(np.outer(np.ones(nL), (2*gm-g-4)/b0) - np.outer(LT,(g-1)/b0)) )

C = q*( np.outer(q*(LT-2)+1, np.ones(nB)) 
       - (gm/g)*( np.outer( q*(2+g-LT*(1+g)-gm)-1,1/b0) - np.outer(LT,q*gm/b0**2)) )

Det = B**2 - 4*A*C

Det_sgn = np.log10(np.abs(Det))*Det/np.abs(Det)

fig = plt.figure(figsize=(13,10))

cs = plt.contourf(b0,LT,Det,40, norm=colors.Normalize(vmin=-1e8,vmax=.1e8), cmap='seismic')
plt.contour(cs, colors='black', levels=[0], linewidths=2.5, linestyles='dashed')

cs = plt.contourf(b0,LT,Det_sgn, 500, norm=colors.Normalize(), cmap='seismic')
plt.text(0.02, 3, r'no transonic solution', fontsize=28, color='white')
plt.tick_params(axis='y', which='major', labelsize=28, direction="out", pad=5)
plt.tick_params(axis='y', which='minor', labelsize=28, direction="out", pad=5)
plt.tick_params(axis='x', which='major', labelsize=28, direction="out", pad=9)
plt.tick_params(axis='x', which='minor', labelsize=28, direction="out", pad=9)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=25, pad=5) 
cb.set_label(r'sign($\rm \Delta$) $\rm \log_{10}(|\Delta |)$', size=30)
plt.xlabel(r'$\beta_0$', size=32)
plt.ylabel(r'$\Lambda_T$', size=32)
plt.xscale('log')
plt.title(r'$\rm \Delta=B^2-4AC$ for $q=2,~\gamma_m=4/3$', fontsize=32)
plt.savefig('discriminant_MHD_spherical.png', transparent=True, bbox_inches='tight')
plt.show()