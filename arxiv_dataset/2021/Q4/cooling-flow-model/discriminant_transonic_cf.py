#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:51:44 2020

@author: prateek
discriminant for the criterion of a real root
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(13,10))

gamma = 5/3;

alpha = np.linspace(-3,6,10000)

q = 2
Dis = q*q*(gamma*(alpha-1)+4-alpha)**2 - 4*(gamma+1)*q*(q*(alpha-2)+1)
plt.plot(alpha, Dis, linestyle='-', label='q=2, spherical geometry', color='tab:blue', linewidth=5)

q = 1
Dis = q*q*(gamma*(alpha-1)+4-alpha)**2 - 4*(gamma+1)*q*(q*(alpha-2)+1)
plt.plot(alpha, Dis, linestyle='--', label='q=1, cylindrical geometry', color='firebrick', linewidth=5)

plt.legend(loc='best', prop={'size': 28}, ncol=1, shadow=False, fancybox=True, framealpha=0.4)
plt.grid()
plt.xlim([-2,5])
plt.ylim([-25,150])
plt.xlabel(r'$\rm \Lambda_T$',size=32)
plt.ylabel(r'Discriminant ($\rm \Delta$)',size=32)
plt.fill([-2, 5, 5, -2],[-25, -25, 0, 0], alpha=0.4, color='khaki')
plt.text(-1, -14, r'$\Delta$<0, no transonic solution', fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=28, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=28, direction="out", pad=5)
plt.savefig('discriminant_vs_lamT.png', transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(13,10))
q=2
Dis = q*q*(gamma*(alpha-1)+4-alpha)**2 - 4*(gamma+1)*q*(q*(alpha-2)+1)
vp1 = (-q*(gamma*(alpha-1)+4-alpha) + np.sqrt(Dis))/(2*(gamma+1))
vp2 = (-q*(gamma*(alpha-1)+4-alpha) - np.sqrt(Dis))/(2*(gamma+1))

plt.plot(alpha, vp1, linestyle='-', label='q=2, spherical geometry', color='tab:blue', linewidth=5)
plt.plot(alpha, vp2, linestyle='-', color='tab:blue', linewidth=5)

q=1
Dis = q*q*(gamma*(alpha-1)+4-alpha)**2 - 4*(gamma+1)*q*(q*(alpha-2)+1)
vp1 = (-q*(gamma*(alpha-1)+4-alpha) + np.sqrt(Dis))/(2*(gamma+1))
vp2 = (-q*(gamma*(alpha-1)+4-alpha) - np.sqrt(Dis))/(2*(gamma+1))

plt.plot(alpha, vp1, linestyle='--', label='q=1, cylindrical geometry', color='firebrick', linewidth=5)
plt.plot(alpha, vp2,'--', linestyle='--', color='firebrick', linewidth=5)

plt.grid()
plt.xlim([-2,5])
plt.ylim([-3,2])
plt.legend(loc='best', prop={'size': 28}, ncol=1, shadow=False, fancybox=True, framealpha=0.4)
plt.xlabel(r'$\rm \Lambda_T$',size=32)
plt.ylabel(r'two roots of $\rm \tilde{v}^\prime$ at sonic point', size=32)
plt.fill([-2, 5, 5, -2],[-3, -3, 0, 0], alpha=0.4, color='khaki')
plt.text(0.5, -1.8, r'negative roots', fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=28, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=28, direction="out", pad=5)
plt.savefig('two_roots.png', transparent=True, bbox_inches='tight')
plt.show()
plt.close(fig)