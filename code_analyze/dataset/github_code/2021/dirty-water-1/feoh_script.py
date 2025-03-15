#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:40:08 2021

@author: sukrit
"""
import numpy as np
import matplotlib.pyplot as plt

#Key numbers: 
#Ksp of Fe(OH)2 = 5E-17 from CRC Handbook of Chemistry and Phyids
# pKa of Fe2+ + H2O ---> FeOH+ + H+ is -log10(3.2E-10)=9.5 (Braterman+1983)

ph=np.arange(6,10, step=0.01)

precip_limited=(5*10**(11-2*ph))*10**(ph-9.5)#First term is how much Fe2+ there is at saturation as function of pH. Second term is how much FeOH+ that implies
fe2_limited=1.0E-4*10**(ph-9.5) #100 uM Fe2+, how much FeOH corresponds to that. 

feoh=np.minimum(precip_limited, fe2_limited)

plt.plot(ph, feoh, color='black')
plt.plot(ph, precip_limited, color='red', linestyle='--')
plt.plot(ph, fe2_limited, color='blue', linestyle='--')

plt.ylim(1.0E-9, 1.0E-4)
plt.yscale('log')

###What depth is OD=1 (ignoring slant angle for now)?
print(np.log10(np.exp(1))/(3.55E2*2.0E-6))