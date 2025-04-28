# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:47:13 2023

@author: hsauro
"""

import tellurium as te
import roadrunner
import numpy as np
import matplotlib.pyplot as plt
from freqResponse import *

# Two step pathway
r0= te.loada('''
   $Xo -> S1; k1*Xo
    S1 -> ;   k2*S1
    
    k1 = 5; k2 = 0.01;
    Xo = 1
''')

# six-step pathway
r1 = te.loada('''
    J1: $Xo -> S1; k1*Xo
    J2: S1 -> S2; k2*S1
    J3: S2 -> S3; k3*S2
    J4: S3 -> S4; k4*S3
    J5: S4 -> S5; k5*S4
    J6: S5 -> ;   k5*S5
    
    k1 = 0.1; k2 = 0.23; k3 = 0.4
    k4 = 0.33; k5 = 0.26
    Xo = 1
''')
 
 # Single conserved cycle
r2 = te.loada('''
       J1: S1 -> S2; k1*S1
       J2: S2 -> S1; k2*S2
       k1 = 0.1; k2 = 0.12   
       S1 = 10; S2 = 0
''')


# Almost oscillating model
r3 = te.loada ('''
  // Reactions:
  J0: $X0 -> S1; (VM1 * X0)/(1 + X0 + S3^h);
  J1: S1 -> S2; k2*S1
  J2: S2 -> S3; k4*S2
  J3: S3 -> $X1; k5*S3

  // Species initializations:
  S1 = 0; S2 = 0; S3 = 0;
  S4 = 0; X0 = 10; X1 = 0;
  k2 = 0.1; k3 = 0.12; k4 = 0.11; k5 = 0.14

  // Variable initialization:
  VM1 = 10; Keq1 = 10; h = 6; V4 = 2.5; KS4 = 0.5;
''')


fr = FreqencyResponse(r0)
results = fr.getSpeciesFrequencyResponse(0.0001, 4, 100, 'k1', 'S1')
fr.plot()

fr = FreqencyResponse(r1)
results = fr.getSpeciesFrequencyResponse(0.01, 4, 100, 'k1', 'S2')
fr.plot()

fr = FreqencyResponse(r2)
results = fr.getFluxFrequencyResponse(0.01, 3, 100, 'k1', 'J1')
fr.plot()    

fr = FreqencyResponse(r2)
results = fr.getSpeciesFrequencyResponse(0.01, 3, 100, 'k1', 'S2')
fr.plot() 
    
fr = FreqencyResponse(r3)
results = fr.getSpeciesFrequencyResponse(0.01, 3, 1000, 'X0', 'S3', useDB=False)
fr.plot()
   


    