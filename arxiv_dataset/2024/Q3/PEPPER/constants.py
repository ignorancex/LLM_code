# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:29:11 2022

@author: madir
"""
import numpy as np

d2s=86400
Au2cm=1.496e13


kb = 1.38065e-16 # boltzmann's constant
mh = 1.673558e-24 # mass of hydrogen atom in grams
sigma = 5.6704e-5 # stefan-Boltzmann constant
mu=2.35 *mh
G=6.67408e-8 # Gravitational constant
alpha = 0.68 # pressure dependence of opacity
beta = 0.45 # temperature dependence of opacity
kappa0 = 10**(-7.32) # opacity constant
gamma=5/3


Mearth2g = lambda mass: mass*5.972e27
Rearth2cm = lambda rad: rad*6.371e8
cm2Rearth = lambda rad: rad/6.371e8
Msun2g= lambda mass: mass*1.9885e33
Rsun2cm = lambda rad: rad*6.9551e10
Myr2sec= lambda val: val*1e6*365*86400

radtable = np.loadtxt('radiustable.csv', delimiter=',')
masstable = np.loadtxt('masstable.csv', delimiter=',')
eff_scalings=np.load("eff_shape_file.npy", allow_pickle=True)

f=1.3