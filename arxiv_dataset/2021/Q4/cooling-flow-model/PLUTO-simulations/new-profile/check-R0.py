#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 00:17:13 2021

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import h5py
import sys
from scipy import interpolate
import sys
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

X = 0.7154
Y = 0.2703
Z = 0.0143
mu = 1./(2*X+0.75*Y+0.5625*Z)
mue = 2./(1+X)
mui = 1./(1/mu-1/mue)
Tfloor = 1.e4

def deriv(x, y):
    if y[0]>0:
        print ("negative density!",x)
        #sys.exit()
    d = np.abs(-1./(y[0]*x**q))
    if d<=0: d = -d
    p = y[1]*d**gamma # tilde density and pressure 
    
    T = mu*mp*c0*c0*p/(kB*d*gamma) # temperature in CGS
    Lam = Lambda(T)/Lam0 #tilde Lambda
    if (T<=Tfloor):
        T = Tfloor
        Lam = 0. #Lambda(T)*(T/Tfloor)**(-10.)/Lam0
    Num1 = q*(c0/v0)**2*( d*Lam*(1+gamma_m/(gamma*beta0))/y[0] + (p+gamma_m*d**gamma_m/(gamma*beta0))/(x*d) )
    Den1 =  (1 - (c0/v0)**2*(p+gamma_m*d**gamma_m/(gamma*beta0))/(d*y[0]**2))*y[0]
    return [ Num1/Den1, -q*gamma*(1+gamma_m/(gamma*beta0))*Lam*d**(2-gamma)/y[0] ]

def Lambda(temp): #returns cooling function in cgs
    klo=0; khi=tab_sz-1
    while (klo != (khi-1)):
        kmid = int((khi+klo)/2)
        Tmid = Ttab[kmid]
        if (temp<=Tmid):
            khi = kmid
        if (temp>Tmid):
            klo = kmid
    dT = Ttab[khi] - Ttab[klo]
    scrh = Ltab[klo]*(Ttab[khi]-temp)/dT + Ltab[khi]*(temp-Ttab[klo])/dT; #linear interpolation
    return scrh
'''
def Lambda(temp):
    lam = 0.0  
    if (temp<=1.e7 and temp>=1.e4):
        lam = 2.7e-23*(temp/1.e7)**-1.0
    if (temp<1.e4):
        lam = 2.7e-20*(temp/1.e4)**20
        #lam = 1.e-100
    if (temp>1.e7):
        lam = 2.7e-23*(temp/1.e7)**-20
        #lam = 1.e-100
    return lam
'''

D = np.loadtxt('./cooltable.dat')
global Ttab, Ltab, tab_sz
Ttab = D[:,0]; Ltab = D[:,1]; tab_sz = np.size(Ttab)

global q, gamma, gamma_m, beta0, c0, v0, Lam0, T0, d0, v0byc0
gamma_m = 1.03; beta0 = 1e10
q=2; gamma=5./3.
mu = 0.62; mue = 1.17; mui = 1./(1./mu - 1./mue)
#v0byc0 = 0.099; T0 = 5.1e5; d0 = 1.96e-3*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)
#v0byc0 = 0.049; T0 = 3.8e5; d0 = 7.9e-4*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)

for logT0 in np.linspace(4,7,1000):
    for logn0 in np.linspace(-4,-1,1000):
        
        v0byc0 = 0.05; T0 = 10**logT0; d0 = (10**logn0)*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp) #5kpc
        shift = 1.0

        ct0=c0*np.sqrt(1+gamma_m/(gamma*beta0)) #sound speed including B-fields
        v0 = v0byc0*c0; Lam0 = Lambda(T0); v0byct0 = v0/ct0
        tcool0cgs = mp*mue*mui*kB*T0/((gamma-1)*mu*d0*Lam0)
        R0 = q*gamma*v0*tcool0cgs*(1+gamma_m/(gamma*beta0))/kpc
        if R0>0.8 and R0<1.0 and (10**logn0*T0)>99 and (10**logn0*T0)<100: 
            print('%.2f kpc'%R0, '%.3e cm^-3'%(10**logn0), '%.3e K'%T0, '%.2e'%(10**logn0*T0) )