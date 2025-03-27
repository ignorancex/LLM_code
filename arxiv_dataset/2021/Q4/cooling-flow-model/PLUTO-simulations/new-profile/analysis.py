#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 07:47:29 2021

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
v0byc0 = 0.099; T0 = 5.1e5; d0 = 1.96e-3*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)
#v0byc0 = 0.049; T0 = 3.8e5; d0 = 7.9e-4*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)
shift = 1.0

ct0=c0*np.sqrt(1+gamma_m/(gamma*beta0)) #sound speed including B-fields
v0 = v0byc0*c0; Lam0 = Lambda(T0); v0byct0 = v0/ct0
tcool0cgs = mp*mue*mui*kB*T0/((gamma-1)*mu*d0*Lam0)
R0 = q*gamma*v0*tcool0cgs*(1+gamma_m/(gamma*beta0))
print(R0/pc)

if (v0byct0==1):
    epsT=0.01
    LamT = np.log(Lambda(T0*(1+epsT))/Lambda(T0*(1-epsT)))/np.log((1+epsT)/(1-epsT)) #dlnLambda/dlnT
    Aq = (gamma+1) + gamma_m*(1+gamma_m)/(gamma*beta0)
    Bq = q*( LamT*(gamma-1)+4-gamma -gamma_m*(2*gamma_m-gamma-4-LamT*(gamma-1))/(gamma*beta0) )
    Cq = q*( q*(LamT-2)+1 - gamma_m*(q*(2+gamma-LamT*(1+gamma+gamma_m/beta0)-gamma_m)-1)/(gamma*beta0) )
    Disc = Bq*Bq - 4*Aq*Cq #Discriminant 
    if (Disc<0):
        print ("no transonic solution exists")
        sys.exit()
    vp = (-Bq + np.sqrt(Disc))/(2*Aq) # physically relevant root for the slope of velocity at sonic point
    #eps = 0.00001; 
else:
    #eps = 0.0
    vp = 0.0
eps = 0.00001
sp = (gamma+gamma_m/beta0)*q #slope of entropy at sonic point, same at all critical point
#method_ODE = "DOP853"
method_ODE = "LSODA"
reltol = 5e-13; abstol = 1.e-12
#start at the critical point and move out
Rout = 2/(R0/kpc); nR = 100
R_out = np.logspace(np.log10(1.+eps),np.log10(Rout),nR)
sol_out = solve_ivp(deriv, [1.+eps,R_out[-1]], [-1.+vp*eps,1.+sp*eps], t_eval=R_out, 
                    method=method_ODE, rtol=reltol, atol=abstol) #outer solution
v_out = sol_out.y[0]; d_out = -1./(sol_out.y[0]*sol_out.t**q); p_out = sol_out.y[1]*d_out**gamma
#inner solution
Rin = 0.18 #starting of the inner radius
R = np.logspace(np.log10(1.-eps),np.log10(Rin),nR)
sol = solve_ivp(deriv, [1.-eps,R[-1]], [-1.-vp*eps, 1.-sp*eps], t_eval=R, 
                method=method_ODE, rtol=reltol, atol=abstol) #inner solution
v = sol.y[0]; d = -1./(sol.y[0]*sol.t**q); p = sol.y[1]*d**gamma

#analyze results
R = np.concatenate((np.flip(R), R_out)); v = np.concatenate((np.flip(v), v_out)) #arranging inner & outer solutions in increasing order
d = np.concatenate((np.flip(d), d_out)); p = np.concatenate((np.flip(p), p_out))
Mach = -v*(v0/c0)/np.sqrt( p/d + gamma_m*d**(gamma_m-1)/(beta0*gamma) ) #including B-fields in sound speed
Be = 0.5*v**2 + p*(c0/v0)**2/((gamma-1.)*d) + gamma_m*d**(gamma_m-1)*(c0/v0)**2/(gamma*(gamma_m-1)*beta0) #including B-fields, normalized to v_0^2

T = mu*mp*c0*c0*p/(gamma*kB*d); Lam=np.zeros(np.size(T))
for i in range(np.size(T)):
    Lam[i]=Lambda(T[i])/Lam0

vcgs = v0*v 
pcgs = p*d0*c0**2/gamma
pmagcgs = d0*c0**2*d**gamma_m/(gamma*beta0)
beta = pcgs/pmagcgs
Rcgs = R0*R
dcgs = d*d0
'''
vdvdr = vcgs*np.gradient(vcgs,Rcgs)
dpdrbyrho = np.gradient(pcgs,Rcgs)/dcgs
dpmagbydrho = np.gradient(pmagcgs,Rcgs)/dcgs
test_mom = vdvdr + dpdrbyrho + dpmagbydrho
test_mom /= vdvdr
dsbydr = np.gradient(np.log(pcgs/dcgs**gamma),Rcgs)
'''
Tcgs = (mu*mp/kB)*pcgs/dcgs
pmag_cgs = (d0*c0**2/gamma)*(dcgs/d0)**gamma_m/beta0
neniL = dcgs**2*Lam*Lam0/(mue*mui*mp**2)
'''
test_energy = pcgs*vcgs*dsbydr/(gamma-1) + neniL 
test_energy = test_energy/neniL
Becgs = Be*v0**2
'''
Mdot = -Rcgs**q*dcgs*vcgs*np.pi*4
entcgs = pcgs/dcgs**gamma

Myr = 1e6*365*24-60**2
tcoolouter = (mp*mue*mui*kB*T[-1]/((gamma-1)*mu*dcgs[-1]*Lam[-1]*Lam0))/Myr
tcool0 = tcool0cgs/Myr
tsc0 = (R0/np.sqrt(gamma*kB*T0/(mu*mp)))/Myr
tscouter = (Rcgs[-1]/np.sqrt(gamma*kB*T[-1]/(mu*mp)))/Myr