#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:06:39 2019

@author: prateek

pressure-driven cooling flow in cylindrical, spherical geometry

12 Jan 2021: added magnetic fields

hydro results can be produced by setting beta>>1
can reproduce Fig. 1, 3 in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys

plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 16})

def deriv(x, y):
    if y[0]>0:
        print ("negative density!",x)
        sys.exit()
    d = -1./(y[0]*x**q); p = y[1]*d**gamma; # tilde density and pressure 
    T = mu*mp*c0*c0*p/(kB*d*gamma) # temperature in CGS
    Lam = Lambda(T)/Lam0 #tilde Lambda
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
    #if (temp<=1.e4 or temp>=8.e6):
    #    scrh = 0.0
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

#D=np.loadtxt('m-00.cie')
D=np.loadtxt('cooltable.dat')
global Ttab, Ltab, tab_sz
Ttab=D[:,0]; Ltab=D[:,1]; tab_sz=np.size(Ttab)
#Ttab=10.**(D[:,0]); Ltab=10.**D[:,5]; tab_sz=np.size(Ttab)

global q, gamma, gamma_m, beta0, kB, mp, mu, c0, v0, Lam0
#gamma_m = 1.1; beta0 = 0.5 #not working with B for now!
#q=2; gamma=5./3.; kB = 1.380649e-16; mp = 1.6726219e-24; mu = 0.62; mue = 1.17; mui = 1./(1./mu - 1./mue)
#v0byct0 = 0.4
#c0 = 1.e7; 
gamma_m = 4./3; beta0 = 0.5
q=1; gamma=5./3.; kB = 1.380649e-16; mp = 1.6726219e-24; mu = 0.62; mue = 1.17; mui = 1./(1./mu - 1./mue)
#T0 = 2.e3; d0 = 5.e-3*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)
#T0 = 2.e3; d0 = 5.e-3*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)
T0 = 4.e4; d0 = 1.e-4*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp)
v0byct0 = 2
ct0=c0*np.sqrt(1+gamma_m/(gamma*beta0)) #sound speed including B-fields
v0 = v0byct0*ct0; Lam0 = Lambda(T0); v0byc0 = v0/c0
#v0 = v0byc0*c0; Lam0 = Lambda(T0); v0byct0 = v0/ct0
#v0=v0byct0*ct0; d0 = 1.e-3*mp; T0 = mu*mp*c0*c0/(gamma*kB); Lam0 = Lambda(T0) #normalization
#v0byct0 = 0.2; v0 = v0byct0*ct0; #d0 = 1.e-3*mp; T0 = mu*mp*c0*c0/(gamma*kB); 
#Lam0 = Lambda(T0);

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
sp = (gamma+gamma_m/beta0)*q #slope of entropy at sonic point, same at all critical points
method_ODE = "DOP853"
#method_ODE = "LSODA"
reltol = 5e-13; abstol = 1.e-12
#start at the sonic point and move out
Rout = 1.22; nR = 200
R_out = np.logspace(np.log10(1.+eps),np.log10(Rout),nR)
sol_out = solve_ivp(deriv, [1.+eps,R_out[-1]], [-1.+vp*eps,1.+sp*eps], t_eval=R_out, method=method_ODE, rtol=reltol, atol=abstol) #outer solution
v_out = sol_out.y[0]; d_out = -1./(sol_out.y[0]*sol_out.t**q); p_out = sol_out.y[1]*d_out**gamma
#inner solution
Rin = 0.35 #starting of the inner radius
R = np.logspace(np.log10(1.-eps),np.log10(Rin),nR)
sol = solve_ivp(deriv, [1.-eps,R[-1]], [-1.-vp*eps, 1.-sp*eps], t_eval=R, method=method_ODE, rtol=reltol, atol=abstol) #inner solution
v = sol.y[0]; d = -1./(sol.y[0]*sol.t**q); p = sol.y[1]*d**gamma


#analyze results
R = np.concatenate((np.flip(R), R_out)); v = np.concatenate((np.flip(v), v_out)) #arranging inner & outer solutions in increasing order
d = np.concatenate((np.flip(d), d_out)); p = np.concatenate((np.flip(p), p_out))

#R = R_out; v = v_out; d = d_out; p = p_out

Mach = -v*(v0/c0)/np.sqrt( p/d + gamma_m*d**(gamma_m-1)/(beta0*gamma) ) #including B-fields in sound speed
Be = 0.5*v**2 + p*(c0/v0)**2/((gamma-1.)*d) + gamma_m*d**(gamma_m-1)*(c0/v0)**2/(gamma*(gamma_m-1)*beta0) #including B-fields, normalized to v_0^2

T = mu*mp*c0*c0*p/(gamma*kB*d); Lam=np.zeros(np.size(T))
for i in range(np.size(T)):
    Lam[i]=Lambda(T[i])/Lam0

tcool0cgs = mp*mue*mui*kB*T0/((gamma-1)*mu*d0*Lam0)
R0 = q*gamma*v0*tcool0cgs*(1+gamma_m/(gamma*beta0))
vcgs = v0*v; pcgs = p*d0*c0**2/gamma; pmagcgs = d0*c0**2*d**gamma_m/(gamma*beta0); Rcgs = R0*R; dcgs = d*d0
vdvdr = vcgs*np.gradient(vcgs,Rcgs); dpdrbyrho = np.gradient(pcgs,Rcgs)/dcgs; dpmagbydrho = np.gradient(pmagcgs,Rcgs)/dcgs
test_mom = vdvdr + dpdrbyrho + dpmagbydrho
test_mom /= vdvdr
dsbydr = np.gradient(np.log(pcgs/dcgs**gamma),Rcgs)
Tcgs = (mu*mp/kB)*pcgs/dcgs
pmag_cgs = (d0*c0**2/gamma)*(dcgs/d0)**gamma_m/beta0
neniL = dcgs**2*Lam*Lam0/(mue*mui*mp**2)
test_energy = pcgs*vcgs*dsbydr/(gamma-1) + neniL 
test_energy = test_energy/neniL
#Becgs = 0.5*vcgs**2 + gamma*pcgs/(dcgs*(gamma-1)) + gamma_m*pmag_cgs/(dcgs*(gamma_m-1))
Becgs = Be*v0**2
if (q==2):
    Mdot = -4.*np.pi*Rcgs**2*dcgs*vcgs
    dEdotdT = 4.*np.pi*Rcgs**2*neniL/np.gradient(Tcgs,Rcgs)
    MdotdBedT = Mdot*np.gradient(Becgs,Tcgs)
    
kpc = 3.086e21
plt.plot(Rcgs/R0, vcgs/np.sqrt((gamma*pcgs+gamma_m*pmag_cgs)/dcgs), '--', label=r'q=1, $v_0/c_{t0}=2$')
plt.show()

    #plt.loglog(Tcgs,Tcgs*dEdotdT,label="dEdot/dT")
    #plt.loglog(Tcgs,Tcgs*MdotdBedT,label="Mdot*dBe/dT")
    #plt.ylabel(r'erg s$^{-1}$ K$^{-1}$')
    #plt.xlabel('T(K)')
    #plt.legend()
#plt.figure()
#plt.plot(R,test_mom,label="momentum error; [vdv/dr+dp/dr/rho+dpmag/dr/rho]/vdv/dr")
#plt.plot(R,test_energy,label="energy error; [pvds/dr/(gamma-1) + neniL]/neniL")
#plt.legend()
#plt.xlabel('dedimensionalized radius')
#plt.figure()
#plt.loglog(R,Mach,label="Mach#")
#plt.loglog(R,Be,label="Bernoulli#")
#plt.loglog(R,-v,label='velocity')
#plt.loglog(R,p,label='pressure')
#plt.loglog(R,d,label='density')
#plt.loglog(R,np.sqrt(gamma*p/d),label='sound speed')
#plt.loglog(R,T,label='temperature')
#plt.loglog(R,p/d**gamma,label='entropy')
#plt.semilogx(R,p/(d*R)+d*Lam/v)
#plt.semilogx(R,v-p/(v*d))
#plt.semilogx(R,-q*gamma*d**(2.-gamma)*Lam/v)
#plt.xlabel('dedimensionalized radius')
#plt.show()
#plt.grid()
#plt.legend()
#=============================================================================
# kpc = 3.086e21
# plt.subplot(321)
# plt.plot(Rcgs/kpc, np.log10(dcgs/(mu*mp)))
# plt.grid()
# plt.subplot(322)
# plt.plot(Rcgs/kpc,np.log10(pcgs/kB))
# plt.grid()
# plt.subplot(323)
# plt.plot(Rcgs/kpc,vcgs/1.e5)
# plt.grid()
# plt.subplot(324)
# plt.plot(Rcgs/kpc,np.log10(pcgs/pmag_cgs))
# plt.grid()
# plt.subplot(325)
# plt.plot(Rcgs/kpc, np.log10(Tcgs))
# plt.grid()
# plt.subplot(326)
# plt.plot(Rcgs/kpc, np.log10((pcgs+pmag_cgs)/kB))
# plt.grid()
#=============================================================================
