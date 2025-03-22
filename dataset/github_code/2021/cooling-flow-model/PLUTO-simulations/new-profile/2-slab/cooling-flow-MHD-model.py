#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:06:39 2019

@author: alankar

pressure-driven cooling flow in cylindrical, spherical geometry

runfile('/home/alankar/Documents/cool-flow/new-profile/cooling-flow-MHD-model.py', args='fields ./output-2048/', wdir='/home/alankar/Documents/cool-flow/new-profile')
12 Jan 2021: added magnetic fields
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
v0byc0 = 0.4; T0 = 4.25e5; d0 = 2.e-3*mu*mp ; c0 = np.sqrt(gamma*kB*T0/mu/mp) #5kpc
shift = 1.0

ct0=c0*np.sqrt(1+gamma_m/(gamma*beta0)) #sound speed including B-fields
v0 = v0byc0*c0; Lam0 = Lambda(T0); v0byct0 = v0/ct0
tcool0cgs = mp*mue*mui*kB*T0/((gamma-1)*mu*d0*Lam0)
R0 = q*gamma*v0*tcool0cgs*(1+gamma_m/(gamma*beta0))
print(R0/kpc)

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
Rout = 6/(R0/kpc); nR = 100 #outer radius in kpc
R_out = np.logspace(np.log10(1.+eps),np.log10(Rout),nR)
sol_out = solve_ivp(deriv, [1.+eps,R_out[-1]], [-1.+vp*eps,1.+sp*eps], t_eval=R_out, 
                    method=method_ODE, rtol=reltol, atol=abstol) #outer solution
v_out = sol_out.y[0]; d_out = -1./(sol_out.y[0]*sol_out.t**q); p_out = sol_out.y[1]*d_out**gamma
#inner solution
Rin = 0.3 #starting of the inner radius
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
'''
checks = False
if checks:
    if (q==2):
        Mdot *= 4.*np.pi
        dEdotdT = 4.*np.pi*Rcgs**2*neniL/np.gradient(Tcgs,Rcgs)
        MdotdBedT = Mdot*np.gradient(Becgs,Tcgs)
        plt.loglog(Tcgs,Tcgs*dEdotdT,label="dEdot/dT")
        plt.loglog(Tcgs,Tcgs*MdotdBedT,label="Mdot*dBe/dT")
        plt.ylabel(r'erg s$^{-1}$ K$^{-1}$')
        plt.xlabel('T(K)')
        plt.legend()
    plt.figure()
    plt.plot(R,test_mom,label="momentum error; [vdv/dr+dp/dr/rho+dpmag/dr/rho]/vdv/dr")
    plt.plot(R,test_energy,label="energy error; [pvds/dr/(gamma-1) + neniL]/neniL")
    plt.legend()
    plt.xlabel('dedimensionalized radius')
    plt.figure()
    
    plt.loglog(R,-v,label='velocity')
    plt.loglog(R,p,label='pressure')
    plt.loglog(R,d,label='density')
    plt.loglog(R,T,label='temperature')
    plt.loglog(R,p/d**gamma,label='entropy')
    plt.xlabel('dedimensionalized radius')
    plt.grid()
    plt.legend()
    plt.show()

np.savetxt('bernoulli-model(rTBeMdot).txt',
           np.vstack((Rcgs,Tcgs,Becgs,Mdot)).T)
'''
#--------------------- Comparing ---------------------------

base_dir = sys.argv[2]
hfile = h5py.File('%s/data.%04d.dbl.h5'%(base_dir, 0),'r')
res   = np.array(hfile['cell_coords/X']).shape[0]
hfile.close()
data = np.loadtxt('%s/plots/%d_%s-avg.txt'%(base_dir,res,sys.argv[1]), skiprows=2)

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


#----------------------------------

Rcgs *= shift
rightedge = 5.0

#Plotting model and PDE data together
fig, axs = plt.subplots(2, 3, figsize=(20,11))
gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.32, hspace=0.02) # set the spacing between axes.
axs = np.array([ [plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs1[2])], 
                 [plt.subplot(gs1[3]), plt.subplot(gs1[4]), plt.subplot(gs1[5])] ])

axs[0, 0].plot(Rcgs/kpc, np.log10(dcgs/(mu*mp)), linewidth=5)
axs[0, 0].plot(radius*pc/kpc, n_50, c='tab:gray', linewidth=5)
axs[0, 0].fill_between(radius*pc/kpc, n_16, n_84, color='tab:gray', alpha=0.3)
axs[0, 0].grid()
axs[0, 0].set_ylim(ymax=-0.3)
axs[0, 0].set_ylabel(r'number density [$\rm cm^{-3}\ (log)$]', size=20 )
axs[0, 0].get_xaxis().set_ticklabels([])
axs[0, 0].set_xlim(xmin=0., xmax=rightedge)
axs[0, 0].tick_params(axis='both', which='major', labelsize=18)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=16)
axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(6))

axs[0, 1].plot(Rcgs/kpc, np.log10(pcgs/kB), 'tab:orange', linewidth=5)
axs[0, 1].plot(radius*pc/kpc, Ptot_50, c='tab:gray', linewidth=5)
axs[0, 1].fill_between(radius*pc/kpc, Ptot_16, Ptot_84, color='tab:gray', alpha=0.3)
axs[0, 1].grid()
axs[0, 1].set_ylim(ymin=2.5, ymax=3.5)
axs[0, 1].set_xlim(xmin=0., xmax=rightedge)
axs[0, 1].set_ylabel(r'$\rm p_{gas}$ [$\rm \times k_B \ K cm^{-3}\ (log)$]', size=20 )
axs[0, 1].get_xaxis().set_ticklabels([])
axs[0, 1].tick_params(axis='both', which='major', labelsize=18)
axs[0, 1].tick_params(axis='both', which='minor', labelsize=16)
axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(6))

axs[0, 2].plot(Rcgs/kpc, vcgs/1e5, 'tab:green', linewidth=5)
axs[0, 2].plot(radius*pc/kpc, vel_50, c='tab:gray', linewidth=5)
axs[0, 2].fill_between(radius*pc/kpc, vel_16, vel_84, color='tab:gray', alpha=0.3)
axs[0, 2].grid()
axs[0, 2].set_ylabel(r'velocity [$\rm km s^{-1}$]', size=20 )
axs[0, 2].set_xlim(xmin=0., xmax=rightedge)
axs[0, 2].get_xaxis().set_ticklabels([])
axs[0, 2].tick_params(axis='both', which='major', labelsize=18)
axs[0, 2].tick_params(axis='both', which='minor', labelsize=16)
axs[0, 2].xaxis.set_major_locator(plt.MaxNLocator(6))

axs[1, 0].plot(Rcgs/kpc, np.log10(entcgs/1e32), 'tab:red', linewidth=5)
axs[1, 0].plot(radius*pc/kpc, ent_50, c='tab:gray', linewidth=5)
axs[1, 0].fill_between(radius*pc/kpc, ent_16, ent_84, color='tab:gray', alpha=0.3)
axs[1, 0].grid()
axs[1, 0].set_xlabel(r'distance [$\rm kpc$]', size=20 )
axs[1, 0].set_ylabel(r'$\rm p_{gas}/\rho ^{\gamma}\ [\rm CGS \times 10^{32}\ (log)$]', size=20 )
axs[1, 0].set_xlim(xmin=0., xmax=rightedge)
#axs[1, 0].set_ylim(ymin=-1.2, ymax=2.2)
axs[1, 0].tick_params(axis='both', which='major', labelsize=18)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=16)
axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(6))

axs[1, 1].plot(Rcgs/kpc, np.log10(Tcgs), 'tab:brown', linewidth=5)
axs[1, 1].plot(radius*pc/kpc, temp_50, c='tab:gray', linewidth=5)
axs[1, 1].fill_between(radius*pc/kpc, temp_16, temp_84, color='tab:gray', alpha=0.3)
axs[1, 1].grid()
axs[1, 1].set_xlim(xmin=0., xmax=rightedge)
#axs[1, 1].set_ylim(ymin=3.5)
axs[1, 1].set_ylabel(r' temperature [$\rm K\ (log)$]', size=20 )
axs[1, 1].set_xlabel(r'distance [$\rm kpc$]', size=20 )
axs[1, 1].tick_params(axis='both', which='major', labelsize=18)
axs[1, 1].tick_params(axis='both', which='minor', labelsize=16)
axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(6))

axs[1, 2].plot(Rcgs/kpc, Mdot/(Msun/yr), 'tab:purple', linewidth=5)
axs[1, 2].plot(radius*pc/kpc, Mdot_50, c='tab:gray', linewidth=5)
axs[1, 2].fill_between(radius*pc/kpc, Mdot_16, Mdot_84, color='tab:gray', alpha=0.3)
axs[1, 2].grid()
axs[1, 2].set_xlim(xmin=0., xmax=rightedge)
#axs[1, 2].set_ylim(ymin=5, ymax = 55)
axs[1, 2].set_ylabel(r'$\rm \dot{M}$ [$\rm M_\odot yr^{-1}\ (log)$] ', size=20 )
axs[1, 2].set_xlabel(r'distance [$\rm kpc$]', size=20 )
axs[1, 2].tick_params(axis='both', which='major', labelsize=18)
axs[1, 2].tick_params(axis='both', which='minor', labelsize=16)
axs[1, 2].xaxis.set_major_locator(plt.MaxNLocator(6))
'''
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
'''
Mdot_avg = np.mean(Mdot/(Msun/yr))
#fig.suptitle('Critical radius is approximately set to %.2f kpc\n'%(R0/kpc*shift)+r'$\dot{\rm M} \rm \approx %.1f \times 10^{%d}\ M_{\odot} \ yr^{-1}$'%( fman(Mdot_avg) ,fexp(Mdot_avg)), size=25)
fig.subplots_adjust(top=0.92)
plt.savefig('%s/plots/%s.png'%(base_dir,'pde_compare-marchinout'), transparent=True, bbox_inches='tight')
plt.show()

'''
fig = plt.figure()

select = radius >= 550
P_avg_50 = np.percentile(10**Ptot_50[select], 50)
delta_P_50 = (10**Ptot_50[select] - P_avg_50 )/P_avg_50

p_ode = pcgs/kB
select_ode = np.logical_and( (Rcgs/pc) >= 550, (Rcgs/pc) <= rightedge)
p_ode_avg = np.percentile(p_ode[select_ode], 50)
delta_p_ode = (p_ode[select_ode] - p_ode_avg )/p_ode_avg
#P_avg_16 = np.percentile(10**Ptot_16[select],16)
#delta_P_16 = np.log10(np.abs(10**(Ptot_16[select]) - P_avg_16 )/P_avg_16)
#P_avg_84 = np.percentile(10**Ptot_84[select],84)
#delta_P_84 = np.log10(np.abs(10**(Ptot_84[select]) - P_avg_84 )/P_avg_84)

radius_ode = Rcgs[select_ode]/pc
radius = radius[select]
plt.plot(radius, delta_P_50)
plt.plot(radius_ode, delta_p_ode)
#plt.fill_between(radius, delta_P_16, delta_P_84, alpha=0.5)
plt.grid()
plt.show()
'''