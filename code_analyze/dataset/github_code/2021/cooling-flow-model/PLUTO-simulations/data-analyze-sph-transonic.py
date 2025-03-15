# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:26:15 2020

@author: alankar
"""

import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.use('pdf')

Msun = 2e33
yr = 365*24*60**2
mp = 1.6726219e-24
kB = 1.380649e-16
pc = 3.086e18
UNIT_VELOCITY = 1e5
UNIT_DENSITY = mp
UNIT_LENGTH = 1e3*pc


coolT = np.loadtxt('./cooltable.dat')
coolT = coolT[:-2,:]
cool = interpolate.interp1d(coolT[:,0], coolT[:,1], fill_value='extrapolate')
X, Y, Z = 0.7154, 0.2703, 0.0143
mu = 1/(2*X+0.75*Y+0.5625*Z)
mue = 2./(1+X)
mui = 1/(1/mu-1/mue)
gamma = 5./3

T0 = 4e4 #K
Mach0 = 1.0
n0 = 1e-4 #cm^-3
rho0 = n0*mu*mp #g cm^-3
ne0, ni0 = n0*mu/mue, n0*mu/mui
P0 = n0*kB*T0 #dyne cm^-2
cs0 = np.sqrt(gamma*P0/rho0) #cm s^-1
v0 = -Mach0*cs0
s0 = P0/rho0**gamma
cool_0 = np.float(cool(T0))
tcool0 = (1/(gamma-1))*n0*kB*T0/(ne0*ni0*cool_0)
q, K = 2, 4*np.pi #Geometry factors
r0 = -q*v0*gamma*tcool0
M_dot = -K*r0**q*rho0*v0

transonic = np.loadtxt('./Spherical_CF/transonic/transonic_rdpv.txt')

fig = plt.figure(figsize=(13,10))
cs_analytic = np.sqrt(transonic[:,2]/transonic[:,1]) #de-dimensionalized
plt.semilogx(transonic[:,0], transonic[:,-1]*v0/(cs_analytic*cs0), linestyle='--', label=r'Steady state',
             linewidth=8, color='gray', alpha=0.8)
plt.semilogx(transonic[:,0], transonic[:,2], linestyle='--',linewidth=8, color='gray', alpha=0.8)
plt.semilogx(transonic[:,0], transonic[:,1], linestyle='--', linewidth=8, color='gray', alpha=0.8)
plt.semilogx(transonic[:,0], -transonic[:,-1], linestyle='--', linewidth=8, color='gray', alpha=0.8)
plt.semilogx(transonic[:,0], transonic[:,2]/transonic[:,1]/100., linestyle='--', linewidth=8, color='gray', alpha=0.8)

lastfile = 100  #if == 0 print only IC
startfile = 50
if (lastfile==0): startfile=0
if(lastfile<startfile and lastfile!=0):
    print('Lastfile must be greater or equal to startfile')
    sys.exit(0)
tcool_max = 0.
tcool_min = 0.

#read initial file to know array sizes
file = np.loadtxt('./Spherical_CF/transonic/Output/data.%04d.tab'%0)   
r = file[:,0]
v = file[:,3]
prs = file[:,4]
rho = file[:,2]

r = np.zeros((len(r),(lastfile-startfile)+1),dtype=np.float64)
v, prs, rho, cs = np.copy(r), np.copy(r), np.copy(r), np.copy(r)

for fileno in range(0,(lastfile-startfile)+1):
    file = np.loadtxt('./Spherical_CF/transonic/Output/data.%04d.tab'%fileno)
    
    r[:,fileno] = file[:,0]
    v[:,fileno] = file[:,3]
    prs[:,fileno] = file[:,4]
    rho[:,fileno] = file[:,2]
    cs[:,fileno] = np.sqrt(gamma*prs[:,fileno]/rho[:,fileno])
    if (fileno==0):
        tcool_max = np.max( (1/(gamma-1))*prs[:,fileno]*UNIT_DENSITY*UNIT_VELOCITY**2/\
            ((rho[:,fileno]*UNIT_DENSITY/(mue*mp))*(rho[:,fileno]*UNIT_DENSITY/(mui*mp))*\
            cool(mu*mp/kB * prs[:,fileno]/rho[:,fileno]*UNIT_VELOCITY**2)) )
        tcool_min = np.min( (1/(gamma-1))*prs[:,fileno]*UNIT_DENSITY*UNIT_VELOCITY**2/\
            ((rho[:,fileno]*UNIT_DENSITY/(mue*mp))*(rho[:,fileno]*UNIT_DENSITY/(mui*mp))*\
            cool(mu*mp/kB * prs[:,fileno]/rho[:,fileno]*UNIT_VELOCITY**2)) )
        tcool_max *= (UNIT_VELOCITY/UNIT_LENGTH)
        tcool_min *= (UNIT_VELOCITY/UNIT_LENGTH)
    
r = np.percentile(r, 50, axis=1)
print('IC has a min cooling time of %.3e [code uints] and max cooling time of %.3e [code uints]'%(tcool_min,tcool_max))

if lastfile==0: #Plot only IC
    
    prs = prs[:,0]
    v = v[:,0]
    rho = rho[:,0]
    cs = cs[:,0]
    
    plt.semilogx(r*UNIT_LENGTH/r0, -v/cs, '-',label=r'$\rm \mathcal{M}=\frac{|v|}{c_s}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, prs*UNIT_DENSITY*UNIT_VELOCITY**2/P0, '-', label=r'$\rm \tilde{p}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, rho*UNIT_DENSITY/rho0, '-', label=r'$\rm \tilde{\rho}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, v*UNIT_VELOCITY/v0, '-', label=r'$\rm \tilde{v}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, (prs/rho)/(100.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, '-', label=r'$\rm \tilde{T}/100$',linewidth=5)
    plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
    
    plt.grid()
    plt.ylim(-0.05,1.8)
    lgd = plt.legend(loc='upper right',prop={'size': 28,}, ncol=2, shadow=False, fancybox=True, framealpha=0.6)
    plt.xlabel(r'De-dimensionalized distance $\rm \tilde{r}$', size=70)
    plt.ylabel('De-dimensionalized fluid fields', size=28)
    #plt.title('Steady state of cooling flow (transonic)', size=40)
    plt.tick_params(axis='both', which='major', labelsize=28, direction="out", pad=5)
    plt.tick_params(axis='both', which='minor', labelsize=28, direction="out", pad=5)
    #plt.xlim(1.0,20.)
    #plt.tight_layout()
    plt.savefig('cool-flow-pde-transonic_outflow(IC).pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', transparent =True)
    #plt.show()
    sys.exit(0)

prs_16 = np.percentile(prs, 16, axis=1)
v_16 = np.percentile(v, 16, axis=1)
rho_16 = np.percentile(rho, 16, axis=1)
cs_16 = np.percentile(cs, 16, axis=1)

prs_84 = np.percentile(prs, 84, axis=1)
v_84 = np.percentile(v, 84, axis=1)
rho_84 = np.percentile(rho, 84, axis=1)
cs_84 = np.percentile(cs, 84, axis=1)

prs_50 = np.percentile(prs, 50, axis=1)
v_50 = np.percentile(v, 50, axis=1)
rho_50 = np.percentile(rho, 50, axis=1)
cs_50 = np.percentile(cs, 50, axis=1)

plt.semilogx(r*UNIT_LENGTH/r0, -v_50/cs_50, '-',label=r'$\rm \mathcal{M}=\frac{|v|}{c_s}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, -v_16/cs_16, -v_84/cs_84, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, prs_50*UNIT_DENSITY*UNIT_VELOCITY**2/P0, '-', label=r'$\rm \tilde{p}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, prs_16*UNIT_DENSITY*UNIT_VELOCITY**2/P0, prs_84*UNIT_DENSITY*UNIT_VELOCITY**2/P0, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, rho_50*UNIT_DENSITY/rho0, '-', label=r'$\rm\tilde{\rho}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, rho_16*UNIT_DENSITY/rho0, rho_84*UNIT_DENSITY/rho0, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, v_50*UNIT_VELOCITY/v0, '-', label=r'$\rm\tilde{v}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, v_16*UNIT_VELOCITY/v0, v_84*UNIT_VELOCITY/v0, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, (prs_50/rho_50)/(100.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, '-', label=r'$\rm\tilde{T}/100$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, (prs_16/rho_16)/(100.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, (prs_84/rho_84)/(100.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')

plt.grid()
plt.ylim(-0.05,1.8)
lgd = plt.legend(loc='upper right',prop={'size': 28,}, ncol=2, shadow=False, fancybox=True, framealpha=0.6)#,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'De-dimensionalized distance $\rm \tilde{r}$', size=32)
plt.ylabel(r'De-dimensionalized variables', size=32)
#plt.title('Steady state of cooling flow (transonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=28, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=28, direction="out", pad=5)
#plt.xlim(1.0,20.)
plt.savefig('cool-flow-pde-transonic_outflow-sph.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

if lastfile == 0: sys.exit(0) 
def decompose(x): 
    """decomposes a float32 into mantissa & exponent"""
    string = '%.2e'%x
    exponent = int(string[-3:])
    mantissa = np.float64(string[:-4])
    return (mantissa, exponent)

M_dot0 = -K*rho0*v0*r0**q   
Mdot_50 = -K*(r*UNIT_LENGTH)**q*(rho_50*UNIT_DENSITY)*(v_50*UNIT_VELOCITY)/M_dot0
Mdot_16 = -K*(r*UNIT_LENGTH)**q*(rho_16*UNIT_DENSITY)*(v_16*UNIT_VELOCITY)/M_dot0
Mdot_84 = -K*(r*UNIT_LENGTH)**q*(rho_84*UNIT_DENSITY)*(v_84*UNIT_VELOCITY)/M_dot0
