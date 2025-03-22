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
matplotlib.use('pdf')

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
q, K = 1, 2*np.pi #Geometry factors
r0 = -q*v0*gamma*tcool0
M_dot = -K*r0**q*rho0*v0

transonic = np.loadtxt('./Cylindrical_CF/transonic/transonic_rdpv.txt')

fig = plt.figure(figsize=(20,20))
cs_analytic = np.sqrt(transonic[:,2]/transonic[:,1]) #de-dimensionalized
plt.semilogx(transonic[:,0], transonic[:,-1]*v0/(cs_analytic*cs0), linestyle='--', label=r'Steady state',linewidth=5, color='gray')
plt.semilogx(transonic[:,0], transonic[:,2], linestyle='--',linewidth=5, color='gray')
plt.semilogx(transonic[:,0], transonic[:,1], linestyle='--', linewidth=5, color='gray')
plt.semilogx(transonic[:,0], -transonic[:,-1], linestyle='--', linewidth=5, color='gray')
plt.semilogx(transonic[:,0], transonic[:,2]/transonic[:,1], linestyle='--', linewidth=5, color='gray')

lastfile = 100  #if == 0 print only IC
startfile = 50
if (lastfile==0): startfile=0
if(lastfile<startfile and lastfile!=0):
    print('Lastfile must be greater or equal to startfile')
    sys.exit(0)
tcool_max = 0.
tcool_min = 0.

#read initial file to know array sizes
file = np.loadtxt('./Cylindrical_CF/transonic/Output/data.%04d.tab'%0)   
r = file[:,0]
v = file[:,3]
prs = file[:,4]
rho = file[:,2]

r = np.zeros((len(r),(lastfile-startfile)+1),dtype=np.float64)
v, prs, rho, cs = np.copy(r), np.copy(r), np.copy(r), np.copy(r)

for fileno in range(0,(lastfile-startfile)+1):
    file = np.loadtxt('./Cylindrical_CF/transonic/Output/data.%04d.tab'%fileno)
    
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
    
    plt.semilogx(r*UNIT_LENGTH/r0, -v/cs, '-',label=r'$\mathcal{M}=\frac{|v|}{c_s}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, prs*UNIT_DENSITY*UNIT_VELOCITY**2/P0, '-', label=r'$\tilde{P}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, rho*UNIT_DENSITY/rho0, '-', label=r'$\tilde{\rho}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, v*UNIT_VELOCITY/v0, '-', label=r'$\tilde{v}$',linewidth=5)
    
    plt.semilogx(r*UNIT_LENGTH/r0, (prs/rho)/(1.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, '-', label=r'$\tilde{T}$',linewidth=5)
    plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
    
    plt.grid()
    #plt.ylim(-0.05,1.8)
    lgd = plt.legend(loc='upper right',prop={'size': 42,})#,bbox_to_anchor=(1.001, 1.001))
    plt.xlabel(r'$\tilde{r}$', size=70)
    plt.ylabel('De-dimensionalized fluid fields', size=65)
    #plt.title('Steady state of cooling flow (Subsonic)', size=40)
    plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
    plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
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

plt.semilogx(r*UNIT_LENGTH/r0, -v_50/cs_50, '-',label=r'$\mathcal{M}=\frac{|v|}{c_s}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, -v_16/cs_16, -v_84/cs_84, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, prs_50*UNIT_DENSITY*UNIT_VELOCITY**2/P0, '-', label=r'$\tilde{P}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, prs_16*UNIT_DENSITY*UNIT_VELOCITY**2/P0, prs_84*UNIT_DENSITY*UNIT_VELOCITY**2/P0, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, prs_50*UNIT_DENSITY*UNIT_VELOCITY**2/P0/(rho_50*UNIT_DENSITY/rho0)**gamma, '-', label=r'$\tilde{s}$',linewidth=5)


plt.semilogx(r*UNIT_LENGTH/r0, rho_50*UNIT_DENSITY/rho0, '-', label=r'$\tilde{\rho}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, rho_16*UNIT_DENSITY/rho0, rho_84*UNIT_DENSITY/rho0, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, v_50*UNIT_VELOCITY/v0, '-', label=r'$\tilde{v}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, v_16*UNIT_VELOCITY/v0, v_84*UNIT_VELOCITY/v0, alpha=0.3)

plt.semilogx(r*UNIT_LENGTH/r0, (prs_50/rho_50)/(1.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, '-', label=r'$\tilde{T}$',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, (prs_16/rho_16)/(1.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, (prs_84/rho_84)/(1.*T0)*(mu*mp/kB)*UNIT_VELOCITY**2, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')

plt.grid()
#plt.ylim(-0.05,1.8)
lgd = plt.legend(loc='upper right',prop={'size': 42,})
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel('Dedimensionalized fluid fields', size=65)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('cool-flow-pde-transonic_outflow-cyl.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

if lastfile == 0: sys.exit(0) 
def decompose(x): 
    """decomposes a float32 into mantissa & exponent"""
    string = '%.2e'%x
    exponent = int(string[-3:])
    mantissa = np.float64(string[:-4])
    return (mantissa, exponent)

M_dot0 = -K*rho0*v0*r0**q*pc*1e3      
Mdot_50 = -K*(r*UNIT_LENGTH)**q*(rho_50*UNIT_DENSITY)*(v_50*UNIT_VELOCITY)/M_dot0
Mdot_16 = -K*(r*UNIT_LENGTH)**q*(rho_16*UNIT_DENSITY)*(v_16*UNIT_VELOCITY)/M_dot0
Mdot_84 = -K*(r*UNIT_LENGTH)**q*(rho_84*UNIT_DENSITY)*(v_84*UNIT_VELOCITY)/M_dot0

fig = plt.figure(figsize=(20,20))
plt.semilogx(r*UNIT_LENGTH/r0, Mdot_50, '-',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, Mdot_16, Mdot_84, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')

M_dot0 /= (Msun/(365*24*60**2))
M_dot0 = decompose(M_dot0)
plt.grid()
#plt.ylim(-0.05,1.8)
#plt.legend(loc='upper right',prop={'size': 42,}) #,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel(r'$\dot{M}$ [$%.2f \times 10 ^{%d} M_\odot yr^{-1} kpc^{-1}$]'%(M_dot0[0],M_dot0[1]), size=65)
#plt.title('Mass flux in cooling flow (Subsonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('Massflux-transonic.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

start = 10
dr = r[1:]-r[:-1]
dr = np.hstack((dr,dr[-1]))
enthalpy_50 = -(gamma/(gamma-1))*(prs_50*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_50*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q + (gamma/(gamma-1))*(prs_50[start]*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_50[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q
enthalpy_16 = -(gamma/(gamma-1))*(prs_16*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_16*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q + (gamma/(gamma-1))*(prs_16[start]*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_16[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q
enthalpy_84 = -(gamma/(gamma-1))*(prs_84*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_84*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q + (gamma/(gamma-1))*(prs_84[start]*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_84[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q

energy_th_50 = -(1./(gamma-1))*(prs_50*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_50*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q + (1./(gamma-1))*(prs_50[start]*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_50[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q
energy_th_16 = -(1./(gamma-1))*(prs_16*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_16*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q + (1./(gamma-1))*(prs_16[start]*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_16[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q
energy_th_84 = -(1./(gamma-1))*(prs_84*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_84*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q + (1./(gamma-1))*(prs_84[start]*UNIT_DENSITY*UNIT_VELOCITY**2)*(v_84[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q

energy_tot_50 = enthalpy_50 - 0.5*(rho_50*UNIT_DENSITY)*(v_50*UNIT_VELOCITY)**2*(v_50*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q - ( - 0.5*(rho_50[start]*UNIT_DENSITY)*(v_50[0]*UNIT_VELOCITY)**2*(v_50[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q)
energy_tot_16 = enthalpy_16 - 0.5*(rho_16*UNIT_DENSITY)*(v_16*UNIT_VELOCITY)**2*(v_16*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q - ( - 0.5*(rho_16[start]*UNIT_DENSITY)*(v_16[0]*UNIT_VELOCITY)**2*(v_16[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q)
energy_tot_84 = enthalpy_84 - 0.5*(rho_84*UNIT_DENSITY)*(v_84*UNIT_VELOCITY)**2*(v_84*UNIT_VELOCITY)*(r*UNIT_LENGTH)**q - ( - 0.5*(rho_84[start]*UNIT_DENSITY)*(v_84[0]*UNIT_VELOCITY)**2*(v_84[start]*UNIT_VELOCITY)*(r[start]*UNIT_LENGTH)**q)

Temperature_50 = (mu*mp/kB) * (prs_50*UNIT_DENSITY*UNIT_VELOCITY**2)/(rho_50*UNIT_DENSITY)
Temperature_16 = (mu*mp/kB) * (prs_16*UNIT_DENSITY*UNIT_VELOCITY**2)/(rho_16*UNIT_DENSITY)
Temperature_84 = (mu*mp/kB) * (prs_84*UNIT_DENSITY*UNIT_VELOCITY**2)/(rho_84*UNIT_DENSITY)

cool_loss_50 = np.array([ np.trapz(((rho_50[:i]*UNIT_DENSITY)/(mue*mp))*((rho_50[:i]*UNIT_DENSITY)/(mui*mp))*cool(Temperature_50[:i])*(r[:i]*UNIT_LENGTH)**q, r[:i]*UNIT_LENGTH) for i in range(start,len(r))], dtype=np.float64)
cool_loss_16 = np.array([ np.trapz(((rho_50[:i]*UNIT_DENSITY)/(mue*mp))*((rho_16[:i]*UNIT_DENSITY)/(mui*mp))*cool(Temperature_16[:i])*(r[:i]*UNIT_LENGTH)**q, r[:i]*UNIT_LENGTH) for i in range(start,len(r))], dtype=np.float64)
cool_loss_84 = np.array([ np.trapz(((rho_50[:i]*UNIT_DENSITY)/(mue*mp))*((rho_84[:i]*UNIT_DENSITY)/(mui*mp))*cool(Temperature_84[:i])*(r[:i]*UNIT_LENGTH)**q, r[:i]*UNIT_LENGTH) for i in range(start,len(r))], dtype=np.float64)

fig = plt.figure(figsize=(20,20))
plt.semilogx(r[start:]*UNIT_LENGTH/r0, energy_tot_50[start:]/cool_loss_50, '-',linewidth=5)
plt.fill_between(r[start:]*UNIT_LENGTH/r0, energy_tot_16[start:]/cool_loss_16, energy_tot_84[start:]/cool_loss_84, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
plt.grid()
#plt.ylim(0.,3.)
#plt.legend(loc='upper right',prop={'size': 42,})#,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel(r'$\left. \left( r^q E_{tot} v - r_\mathcal{O}^q E_{tot,\mathcal{O}} v_\mathcal{O} \right) \right/ \int_{r_\mathcal{O}} ^ r n_e n_i \Lambda (T) r^{\prime q} dr^\prime $', size=55)
#plt.title('Cooling-Total energy flux blance (Subsonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('Energtotcool-transonic.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

M_dot0 = -K*rho0*v0*r0**q   
coolflow_dTdr_50 = Temperature_50*(gamma/(gamma-1))*(M_dot0/K)*(kB/(mu*mp)) - Temperature_50[start]*(gamma/(gamma-1))*(M_dot0/K)*(kB/(mu*mp))
coolflow_dTdr_16 = Temperature_16*(gamma/(gamma-1))*(M_dot0/K)*(kB/(mu*mp)) - Temperature_16[start]*(gamma/(gamma-1))*(M_dot0/K)*(kB/(mu*mp))
coolflow_dTdr_84 = Temperature_84*(gamma/(gamma-1))*(M_dot0/K)*(kB/(mu*mp)) - Temperature_84[start]*(gamma/(gamma-1))*(M_dot0/K)*(kB/(mu*mp))

fig = plt.figure(figsize=(20,20))
plt.semilogx(r[start:]*UNIT_LENGTH/r0, coolflow_dTdr_50[start:]/cool_loss_50, '-',linewidth=5)
plt.fill_between(r[start:]*UNIT_LENGTH/r0, coolflow_dTdr_16[start:]/cool_loss_16, coolflow_dTdr_84[start:]/cool_loss_84, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
plt.grid()
#plt.ylim(0.,3.)
#plt.legend(loc='upper right',prop={'size': 42,})#,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel(r'$\left. \left(\frac{\gamma}{\gamma -1}\right) \left(\frac{\dot{M}}{K}\right) \left(\frac{k_B}{\mu m_p}\right) (T - T_{\mathcal{O}}) \right/ \int_{r_\mathcal{O}} ^ r n_e n_i \Lambda (T) r^{\prime q} dr^\prime $', size=55)
#plt.title('Cooling-Total energy flux blance (Subsonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('EdotMdot-transonic.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

fig = plt.figure(figsize=(20,20))
plt.semilogx(r[start:]*UNIT_LENGTH/r0, enthalpy_50[start:]/cool_loss_50, '-',linewidth=5, label=r'Enthalpy $(\times \gamma)$')
plt.fill_between(r[start:]*UNIT_LENGTH/r0, enthalpy_16[start:]/cool_loss_16, enthalpy_84[start:]/cool_loss_84, alpha=0.3)
plt.semilogx(r[start:]*UNIT_LENGTH/r0, energy_th_50[start:]/cool_loss_50, '-',linewidth=5, label=r'Thermal energy')
plt.fill_between(r[start:]*UNIT_LENGTH/r0, energy_th_16[start:]/cool_loss_16, energy_th_84[start:]/cool_loss_84, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
plt.grid()
#plt.ylim(0.,3.)
plt.legend(loc='upper right',prop={'size': 42,}) #,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel(r'$\left. \left( r^q \frac{ Pv}{\gamma - 1} - r_\mathcal{O}^q \frac{P_\mathcal{O} v_\mathcal{O}}{\gamma - 1} \right) \right/ \int_{r_\mathcal{O}} ^ r n_e n_i \Lambda (T) r^{\prime q} dr^\prime $', size=65)
#plt.title('Cooling-Enthalpy flux blance (Subsonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('Enthalpycool-transonic.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

energy_kin_50 =  energy_tot_50 - enthalpy_50
energy_kin_16 = energy_tot_16 - enthalpy_16
energy_kin_84 = energy_tot_84 - enthalpy_84

fig = plt.figure(figsize=(20,20))
plt.semilogx(r[start:]*UNIT_LENGTH/r0, energy_kin_50[start:]/cool_loss_50, '-',linewidth=5)
plt.fill_between(r[start:]*UNIT_LENGTH/r0, energy_kin_16[start:]/cool_loss_16, energy_kin_84[start:]/cool_loss_84, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
plt.grid()
#plt.ylim(-1.,0.75)
#plt.legend(loc='upper right',prop={'size': 42,}) #,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel(r'$\left. \left[ r^q \left(\frac{1}{2} \rho v^2\right) v - r_\mathcal{O}^q \left(\frac{1}{2} \rho_\mathcal{O} v_\mathcal{O}^2 \right) v_\mathcal{O} \right] \right/ \int_{r_\mathcal{O}} ^ r n_e n_i \Lambda (T) r^{\prime q} dr^\prime $', size=65)
#plt.title('Cooling-Kinetic energy flux blance (Subsonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('Energkincool-transonic.pdf',transparent =True ,bbox_inches='tight')
#plt.show()

eth_50 = (1./(gamma-1))*(prs_50*UNIT_DENSITY*UNIT_VELOCITY**2)
eth_16 = (1./(gamma-1))*(prs_16*UNIT_DENSITY*UNIT_VELOCITY**2)
eth_84 = (1./(gamma-1))*(prs_84*UNIT_DENSITY*UNIT_VELOCITY**2)

kinetic_50 = 0.5*(rho_50*UNIT_DENSITY)*(v_50*UNIT_VELOCITY)**2
kinetic_16 = 0.5*(rho_16*UNIT_DENSITY)*(v_16*UNIT_VELOCITY)**2
kinetic_84 = 0.5*(rho_84*UNIT_DENSITY)*(v_84*UNIT_VELOCITY)**2

fig = plt.figure(figsize=(20,20))
plt.semilogx(r*UNIT_LENGTH/r0, kinetic_50/eth_50, '-',linewidth=5)
plt.fill_between(r*UNIT_LENGTH/r0, kinetic_16/eth_16, kinetic_84/eth_84, alpha=0.3)
plt.axvline(x=1.0, color='black',linewidth=5,linestyle=':')
plt.grid()
#plt.ylim(-0.05,1.8)
#plt.legend(loc='upper right',prop={'size': 42,}) #,bbox_to_anchor=(1.001, 1.001))
plt.xlabel(r'$\tilde{r}$', size=70)
plt.ylabel(r'Kinetic energy/Thermal energy', size=65)
#plt.title('Fluid Kinetic energy in cooling flow (Subsonic)', size=40)
plt.tick_params(axis='both', which='major', labelsize=50, direction="out", pad=15)
plt.tick_params(axis='both', which='minor', labelsize=48, direction="out", pad=15)
#plt.xlim(1.0,20.)
#plt.tight_layout()
plt.savefig('KinE-transonic.pdf',transparent =True ,bbox_inches='tight')
#plt.show()