import numpy as np

import sys
sys.path.append('..') # point this to the location of the repository
from cusp_halo_relation import CuspHaloWDM

# virial overdensity
Dvir = 200.

# primordial power spectrum
n_s = 0.9649
A_s = 2.100e-9

# range of parameters to plot
zcoll = np.array([0,2,6,15,32])
zlist = np.array([2,0])
mXlist = np.geomspace(.8,80,50)
Mlist = 10.**np.arange(6,11)

# evaluate cusp/halo properties, save to these arrays
A = np.zeros((len(zlist),len(Mlist),len(mXlist)))
m = np.zeros((len(zlist),len(Mlist),len(mXlist)))
Mhmlist = np.zeros_like(mXlist)
A0 = np.zeros(len(mXlist))
m0 = np.zeros(len(mXlist))
Acoll = np.zeros((len(zcoll),len(mXlist)))
mcoll = np.zeros((len(zcoll),len(mXlist)))
for i,mX in enumerate(mXlist): # DM particle mass
  model = CuspHaloWDM(mX=mX,cutoff='VA23',n_s=n_s,A_s=A_s,transfer='table',verbose=True)
  Mhmlist[i] = model.Mhm
  for j,M in enumerate(Mlist): # halo mass
    for k,z in enumerate(zlist): # redshift
      A[k,j,i] = model.A_at_z(M,z)
      m[k,j,i] = model.m_at_z(M,z)
  A0[i] = model.characteristic_A(model.a0)
  m0[i] = model.characteristic_m(model.a0)
  for k,z in enumerate(zcoll): # formation redshift
    Acoll[k,i] = model.characteristic_A(1./(1+z))
    mcoll[k,i] = model.characteristic_m(1./(1+z))

# now make the plots
import matplotlib.pyplot as plt
fig1,ax1 = plt.subplots(nrows=1,ncols=1)
fig2,ax2 = plt.subplots(nrows=1,ncols=1)
for k,z in enumerate(zlist):
  ls = ['-','--'][k]
  for j,M in enumerate(Mlist):
    color = 'C%d'%(j)
    ax1.loglog(mXlist,A[k,j]*1e3**-1.5,color=color,ls=ls,lw=1) # convert from Mpc to kpc
    ax2.loglog(mXlist,m[k,j]/M,color=color,ls=ls,lw=1)
ax1.plot(mXlist,A0*1e3**-1.5,color='darkgray',ls='-',zorder=1,lw=.7) # convert from Mpc to kpc
for k,z in enumerate(zcoll):
  ax1.plot(mXlist,Acoll[k]*1e3**-1.5,color='darkgray',ls=':',lw=.7,zorder=1) # convert from Mpc to kpc
ax1.set_ylim(4e4,3e7)
ax2.set_ylim(.3e-4,1e0)
ax1.set_ylabel(r'cusp $A$ ($\mathrm{M}_\odot\mathrm{kpc}^{-1.5}$)')
ax2.set_ylabel(r'cusp mass fraction $m/M$')
mXtoMhm = lambda mX: np.exp(np.interp(np.log(mX),np.log(mXlist),np.log(Mhmlist)))
MhmtomX = lambda Mhm: np.exp(np.interp(np.log(Mhm),np.log(Mhmlist[::-1]),np.log(mXlist[::-1])))
for ax in ax1,ax2:
  ax.set_xlabel(r'dark matter mass $m_\chi$ (keV)')
  ax.tick_params(which='both',right=True)
  ax.set_xlim(1,70)
  xax = ax.secondary_xaxis('top',functions=(mXtoMhm,MhmtomX))
  xax.set_xlabel(r'half-mode mass scale $M_\mathrm{hm}$ (M$_\odot$)',labelpad=6)
fig1.savefig('wdm_cusps_A.png',bbox_inches='tight',pad_inches=0.03,dpi=600)
fig2.savefig('wdm_cusps_m.png',bbox_inches='tight',pad_inches=0.03,dpi=600)
