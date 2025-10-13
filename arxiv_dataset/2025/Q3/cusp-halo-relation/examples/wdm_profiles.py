import numpy as np

import sys
sys.path.append('..')
from cusp_halo_relation import CuspHaloWDM, cuspNFW

# virial overdensity
Dvir = 200.

# primordial power spectrum
n_s = 0.9649
A_s = 2.100e-9

mX = 10 # dark matter mass in keV
model = CuspHaloWDM(mX=mX,cutoff='VA23',n_s=n_s,A_s=A_s,transfer='table',verbose=True)

# power of r by which to scale vertical axis
rscale = 1.5

import matplotlib.pyplot as plt
fig,axs = plt.subplots(nrows=3,ncols=1,figsize=(4.8,6.4),gridspec_kw={'hspace':0.,'wspace':.0},sharex=True,sharey=True)
for i,ax in enumerate(axs):
  ax.set_ylim(.5e6,5e7)
  ax.set_xlim(2e-3,20)
  if i == 2:
    ax.set_xlabel(r'radius $r$ (kpc)')
  ax.set_ylabel(r'$r^{1.5}\rho(r)$ (M$_\odot$kpc$^{-1.5}$)')
  ax.tick_params(which='both',top=i==0,bottom=i==2,labelbottom=i==2,right=True)
colors = ['C0','k','C1','C2','gray']
zorders = [1.9,2,1.8,1.7,1.6]
lws = [1,1.5,1,1,.7]

# base halo parameters (for black curve)
z = 2
M0 = 3e8
c0 = model.c_at_z(z)
A0 = model.A_at_z(M0,z)*1e3**-1.5

# virial density
rho_vir = Dvir * model.rhoCrit_at_z(z) * 1e-9 # solar mass/kpc^3

# plot profiles of varying A
ax = axs[0]

M = M0
c = c0
Apred = A0
Amax = cuspNFW.A_max(c,M,rho_vir)*.9999999
for i,A in enumerate([Apred*2,Apred,Apred/2,Apred/4,0]):
  color = colors[i]
  R = cuspNFW.R_from_M(M,rho_vir)
  r = np.geomspace(1e-5,1,1000)*R
  rs,rhos = cuspNFW.scale_from_c(c,M,A,rho_vir)
  ax.loglog(r,cuspNFW.density(r,rs,rhos,A)*r**rscale,color=color,zorder=zorders[i],lw=lws[i],)
ax.text(.5,.1,r'varying $A$',transform=ax.transAxes,color='k',size=8,va='bottom',ha='center',bbox=dict(facecolor='w', edgecolor='lightgray', boxstyle='round,pad=.4'))
  
# plot profiles of varying c
ax = axs[1]

M = M0
A = A0
cpred = c0
cmin = cuspNFW.c_min(M,A,rho_vir)*1.0000001
for i,c in enumerate([cpred-4,cpred,cpred+4,cpred+8]):
  color = colors[i]
  R = cuspNFW.R_from_M(M,rho_vir)
  r = np.geomspace(1e-5,1,1000)*R
  rs,rhos = cuspNFW.scale_from_c(c,M,A,rho_vir)
  ax.loglog(r,cuspNFW.density(r,rs,rhos,A)*r**rscale,color=color,zorder=zorders[i],lw=lws[i],)
ax.text(.5,.1,r'varying $c$',transform=ax.transAxes,color='k',size=8,va='bottom',ha='center',bbox=dict(facecolor='w', edgecolor='lightgray', boxstyle='round,pad=.4'))
  
# plot profiles of varying M
ax = axs[2]

c = c0
A = A0
Mpred = M0
Mmin = cuspNFW.M_min(c,A,rho_vir)*1.0001
for i,M in enumerate([Mpred/3,Mpred,Mpred*10/3,Mpred*10]):
  color = colors[i]
  R = cuspNFW.R_from_M(M,rho_vir)
  r = np.geomspace(1e-5,1,1000)*R
  rs,rhos = cuspNFW.scale_from_c(c,M,A,rho_vir)
  ax.loglog(r,cuspNFW.density(r,rs,rhos,A)*r**rscale,color=color,zorder=zorders[i],lw=lws[i],)
ax.text(.5,.1,r'varying $M$',transform=ax.transAxes,color='k',size=8,va='bottom',ha='center',bbox=dict(facecolor='w', edgecolor='lightgray', boxstyle='round,pad=.4'))
  
# done
fig.savefig('wdm_profiles.png',bbox_inches='tight',pad_inches=0.03,dpi=600)