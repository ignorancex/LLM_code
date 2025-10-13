import numpy as np
from scipy.special import gamma,gammainc
from scipy.optimize import brentq, curve_fit

import sys
sys.path.append('..') # point this to the location of the repository
from cusp_halo_relation import cuspNFW

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.set_ylim(1e-5,1e11)
ax.set_xlim(3e-5,30)
ax.set_xlabel(r'energy $E/(G \rho_\mathrm{s} r_\mathrm{s}^2)$')
ax.set_ylabel(r'$G^{3/2}r_\mathrm{s}^3\rho_\mathrm{s}^{1/2}f(E)$')
ax.tick_params(which='both',top=True,right=True)

###
rs = rhos = 1.
for i,y in enumerate([0.,.01,.1,.3,.6,1]):
  E = np.geomspace(1e-7,1.,1000)*cuspNFW.potential(1e12,rs,rhos,y)
  color = 'C%d'%i
  ax.loglog(E,cuspNFW.df(E,rs,rhos,y),color=color)
  label = r'$y=%g$'%y
  ax.text(.02,.06+i*.1,label,transform=ax.transAxes,color=color,size=8)

###
fig.savefig('stability.png',bbox_inches='tight',pad_inches=0.03,dpi=600)