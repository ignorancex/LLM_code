import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import paths
import matplotlib as mpl
#mpl.use('macosx')

FEB_spec=fits.getdata(paths.data/'FEB_spec.fits')
FEB_idx=fits.getdata(paths.data/'FEB_idx.fits')
NO_FEB_idx=fits.getdata(paths.data/'NO_FEB_idx.fits')
FEB_depths=fits.getdata(paths.data/'FEB_depths.fits')
FEB_vel=fits.getdata(paths.data/'FEB_velocities.fits')

spec_FEB=FEB_spec[:,FEB_idx]
spec_NO_FEB=FEB_spec[:,NO_FEB_idx]

xlow=0.15
xhig=0.60

f,ax=plt.subplots(1,1,figsize=(8,4))

ax.plot(FEB_vel,FEB_depths,'k.')
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('Normalised continuum')
ax.set_ylim(-0.1,1.1)
ax.hlines(xlow,-300,300,linestyle='dashed')
ax.hlines(xhig,-300,300,linestyle='dashed')
ax.hlines(0,-300,300,linestyle='dotted')
ax.set_xlim(-25,80)
plt.savefig(paths.figures/'FEB_velocity_depth.pdf')
#plt.show()
