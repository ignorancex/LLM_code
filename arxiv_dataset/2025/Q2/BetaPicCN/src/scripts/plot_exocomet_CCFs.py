import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
import paths

hdu=fits.open(paths.data/'v_phasefold.fits')
v=hdu[0].data
hdu.close()

hdu=fits.open(paths.data/'phase_cube_comet.fits')
d=hdu[0].data
hdu.close()

hdu=fits.open(paths.data/'FEB_idx.fits')
idx=hdu[0].data.astype(bool)
hdu.close()

hdu=fits.open(paths.data/'T_gas.fits')
T_gas=hdu[0].data
hdu.close()


ord=3


f,ax=plt.subplots(4,5,sharex=True,sharey=True,figsize=(15,9))
for i in range(5):
    for j in range(4):
        ax[j,i].pcolormesh(v,np.arange(idx.sum()),d[i+j*5,33,0,idx,:])
        ax[j,i].annotate("{0:} K".format(T_gas[i+j*5]),(84,30),ha='right')
        if i==0:
            ax[j,i].set_ylabel('Night Number')
    ax[3,i].set_xlabel('v$_{rest,comet}$ [km/s]')
f.tight_layout()
f.subplots_adjust(hspace=0,wspace=0)
plt.savefig(paths.figures/'CN_CCF_ExocometRestframe.pdf')
