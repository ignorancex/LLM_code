import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits
from analyse_HARPS_spectra import get_mdl

wave=fits.getdata(paths.data/"wave_corrected_spectra.fits")
stellar=fits.getdata(paths.data/"corrected_spectra_stellar.fits")
comet=fits.getdata(paths.data/"corrected_spectra_comet.fits")
FEB_idx=fits.getdata(paths.data/'FEB_idx.fits').astype(bool)
w_CN,s_CN,i_CN=get_mdl(fn='CN/CN_0300K.npy',N=1e14,FWHM_kernel=3e5/1.1e5)




fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_xlabel('Wavelength [$\AA$] (Stellar restframe)',fontsize=16)
ax.set_ylabel('Normalised spectrum [a.u.]',fontsize=16)


ax.plot(wave[3,215:3860],np.nanmean(stellar[:,3,215:3860],axis=0),label='stacked spectrum')
ax.plot(w_CN,s_CN*0.98,label='Model spectrum CN at 300K (N=10$^{14}$)')
ax.set_xlim( (3855,3886) )
ax.set_ylim( (0.95,1.02) )
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'spec_stack_CN_stellar.pdf')




fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_xlabel('Wavelength [$\AA$] (Comet restframe)',fontsize=16)
ax.set_ylabel('Normalised spectrum [a.u.]',fontsize=16)


ax.plot(wave[3,215:3860],np.nanmean(comet[FEB_idx,3,215:3860],axis=0),label='stacked spectrum')
ax.plot(w_CN,s_CN*0.98,label='Model spectrum CN at 300K (N=10$^{14}$)')
ax.set_xlim( (3855,3886) )
ax.set_ylim( (0.95,1.02) )
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'spec_stack_CN_comet.pdf')



fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_xlabel('Wavelength [$\AA$] (Comet restframe)',fontsize=16)
ax.set_ylabel('Normalised spectrum [a.u.]',fontsize=16)


ax.plot(wave[7,:],np.nanmean(comet[FEB_idx,7,:],axis=0),label='stacked spectrum')
ylim=ax.get_ylim()
ax.vlines(3968.469,ymin=ylim[0],ymax=ylim[1],color='k',linestyle='--')
ax.set_xlim( (3968.469/1.001,3968.469*1.001) )
ax.set_ylim(ylim)
ax.legend()
plt.draw()

plt.tight_layout()

fig.savefig(paths.figures/'spec_stack_Ca_comet.pdf')



plt.figure()
plt.imshow(stellar[:,3,:],aspect='auto',origin='lower',vmin=0.98,vmax=1.02)
plt.show()
