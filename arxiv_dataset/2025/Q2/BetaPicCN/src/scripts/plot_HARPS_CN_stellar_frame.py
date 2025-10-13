import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits
import matplotlib as mpl

#mpl.use('macosx')

T0=fits.getdata(paths.data/'T_gas.fits')
N0=fits.getdata(paths.data/'Nf.fits')
w=fits.getdata(paths.data/'broadening.fits')

##run for simulation

fit=fits.getdata(paths.data/'sim_fit_cube_stellar.fits')
sig0=fits.getdata(paths.data/'sim_fit_err_cube_stellar.fits')

mean=fit[:,:,:,-1,:]
sig=sig0[:,:,:,-1,:]
SN=mean/sig

def fmt(x):
    s = f"{x:.0f}"
    return rf"{s} \sigma" if plt.rcParams["text.usetex"] else f"{s} $\sigma$"

nplot = 2
f,ax=plt.subplots(1,nplot,figsize=(8,6),gridspec_kw={"width_ratios":[1,0.08]})
for i in range(nplot-1):
    SNclip = SN[:,:,i,0]
    SNclip[(SNclip<1)]=1
    im=ax[i].pcolormesh(N0,T0,SN[:,:,i,0],
        shading='auto',
        vmin=1,vmax=600,norm='log')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    (xx,yy) = np.meshgrid(T0,N0)
    CS = ax[i].contour(N0, T0, SN[:,:,i,0], [5.7,20,50,200],
        colors=['#AAAAAA', '#AA8888', '#AA4444','#AA2222'],
        extend='both')

    ax[i].clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=20)

    ax[i].set_xlim([10**11.5,10**15.5])
    ax[i].set_ylim([10**1,10**3.4])

    #ax[i].scatter(yy,xx)
    ax[i].tick_params(axis='both',labelsize=14)

f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=18)
ax[0].set_xlabel(f'N$\cdot$f [$cm^{{-2}}$] width={w[0]:.1f} km/s',fontsize=18)
    
f.tight_layout()

plt.savefig(paths.figures/'HARPS_CN_stellar_frame.pdf')
#plt.show()
