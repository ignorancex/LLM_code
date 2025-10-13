import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits
import matplotlib as mpl

#mpl.use('macosx')


T0=fits.getdata(paths.data/'T_gas.fits')
N0=fits.getdata(paths.data/'Nf.fits')
w=fits.getdata(paths.data/'broadening.fits')

midpts_T=(T0[1:]+T0[:-1])/2.
x=np.arange(midpts_T.shape[0])+1
#T=10**np.polyval(np.polyfit(x,np.log10(midpts_T),1),np.arange(T0.shape[0]+1))
T=np.hstack((0.,midpts_T,1.5*T0[-1]))

midpts_N=(N0[1:]+N0[:-1])/2.
x=np.arange(midpts_N.shape[0])+1
N=10**np.polyval(np.polyfit(x,np.log10(midpts_N),1),np.arange(N0.shape[0]+1))



# fit=fits.getdata(paths.data/'fit_cube_comet.fits')
# sig0=fits.getdata(paths.data/'fit_err_cube_comet.fits')

# mean=fit[:,:,:,-1,:]
# sig=sig0[:,:,:,-1,:]#np.std(fit[:,:,:,:-1,:],axis=-2)/np.sqrt(fit.shape[-2])
# SN=mean/sig
# print(SN.shape,T.shape,N.shape,w.shape)

# f,ax=plt.subplots(1,6,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
# vmin=np.min(SN[:,:,:,0])
# vmax=np.max(SN[:,:,:,0])
# for i in range(5):
#     im=ax[i].pcolormesh(N,T,SN[:,:,i,0],vmin=vmin,vmax=vmax,shading='flat')
#     ax[i].set_xscale('log')
# f.colorbar(im,cax=ax[-1])

# ax[0].set_ylabel('T [K]',fontsize=12)
# for i in range(5):
#     ax[i].set_xlabel('N$\cdot$f [cm$^{-2}$]',fontsize=12)
    
# f.tight_layout()




##run for simulation

# fit=fits.getdata(paths.data/'sim_fit_cube_comet.fits')

# mean=fit[:,:,:,-1,:]#np.mean(fit,axis=-2)
# sig=np.std(fit[:,:,:,:-1,:],axis=-2)/np.sqrt(fit.shape[-2])



fit=fits.getdata(paths.data/'sim_fit_cube_comet.fits')
sig0=fits.getdata(paths.data/'sim_fit_err_cube_comet.fits')

mean=fit[:,:,:,-1,:]
sig=sig0[:,:,:,-1,:]
SN=mean/sig

def fmt(x):
    s = f"{x:.0f}"
    return rf"{s} \sigma" if plt.rcParams["text.usetex"] else f"{s} $\sigma$"

f,ax=plt.subplots(1,4,figsize=(16,6),gridspec_kw={"width_ratios":[1,1,1,0.1]})
# vmin=np.min(SN[:,:,:,0])
# vmax=np.max(SN[:,:,:,0])
for aa,(i) in enumerate((0,2,4)):
    SNclip = SN[:,:,i,0]
    SNclip[(SNclip<1)]=1
    im=ax[aa].pcolormesh(N0,T0,SN[:,:,i,0],
        shading='auto',
        vmin=1,vmax=600,norm='log')
    ax[aa].set_xscale('log')
    ax[aa].set_yscale('log')
    CS = ax[aa].contour(N0, T0, SN[:,:,i,0], [5,20,50,200],
        colors=['#AAAAAA', '#AA8888', '#AA4444','#AA2222'],
        extend='both')

    ax[aa].clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=20)

    ax[aa].set_xlim([10**11.5,10**15.5])
    ax[aa].set_ylim([10**1,10**3.4])

    ax[aa].set_xlabel(f'N$\cdot$f [$cm^{{-2}}$] $\sigma$={w[i]:.1f} km/s',fontsize=18)
    #ax[i].scatter(yy,xx)
    ax[aa].tick_params(axis='both',labelsize=14)

f.colorbar(im,cax=ax[-1])

ax[0].set_ylabel('T [K]',fontsize=18)
    
f.tight_layout()

plt.savefig(paths.figures/'HARPS_CN_exocomet_frame.pdf')
#plt.show()
