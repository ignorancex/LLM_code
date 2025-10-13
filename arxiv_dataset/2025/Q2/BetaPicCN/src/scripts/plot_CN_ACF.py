import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
import paths

#mpl.use('macosx')
mpl.rcParams['figure.figsize'] = 12, 8
mpl.rc('image', interpolation='nearest', origin='lower', cmap = 'gray')

#from pylab import rcParams
from scipy import interpolate
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
from astropy.stats.sigma_clipping import sigma_clipped_stats
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial as P
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats as stats
import time
import gc

from analyse_HARPS_spectra import c_correlate,get_mdl,read_data

nspec,spec,wlen,bary,obsdate=read_data(order=3)


###Perform autocorrelation of the model to show the strength and pattern in the different orders
order_start= 26
order_end  =4060

dv=np.linspace(-300,300,601)
col=['r','g','b','c','y','m','k','orange']
T_gas0=np.asarray([10, 20, 50, 100, 200, 300, 1000, 2000])
T_gas0=np.asarray([10, 50, 100,  300, 1000, 2000])
f,ax=plt.subplots(2,1,sharex=True)


colors = plt.cm.viridis(np.linspace(0,1,np.size(T_gas0)))
##: left panel: un-normalised ACF, right panel: normalised ACF
for i,T in enumerate(T_gas0):
    w,prof,imdl=get_mdl('CN/CN_{0:04d}K.npy'.format(T),N=1e13)
    p2=imdl(wlen[order_start:order_end])
    ccf,ccf0=c_correlate(wlen[order_start:order_end],np.vstack((p2,p2)),imdl,dv=dv)
    ax[0].plot(dv,ccf[0,:],color=colors[i],linewidth=3,label='T={0:}K'.format(T))
    ax[1].plot(dv,ccf[0,:]/np.max(ccf[0,:]),color=colors[i],linewidth=3,label='T={0:}K'.format(T))

ax[0].tick_params(axis='both',labelsize=14)
ax[1].tick_params(axis='both',labelsize=14)
#f.tight_layout()

ax[0].legend()

ax[1].set_xlabel(f'Rest frame velocity [km/s]',fontsize=18)
ax[1].set_ylabel(f'Peak normalised',fontsize=18)
ax[0].set_ylabel(f'Strength',fontsize=18)

ax[0].set_xlim(-150,150)

f.savefig(paths.figures/'CN_ACF.pdf') 

#plt.show()
