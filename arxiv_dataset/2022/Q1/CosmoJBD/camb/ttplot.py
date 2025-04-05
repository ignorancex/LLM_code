from numpy import *
from scipy import *
from pylab import *
from astropy.io import fits
import matplotlib.pyplot as pyplot

grid = genfromtxt('eftcamb_scalCls_grbd.dat')
grid_lamda = genfromtxt('eftcamb_scalCls_truelamda.dat')
data = fits.open('COM_PowerSpect_CMB_R2.02.fits')
print data.info()
cmb_tt_low = data[1].data
cmb_tt_high = data[7].data

#theoretical data
l = grid[:,0]
tt_model_new = grid[:,1]
#tt_model_new_2 = grid2[:,1]
tt_lamda = grid_lamda[:,1]

f,axarr = pyplot.subplots(3, sharex='True')
axarr[0].plot(l,tt_lamda,'k-',label = r"$\Lambda \rm{CDM}$")
axarr[0].plot(l,tt_model_new,'r-',label = r"Model")
#axarr[0].plot(l,tt_model_new_2,label = r"$f_{\mathcal{R}}(z_{\rm{i}}) = 1 \times 10^{-1}$")
axarr[1].plot(l,(tt_model_new-tt_lamda),'r-',label = r"$\rm{Model}-\Lambda \rm{CDM}$")
axarr[2].plot(l,abs(tt_model_new-tt_lamda)/tt_lamda*100,'r-')

axarr[0].set_ylabel(r'$\rm{l}(\rm{l}+1)C_{\rm{l}}^{\rm{TT}}/2\pi$',fontsize=15)
axarr[1].set_ylabel(r'$\Delta_{\rm{rel}}$',fontsize=15)
axarr[2].set_ylabel(r'$|TT^{\rm{Model}} - TT^{\Lambda}|/TT^{\Lambda} \times 100$',fontsize=15)
#axarr[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
xlim([2.0,2500.0])
axarr[0].set_xscale('log')
axarr[1].set_xscale('log')
axarr[2].set_xscale('log')
axarr[0].set_ylim([0.0,6000.0])
axarr[0].legend(loc = 0)
axarr[1].legend(loc = 0)
axarr[2].legend(loc = 0)
savefig('cmb_ttcomp.pdf')
show()



