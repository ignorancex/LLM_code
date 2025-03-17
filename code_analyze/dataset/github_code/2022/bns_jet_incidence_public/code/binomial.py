import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta
from scipy.integrate import cumtrapz

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

fjet = np.linspace(0.,1.,1000)

CDF_up = fjet**2
CDF_Jp = (np.arcsin(fjet**0.5)-np.sqrt(fjet-fjet**2))*2/np.pi

ll90_up = np.interp(0.1,CDF_up,fjet)
ll90_Jp = np.interp(0.1,CDF_Jp,fjet)

ll3s_up = np.interp(1.-0.9973,CDF_up,fjet)
ll3s_Jp = np.interp(1.-0.9973,CDF_Jp,fjet)

plt.figure(figsize=(4.,3.5),tight_layout=True)

plt.plot(fjet,CDF_up,'-b',lw=3,label=r'$f_\mathrm{j,GW}$, uniform prior')
plt.plot(fjet,CDF_Jp,'-r',lw=3,label=r'$f_\mathrm{j,GW}$, Jeffreys prior')

# add fj from R0/RBNS
fj,cdf = np.load('../data/fj_from_R0_RBNS.npy')

ll90_r0 = np.interp(0.1,cdf,fj)
ll3s_r0 = np.interp(1.-0.9973,cdf,fj)

plt.plot(fj,cdf,ls='-',c='grey',lw=3,label=r'$f_\mathrm{j,tot}=R_\mathrm{0,SJ}/R_\mathrm{0,BNS}$',zorder=0.1)

print(ll3s_up,ll3s_Jp,ll3s_r0)


plt.axhline(y=0.1,ls='--',color='k')

plt.annotate(xy=(0.6,0.11),text='90% lower limit',ha='left')

plt.annotate(xy=(ll90_up,0.1),xytext=(ll90_up,0.3),text=r'{:.1%}'.format(ll90_up),ha='right',arrowprops={'arrowstyle':'-'})

plt.annotate(xy=(ll90_Jp,0.1),xytext=(ll90_Jp,0.45),text=r'{:.1%}'.format(ll90_Jp),ha='center',arrowprops={'arrowstyle':'-'})

plt.annotate(xy=(ll90_r0,0.1),xytext=(ll90_r0-0.1,0.45),text=r'{:.1%}'.format(ll90_r0),ha='center',arrowprops={'arrowstyle':'-'})


plt.xlim(0.,1.)
plt.ylim(0.,1.)

plt.xlabel(r'$f_\mathrm{j}$')
plt.ylabel(r'$C(f_\mathrm{j}\,|\,\mathbf{d}_\mathrm{j})$')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.legend(frameon=True,framealpha=1)

plt.grid()

plt.savefig('../figures/cumulative_posterior.pdf')

plt.show()


