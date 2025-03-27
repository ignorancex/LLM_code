import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta
from scipy.integrate import cumtrapz

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

fjet = np.linspace(0.,1.,10000)

CDF_up = fjet**2
CDF_Jp = (np.arcsin(fjet**0.5)-np.sqrt(fjet-fjet**2))*2/np.pi

CDF2_up = cumtrapz(fjet*(1.-0.06*fjet),fjet,initial=0.)
CDF2_up /= CDF2_up[-1]

CDF2_Jp = cumtrapz((1.-0.06*fjet)*fjet**0.5*(1.-fjet)**-0.5,fjet,initial=0.)
CDF2_Jp[-1]=CDF2_Jp[-2]
CDF2_Jp /= CDF2_Jp[-1]


ll90_up = np.interp(0.1,CDF_up,fjet)
ll90_Jp = np.interp(0.1,CDF_Jp,fjet)

ll3s_up = np.interp(1.-0.9973,CDF_up,fjet)
ll3s_Jp = np.interp(1.-0.9973,CDF_Jp,fjet)

ll90_2up = np.interp(0.1,CDF2_up,fjet)
ll90_2Jp = np.interp(0.1,CDF2_Jp,fjet)

ll3s_2up = np.interp(1.-0.9973,CDF2_up,fjet)
ll3s_2Jp = np.interp(1.-0.9973,CDF2_Jp,fjet)


plt.figure(figsize=(4.,3.5),tight_layout=True)

plt.plot(fjet,CDF_up,'-b',lw=3,label=r'Uniform prior (GW170817)',alpha=0.5)
plt.plot(fjet,CDF_Jp,'-r',lw=3,label=r'Jeffreys pr. (GW170817)',alpha=0.5)

plt.plot(fjet,CDF2_up,'--b',lw=3,label=r'Unif. pr. (GW170817+GW190425)')
plt.plot(fjet,CDF2_Jp,'--r',lw=3,label=r'Jeffr. pr. (GW170817+GW190425)')




plt.xlim(0.,1.)
plt.ylim(0.,1.)

plt.xlabel(r'$f_\mathrm{j,GW}$')
plt.ylabel(r'$C(f_\mathrm{j,GW}\,|\,\mathbf{d}_\mathrm{GW})$')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.legend(frameon=True,framealpha=1)

plt.grid()

plt.savefig('../figures/cumulative_posterior_2events.pdf')

plt.show()


