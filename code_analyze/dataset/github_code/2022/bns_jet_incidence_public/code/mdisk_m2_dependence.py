import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import fjet
from fjet import NS
from MTOV_R14_priors import p_R14
from MTOV_R14_priors import p_MTOV
from fjet import fitting_formulae

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4.,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

EoSs = ['APR4','SFHo','DD2']
colors = {'APR4':'blue','SFHo':'red','DD2':'grey'}

for EoS in EoSs:
    m2 = np.linspace(1.2,NS.M_max[EoS],1000)
    c2 = NS.C(m2,EoS)
    md = fitting_formulae.Mdisk_KF20(m2,c2)
    
    plt.plot(m2,md,label=EoS,lw=3,c=colors[EoS])


plt.ylim(6e-4,0.5)
plt.xlim(1.2,1.7)

plt.semilogy()

plt.xlabel(r'$M_2\,\mathrm{[M_\odot]}$')
plt.ylabel(r'$M_\mathrm{disc}\,\mathrm{[M_\odot]}$')

plt.axhline(y=1e-2,ls='--',color='grey')
plt.axhline(y=1e-3,ls='--',color='grey')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.legend(frameon=False)

plt.savefig('../figures/mdisc_m2_dependence.pdf')

plt.show()
