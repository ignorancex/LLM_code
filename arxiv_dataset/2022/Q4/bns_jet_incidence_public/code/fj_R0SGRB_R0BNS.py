import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from KDEpy import FFTKDE

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'


# load R_0_SGRB lower limit posterior samples
R0_SGRB_lolim = np.load('data/R0_GW170817.npy')

# set number of samples
Nsamples = len(R0_SGRB_lolim)

# sample the R_0_BNS distribution
r0mw,pr0 = np.loadtxt('data/Grunthal2021_RMW.txt',unpack=True) # Grunthal+2021 posterior
r0 = r0mw*0.0116*1e3 # 0.0116 is # MWEG density in Mpc-3 (Abadie+2010); 1e3 is Mpc-3 to Gpc-3 and Myr-1 to yr-1

cr0 = cumtrapz(pr0,r0,initial=0.)
cr0/=cr0[-1]

R0_BNS = np.interp(np.random.uniform(0.,1.,Nsamples),cr0,r0)
R0_BNS *= np.maximum(0.01,np.random.normal(1.,0.3,Nsamples)) # add 30% uncertainty on MWEG density

# the SGRB rate is sampled from a uniform-in-log distribution limited from below by R0_SGRB_lolim 
R0_SGRB = 10**np.random.uniform(-1,4,Nsamples)
while np.any(R0_SGRB<R0_SGRB_lolim):
    to_be_resampled = R0_SGRB<R0_SGRB_lolim
    Nresample = int(np.sum(to_be_resampled))
    R0_SGRB[to_be_resampled] = 10**np.random.uniform(-1,6,Nresample)

fj = R0_SGRB/R0_BNS

fj = fj[(fj<=1.)&(fj>0.)]
c = np.linspace(0.,1.,len(fj))

plt.plot(np.sort(fj),c)
plt.show()

a = np.linspace(0.35,0.55,300)
logpa = np.zeros_like(a)

for i in range(len(a)):
    logpa[i] = len(fj)*np.log(a[i]+1.) + a[i]*np.sum(np.log(fj))

pa = np.exp(logpa-logpa.max())

pa/=np.trapz(pa,a)
    
plt.plot(a,pa,lw=3,c='r')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Posterior probability density')
plt.tick_params(which='both',direction='in',top=True,right=True)
plt.show()

a0 = a[np.argmax(pa)]
print(a0)
f0 = np.linspace(0.,1.,10000)

plt.hist(fj,bins=50,histtype='step',density=True,color='b',label='$f_\mathrm{j,tot}$ samples')
plt.plot(f0,(a0+1)*f0**a0,'-r',label='power law approximation (Eq. 8)')

plt.xlabel(r'$f_\mathrm{j,tot}$')
plt.ylabel('Probability density')

plt.xlim(0.,1.)
plt.ylim(1e-1,2)

plt.yscale('log')

plt.tick_params(which='both',direction='in',top=True,right=True)
plt.legend(frameon=False,markerfirst=False,loc='lower right')

plt.savefig('../figures/fj_R0comparison_posterior.pdf')

plt.show()

np.save('data/fj_from_R0_RBNS.npy',(np.sort(fj),c))
