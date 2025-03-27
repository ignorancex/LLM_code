import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter
from MTOV_R14_priors import p_R14
from MTOV_R14_priors import p_MTOV
from fjet import fitting_formulae

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4.,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'


def pm_plaw(m,alpha,Mmin,Mmax=2.5):
    p = m**alpha*np.heaviside(m-Mmin,0.)*np.heaviside(Mmax-m,0.)
    return (1.+alpha)*p/(Mmax**(1.+alpha)-Mmin**(1.+alpha))

def pm_gauss(m,mu,sigma,Mmin=1.,Mmax=2.5):
    p = np.exp(-0.5*((m-mu)/sigma)**2)*np.heaviside(m-Mmin,0.)*np.heaviside(Mmax-m,0.)
    return p

alpha = np.linspace(-20.,10.,80)
Mmin = np.linspace(1.,1.25,31)

Pm_a_MminGW = np.load('data/Pm_a_MminGW.npy')

# sample distribution and plot constraint in mass distribution space
w = Pm_a_MminGW.ravel()
w/=np.sum(w)

mtov0 = np.linspace(1.85,2.4,100)
pmtov0 = p_MTOV(mtov0)
cmtov0 = cumtrapz(pmtov0,mtov0,initial=0.)
cmtov0/=cmtov0[-1]
mtov = np.interp(np.random.uniform(0.,1.,size=Pm_a_MminGW.T.shape),cmtov0,mtov0)

m = np.linspace(1.,2.4,100)
pms = np.zeros([len(alpha)*len(Mmin),len(m)])

k = 0
for i in range(len(Mmin)):
        for j in range(len(alpha)):
            pms[k] = pm_plaw(m,alpha[j],Mmin[i],Mmax=mtov[j,i])
            k = k+1


pm_05,pm_25,pm_50,pm_75,pm_95 = np.zeros([5,len(m)])

for i in range(len(m)):
    s = np.argsort(pms[:,i])
    c = np.cumsum(w[s])
    c/=c[-1]
    cprior = np.linspace(0.,1.,len(c))
    pm_05[i],pm_25[i],pm_50[i],pm_75[i],pm_95[i] = np.interp([0.05,0.25,0.50,0.75,0.95],c,pms[s,i])
    

plt.fill_between(m,pm_05,pm_95,edgecolor='red',facecolor='#FF9999',alpha=0.3,zorder=10,lw=1)
plt.fill_between(m,pm_25,pm_75,edgecolor='red',facecolor='#FF9999',alpha=0.5,zorder=11,lw=1)
plt.plot(m,pm_50,ls='-',color='r',label='This work (power law model)',zorder=12)

# GTWC3 result (POWER model)

a,mmin,mmax = np.load('data/GWTC3_pop_alpha_mmin_mmax.npy')

pms = pm_plaw(m.reshape([len(m),1]),a,mmin,mmax)
pm_05,pm_25,pm_50,pm_75,pm_95 = np.percentile(pms,[5.,25.,50.,75.,95.],axis=1)

plt.fill_between(m,pm_05,pm_95,edgecolor='green',facecolor='#99FF99',alpha=0.3,zorder=10,lw=1)
plt.fill_between(m,pm_25,pm_75,edgecolor='green',facecolor='#99FF99',alpha=0.5,zorder=11,lw=1)
plt.plot(m,pm_50,ls='-',color='g',label='GWTC-3 (POWER model)',zorder=12)


plt.semilogy()

plt.ylim(1e-1,10)
plt.xlim(1.,2.3)

plt.xlabel(r'$m\,[\mathrm{M_\odot}]$')
plt.ylabel(r'$P(m\,|\,\theta_\mathrm{m})\,[\mathrm{M_\odot}^{-1}]$')

plt.grid(color='#EDEDED',zorder=-100)

plt.tick_params(which='both',direction='in',top=True,right=True)

# plt.title('Power law model')

plt.gca().set_axisbelow(True)

plt.xticks([1.,1.2,1.4,1.6,1.8,2.0,2.2])

plt.legend(frameon=False,loc='upper right',markerfirst=False)

plt.savefig('../figures/Plaw_GWTC3_comparison.pdf')
plt.show()    


mu = np.linspace(1.,2.,30)
sigma = np.linspace(0.01,0.5,31)
Pm_mu_sigmaGW = np.load('data/Pm_mu_sigmaGW.npy')
w = Pm_mu_sigmaGW.ravel()
w/=np.sum(w)

mtov = np.interp(np.random.uniform(0.,1.,size=Pm_mu_sigmaGW.T.shape),cmtov0,mtov0)

pms = np.zeros([len(mu)*len(sigma),len(m)])

k = 0
for i in range(len(sigma)):
        for j in range(len(mu)):
            pms[k] = pm_gauss(m,mu[j],sigma[i],Mmax=mtov[j,i])
            pms[k]/=np.trapz(pms[k],m)
            k = k+1

pm_05,pm_25,pm_50,pm_75,pm_95 = np.zeros([5,len(m)])

for i in range(len(m)):
    s = np.argsort(pms[:,i])
    c = np.cumsum(w[s])
    c/=c[-1]
    cprior = np.linspace(0.,1.,len(c))
    pm_05[i],pm_25[i],pm_50[i],pm_75[i],pm_95[i] = np.interp([0.05,0.25,0.50,0.75,0.95],c,pms[s,i])
    

plt.fill_between(m,pm_05,pm_95,edgecolor='blue',facecolor='#9999FF',alpha=0.3,zorder=10,lw=1)
plt.fill_between(m,pm_25,pm_75,edgecolor='blue',facecolor='#9999FF',alpha=0.5,zorder=11,lw=1)
plt.plot(m,pm_50,ls='-',color='b',label='This work (Gaussian model)',zorder=12)

# GTWC3 result (PEAK model)

mu,sigma,mmin,mmax = np.load('data/GWTC3_pop_mu_sigma_mmin_mmax.npy')

pms = pm_gauss(m.reshape([len(m),1]),mu,sigma,mmin,mmax)
pms/=np.trapz(pms,m,axis=0)
pm_05,pm_25,pm_50,pm_75,pm_95 = np.percentile(pms,[5.,25.,50.,75.,95.],axis=1)

plt.fill_between(m,pm_05,pm_95,edgecolor='green',facecolor='#99FF99',alpha=0.3,zorder=10,lw=1)
plt.fill_between(m,pm_25,pm_75,edgecolor='green',facecolor='#99FF99',alpha=0.5,zorder=11,lw=1)
plt.plot(m,pm_50,ls='-',color='green',label='GWTC-3 (PEAK model)',zorder=12)


plt.semilogy()

plt.ylim(1e-1,10)
plt.xlim(1.,2.3)

plt.xlabel(r'$m\,[\mathrm{M_\odot}]$')
plt.ylabel(r'$P(m\,|\,\theta_\mathrm{m})\,[\mathrm{M_\odot}^{-1}]$')

plt.grid(color='#EDEDED',zorder=-100)

plt.tick_params(which='both',direction='in',top=True,right=True)

# plt.title('Gaussian model')

plt.gca().set_axisbelow(True)

plt.xticks([1.,1.2,1.4,1.6,1.8,2.0,2.2])

plt.legend(frameon=False,loc='upper right',markerfirst=False)

plt.savefig('../figures/Gauss_GWTC3_comparison.pdf')
plt.show()    


