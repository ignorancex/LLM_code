import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter
from MTOV_R14_priors import p_R14
from MTOV_R14_priors import p_MTOV_R14
from fjet import fitting_formulae

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=8.,3.
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'


def pm1_plaw(m1,alpha,Mmin,Mmax=2.5):
    p = m1**alpha*np.heaviside(m1-Mmin,0.)*np.heaviside(Mmax-m1,0.)
    return p/np.trapz(p,m1)

def pm1_gauss(m1,mu,sigma,Mmin=1.,Mmax=2.5):
    p = np.exp(-0.5*((m1-mu)/sigma)**2)*np.heaviside(m1-Mmin,0.)*np.heaviside(Mmax-m1,0.)
    return p/np.trapz(p,m1)



if __name__=='__main__':
    
    # first create axis for combined plot panel
    ax = plt.subplot(133)
    
    # --------------------------------------- first panel: power law model  ---------------------------
    
    plt.subplot(131)
    
    
    alpha = np.linspace(-20.,10.,80)
    Mmin = np.linspace(1.,1.25,31)
    
    Pm_a_MminGW = np.load('data/Pm_a_MminGW.npy')
    
    # sample distribution and plot constraint in mass distribution space
    w = Pm_a_MminGW.ravel()
    w/=np.sum(w)
    
    mtov0 = np.linspace(1.85,2.4,100)
    r = np.random.normal(12.45,0.65,300)
    pmtov0_r = np.array([p_MTOV_R14(mtov0,r[i]) for i in range(len(r))]).T
    pmtov0 = np.mean(pmtov0_r,axis=1)
    cmtov0 = cumtrapz(pmtov0,mtov0,initial=0.)
    cmtov0/=cmtov0[-1]
    mtov = np.interp(np.random.uniform(0.,1.,size=Pm_a_MminGW.T.shape),cmtov0,mtov0)
    
    m1 = np.linspace(1.,2.4,100)
    pm1s = np.zeros([len(alpha)*len(Mmin),len(m1)])
    
    k = 0
    for i in range(len(Mmin)):
            for j in range(len(alpha)):
                pm1s[k] = pm1_plaw(m1,alpha[j],Mmin[i],Mmax=mtov[j,i])
                k = k+1
    
    pm1_05,pm1_25,pm1_50,pm1_75,pm1_95 = np.zeros([5,len(m1)])
    pm1_05_prior,pm1_25_prior,pm1_50_prior,pm1_75_prior,pm1_95_prior = np.zeros([5,len(m1)])
    for i in range(len(m1)):
        s = np.argsort(pm1s[:,i])
        c = np.cumsum(w[s])
        c/=c[-1]
        cprior = np.linspace(0.,1.,len(c))
        pm1_05[i],pm1_25[i],pm1_50[i],pm1_75[i],pm1_95[i] = np.interp([0.05,0.25,0.50,0.75,0.95],c,pm1s[s,i])
        pm1_05_prior[i],pm1_25_prior[i],pm1_50_prior[i],pm1_75_prior[i],pm1_95_prior[i] = np.interp([0.05,0.25,0.50,0.75,0.95],cprior,pm1s[s,i])
    
    plt.fill_between(m1,pm1_05,pm1_95,edgecolor='red',facecolor='#FF9999',alpha=0.3,zorder=10,lw=1)
    plt.fill_between(m1,pm1_25,pm1_75,edgecolor='red',facecolor='#FF9999',alpha=0.5,zorder=11,lw=1)
    plt.plot(m1,pm1_50,ls='-',color='r',label='Posterior',zorder=12)
    
    plt.fill_between(m1,pm1_05_prior,pm1_95_prior,edgecolor='grey',facecolor='None',alpha=1,zorder=10,ls='--',lw=1)
    plt.fill_between(m1,pm1_25_prior,pm1_75_prior,edgecolor='grey',facecolor='None',alpha=1,zorder=10,ls='--',lw=1)
    plt.plot(m1,pm1_50_prior*0.,ls='--',color='grey',label='Prior',zorder=10,lw=1)
    
    # plt.semilogy()
    
    plt.ylim(0.,3.5)
    plt.xlim(1.,2.4)
    
    plt.xlabel(r'$m\,[\mathrm{M_\odot}]$')
    plt.ylabel(r'$P(m\,|\,\theta_\mathrm{m})\,[\mathrm{M_\odot}^{-1}]$')
    
    plt.grid(color='#EDEDED',zorder=-100)
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.title('Power law model')
    
    plt.gca().set_axisbelow(True)
    
    plt.xticks([1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4])
    
    plt.legend(frameon=False,loc='upper center')
    
    # show the result aso on the combined plot panel
    
    ax.fill_between(m1,pm1_05,pm1_95,edgecolor='red',facecolor='#FF9999',alpha=0.3,zorder=10)
    ax.fill_between(m1,pm1_25,pm1_75,edgecolor='red',facecolor='#FF9999',alpha=0.5,zorder=11)
    ax.plot(m1,pm1_50,ls='-',color='r',label='Power law',zorder=12)
    
    # ------------------------------------------------------ second panel: Gaussian model ----------------------------------
    
    plt.subplot(132)
    

    mu = np.linspace(1.,2.,30)
    sigma = np.linspace(0.01,0.5,31)
    Pm_mu_sigmaGW = np.load('data/Pm_mu_sigmaGW.npy')
    w = Pm_mu_sigmaGW.ravel()
    w/=np.sum(w)
    
    mtov0 = np.linspace(1.85,2.4,100)
    r = np.random.normal(12.45,0.65,300)
    pmtov0_r = np.array([p_MTOV_R14(mtov0,r[i]) for i in range(len(r))]).T
    pmtov0 = np.mean(pmtov0_r,axis=1)
    cmtov0 = cumtrapz(pmtov0,mtov0,initial=0.)
    cmtov0/=cmtov0[-1]
    mtov = np.interp(np.random.uniform(0.,1.,size=Pm_mu_sigmaGW.T.shape),cmtov0,mtov0)
    
    m1 = np.linspace(1.,2.4,100)
    pm1s = np.zeros([len(mu)*len(sigma),len(m1)])
    
    k = 0
    for i in range(len(sigma)):
            for j in range(len(mu)):
                pm1s[k] = pm1_gauss(m1,mu[j],sigma[i],Mmin=1.,Mmax=mtov[j,i])
                k = k+1
    
    pm1_05,pm1_25,pm1_50,pm1_75,pm1_95 = np.zeros([5,len(m1)])
    pm1_05_prior,pm1_25_prior,pm1_50_prior,pm1_75_prior,pm1_95_prior = np.zeros([5,len(m1)])
    for i in range(len(m1)):
        s = np.argsort(pm1s[:,i])
        c = np.cumsum(w[s])
        c/=c[-1]
        cprior = np.linspace(0.,1.,len(c))
        pm1_05[i],pm1_25[i],pm1_50[i],pm1_75[i],pm1_95[i] = np.interp([0.05,0.25,0.50,0.75,0.95],c,pm1s[s,i])
        pm1_05_prior[i],pm1_25_prior[i],pm1_50_prior[i],pm1_75_prior[i],pm1_95_prior[i] = np.interp([0.05,0.25,0.50,0.75,0.95],cprior,pm1s[s,i])
        
       
    plt.fill_between(m1,pm1_05,pm1_95,edgecolor='blue',facecolor='#9999FF',alpha=0.3,zorder=13)
    plt.fill_between(m1,pm1_25,pm1_75,edgecolor='blue',facecolor='#9999FF',alpha=0.5,zorder=13)
    plt.plot(m1,pm1_50,ls='-',color='blue',label='Posterior',zorder=13)
    
    plt.fill_between(m1,pm1_05_prior,pm1_95_prior,edgecolor='grey',facecolor='None',alpha=1,zorder=10,ls='--',lw=1)
    plt.fill_between(m1,pm1_25_prior,pm1_75_prior,edgecolor='grey',facecolor='None',alpha=1,zorder=10,ls='--',lw=1)
    plt.plot(m1,pm1_50_prior*0.,ls='--',color='grey',label='Prior',zorder=10,lw=1)
    
    plt.ylim(0.,3.5)
    plt.xlim(1.,2.4)
    
    plt.xlabel(r'$m\,[\mathrm{M_\odot}]$')
    
    plt.grid(color='#EDEDED',zorder=-100)
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.title('Gaussian model')
    
    plt.gca().set_axisbelow(True)
    
    plt.xticks([1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4])
    
    plt.legend(frameon=False)
    
    # plt.savefig('../figures/dP_dm1_projected_gaussian.pdf')
    
    # plt.show()
    
    # -------------------------------- combined -------------------------------------------------------
    
    plt.sca(ax)
    
    ax.fill_between(m1,pm1_05,pm1_95,edgecolor='blue',facecolor='#9999FF',alpha=0.2,zorder=13)
    ax.fill_between(m1,pm1_25,pm1_75,edgecolor='blue',facecolor='#9999FF',alpha=0.4,zorder=13)
    ax.plot(m1,pm1_50,ls='-',color='blue',label='Gaussian',zorder=13)
    
    plt.semilogy()
    
    plt.ylim(3e-2,20.)
    plt.xlim(1.,2.4)
    
    plt.xlabel(r'$m\,[\mathrm{M_\odot}]$')
    # plt.ylabel(r'$P(M_1)\,[\mathrm{M_\odot}^{-1}]$')
    
    plt.grid(color='#EDEDED',zorder=-100)
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.title('Comparison')
    
    plt.gca().set_axisbelow(True)
    
    plt.xticks([1.,1.2,1.4,1.6,1.8,2.0,2.2,2.4])
    
    plt.legend(frameon=False)
    
    plt.savefig('../figures/dP_dm1_projected.pdf')
    
    plt.show()
    
    
    
