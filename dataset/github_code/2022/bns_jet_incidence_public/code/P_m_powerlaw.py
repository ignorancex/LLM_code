import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter
from MTOV_R14_priors import p_R14
from MTOV_R14_priors import p_MTOV
from fjet import fitting_formulae

recomp_EoS_prior_samples = False

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

# load fj posterior from R0/RBNS
fj0,cdf_fj0 = np.load('data/fj_from_R0_RBNS.npy')
fj = np.interp(np.random.uniform(0.,1.,100000),cdf_fj0,fj0)
fj0 = np.linspace(0.,1.,100)
pdf_fj0,b = np.histogram(fj,bins=fj0)
fj0 = 0.5*(fj0[1:]+fj0[:-1])

# MTOV and R14 samples
R140 = np.linspace(9.,16.,101)
MTOV0 = np.linspace(1.8,2.4,31)


# compute cumulatives
PR14 = p_R14(R140)
CR14 = cumtrapz(PR14,R140,initial=0.)
CR14/=CR14[-1]

if recomp_EoS_prior_samples:
    R14s = np.interp(np.random.uniform(0.,1.,10000),CR14,R140)
    MTOVs = np.zeros(10000)
    for i in range(10000): # it is quite inefficient to do this sampling here, as it involves this long cycle...would be better to pre-sample R14 and MTOV outside, and then pick
        PMTOV = p_MTOV_R14(MTOV0,R14s[i])
        CMTOV = cumtrapz(PMTOV,MTOV0,initial=0.)
        CMTOV/=CMTOV[-1]
        MTOVs[i] = np.interp(np.random.uniform(0.,1.,1),CMTOV,MTOV0)
    
    np.save('data/R14_samples.npy',R14s)
    np.save('data/MTOV_samples.npy',MTOVs)
else:
    R14s = np.load('data/R14_samples.npy')
    MTOVs = np.load('data/MTOV_samples.npy')


def pm1(m1,alpha,Mmin,Mmax=2.5):
    p = m1**alpha*np.heaviside(m1-Mmin,0.)*np.heaviside(Mmax-m1,0.)
    return p/np.trapz(p,m1)

def compute_Mrem_Mdisk(m1,m2,R14):
    """
    Function to compute the remnant mass and disk mass according to our simplified prescription
    """
    C2 = 1.48*m2/R14
    md = fitting_formulae.Mdisk_KF20(m2,C2)
    q = m2/m1
    M = m1+m2
        
    return M*(1-0.25*q/(1.+q)**2*1.48*M/R14)-md,md

    
def Pm(alpha,Mmin,Mdisk_min=1e-3,N=3000,NEoS=300,alpha_fj=0.42):
    """
    This is the posterior probability on the mass distribution parameters \alpha and Mmin,
    marginalized over the EoS parameters MTOV and R14.
    """
    
    m0 = np.linspace(Mmin,2.7,100)
    pm0 = pm1(m0,alpha,Mmin,2.7)
    cm0 = cumtrapz(pm0,m0,initial=0.)
    cm0/=cm0[-1]
    
    ma = np.interp(np.random.uniform(0.,1.,N),cm0,m0)
    mb = np.interp(np.random.uniform(0.,1.,N),cm0,m0)
    
    m1 = np.where(ma>=mb,ma,mb).reshape([N,1])
    m2 = np.where(ma>=mb,mb,ma).reshape([N,1])
    
    # sample MTOV and R14 priors 
    ii = np.random.randint(10000,size=NEoS)
    R14 = np.copy(R14s[ii]).reshape([1,NEoS])
    MTOV = np.copy(MTOVs[ii]).reshape([1,NEoS])
    
    # compute Mrem and Mdisk for each m1,m2 and for each R14
    Mrem,Mdisk = compute_Mrem_Mdisk(m1,m2,R14)
    xi_d = np.maximum(0.,np.random.normal(1.,0.5,NEoS))
    Mrem = Mrem + (1.-xi_d)*Mdisk
    Mdisk = xi_d*Mdisk
    
    
    # THETA_j = 1 if our jet-launching conditions are met, 0 otherwise
    THETA_j = (Mrem>=(1.2*MTOV)) & (Mdisk>=Mdisk_min) 
    
    
    # compute effective GW sensitive volume
    Veff = np.nan_to_num((m1*m2)**1.5/(m1+m2)**0.5)
    
    # compute fjGW for each R14 sample,  throwing away samples with m1 or m2 > MTOV
    fjGW = np.zeros(NEoS)
    fj = np.zeros(NEoS)
    for i in range(NEoS):
        self_cons = (m1[:,0]<=MTOV[0,i]) & (m2[:,0]<=MTOV[0,i])
        fjGW[i] = np.sum(THETA_j[self_cons,i]*Veff[self_cons,0],axis=0)/np.sum(Veff[self_cons,0],axis=0)
        fj[i] = np.mean(THETA_j[self_cons,i],axis=0)
    
    # compute the marginalised posterior
    
    return np.mean(fjGW),np.mean(fj**alpha_fj)

if __name__=='__main__':
    recompute = False
    
    alpha = np.linspace(-20.,10.,80)
    Mmin = np.linspace(1.,1.25,31)
    
    if recompute:
        Pm_a_MminGW = np.zeros([len(Mmin),len(alpha)])
        Pm_a_Mmin = np.zeros([len(Mmin),len(alpha)])
        
        print('\n')
        for i in range(len(alpha)):
            print('{0:.2f} percent completed...    '.format(100*i/len(alpha)),end='\r')
            for j in range(len(Mmin)):
                Pm_a_MminGW[j,i],Pm_a_Mmin[j,i]=Pm(alpha[i],Mmin[j])
        
        
        Pm_a_MminGW[~np.isfinite(Pm_a_MminGW)]=0
        
        np.save('data/Pm_a_MminGW.npy',Pm_a_MminGW)
        np.save('data/Pm_a_Mmin.npy',Pm_a_Mmin)
    else:
        Pm_a_MminGW = np.nan_to_num(np.load('data/Pm_a_MminGW.npy'))
        Pm_a_Mmin = np.nan_to_num(np.load('data/Pm_a_Mmin.npy'))
            
    cs = plt.contourf(alpha,Mmin,gaussian_filter(Pm_a_MminGW,1.2),levels=np.linspace(0.,0.7,8))
    for c in cs.collections:
        c.set_rasterized(True)

    plt.colorbar(label=r'$P(\alpha,M_\mathrm{min}\,|\,d)$')
    
    cmap = plt.cm.hot_r
    levs = np.linspace(0.,0.7,8)
    cs2 = plt.contour(alpha,Mmin,gaussian_filter(Pm_a_Mmin,1.8),levels=levs,colors=cmap(levs),linewidths=0.5)
    cpos = np.array([(-14.2,1.06),(-12.,1.09),(-9.9,1.1),(-8.65,1.11),(-7.1,1.12),(-5.5,1.13),(-3.,1.14)])
    for i in range(len(cpos)):
        plt.annotate(xy=cpos[i],text='{0:.1f}'.format(levs[i+1]),color=cmap(levs[i+1]),fontsize=8.,va='center',ha='right')


    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$M_\mathrm{min}\,[\mathrm{M_\odot}]$')
    
    
    plt.savefig('../figures/Pm_a_Mmin.pdf')
    
    plt.show()
    
    # sample distribution and plot constraint in mass distribution space
    w = Pm_a_MminGW.ravel()
    w/=np.sum(w)
    
    mtov0 = np.linspace(1.85,2.4,100)
    pmtov0 = p_MTOV(mtov0)
    cmtov0 = cumtrapz(pmtov0,mtov0,initial=0.)
    cmtov0/=cmtov0[-1]
    mtov = np.interp(np.random.uniform(0.,1.,size=Pm_a_MminGW.T.shape),cmtov0,mtov0)
    
    m1 = np.linspace(1.,2.4,100)
    pm1s = np.zeros([len(alpha)*len(Mmin),len(m1)])
    
    k = 0
    for i in range(len(alpha)):
            for j in range(len(Mmin)):
                pm1s[k] = pm1(m1,alpha[i],Mmin[j],Mmax=mtov[i,j])
                k = k+1
    
    pm1_05,pm1_25,pm1_50,pm1_75,pm1_95 = np.zeros([5,len(m1)])
    for i in range(len(m1)):
        s = np.argsort(pm1s[:,i])
        c = np.cumsum(w[s])
        c/=c[-1]
        pm1_05[i],pm1_25[i],pm1_50[i],pm1_75[i],pm1_95[i] = np.interp([0.05,0.25,0.50,0.75,0.95],c,pm1s[s,i])
        
    plt.fill_between(m1,pm1_05,pm1_95,edgecolor='None',facecolor='blue',alpha=0.2)
    plt.fill_between(m1,pm1_25,pm1_75,edgecolor='None',facecolor='blue',alpha=0.2)
    plt.plot(m1,pm1_50,'-b')
    
    plt.semilogy()
    
    plt.ylim(1e-2,1e1)
    
    plt.show()

    
                
    
    
    
    

