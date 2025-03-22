import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
from fjet.fitting_formulae import Mdisk_KF20 as MD_K
from fjet.fitting_formulae import Mdisk_B21 as MD_B
from fjet.fitting_formulae import Mej_KF20 as Mej_K

m1_17,m2_17 = np.load('data/GW170817_m1m2_posterior_samples_HS.npy')

def p_MTOV(m):
    # larger than PSRJ0740+6620 but smaller than GW170817
    mpsr = np.random.normal(2.08,0.07,10000)
    m1,m2 = m1_17,m2_17
    m17 = m1+m2
    cpsr = np.linspace(0.,1.,len(mpsr))
    c17 = np.linspace(0.,1.,len(m17))
    
    return np.interp(m,np.sort(mpsr),cpsr)*(1.-np.interp(m,np.sort(m17)/1.2,c17))

def p_R14(r):
    # from Miller et al. 2021 (NICER)
    return np.exp(-0.5*((r-12.45)/0.65)**2)

def p_MTOV_R14(m,r,mu=-0.112,si=0.472,lognormal=False,xi_d=None):
    
    mpsr = np.random.normal(2.08,0.07,10000)
    m1 = m1_17
    m2 = m2_17
    c1_17 = 1.48*m1/r
    c2_17 = 1.48*m2/r
    m17 = m1+m2
    nu17 = m1*m2/(m1+m2)**2
    C17 = 1.48*m17/r
    mgw17 = 0.25*nu17*C17*m17
    md170 =MD_K(m2,c2_17)
    #md170 =MD_B(m1,m2,c1_17,c2_17) # use the Barbieri+21 M_disk formula instead
    if xi_d is None:
        if lognormal:
            md17 = np.maximum(0.,np.random.normal(loc=0.,scale=1e-4,size=len(md170)) + md170*(np.random.lognormal(mean=mu,sigma=si,size=len(md170))))
        else:
            md17 = np.maximum(0.,np.random.normal(loc=0.,scale=1e-4,size=len(md170)) + md170*(np.random.normal(loc=1.,scale=0.5,size=len(md170))))
    else:
        md17 = xi_d*md170
    #mej170 = Mej_K(m1,m2,c1_17,c2_17)
    #mej17 = np.maximum(0.,mej170*np.random.normal(loc=1.,scale=0.5,size=len(md170)))
    mrem_17 = m17-mgw17-md17 #-mej17
    cpsr = np.linspace(0.,1.,len(mpsr))
    c17 = np.linspace(0.,1.,len(m17))
    
    return np.interp(m,np.sort(mpsr),cpsr)*(1.-np.interp(m,np.sort(mrem_17)/1.2,c17))*np.exp(-0.5*((r-12.45)/0.65)**2)

if __name__=='__main__':
    
    plt.rcParams['figure.figsize']=4,3.5
    plt.rcParams['figure.autolayout']=True
    plt.rcParams['font.family']='Liberation Serif'
    plt.rcParams['mathtext.fontset']='dejavuserif'
    
    # plot our MTOV prior
    m = np.linspace(1.8,2.8,300)
    p = p_MTOV(m)
    p/=np.trapz(p,m)
    
    np.save('data/Our_MTOV_prior.npy',(m,p))
    
    #plt.plot(m,p,color='blue',ls='-',lw=3,label=r'$M_\mathrm{GW17}>1.2 M_\mathrm{TOV}$',zorder=100)
    
    r = np.linspace(10.,15.,60)
    
    p_mr = np.zeros([len(m),len(r)])
    
    for i in range(len(r)):
        p_mr[:,i] = p_MTOV_R14(m,r[i])
    
    pr = np.trapz(p_mr,m,axis=0)
    
    pm = np.trapz(p_mr,r,axis=1)
    pm/=np.trapz(pm,m)
    
    plt.plot(m,pm,color='blue',ls='-',lw=3,zorder=100,label=r'This work (marginalised)')
    
    # plot the Raaijmakers+21 MTOV posteriors
    mpp,ppp = np.loadtxt('data/Raaijmakers_MTOV_PPmodel.txt',comments='#',unpack=True)
    mcs,pcs = np.loadtxt('data/Raaijmakers_MTOV_CSmodel.txt',comments='#',unpack=True)
    
    plt.plot(mpp,gf(ppp,1.),lw=2,label='Raaijmakers+21, PP',color='orange')
    plt.plot(mcs,gf(pcs,1.),lw=2,label='Raaijmakers+21, CS',color='green')
    
    # plot the Legred+21 MTOV posterior
    ml,pl = np.loadtxt('data/Legred_MTOV.txt',comments='#',unpack=True)
    
    pl/=np.trapz(pl,ml)
    
    plt.plot(ml,pl,lw=2,label='Legred+21',color='red')
    
    # plot the Pang+21 MTOV posterior
    mp,pp = np.loadtxt('data/Pang_MTOV.txt',comments='#',unpack=True)
    
    pp/=np.trapz(pp,mp)
    
    plt.plot(mp,pp,lw=2,label='Pang+21',color='purple')
    
    
    # plot the PSRJ0740+6620 mass from Fonseca+21
    ppsr = np.exp(-0.5*((m-2.08)/0.07)**2)/np.sqrt(2*np.pi*0.07**2)
    
    plt.fill_between(m,np.zeros_like(m),ppsr,color='grey',alpha=0.3)
    plt.annotate(xytext=(1.85,6.9),xy=(2.09,4.47),text='PSRJ0740+6620',color='grey',ha='left',arrowprops={'arrowstyle':'->','color':'grey'})
    
    m1,m2 = np.load('data/GW170817_m1m2_posterior_samples_HS.npy')
    m17 = m1+m2
    r = np.random.normal(12.45,0.65,len(m17))
    nu17 = m1*m2/(m1+m2)**2
    C17 = 1.48*m17/r
    c1_17 = 1.48*m1/r
    c2_17 = 1.48*m2/r
    mgw17 = 0.25*nu17*C17*m17
    md170 =MD_K(m2,c2_17)
    #md170 =MD_B(m1,m2,c1_17,c2_17) # use the Barbieri+21 M_disk formula instead
    
    # lognormal Mdisc error
    mu = -0.112
    si = 0.472
    md17 = np.maximum(0.,np.random.normal(loc=0.,scale=1e-4,size=len(md170)) + md170*np.random.lognormal(mu,si,size=len(md170)))
    
    # normal Mdisc error
    md17 = np.maximum(0.,np.random.normal(loc=0.,scale=1e-4,size=len(md170)) + md170*np.random.normal(1.,0.5,size=len(md170)))
    
    mej170 = Mej_K(m1,m2,c1_17,c2_17)
    mej17 = np.maximum(0.,mej170*np.random.normal(loc=1.,scale=0.5,size=len(md170)))
    mrem_17 = m17-mgw17-md17-mej17
    kde = gaussian_kde(mrem_17)
    plt.fill_between(m,np.zeros_like(m),kde.pdf(m*1.2),color='pink',alpha=0.5)
    plt.annotate(xy=(2.15,0.5),xytext=(2.4,2),text=r'$M_\mathrm{rem,GW17}/1.2$',color='#FDA3B3',ha='left',arrowprops={'arrowstyle':'->','color':'#FDA3B3'})
    
    plt.xlim(1.8,2.8)
    plt.ylim(0.,8.)
    
    plt.xlabel(r'$M_\mathrm{TOV}\,[\mathrm{M_\odot}]$')
    plt.ylabel(r'$dP/dM\,[\mathrm{M_\odot}^{-1}]$')
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.legend(frameon=False,markerfirst=False,loc='upper right')
    
    plt.savefig('../figures/MTOV_prior_comparison.pdf')
    
    plt.show()
    
    r = np.linspace(10.,15.,60)
    
    p_mr = np.zeros([len(m),len(r)])
    
    for i in range(len(r)):
        p_mr[:,i] = p_MTOV_R14(m,r[i])
    
    pr = np.trapz(p_mr,m,axis=0)
    
    pm = np.trapz(p_mr,r,axis=1)
    
    plt.plot(r,pr/pr.max())
    plt.plot(r,p_R14(r))
    
    plt.show()
    
    
    plt.fill_between(m,np.zeros_like(m),ppsr,color='grey',alpha=0.3)
    plt.annotate(xy=(1.85,5.9),text='PSRJ0740+6620',color='grey',ha='left')
    
    
    pm/=np.trapz(pm,m)
    cm = cumtrapz(pm,m,initial=0.)
    print(np.interp([0.05,0.5,0.95],cm,m))
    
    pm0 = p_MTOV(m)
    pm0/=np.trapz(pm0,m)
    
    plt.plot(m,pm,label=r'$M_\mathrm{rem,GW17}>1.2 M_\mathrm{TOV}$')
    #plt.plot(m,pm0,label=r'$M_\mathrm{GW17}>1.2 M_\mathrm{TOV}$')
    
    r = np.random.normal(12.45,0.65,len(m17))
    nu17 = m1*m2/(m1+m2)**2
    C17 = 1.48*m17/r
    c1_17 = 1.48*m1/r
    c2_17 = 1.48*m2/r
    mgw17 = 0.25*nu17*C17*m17
    md170 =MD_K(m2,c2_17)
    #md170 =MD_B(m1,m2,c1_17,c2_17) # use the Barbieri+21 M_disk formula instead
    
    # lognormal
    mu = -0.112
    si = 0.472
    md17 = np.maximum(0.,np.random.normal(loc=0.,scale=1e-4,size=len(md170)) + md170*np.random.lognormal(mu,si,size=len(md170)))
    
    # normal
    md17 = np.maximum(0.,np.random.normal(loc=0.,scale=1e-4,size=len(md170)) + md170*np.random.normal(1.,0.5,size=len(md170)))
    
    mej170 = Mej_K(m1,m2,c1_17,c2_17)
    mej17 = np.maximum(0.,mej170*np.random.normal(loc=1.,scale=0.5,size=len(md170)))
    mrem_17 = m17-mgw17-md17-mej17
    
    kde = gaussian_kde(mrem_17)
    
    plt.fill_between(m,np.zeros_like(m),kde.pdf(m*1.2),color='pink',alpha=0.7)
    plt.annotate(xy=(2.27,0.5),xytext=(2.3,2),text=r'$M_\mathrm{rem,GW17}/1.2$',color='#FDA3B3',ha='left',arrowprops={'arrowstyle':'->','color':'#FDA3B3'})
    
    plt.xlabel(r'$M_\mathrm{TOV}\,[\mathrm{M_\odot}]$')
    plt.ylabel(r'$dP/dM\,[\mathrm{M_\odot}^{-1}]$')
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.xlim(1.8,2.65)
    plt.ylim(0.,7.)
    
    plt.legend(frameon=False,markerfirst=False,loc='upper right')
    
    
    
    plt.show()
        
