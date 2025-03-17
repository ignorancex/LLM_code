import numpy as np
from scipy.integrate import cumtrapz

def fjGW_posterior(J_i,P_miss_i,prior='uniform'):
    
    
    f = np.linspace(0.,1.,1000)
    Pji = f.reshape([1000,1])*(1.-P_miss_i)
    likelihoods = J_i*Pji + (1.-J_i)*(1.-Pji)
    
    Pf = np.prod(likelihoods,axis=1)
    Pf/=np.trapz(Pf,f)
    return f,Pf
    
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    plt.rcParams['font.family']='Liberation Serif'
    plt.rcParams['figure.figsize']=4,3.5
    plt.rcParams['figure.autolayout']=True
    plt.rcParams['mathtext.fontset']='dejavuserif'

    cmap = mpl.cm.magma
        
    # test computation for GW170817 and GW190425
    J_i = np.array([1.,0.])
    P_miss_i = np.array([0.,0.94])
    
    
    # mock series of observations
    N = 1000
    fjtrue = 0.5
    ji = np.zeros(N)
    ji[N//2:]=1.
    np.random.shuffle(ji)
    J = np.zeros_like(ji)
    
    print(np.sum(ji)/len(ji))
    fjtrue = np.sum(ji)/len(ji)
    
    # extract distances and inclinations
    cdf_dL = np.linspace(0.,1.,1000)
    dL0 = cdf_dL**0.333
    dLi = np.interp(np.random.uniform(0.,1.,N*100),cdf_dL,dL0)
    costhvi = np.random.uniform(0.,1.,N*100)
    
    dLmax = np.sqrt(0.125*(1.+6*costhvi**2+costhvi**4))
    
    GWdet = dLi<=dLmax
    
    dLe = np.percentile(dLi[GWdet],25.)
    
    dLi = dLi[GWdet][0:N]
    thvi = np.arccos(costhvi[GWdet][0:N])
    
    # compute limiting viewing angle for GRB det
    thv0 = np.linspace(0.,np.pi/2.,1000)
    pthv = (1.+6*np.cos(thv0)**2+np.cos(thv0)**4)**1.5*np.sin(thv0)
    pthv/=np.trapz(pthv,thv0)
        
    thvlim = np.minimum(np.pi/2.,(dLi/dLe)**-2.)
    

    Pmissi = np.zeros(N)
    for i in range(len(ji)):
        if ji[i]==1. and thvi[i]<=thvlim[i]:
            J[i] = 1.
            Pmissi[i] = np.trapz(pthv*(thv0>=thvlim[i]),thv0)
        else:
            J[i] = 0.
            Pmissi[i] = np.trapz(pthv*(thv0>=thvlim[i]),thv0)
        
    # compute posterior after i events
    f,Pf = fjGW_posterior(J[:1],Pmissi[:1])
    
    Pfi = np.zeros([len(Pmissi),len(f)])
    
    for i in range(len(Pmissi)):
        f,Pfi[i,:] = fjGW_posterior(J[:i],Pmissi[:i])
    
    # plot
    plt.subplot(111)
        
    n = np.arange(N)
    f05 = np.zeros(N)
    f25 = np.zeros(N)
    f50 = np.zeros(N)
    f75 = np.zeros(N)
    f95 = np.zeros(N)
    for i in n:
        c = cumtrapz(Pfi[i,:],f,initial=0.)
        c/=c[-1]
        f05[i],f25[i],f50[i],f75[i],f95[i] = np.interp(np.array([0.05,0.25,0.50,0.75,0.95]),c,f)
        
    
    plt.fill_between(n,f05,f95,edgecolor='r',facecolor='r',alpha=0.3)
    plt.fill_between(n,f25,f75,edgecolor='r',facecolor='r',alpha=0.3)
    plt.plot(n,f50,'-r')
    
    plt.ylabel(r'$f_\mathrm{j,GW}$')
    plt.xlabel(r'$N_\mathrm{events}$')
    
    plt.xscale('log')
    
    plt.xlim(1,N)
    plt.ylim(0.,1.)

    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.axhline(y=fjtrue,ls='--',c='k')
    
    plt.savefig('../figures/multi_event_fjGW_inference.pdf',bbox_inches='tight')
    
    plt.show()
