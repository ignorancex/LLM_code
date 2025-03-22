import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter
from MTOV_R14_priors import p_R14
from MTOV_R14_priors import p_MTOV_R14
from fjet import fitting_formulae
import greedy_contours

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

recompute = True

def credible_interval(p,x,level=0.9,res=1000):
    c = cumtrapz(p,x,initial=0.)
    c/=c[-1]
    pmax = np.max(p)
    xmax = x[np.argmax(p)]
    H = np.linspace(0.,pmax,res)
    for h in H:
        xa,xb = x[p>h][[0,-1]]
        I = np.interp(xb,x,c)-np.interp(xa,x,c)
        if I<=level:
            break
    return xmax,xa,xb

def compute_Mrem_Mdisk(m1,m2,R14):
    """
    Function to compute the remnant mass and disk mass according to our simplified prescription
    """
    C2 = 1.48*m2/R14
    md = fitting_formulae.Mdisk_KF20(m2,C2)
    q = m2/m1
    M = m1+m2
        
    return M*(1-0.25*q/(1.+q)**2*1.48*M/R14)-md,md
    
    
def Pm_EoS(MTOV,R14,xi_d,mu,sigma,m1,m2,Mdisk_min=1e-3):
    """
    This is the posterior probability on the mass distribution parameters mu_m and sigma_m,
    and EoS parameters MTOV and R14.
    """
    
    # compute Mrem and Mdisk for each m1,m2 and for each R14
    Mrem,Mdisk = compute_Mrem_Mdisk(m1,m2,R14)
    #xi_d = np.maximum(0.,np.random.normal(1.,0.5,len(m1)))
    Mrem = Mrem + (1.-xi_d)*Mdisk
    Mdisk = xi_d*Mdisk
    
    # THETA_j = 1 if our jet-launching conditions are met, 0 otherwise
    THETA_j = (Mrem>=(1.2*MTOV)) & (Mdisk>=Mdisk_min) 
    
    
    # compute effective GW sensitive volume
    Veff = np.nan_to_num((m1*m2)**1.5/(m1+m2)**0.5)
    
    # compute fjGW for each R14 sample,  throwing away samples with m1 or m2 > MTOV
    self_cons = (m1<=MTOV) & (m2<=MTOV)
    fjGW = np.sum(THETA_j[self_cons]*Veff[self_cons],axis=0)/np.sum(Veff[self_cons],axis=0)
    
    # compute the marginalised posterior 
    return fjGW

if __name__=='__main__':
    mu = 1.33
    sigma = 0.09
    N = 1000
    Mmin = 1.
    
    mtov = np.linspace(1.8,2.4,100)
    r14 = np.linspace(10.,15.,101)
    xi_d = np.maximum(0.,np.random.normal(1.,0.5,10))
    
    if recompute:
        PEOS = np.zeros([len(mtov),len(r14),len(xi_d)])
        LEOS = np.zeros([len(mtov),len(r14),len(xi_d)])
        
        # generate m1,m2 samples
        ma = np.random.normal(loc=mu,scale=sigma,size=N)
        mb = np.random.normal(loc=mu,scale=sigma,size=N)
        
        while len(ma[ma<Mmin])>0:
            ma[ma<Mmin] = np.random.normal(loc=mu,scale=sigma,size=len(ma[ma<Mmin]))
        while len(mb[mb<Mmin])>0:
            mb[mb<Mmin] = np.random.normal(loc=mu,scale=sigma,size=len(mb[mb<Mmin]))
        
        m1 = np.where(ma>=mb,ma,mb)
        m2 = np.where(ma>=mb,mb,ma)
        
        for i in range(len(mtov)):
            print(i)
            for j in range(len(r14)):
                for k in range(len(xi_d)):
                    LEOS[i,j,k]=Pm_EoS(mtov[i],r14[j],xi_d[k],mu,sigma,m1,m2)
                    PEOS[i,j,k]=LEOS[i,j,k]*p_MTOV_R14(mtov[i],r14[j],xi_d=xi_d[k]) #*p_R14(r14[j])
        
        LEOS = np.mean(LEOS,axis=-1)
        PEOS = np.mean(PEOS,axis=-1)
        np.save('data/PEoSgauss.npy',PEOS)
        np.save('data/LEoSgauss.npy',LEOS)
    else:
        PEOS = np.load('data/PEoSgauss.npy')
        LEOS = np.load('data/LEoSgauss.npy')

    
    plt.subplot(223)
    
    pmtovr = np.array([p_MTOV_R14(mtov,r14[i]) for i in range(len(r14))]).T
    C = greedy_contours.mesh_to_mesh(pmtovr)
    plt.contour(r14,mtov,C,levels=(0.68,0.95,0.9973),colors='k',linestyles='--')
    
    C = greedy_contours.mesh_to_mesh(LEOS)
    plt.contour(r14,mtov,C,levels=(0.68,0.95,0.9973),colors='grey',linestyles='-')
    
        
    C = greedy_contours.mesh_to_mesh(PEOS,smooth=1.)
    plt.contour(r14,mtov,C,levels=(0.68,0.95,0.9973),colors=['r','r','r'],linestyles='-')

    plt.annotate(xy=(12.9,2.03),text=r'$1\sigma$',color='r',ha='right',va='bottom',fontsize=8)
    plt.annotate(xy=(13.4,1.98),text=r'$2\sigma$',color='r',ha='right',va='bottom',fontsize=8)
    plt.annotate(xy=(13.9,1.93),text=r'$3\sigma$',color='r',ha='right',va='bottom',fontsize=8)
    
    plt.xticks([10.5,11.5,12.5,13.5,14.5])
    plt.yticks([1.9,2.0,2.1,2.2,2.3])

    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.xlabel(r'$R_\mathrm{1.4}\,[\mathrm{km}]$')
    plt.ylabel(r'$M_\mathrm{TOV}\,[\mathrm{M_\odot}]$')
    
    plt.xlim(r14.min(),r14.max())
    plt.ylim(mtov.min(),mtov.max())
    
    
    plt.subplot(221)
    
    Pr14 = np.trapz(PEOS,mtov,axis=0)
    Pr14/=np.trapz(Pr14,r14)
    Lr14 = np.trapz(LEOS,mtov,axis=0)
    Lr14/=np.trapz(Pr14,r14)
    
    r14best,r14low,r14high = credible_interval(Pr14,r14)
    print('R_1.4 = {0:.3g} -{1:.3g} +{2:.3g} km (90%)'.format(r14best,r14best-r14low,r14high-r14best))
  
  
    mtovg = np.copy(mtov.reshape([1,len(mtov)]))
    r14g = np.copy(r14.reshape([len(r14),1]))
    
    pmtovr = np.array([p_MTOV_R14(mtov,r14[i]) for i in range(len(r14))]) #*p_R14(r14g)
    p0_r14 = np.trapz(pmtovr,mtov,axis=1)
    p0_r14/=np.trapz(p0_r14,r14)
  
    plt.plot(r14,p0_r14,'--k')
    plt.plot(r14,Lr14,'-',color='grey',lw=3,alpha=0.5)
    plt.plot(r14,Pr14,'-r')
    
    plt.xlim(r14.min(),r14.max())
    plt.ylim(0.,Pr14.max()*1.1)
    
    plt.xticks([10.5,11.5,12.5,13.5,14.5])
    
    plt.tick_params(which='both',direction='in',top=False,right=False,left=False,labelbottom=False,labelleft=False)
    plt.grid()
    
    plt.subplot(224)
    
    Pmtov = np.trapz(PEOS,r14,axis=1)
    Pmtov/=np.trapz(Pmtov,mtov)
    Lmtov = np.trapz(LEOS,r14,axis=1)
    Lmtov/=np.trapz(Pmtov,mtov)
    
    mtovbest,mtovlow,mtovhigh = credible_interval(Pmtov,mtov)
    print('M_TOV = {0:.3g} -{1:.3g} +{2:.3g} Msun (90%)'.format(mtovbest,mtovbest-mtovlow,mtovhigh-mtovbest))
    
    r = np.random.normal(12.45,0.65,300)
    pmtov0_r = np.array([p_MTOV_R14(mtov,r[i]) for i in range(len(r))]).T
    pmtov0 = np.mean(pmtov0_r,axis=1)
    
    plt.plot(pmtov0/np.trapz(pmtov0,mtov),mtov,'--k')
    plt.plot(Lmtov,mtov,'-',lw=3,color='grey',alpha=0.5)
    plt.plot(Pmtov,mtov,'-r')
    
    plt.ylim(mtov.min(),mtov.max())
    plt.xlim(0.,Pmtov.max()*1.1)
    
    plt.yticks([1.9,2.0,2.1,2.2,2.3])
    plt.xticks(np.linspace(0.,Pmtov.max()*1.1,5))
    
    plt.tick_params(which='both',direction='in',top=False,right=False,bottom=False,labelbottom=False,labelleft=False)
    plt.grid()
    
    plt.subplot(222)
    plt.gca().set_axis_off()
    
    plt.annotate(xy=(0.5,0.9),text='Gaussian model',ha='center',va='center')
    plt.annotate(xy=(0.5,0.7),text=r'$\mu_\mathrm{m}=1.33\,\mathrm{M_\odot}$' + '\n' + r'$\sigma_\mathrm{m}=0.09\,\mathrm{M_\odot}$',ha='center',va='center')
    
    plt.plot([-1],[-1],'--k',label='prior')
    plt.plot([-1],[-1],'-',lw=3,color='grey',alpha=0.5,label='likelihood')
    plt.plot([-1],[-1],'-r',label='posterior')
    
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    
    plt.legend(loc='lower center',frameon=False)
    
    plt.savefig('../figures/P_MTOV_R14_galactic_gaussian.pdf')
    
    plt.show()
