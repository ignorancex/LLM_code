import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import fjet
from MTOV_R14_priors import p_MTOV_R14,p_R14
import greedy_contours

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

recompute = False
fj0 = 0.75
dfj0 = 0.05

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

def pfj(fj,mu=fj0,sig=dfj0):
    return np.exp(-0.5*((fj-mu)/sig)**2)

def pm1m2(m1,m2,alpha=-2.,Mmin=1.1,Mmax=2.2):
    return (m1*m2)**alpha*np.heaviside(m1-Mmin,0.)*np.heaviside(m2-Mmin,0.)*np.heaviside(m1-m2,0.)*np.heaviside(Mmax-m1,0.)
    


# sample Mmin and alpha priors
Mmin = np.random.uniform(1.1,1.3,300)
alpha = np.random.uniform(-9.,3.3,300)
xi_d = np.maximum(np.random.normal(1.,0.5,300),0.)

# create MTOV and R14 grid
mtov = np.linspace(1.85,2.6,70)
r14 = np.linspace(10,15.,71)


Pfj = np.zeros([len(Mmin),len(mtov),len(r14)])
PEOS = np.zeros([len(mtov),len(r14)])
LEOS = np.zeros([len(mtov),len(r14)])

if recompute:
    print('\n')
    for i in range(len(Mmin)):
        print('{0:.2f} percent completed...    '.format(100*i/len(Mmin)),end='\r')
        for j in range(len(mtov)):
            # create m1 and m2 grid
            m1g = np.linspace(Mmin[i],mtov[j],50).reshape([50,1])
            m2g = np.linspace(Mmin[i],mtov[j],51).reshape([1,51])
            mc = (m1g*m2g)**0.6/(m1g+m2g)**0.2
            pm1m2_ij = pm1m2(m1g,m2g,alpha[i],Mmin[i],mtov[j])

            for k in range(len(r14)):
                M = m1g+m2g
                q = m2g/m1g
                C1 = 1.48*m1g/r14[k]
                C2 = 1.48*m2g/r14[k]
                md = xi_d[i]*fjet.fitting_formulae.Mdisk_KF20(m2g,C2)
                #mej = fjet.fitting_formulae.Mej_KF20(m1g,m2g,C1,C2)
                mrem = M*(1.-0.25*q/(1.+q)**2*1.48*M/r14[k])-md #-mej
    
                thj = (mrem>=(1.2*mtov[j]))*(md>=1e-3)
                
                
                fj_ijk = np.trapz(np.trapz(pm1m2_ij*thj*mc**2.5,m2g[0],axis=1),m1g[:,0],axis=0)/np.trapz(np.trapz(pm1m2_ij*mc**2.5,m2g[0],axis=1),m1g[:,0],axis=0)
                Pfj[i,j,k] = pfj(fj_ijk)
    print('\n')
    LEOS = np.mean(Pfj,axis=0)
    
    for i in range(len(r14)):
        
        leosj = np.zeros([len(mtov),len(xi_d)])
        for j in range(len(xi_d)):
            prior = p_MTOV_R14(mtov,r14[i],xi_d=xi_d[j])
            leosj[:,j] = Pfj[j,:,i]*prior
        
        PEOS[:,i] = np.mean(leosj,axis=1)
    
    np.save('data/PEoSfjknown{0:.2f}.npy'.format(fj0),PEOS)
    np.save('data/LEoSfjknown{0:.2f}.npy'.format(fj0),LEOS)
else:
    PEOS = np.load('data/PEoSfjknown{0:.2f}.npy'.format(fj0))
    LEOS = np.load('data/LEoSfjknown{0:.2f}.npy'.format(fj0))

LEOS/=LEOS.max()

plt.subplot(223)

pmtovr = np.array([p_MTOV_R14(mtov,r14[i]) for i in range(len(r14))]).T

C = greedy_contours.mesh_to_mesh(pmtovr)
plt.contour(r14,mtov,C,levels=(0.68,0.95,0.9973),colors='k',linestyles='--')

"""
C = greedy_contours.mesh_to_mesh(LEOS)
plt.contour(r14,mtov,C,levels=(0.68,0.95,0.9973),colors='grey',linestyles='-')
"""
    
C = greedy_contours.mesh_to_mesh(PEOS,smooth=1.)
plt.contour(r14,mtov,C,levels=(0.68,0.95,0.9973),colors=['r','r','r'],linestyles='-')

"""
plt.annotate(xy=(12.9,2.03),text=r'$1\sigma$',color='r',ha='right',va='bottom',fontsize=8)
plt.annotate(xy=(13.4,1.98),text=r'$2\sigma$',color='r',ha='right',va='bottom',fontsize=8)
plt.annotate(xy=(13.9,1.93),text=r'$3\sigma$',color='r',ha='right',va='bottom',fontsize=8)
"""

plt.xticks([10.5,11.5,12.5,13.5,14.5])
plt.yticks([1.9,2.0,2.1,2.2,2.3])

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.xlabel(r'$R_\mathrm{1.4}\,[\mathrm{km}]$')
plt.ylabel(r'$M_\mathrm{TOV}\,[\mathrm{M_\odot}]$')

plt.xlim(r14.min(),r14.max())
plt.ylim(1.85,2.4)


plt.subplot(221)

mtovg = np.copy(mtov.reshape([1,len(mtov)]))
r14g = np.copy(r14.reshape([len(r14),1]))

#pmtovr = np.array([p_MTOV_R14(mtov,r14[i]) for i in range(len(r14))]) #*p_R14(r14g)
p0_r14 = np.trapz(pmtovr,mtov,axis=0)
p0_r14/=np.trapz(p0_r14,r14)

Pr14 = np.trapz(PEOS,mtov,axis=0)
Pr14/=np.trapz(Pr14,r14)
Lr14 = np.trapz(LEOS,mtov,axis=0)
Lr14/=np.trapz(Lr14,r14)

r14best,r14low,r14high = credible_interval(Pr14,r14)
print('R_1.4 = {0:.3g} -{1:.3g} +{2:.3g} km (90%)'.format(r14best,r14best-r14low,r14high-r14best))

plt.plot(r14,p0_r14,'--k')
#plt.plot(r14,Lr14,'-',color='grey',lw=3,alpha=0.5)
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


pmtov0 = np.trapz(pmtovr,r14,axis=1)

plt.plot(pmtov0/np.trapz(pmtov0,mtov),mtov,'--k')

#plt.plot(Lmtov,mtov,'-',lw=3,color='grey',alpha=0.5)
plt.plot(Pmtov,mtov,'-r')

plt.ylim(1.85,2.4)
plt.xlim(0.,Pmtov.max()*1.1)

plt.yticks([1.9,2.0,2.1,2.2,2.3])
plt.xticks(np.linspace(0.,Pmtov.max()*1.1,5))

plt.tick_params(which='both',direction='in',top=False,right=False,bottom=False,labelbottom=False,labelleft=False)
plt.grid()

plt.subplot(222)
plt.gca().set_axis_off()

plt.annotate(xy=(0.5,0.95),text='Power law model',ha='center',va='center')
plt.annotate(xy=(0.5,0.87),text=r'$-9<\alpha<3.3$' + '\n' + r'$1.1<M_\mathrm{min}/\mathrm{M_\odot}<1.3$',ha='center',va='top')
#plt.annotate(xy=(0.5,0.55),text=r'$f_\mathrm{j,GW}=%.2f\pm %.2f$'%(fj0,dfj0),ha='center',va='top')

plt.plot([-1],[-1],'--k',label='current constraint')
#plt.plot([-1],[-1],'-',lw=3,color='grey',alpha=0.5,label='likelihood')
plt.plot([-1],[-1],'-r',label=r'$f_\mathrm{j,GW}=%.2f\pm %.2f$'%(fj0,dfj0))

plt.xlim(0.,1.)
plt.ylim(0.,1.)

plt.legend(loc=(-0.1,0.12),frameon=False)

plt.savefig('../figures/P_MTOV_R14_fjknown{0:.2f}.pdf'.format(fj0))

plt.show()
