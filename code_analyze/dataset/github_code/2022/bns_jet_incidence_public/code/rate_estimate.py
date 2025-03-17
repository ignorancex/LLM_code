import numpy as np
import matplotlib.pyplot as plt
import pandas                                                           
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

plt.rcParams['font.family']='Liberation Serif'
plt.rcParams['figure.figsize']=4,3.5
plt.rcParams['figure.autolayout']=True
plt.rcParams['mathtext.fontset']='dejavuserif'

def credible(x,dP_dx,level=0.9):
    cum = cumtrapz(dP_dx,x,initial=0.)
    cum = cum/cum[-1]
    H = np.linspace(0.,dP_dx.max(),1000)[::-1]
    for h in H[1:]:
        r = dP_dx>=h
        I = cum[r][-1]-cum[r][0]
        if I>=level:
            return x[r][0],x[r][-1]
    
def sample(x,dP_dx,N):
    cum = cumtrapz(dP_dx,x,initial=0.)
    cum = cum/cum[-1]
    return np.interp(np.random.uniform(0.,1.,N),cum,x)

def dN_dE_Comp(E,Ep,a):
    return E**a*np.exp(-(a+2)*E/Ep)

# read GBM short GRB data and create logN-logP64
d = pandas.read_csv('/home/omsharan/data/GRB_tables/GBM_flux.txt')                                     
short = d['t90']<= 2.                                                  
p64 = d['flux_64'].loc[short]
p64_err = d['flux_64_error'].loc[short]
c = np.arange(len(p64))
                          
p64_sampled = np.array(p64).reshape([1,len(p64)])*(np.random.lognormal(mean=np.zeros_like(p64_err),sigma=np.log((1.+(p64_err/p64)**2)**0.5),size=[1000,len(p64)]))
s = np.argsort(p64)
c_sampled = np.zeros([1000,len(p64)])
for i in range(1000):
    c_sampled[i] = np.interp(p64.iloc[s],np.sort(p64_sampled[i]),np.random.poisson(c))

c_05 = np.percentile(c_sampled,5.,axis=0)
c_95 = np.percentile(c_sampled,95.,axis=0)

# create "theoretical", intrinsic logN-logP64
p = np.logspace(-1,3,100)
N10 = np.interp(10.,p64.iloc[s],c[::-1])                               
N = N10*(p/10.)**-1.5

# ansatz form of the GBM detection efficiency as a function of the peak photon flux
deff = lambda p,a,p12: 0.5*(1. + np.tanh(np.log(p/p12)*a))

def Nobs(p,a,p12): 
    dN_dp = 1.5*N10*10**1.5*p**-2.5    
    nobs = cumtrapz(deff(p,a,p12)*dN_dp,p,initial=0.) 
    return nobs[-1]-nobs 

# fit statistic
def fitstat(x):
    return np.sum((Nobs(p64.iloc[s][c[::-1]>0],x[0],x[1])-c[::-1][c[::-1]>0])**2)

sol = minimize(fitstat,[2.2,5.4],tol=1e-2)
print(sol)

# plot resulting fit
plt.loglog()
plt.fill_between(p64.iloc[s[::-1]],c_05,c_95,edgecolor='None',facecolor='#E5E5E5')
plt.plot(p64.iloc[s[::-1]],c,'-b')
plt.plot(p,Nobs(p,sol.x[0],sol.x[1]),'--r')
plt.xlim(1e0,1e2)
plt.ylim(1e1,1e3)
plt.xlabel(r'$p_{64}$ [cm$^{-2}$ s$^{-1}$]')
plt.ylabel(r'$N_\mathrm{obs}(>p_{64})$')

plt.tick_params(which='both',direction='in',top=True,right=True)

plt.savefig('../figures/logNlogS.pdf')

plt.show()

plt.plot(p64.iloc[s],deff(p64.iloc[s],sol.x[0],sol.x[1]))
plt.show()

                    
# compute effective Vmax of GRB170817A
z = np.linspace(0.00001,0.05,10000)                                     
dL = cosmo.luminosity_distance(z).to('Mpc').value
k = np.zeros_like(dL)
E = np.logspace(1,3,1000) # 10-1000 keV GBM band
Ep = 229.
a = 0.85
dL0 = 41.
print(dL0)
z0 = np.interp(dL0,dL,z)
k0 = np.trapz(dN_dE_Comp(E,Ep,a),E)
for i in range(len(z)):
    k[i] = np.trapz(dN_dE_Comp(E*(1.+z[i])/(1.+z0),Ep,a),E)/k0

plt.plot(z,k)
plt.show()

p0 = 3.7 # peak photon flux of GRB 170817A
pz = p0*(dL/dL0)**-2*k # assume dL^-2 scaling for the photon flux + k-correction

V = np.trapz(4*np.pi*u.sr*cosmo.differential_comoving_volume(z)*deff(pz,sol.x[0],sol.x[1])/(1.+z),z) + cosmo.comoving_volume(z[0])

# compute rate estimate
T = 13. #yrs of Fermi operation at GRB170817's time
dc = 0.59 # duty cycle x sky fraction

R0 = 1./(V.to('Gpc3').value*T*dc)

# estimate errors adopting a Poisson likelihood and the Jeffreys prior
r0 = np.logspace(-4,2.5,10000)*R0
dP_dr0 = (r0/R0)**0.5*np.exp(-(r0/R0))
dP_dr0/=np.trapz(dP_dr0,r0)


# compute R_0SJ posterior
dPSJ_dr0 = r0**-1
for i in range(len(r0)):
    dPSJ_dr0[i] *= np.trapz(dP_dr0*(r0<r0[i]),r0)

dPSJ_dr0/=np.trapz(dPSJ_dr0,r0)

# compute BNS rate posterior distribution
"""
R0BNS = 370./9
dP_dr0BNS = (r0/R0BNS)**9*np.exp(-(r0/R0BNS))
dP_dr0BNS/=np.trapz(dP_dr0BNS,r0)
"""

r,pr = np.loadtxt('data/Grunthal2021_RMW.txt',unpack=True)

r*=32./38.

cr = cumtrapz(pr,r,initial=0.)
cr/=cr[-1]
md = np.interp(0.5,cr,r)
print('Grunthal median: {0:.2f}, mult. by 11.6: {1:.2f}'.format(md,11.6*md))

plt.plot(r,pr)

plt.axvline(x=md)
plt.show()

ri = np.interp(np.random.uniform(0.,1.,int(1e5)),cr,r*11.6)*np.maximum(0.01,np.random.normal(1.,0.3,int(1e5)))

dP_dr0BNS = gaussian_kde(ri).pdf(r0)

plt.plot(r0,dP_dr0/dP_dr0.max(),'-',color='grey',lw=2,label=r'GRB170817A-like, $R_\mathrm{0,17A}$')
plt.plot(r0,dPSJ_dr0/dPSJ_dr0.max(),'-r',lw=2,label='SGRB jets, $R_\mathrm{0,SJ}$')
plt.plot(r0,dP_dr0BNS/dP_dr0BNS.max(),'-b',label='BNS, $R_\mathrm{0,BNS}$ (Grunthal+2021)')
plt.xlabel(r'$R_0\,\mathrm{[Gpc^{-3}yr^{-1}]}$')
plt.ylabel('Probability density (normalized to peak)')
plt.semilogx()
plt.xlim([1e-1,1e5])
plt.ylim([0.,1.5])
plt.tick_params(which='both',direction='in',top=True,right=True)
plt.legend(frameon=False,loc='upper left')

plt.savefig('../figures/R0_R0BNS_comparison.pdf')

plt.show()

R0_low_90,R0_high_90 = credible(r0,dP_dr0,level=0.9)
R0_low_68,R0_high_68 = credible(r0,dP_dr0,level=0.68)
#R0_best = r0[np.argmax(dP_dr0)] # MAP
cr = cumtrapz(dP_dr0,r0,initial=0.)
cr/=cr[-1]
R0_best = np.interp(0.5,cr,r0)

print(' V_eff = {6:.3g} Mpc3\n (V_eff T)^-1 = {5:.2f} Gpc-3 yr-1\n R0 = {0:.0f} -{1:.0f}/+{2:.0f} (68%), -{3:.0f}/+{4:.0f} (90%) Gpc-3 yr-1'.format(R0_best,R0_best-R0_low_68,R0_high_68-R0_best,R0_best-R0_low_90,R0_high_90-R0_best,R0,V.to('Mpc3').value*dc,V.to('Mpc3').value*dc))

R0BNS_low_90,R0BNS_high_90 = credible(r0,dP_dr0BNS,level=0.9)
#R0BNS_best = r0[np.argmax(dP_dr0BNS)] # MAP
cr = cumtrapz(dP_dr0BNS,r0,initial=0.)
cr/=cr[-1]
R0BNS_best = np.interp(0.5,cr,r0)


print('R0BNS = {0:.0f} -{1:.0f}/+{2:.0f} (90%) Gpc-3 yr-1'.format(R0BNS_best,R0BNS_best-R0BNS_low_90,R0BNS_high_90-R0BNS_best))

# fjet from ratio

N = 3000000
R0_samples = sample(r0,dP_dr0,N)
R0BNS_samples = sample(r0,dP_dr0BNS,N)

fjet_samples = R0_samples/R0BNS_samples
fjet_samples = fjet_samples[fjet_samples<=1]

c = np.arange(len(fjet_samples))
c = c/c[-1]

np.save('data/R0_GW170817.npy',R0_samples)
# np.save('data/fj_from_R0_RBNS.npy',(np.sort(fjet_samples),c))

plt.plot(np.sort(fjet_samples),c,lw=3,ls='-',c='r')

fj0 = np.logspace(-3,0,1000)
CU = fj0**2
CJ = 2/np.pi*(np.arcsin(fj0**0.5)-(fj0-fj0**2)**0.5)

plt.plot(fj0,CU,lw=3,ls='-',c='b')
plt.plot(fj0,CJ,lw=3,ls='-',c='g')

plt.loglog()
plt.xlim(1e-3,1)
plt.ylim(1e-3,1)

plt.xlabel(r'$f_\mathrm{jet}$')
plt.ylabel(r'cumulative probability')
plt.semilogx()
plt.show()

print(np.percentile(fjet_samples,[0.7,10.]))
