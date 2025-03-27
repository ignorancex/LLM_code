import numpy as np
import scipy.integrate as itg
import time

# fundamental constants
e=1.6e-19
c=3e8

# masses (all in MeV)
me=.51099895000
mmu= 105.6583755
mtau=1776.86
mp=938.272088
mn=939.565420
alphaEM = 1/137.035999084

# transforming units
erg2MeV=6.2415e5
invs2MeV=1/1.5e24*1e3
km2invMeV=5.1e15
MeV2invfm=5.1e-3
invcmtoMeV= 1/5.1e10

Qbound1=5.68e52 #in erg/s
Qbound2=1.2e53
def getQbound(nSim):
    if nSim==1:
        Qbound = Qbound1
    elif nSim==2:
        Qbound = Qbound2
    else:
        print(f"Warning: Neutrino luminosity for simulation {nSim} not hard-coded. Defaulting to the luminosity for simulation 1.")
        Qbound = Qbound1        
    Qbound*=erg2MeV*invs2MeV #MeV
    return Qbound

### data loader
# enter path to the simulation data of https://arxiv.org/abs/2005.07141 here
start = "data/"
file1 = start+"hydro-SFHo-s18.6-MUONS-T=0.99948092.txt"
file2 = start+"hydro-SFHo-s18.80-MUONS-T=1.0001996.txt"
file3 = start+"hydro-SFHo-s20.0-MUONS-T=1.0010874.txt"
file4= start+"hydro-LS220-T=1.0001464.txt"
files=[file1, file2, file3, file4]

tab1=np.loadtxt(file1, skiprows=5, dtype=np.float64)
tab2=np.loadtxt(file2, skiprows=5, dtype=np.float64)
tab3=np.loadtxt(file3, skiprows=5, dtype=np.float64)
tab4=np.loadtxt(file4, skiprows=5, dtype=np.float64)
tabs=[tab1, tab2, tab3, tab4]
simNames=["SFHo-18.6", "SFHo-18.80", "SFHo-20.0", "LS220"]

def unpack(n):
    R=1e-5*tabs[n][:,0]*km2invMeV #in MeV^-1
    T=tabs[n][:,4] #in MeV
    mu_e=tabs[n][:,9] #in MeV
    mu_mu=tabs[n][:,12] #in MeV
    mu_p=tabs[n][:,11]+mp
    mu_n=tabs[n][:,10]+mn
    mu_nue=tabs[n][:,8]
    mu_numu=tabs[n][:,13]
    return R, T, mu_e, mu_mu, mu_p, mu_n, mu_nue, mu_numu

### Decay widths
def gamZp_i(mZp, mi, gi):
    if mZp < 2*mi:
        return 0.
    gamContr=mZp/(12*np.pi) * gi**2 *(1+2*mi**2/mZp**2) *(1-4*mi**2/mZp**2)**.5
    return gamContr
def GamZpMuTau(mZp, mL, mChi, gL, gChi):    
    gamZp=0.
    if(mZp > 2*mL):
        gamZp+= gamZp_i(mZp, mL, gL)
    if(mZp > 2*mChi):
        gamZp+= gamZp_i(mZp, mChi, gChi)
    gamZp+=2*gamZp_i(mZp, 0., gL)
    return gamZp
def GamZpOne(mZp, mL, mChi, gL, gChi): #if only one SM fermion couples to Z'
    gamZp=0.
    if(mZp > 2*mL):
        gamZp+= gamZp_i(mZp, mL, gL)
    if(mZp > 2*mChi):
        gamZp+= gamZp_i(mZp, mChi, gChi)
    return gamZp

### trapping luminosity
twoThirds=2/3 #in case someone does not trust the two thirds

def getTrappingLumi(mChi, R, T):
    Qtheo=np.ones(len(T))
    for i in range(len(T)):
        Qtheo[i] = getTrappingLumiOne(mChi/T[i], R[i], T[i])
    return Qtheo

def getTrappingLumiOne(xChi, R, T):
    fac=7*np.pi**4/120
    if(xChi>1e-2):
        fac, _=itg.quad(lambda x: x**2*(x**2-xChi**2)**.5 /(np.exp(x)+1), xChi, np.inf)
    Qtheo=2/np.pi * R**2* T**4 * fac #4 DM-Freiheitsgrade
    return Qtheo

def getRadiusSphere(mChi, R, T, nSim=1, out=False):
    Qtheo=getTrappingLumi(mChi, R, T)
    iCrit=len(R)-1
    Qbound = getQbound(nSim)

    while(iCrit > 0):
        if(Qtheo[iCrit] < Qbound):
            iCrit=iCrit-1
        else:
            break
    if(iCrit!=0):
        if(out):
            print("Properties of blackbody radiation (from i={0}): r = {1:.2f} km, T = {2:.2f} MeV".format(iCrit, R[iCrit]/km2invMeV, T[iCrit]))
        return iCrit
    else:
        return None
