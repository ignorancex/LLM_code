import numpy as np
import scipy.special as sp
import scipy.integrate as itg
import scipy.interpolate as itp
import vegas
import time
import cross_sections as cs
import calcIndividual
import helper

'''
Reuse results from calcIndividual with minimal adaptions
Effectively just have to add up the terms in another way
'''

### free-streaming
def dQdR(mL, mChi, mu_mu, mu_numu, T, R, *args, iCompton=0, giveRatios=False, **kwargs):
    dQdR_L = calcIndividual.dQdR_Ann(mL, mChi, mu_mu, T, R, 4, *args, **kwargs)
    dQdR_nu = calcIndividual.dQdR_Ann(0., mChi, mu_numu, T, R, 9, *args, **kwargs)
    dQdR_C = calcIndividual.dQdR_Com(1, mL, mChi, mu_mu, T, R, 4, *args, **kwargs) if iCompton==1 else 0.
    dQdR = dQdR_L + 2*dQdR_nu + dQdR_C

    if giveRatios:
        return dQdR_L/dQdR, dQdR_nu/dQdR, dQdR_C/dQdR
    else:
        return dQdR

### trapping ###

# main function for trapping #
def lambdaInvMean(*args, approx="inv", **kwargs):
    if(approx=="exact"):
        return lambdaInvMean_exact(*args, **kwargs)
    elif(approx=="inv"):
        return lambdaInvMean_inv(*args, **kwargs)
    else:
        raise ValueError(f"ERROR: Approximation approx={approx} not implemented. Use one of approx=inv,exact")

def lambdaInvMean_inv(mL, mChi, mu_mu, mu_numu, T, scat=2, iCompton=0, giveRatios=False, **kwargs):
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegChi=Fdeg(mChi, T, T*0.) #0.9

    def lambdaInvMeanF(iL, mL, mu, iSigma):
        FdegLm = Fdeg(mL, T, mu)
        FdegLp = Fdeg(mL, T, -mu)
        lIM_LAnn = calcIndividual.integ_LAnn_inv(mL, mChi, mu, T, iSigma, **kwargs) *FdegLm *FdegLp
        lIM_LScat = calcIndividual.integ_LScat_inv(mL, mChi, mu, T, iSigma, **kwargs) *FdegLm *FdegChi if (scat==2 or scat==4) else 0.
        lIM_DMAnn = calcIndividual.integ_DMAnn_inv(mL, mChi, T, iSigma, **kwargs) *FdegChi**2 if (scat==4 or scat==3) else 0.
        lIM_DMScat = 2*calcIndividual.integ_DMScat_inv(mChi, T, iSigma, **kwargs) *FdegChi**2 if scat==4 else 0.
        lIM_C = calcIndividual.lambdaInv_Compt(iL, mL, mChi, mu, T, iSigma, **kwargs) if iCompton==1 else 0.
        #lIM = lIM_LAnn + lIM_LScat + lIM_DMAnn + lIM_DMScat + lIM_C
        return lIM_LAnn, lIM_LScat, lIM_DMAnn, lIM_DMScat, lIM_C
    
    lIM_LAnn, lIM_LScat, lIM_DMAnn, lIM_DMScat, lIM_C = lambdaInvMeanF(1, mL, mu_mu, 4)
    lIM_nuAnn, lIM_nuScat, _, _, _ = lambdaInvMeanF(2, 0., mu_numu, 9) #dont double count chi self-interactions
    lIM = lIM_LAnn + lIM_LScat + lIM_DMAnn + lIM_DMScat + lIM_C + 2*(lIM_nuAnn + lIM_nuScat)

    if giveRatios:
        return lIM_LAnn/lIM, lIM_LScat/lIM, lIM_DMAnn/lIM, lIM_DMScat/lIM, lIM_C/lIM, 2*lIM_nuAnn/lIM, 2*lIM_nuScat/lIM
    else:
        return lIM

def lambdaInvMean_exact(mL, mChi, mu_mu, mu_numu, T, scat=2, xMax=1e3, xSteps=50, iCompton=0, **kwargs):
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegChi=Fdeg(mChi, T, T*0.)

    xChi=mChi/T
    xChiReg = 1e-6
    if(xChi<=xChiReg): #regulator
        xChi = xChiReg

    #calculate Gamma on array (2 integrations)
    xMin = xChi * (1.+1.e-6) #avoid boundary effects
    xFirst = np.exp(np.linspace(np.log(xMin), np.log(xMax), xSteps)) #=x2
    Gamma = np.zeros(xSteps)
    for i in range(xSteps):
        def GammaF(iL, mL, mu, iSigma):
            FdegLm = Fdeg(mL, T, mu)
            FdegLp = Fdeg(mL, T, -mu)
            Gamma_LAnn = calcIndividual.integ_LAnn_exact(xFirst[i], mL, mChi, mu, T, iSigma, **kwargs) *FdegLm*FdegLp
            Gamma_LScat = calcIndividual.integ_LScat_exact(xFirst[i], mL, mChi, mu, T, iSigma, **kwargs) *FdegLm*FdegChi if (scat==2 or scat==4) else 0.
            Gamma_DMAnn = calcIndividual.integ_DMAnn_exact(xFirst[i], mL, mChi, T, iSigma, **kwargs) *FdegChi**2 if (scat==4 or scat==3) else 0.
            Gamma_DMScat = 2*calcIndividual.integ_DMScat_exact(xFirst[i], mChi, T, iSigma, **kwargs) *FdegChi**2 if scat==4 else 0.
            Gamma_C = calcIndividual.integ_Compt(iL, xFirst[i], mL, mChi, mu, T, iSigma, **kwargs) if iCompton==2 else 0.
            #Gamma = Gamma_LAnn+Gamma_LScat +Gamma_C + 2*(Gama_DMAnn+Gamma_DMScat) 
            return Gamma_LAnn, Gamma_LScat, Gamma_DMAnn, Gamma_DMScat, Gamma_C
        Gamma_LAnn, Gamma_LScat, Gamma_DMAnn, Gamma_DMScat, Gamma_C = GammaF(1, mL, mu_mu, 4)
        Gamma_nuAnn, Gamma_nuScat, _, _, _ = GammaF(2, 0., mu_numu, 9)
        Gamma[i] = Gamma_LAnn + Gamma_LScat + Gamma_DMAnn + Gamma_DMScat + Gamma_C + 2*(Gamma_nuAnn + Gamma_nuScat)
        
    # calculate integrand from Gammas using interpolation
    GammaReg = 1e-100
    mask = np.array(Gamma<GammaReg)
    Gamma[mask] = GammaReg
    lambd = (1-xChi**2/xFirst**2)**.5 / Gamma
    weighting = calcIndividual.mfpweight(xFirst, xChi)
    integrand_vals = lambd * weighting
    intf = lambda x: np.exp(itp.interp1d(np.log(xFirst), np.log(integrand_vals), kind="linear")(np.log(x)))

    # calculate final integral
    @vegas.batchintegrand
    def intf2(x):
        x=x[...,0]
        integrand = intf(x)
        return integrand
    integ=vegas.Integrator([[xMin, xMax]])
    res=integ(intf2, nitn=10, neval=2000, alpha=.5).mean
    
    norm = calcIndividual.mfpnorm(xChi)
    lambdaInv = norm/res

    if iCompton==1:
        iL=1
        mu=mu_mu
        lambdaInv_Compton = calcIndividual.lambdaInv_Compt(iL, mL, mChi, mu, T, **kwargs)
        lambdaInv += lambdaInv_Compton

    return lambdaInv
