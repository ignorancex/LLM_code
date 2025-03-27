import numpy as np
import scipy.integrate as itg
import scipy.interpolate as itp
import vegas
import math, time
import cross_sections as cs
import helper

'''
Main calculations for the project. The methods dQdR and lambdaInvMean are called by main.py.
- Calculations are performed for each Supernova radius individually
- Integrals: Use scipy.quad for single integrations and vegas for everything else. We
  made the experience that evaluating the multi-dimensional integrals is quite tricky, with
  different methods implemented in scipy.quad and the Mathematica Integrate function giving
  very different results. We find very stable results with the vegas implementation.
- Calculation for the free-streaming case is straight-forward (just the triple-integral)
- Calculation for the trapping case is challenging: The exact calculation is time consuming,
  so we propose the inv approximation that estimates the average mean free path <MFP> by
  <MFP^-1>^-1. This is roughly an order of magnitude faster, but the results may be off when
  the (weighted) integrands MFP and MFP^-1 have different maxima. Further, it is not clear
  how to include scattering and chi self-interactions in the trapping regime, so we implemented
  3 versions scat=1 (only chi chi -> mu mu), scat=2 (add chi mu -> chi mu) and
  scat=4 (add chi chi -> chi chi) to obtain a range in which the "true" result has to lie.
'''

def calc_annihilation_integral(intf, xm1, xm2, xZp, xGammaZp, gL, nIt, nEval, limit, xfac=5):
    '''
    Algorithm for calculating integrals over 3 variables with a peak with width xGammaZp
    located at xZp by splitting up the integration region in 3 parts.
    xm1 is the m/T of the annihilating particles, and xm2 is the m/T of the final-state particles
    '''
    distance = xfac * xZp * xGammaZp
    ypeak1, ypeak2 = xZp**2 - distance, xZp**2 + distance
    ymin = 4*max(xm1, xm2)**2

    if limit=="eft" or gL > .3: # things are trivial in EFT regime
        integ1=vegas.Integrator([[xm1/(1+xm1), 1.], [xm1/(1+xm1), 1.], [ymin/(1+ymin), 1.]])
        res=integ1(intf, nitn=nIt, neval=nEval, alpha=.5).mean
        return res

    # region left of peak
    if ypeak1 > ymin:
        integ1=vegas.Integrator([[xm1/(1+xm1), 1.], [xm1/(1+xm1), 1.], [ymin/(1+ymin), ypeak1/(1+ypeak1)]])
        res1=integ1(intf, nitn=nIt, neval=nEval, alpha=.5).mean
    else:
        res1 = 0.

    # peak region
    ymin = max(ymin, ypeak1)
    if ypeak2 >= ymin:
        # have to be careful with numerical precision -> Define extra function that deals with this
        @vegas.batchintegrand
        def intf2(x):
            z1, z2, diff = x[...,0], x[...,1], x[...,2]
            y1 = xZp**2 + diff #this becomes numerically the same as xZp**2 for diff<<xZp**2
            zy = y1/(1+y1)
            xNew = np.zeros_like(x)
            xNew[:,0], xNew[:,1], xNew[:,2] = z1, z2, zy
            integrand = intf(xNew, diff=diff) #give diff as additional argument to be used in the propagator (avoid numerical issues from diff=0.)
            jacobian = (1-zy)**2 #cancel jacobian from zy -> y1 transformation
            res = integrand * jacobian
            idx = np.isnan(res)
            return res
        integ2=vegas.Integrator([[xm1/(1+xm1), 1.], [xm1/(1+xm1), 1.], [-distance, distance]])
        res2=integ2(intf2, nitn=nIt, neval=nEval, alpha=.5).mean
    else:
        res2 = 0.

    # region right of peak
    ymin = max(ymin, ypeak2)
    integ3=vegas.Integrator([[xm1/(1+xm1), 1.], [xm1/(1+xm1), 1.], [ymin/(1+ymin), 1.]])
    res3=integ3(intf, nitn=nIt, neval=nEval, alpha=.5).mean

    res = res1 + res2 + res3
    return res

def calc_annihilation_integral_2d(intf, xm1, xm2, xZp, xGammaZp, gL, nIt, nEval, limit, xfac=5):
    '''
    Same as above, but for 2d integration. This is needed for the "exact" approach to trapping. 
    '''
    distance = xfac * xZp * xGammaZp
    ypeak1, ypeak2 = xZp**2 - distance, xZp**2 + distance
    ymin = 4*max(xm1, xm2)**2

    if limit=="eft" or gL>.3: # things are trivial in EFT regime
        integ1=vegas.Integrator([[xm1/(1+xm1), 1.], [ymin/(1+ymin), 1.]])
        res=integ1(intf, nitn=nIt, neval=nEval, alpha=.5).mean
        return res

    # region left of peak
    if ypeak1 > ymin:
        integ1=vegas.Integrator([[xm1/(1+xm1), 1.], [ymin/(1+ymin), ypeak1/(1+ypeak1)]])
        res1=integ1(intf, nitn=nIt, neval=nEval, alpha=.5).mean
    else:
        res1 = 0.

    # peak region
    ymin = max(ymin, ypeak1)
    if ypeak2 >= ymin:
        @vegas.batchintegrand
        def intf2(x): 
            z1, diff = x[...,0], x[...,1]
            y1 = xZp**2 + diff #this becomes equivalent to xZp**2 for diff<<xZp**2
            zy = y1/(1+y1)
            xNew = np.zeros_like(x)
            xNew[:,0], xNew[:,1] = z1, zy
            integrand = intf(xNew, diff=diff) #give diff as additional argument to be used in the propagator
            jacobian = (1-zy)**2 #cancel jacobian from zy -> y1 transformation
            res = integrand * jacobian
            return res
        integ2=vegas.Integrator([[xm1/(1+xm1), 1.], [-distance, distance]])
        res2=integ2(intf2, nitn=nIt, neval=nEval, alpha=.5).mean
    else:
        res2 = 0.

    # region right of peak
    ymin = max(ymin, ypeak2)
    integ3=vegas.Integrator([[xm1/(1+xm1), 1.], [ymin/(1+ymin), 1.]])
    res3=integ3(intf, nitn=nIt, neval=nEval, alpha=.5).mean

    res = res1 + res2 + res3
    return res

### free-streaming
def dQdR(iL, *args, iCompton=0, **kwargs):
    dQdR_A = dQdR_Ann(*args, **kwargs)
    dQdR_C = dQdR_Com(iL, *args, **kwargs) if iCompton==1 else 0.
    dQdR = dQdR_A + dQdR_C
    
    return dQdR

def dQdR_Ann(mL, mChi, mu, T, R, iSigma, nIt=10, nEval=5000, xfac=5, mZp=1., gL=1., gChi=1., limit="full", **kwargs):
    xL = mL/T
    xChi = mChi/T
    xZp = mZp/T
    xGammaZp = helper.GamZpOne(mZp, mL, mChi, gL, gChi) / T
    def intf1(x1, x2, y1, diff=None):
        sigmaVal=cs.sigmaFS(mL, mChi, T, y1, iSigma, mZp=mZp, gL=gL, gChi=gChi, diff=diff, limit=limit, **kwargs)
        facAverage=(x1+x2) * (x1**2-xL**2)**.5/(np.exp(x1+mu/T)+1) * (x2**2-xL**2)**.5/(np.exp(x2-mu/T)+1)
        facv=(1-4*xL**2/y1)**.5 *y1
        ret=facAverage*facv*sigmaVal
        return ret
    @vegas.batchintegrand
    def intf2(x, diff=None):
        z1, z2, zy=x[...,0], x[...,1], x[...,2]
        x1, x2, y1=z1/(1-z1), z2/(1-z2), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-z2)**2 /(1-zy)**2
        jacobian_2=1/(2* (x1**2-xL**2)**.5 * (x2**2-xL**2)**.5)
        integrand=intf1(x1, x2, y1, diff=diff)
        res= integrand * jacobian_1 * jacobian_2
        mask1 = np.array(y1 > cs.y_s(xL, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_s(xL, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        return res
    res = calc_annihilation_integral(intf2, xL, xChi, xZp, xGammaZp, gL, nIt, nEval, limit)
    res *= 1/(4*np.pi**4)*T**7 #prefactors
    res *= 4*np.pi * R**2 #from dQ/dV to dQ/dR
    res *= 1/2 if iSigma==9 else 1. # factor 1/2 for neutrinos
    return res

def dQdR_Com(iL, mL, mChi, mu, T, R, iSigma, nIt=10, nEval=2000, xfac=5, mZp=1., gL=1., gChi=1., **kwargs):
    if iL>1: #no Compton for neutrinos
        return 0.

    xL = mL/T
    xChi = mChi/T
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    def n(m, T, mu):
        fac, _ = itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return 2/(2*np.pi**2) * T**3 * fac
    
    if iL==0: #electrons (relativistic)
        def intf1(xphoton, xe, costheta):
            weight = xphoton / (np.exp(xphoton) -1) * xe *(xphoton+xe) /(np.exp(xe -mu/T)+1)
            shat = 2 * xphoton * xe * (1-costheta) / xL**2
            sigma = cs.sigmaFS_Compton_resonant(shat, mZp, mL, mChi, gChi, gL, **kwargs)
            res = weight * sigma * shat * xL**2
            mask = np.array(shat < (1+mZp/mL)**2)
            res[mask] = 0.
            return res
        @vegas.batchintegrand
        def intf2(x):
            z1, z2, costheta= x[...,0], x[...,1], x[...,2]
            xphoton=z1/(1-z1)
            xe = z2/(1-z2)
            jacobian = 1/(1-z1)**2 / (1-z2)**2
            res = jacobian * intf1(xphoton, xe, costheta)
            return res
        integ = vegas.Integrator([[0., 1.], [xL/(1+xL), 1.], [-1., 1.]])
        res = integ(intf2, nitn=nIt, neval=nEval, alpha=.5).mean
        res = Fdeg(mL, T, mu) /(8 * np.pi**4) * T**7 * res
        
    elif iL==1: #muons (non-relativistic)
        def intf1(xphoton): #xphoton = omega / T
            weight = xphoton**3 / (np.exp(xphoton) -1)
            shat = 1 + 2 * xphoton /xL
            sigma = cs.sigmaFS_Compton_resonant(shat, mZp, mL, mChi, gChi, gL, **kwargs)
            res = weight * sigma
            mask = np.array(shat < (1+mZp/mL)**2)
            res[mask] = 0.
            return res
        @vegas.batchintegrand
        def intf2(z):
            z = z[...,0]
            x=z/(1-z)
            jacobian = 1/(1-z)**2
            res= jacobian * intf1(x)
            return res
        integ = vegas.Integrator([[0., 1.]])
        res = integ(intf2, nitn=nIt, neval=nEval, alpha=.5).mean
        res = Fdeg(mL, T, mu) * n(mL, T, mu)/np.pi**2 * T**4 * res
        
    res *= 4*np.pi * R**2
    return res

### trapping ###
# main function for trapping #
def lambdaInvMean(*args, approx="inv", **kwargs):
    if(approx=="exact"):
        return lambdaInvMean_exact(*args, **kwargs)
    elif(approx=="inv"):
        return lambdaInvMean_inv(*args, **kwargs)
    else:
        raise ValueError(f"ERROR: Approximation approx={approx} not implemented. Use one of approx=inv,exact")

# helper functions
def mfpweight(x, xChi, approach="thermal"):
    if approach == "rosseland":
        weight1 = (1-xChi**2/x**2) * x**4
        if type(x) is np.ndarray:
            weight2 = np.exp(x)/(np.exp(x)+1)**2
            mask = np.array(x > 1e2)
            weight2[mask] = np.exp(-x[mask])
        else:
            weight2 = np.exp(x)/(np.exp(x)+1)**2 if x<2e2 else np.exp(-x)    
        return weight1 * weight2
    elif approach == "thermal":
        return 1/(np.exp(x)+1)
    else:
        raise ValueError(f"Approach {approach} for calculating MFP not implemented")

def mfpnorm(xChi, approach="thermal"):
    '''Calculate normalization factor in the Rosseland average'''
    normR, _=itg.quad(lambda x: mfpweight(x, xChi, approach=approach), xChi, np.inf)
    return normR

# inv approximation #
def integ_LAnn_inv(mL, mChi, mu, T, iSigma, mZp=1., gL=1., gChi=1., nIt=10, nEval=2000, al=.5, limit="full", **kwargs):
    xChi=mChi/T
    xL=mL/T
    xZp = mZp/T
    xGammaZp = helper.GamZpOne(mZp, mL, mChi, gL, gChi) / T
    def intf1(x1, x2, y1, diff=None): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaTR_s(mL, mChi, T, y1, iSigma, mZp=mZp, gL=gL, gChi=gChi, diff=diff, limit=limit, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    @vegas.batchintegrand
    def intf2(x, diff=None):
        z1, z2, zy=x[...,0], x[...,1], x[...,2]
        x1, x2, y1=z1/(1-z1), z2/(1-z2), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-z2)**2 /(1-zy)**2
        jacobian_2=1/(2* (x1**2-xChi**2)**.5 * (x2**2-xChi**2)**.5)
        integrand=intf1(x1, x2, y1, diff=diff)
        weight = mfpweight(x2,xChi)
        res= integrand * jacobian_1 * jacobian_2 * weight
        mask1 = np.array(y1 > cs.y_s(xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_s(xChi, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        return res
    res = calc_annihilation_integral(intf2, xChi, xL, xZp, xGammaZp, gL, nIt, nEval, limit)
    res *= 1/2 if iSigma==9 else 1. # factor 1/2 for neutrinos
    norm = mfpnorm(xChi)
    return res/norm

def integ_LScat_inv(mL, mChi, mu, T, iSigma, nIt=10, nEval=2000, al=.5, **kwargs):
    xChi=mChi/T
    xL=mL/T
    def intf1(x1, x2, y1): #x1: lepton, x2: chi
        sigmaVal=cs.sigmaTR_t(mL, mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xL**2)**.5/(np.exp(x1-mu/T)+1) #set muChi=0
        facGamma2=( (y1-xChi**2-xL**2)**2-4*xChi**2*xL**2)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    @vegas.batchintegrand
    def intf2(x):
        z1, z2, zy=x[...,0], x[...,1], x[...,2]
        x1, x2, y1=z1/(1-z1), z2/(1-z2), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-z2)**2
        jacobian_2=1/(2*(x1**2-xL**2)**.5 * (x2**2-xChi**2)**.5)
        integrand = intf1(x1, x2, y1)
        weight = mfpweight(x2,xChi)
        res=jacobian_1 * jacobian_2 * integrand * weight
        mask1 = np.array(y1 > cs.y_t(xL, xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_t(xL, xChi, x1, x2, 1.))
        res[mask1]=0.
        res[mask2]=0.
        return res
    ymin = 4*max(xChi, xL)**2
    integ=vegas.Integrator([[xL/(1+xL), 1.], [xChi/(1+xChi), 1.], [ymin/(1+ymin),1.]])
    res=integ(intf2, nitn=nIt, neval=nEval, alpha=al).mean
    res *= 1/2 if iSigma==9 else 1. # factor 1/2 for neutrinos
    norm = mfpnorm(xChi)
    return res/norm

def integ_DMAnn_inv(mL, mChi, T, iSigma, nIt=10, mZp=1., gL=1., gChi=1., nEval=2000, al=.5, limit="full", **kwargs): #need mL for GammaZp
    xChi=mChi/T
    xZp = mZp/T
    xGammaZp = helper.GamZpOne(mZp, mL, mChi, gL, gChi) / T  
    def intf1(x1, x2, y1, diff=None): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaTR_DMself_s(mChi, T, y1, iSigma, mZp=mZp, gL=gL, gChi=gChi, diff=diff, limit=limit, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    @vegas.batchintegrand
    def intf2(x, diff=None):
        z1, z2, zy=x[...,0], x[...,1], x[...,2]
        x1, x2, y1=z1/(1-z1), z2/(1-z2), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-z2)**2 /(1-zy)**2
        jacobian_2=1/(2* (x1**2-xChi**2)**.5 * (x2**2-xChi**2)**.5)
        integrand = intf1(x1, x2, y1, diff=diff)
        weight= mfpweight(x2,xChi)
        res= jacobian_1 * jacobian_2 * integrand * weight
        mask1 = np.array(y1 > cs.y_s(xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_s(xChi, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        return res
    res = calc_annihilation_integral(intf2, xChi, xChi, xZp, xGammaZp, gL, nIt, nEval, limit)
    norm = mfpnorm(xChi)
    return res/norm

def integ_DMScat_inv(mChi, T, iSigma, nIt=10, nEval=2000, al=.5, **kwargs):
    xChi=mChi/T
    def intf1(x1, x2, y1): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaTR_DMself_t(mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=( (y1-2*xChi**2)**2-4*xChi**4)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        lambdaInv=Gamma/(1-xChi**2/x2**2)**.5
        return lambdaInv
    @vegas.batchintegrand
    def intf2(x):
        z1, z2, zy=x[...,0], x[...,1], x[...,2]
        x1, x2, y1=z1/(1-z1), z2/(1-z2), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-z2)**2
        jacobian_2=1/(2*(x1**2-xChi**2)**.5 * (x2**2-xChi**2)**.5)
        integrand = intf1(x1, x2, y1)
        weight= mfpweight(x2,xChi)
        res=jacobian_1 * jacobian_2 * integrand * weight
        mask1 = np.array(y1 > cs.y_t(xChi, xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_t(xChi, xChi, x1, x2, 1.))
        res[mask1]=0.
        res[mask2]=0.
        return res
    ymin = 4*xChi**2
    integ=vegas.Integrator([[xChi/(1+xChi), 1.], [xChi/(1+xChi), 1.], [ymin/(1+ymin),1.]])
    res=integ(intf2, nitn=nIt, neval=nEval, alpha=al).mean
    norm = mfpnorm(xChi)
    return res/norm

def lambdaInvMean_inv(iL, mL, mChi, mu, T, iSigma, scat=2, iCompton=0, **kwargs):
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegLm=Fdeg(mL, T, mu)
    FdegLp=Fdeg(mL, T, -mu)
    FdegChi=Fdeg(mChi, T, T*0.)

    # lIM = lambdaInverseMean
    lIM_LAnn = integ_LAnn_inv(mL, mChi, mu, T, iSigma, **kwargs) *FdegLm *FdegLp
    lIM_LScat = integ_LScat_inv(mL, mChi, mu, T, iSigma, **kwargs) *FdegLm *FdegChi if (scat==2 or scat==4) else 0.
    lIM_DMAnn = integ_DMAnn_inv(mL, mChi, T, iSigma, **kwargs) *FdegChi**2 if (scat==4 or scat==3) else 0.
    lIM_DMScat = 2*integ_DMScat_inv(mChi, T, iSigma, **kwargs) *FdegChi**2 if scat==4 else 0.
    lIM_C = lambdaInv_Compt(iL, mL, mChi, mu, T, iSigma, **kwargs) if iCompton==1 else 0.

    lIM = lIM_LAnn + lIM_LScat + lIM_DMAnn + lIM_DMScat + lIM_C

    return lIM

# exact treatment #
def integ_LAnn_exact(x2, mL, mChi, mu, T, iSigma, nIt=10, nEval=2000, al=.5, mZp=1., gL=1., gChi=1., limit="full", **kwargs):
    xChi=mChi/T
    xL=mL/T
    xZp = mZp/T
    xGammaZp = helper.GamZpOne(mZp, mL, mChi, gL, gChi) / T   
    def intf1(x1, y1, diff=None): #x1: boring chi, x2: interesting chi (integrated last)
        sigmaVal=cs.sigmaTR_s(mL, mChi, T, y1, iSigma, mZp=mZp, gL=gL, gChi=gChi, diff=diff, limit=limit, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    @vegas.batchintegrand
    def intf2(x, diff=None):
        z1, zy=x[...,0], x[...,1]
        x1, y1=z1/(1-z1), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-zy)**2
        jacobian_2=1/(2* (x1**2-xChi**2)**.5 * (x2**2-xChi**2)**.5)
        integrand=intf1(x1, y1, diff=diff)
        res = jacobian_1 * jacobian_2 * integrand
        
        mask1 = np.array(y1 > cs.y_s(xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_s(xChi, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        return res
    res=calc_annihilation_integral_2d(intf2, xChi, xL, xZp, xGammaZp, gL, nIt, nEval, limit)
    res *= 1/2 if iSigma==9 else 1. # factor 1/2 for neutrinos
    return res

def integ_LScat_exact(x2, mL, mChi, mu, T, iSigma, nIt=10, nEval=2000, al=.5, **kwargs):
    xChi=mChi/T
    xL=mL/T
    def intf1(x1, y1): #x1: lepton, x2: chi #could factorize integral & calc with quad
        sigmaVal=cs.sigmaTR_t(mL, mChi, T, y1, iSigma, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xL**2)**.5/(np.exp(x1-mu/T)+1) #set muChi=0
        facGamma2=( (y1-xChi**2-xL**2)**2-4*xChi**2*xL**2)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    @vegas.batchintegrand
    def intf2(x):
        z1, zy = x[...,0], x[...,1]
        x1=z1/(1-z1)
        y1=zy/(1-zy)
        jacobian_1 = 1/(1-z1)**2 /(1-zy)**2
        jacobian_2 = 1/(2* (x1**2-xL**2)**.5 * (x2**2-xChi**2)**.5)
        integrand = intf1(x1, y1)
        res=jacobian_1 * jacobian_2 * integrand
        mask1 = np.array(y1 > cs.y_t(xL, xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_t(xL, xChi, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        mask3 = np.array(res<0)
        res[mask3] = 0.
        return res
    ymin = (xL+xChi)**2
    integ=vegas.Integrator([[xL/(1+xL), 1.], [ymin/(1.+ymin), 1.]])
    res=integ(intf2, nitn=nIt, neval=nEval, alpha=al).mean
    res *= 1/2 if iSigma==9 else 1. # factor 1/2 for neutrinos
    return res

def integ_DMAnn_exact(x2, mL, mChi, T, iSigma, mZp=1., gL=1., gChi=1., nIt=10, nEval=2000, al=.5, limit="full", **kwargs):
    xChi=mChi/T
    xZp = mZp/T
    xGammaZp = helper.GamZpOne(mZp, mL, mChi, gL, gChi) / T  
    def intf1(x1, y1, diff=None): #x1: boring chi, x2: interesting chi  (integrated last)
        sigmaVal=cs.sigmaTR_DMself_s(mChi, T, y1, iSigma, mZp=mZp, gL=gL, gChi=gChi, diff=diff, limit=limit, **kwargs)
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=(1-4*xChi**2/y1)**.5 * y1
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    @vegas.batchintegrand
    def intf2(x, diff=None):
        z1, zy=x[...,0], x[...,1]
        x1, y1=z1/(1-z1), zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-zy)**2
        jacobian_2=1/(2* (x1**2-xChi**2)**.5 * (x2**2-xChi**2)**.5)
        integrand=intf1(x1, y1, diff=diff)
        res=jacobian_1 * jacobian_2 * integrand
        
        mask1 = np.array(y1 > cs.y_s(xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_s(xChi, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        return res
    res=calc_annihilation_integral_2d(intf2, xChi, xChi, xZp, xGammaZp, gL, nIt, nEval, limit)
    return res
def integ_DMScat_exact(x2, mChi, T, iSigma, nIt=10, nEval=2000, al=.5, **kwargs):
    xChi=mChi/T
    def intf1(x1, y1): #x1: boring chi, x2: interesting chi  (integrated last)
        sigmaVal=cs.sigmaTR_DMself_t(mChi, T, y1, iSigma, **kwargs)        
        facGamma=T**3/(4*np.pi**2*x2)*(x1**2-xChi**2)**.5/(np.exp(x1)+1) #set muChi=0
        facGamma2=( (y1-2*xChi**2)**2-4*xChi**4)**.5
        Gamma=facGamma * facGamma2 * sigmaVal
        return Gamma
    @vegas.batchintegrand
    def intf2(x):
        z1, zy=x[...,0], x[...,1]
        x1=z1/(1-z1)
        y1=zy/(1-zy)
        jacobian_1=1/(1-z1)**2 /(1-zy)**2
        jacobian_2=1/(2*(x1**2-xChi**2)**.5 * (x2**2-xChi**2)**.5)
        integrand = intf1(x1, y1)
        res=jacobian_1 * jacobian_2 * integrand
        
        mask1 = np.array(y1 > cs.y_t(xChi, xChi, x1, x2, -1.))
        mask2 = np.array(y1 < cs.y_t(xChi, xChi, x1, x2, 1.))
        res[mask1] = 0.
        res[mask2] = 0.
        return res
    integ=vegas.Integrator([[xChi/(1+xChi), 1.], [4*xChi**2/(1+4*xChi**2),1.]])
    res=integ(intf2, nitn=nIt, neval=nEval, alpha=al).mean
    return res

def lambdaInv_Compt(iL, mL, mChi, mu, T, nIt=10, nEval=2000, mZp=1., gL=1., gChi=1., **kwargs):
    if iL>1: #no Compton for neutrinos
        return 0.

    xL = mL/T
    xChi = mChi/T
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    def n(m, T, mu):
        fac, _ = itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return 2/(2*np.pi**2) * T**3 * fac
    
    if iL==0: #electrons (relativistic)
        def intf1(xphoton, xe, costheta):
            weight = xphoton / (np.exp(xphoton) -1) * xe /(np.exp(xe -mu/T)+1)
            shat = 2 * xphoton * xe * (1-costheta) / xL**2
            sigma = cs.sigmaFS_Compton_resonant(shat, mZp, mL, mChi, gChi, gL, **kwargs)
            res = weight * sigma * shat * xL**2
            mask = np.array(shat < (1+mZp/mL)**2)
            res[mask] = 0.
            return res
        @vegas.batchintegrand
        def intf2(x):
            z1, z2, costheta= x[...,0], x[...,1], x[...,2]
            xphoton=z1/(1-z1)
            xe = z2/(1-z2)
            jacobian = 1/(1-z1)**2 / (1-z2)**2 
            res = jacobian * intf1(xphoton, xe, costheta)
            return res
        integ = vegas.Integrator([[0., 1.], [xL/(1+xL), 1.], [-1., 1.]])
        res = integ(intf2, nitn=nIt, neval=nEval, alpha=.5).mean
        res = Fdeg(mL, T, mu) /(4 * np.pi**4) * T**7 * res
    elif iL==1: #muons (non-relativistic)
        def intf1(xphoton): #xphoton = omega / T
            weight = xphoton**2 / (np.exp(xphoton) -1)
            shat = 1 + 2 * xphoton * T/mL
            sigma = cs.sigmaFS_Compton_resonant(shat, mZp, mL, mChi, gChi, gL, **kwargs)
            res = weight * sigma
            mask = np.array(shat < (1+mZp/mL)**2)
            res[mask] = 0.
            return res
        @vegas.batchintegrand
        def intf2(z):
            z = z[...,0]
            x=z/(1-z)
            jacobian = 1/(1-z)**2
            res = jacobian * intf1(x)
            return res
        integ = vegas.Integrator([[0., 1.]])
        res = integ(intf2, nitn=nIt, neval=nEval, alpha=.5).mean
        res = Fdeg(mL, T, mu) * n(mL, T, mu)/np.pi**2 * T**3 *res
    lambdaInv = res/n(mChi, T, 0.)
    return lambdaInv
        
def lambdaInvMean_exact(iL, mL, mChi, mu, T, iSigma, scat=2, xMax=1e2, xSteps=50, iCompton=0,
                        mZp=1., **kwargs):
    def Fdeg(m, T, mu):
        num, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1) * (1-1/(np.exp(x-mu/T)+1)), m/T, np.inf)
        denom, _=itg.quad(lambda x: x*(x**2-(m/T)**2)**.5/(np.exp(x-mu/T)+1), m/T, np.inf)
        return num/denom
    FdegLm=Fdeg(mL, T, mu)
    FdegLp=Fdeg(mL, T, -mu)
    FdegChi=Fdeg(mChi, T, T*0.)

    xL = mL/T
    xChi=mChi/T
    xChiReg = 1.e-2
    if(xChi<=xChiReg): #regulator
        xChi = xChiReg

    #calculate Gamma on array (2 integrations)
    xMin = xChi * (1.+1.e-2)
    xFirst = np.exp(np.linspace(np.log(xMin), np.log(xMax), xSteps)) #=x2
    Gamma = np.zeros(xSteps)

    Gamma_LAnn, Gamma_LScat, Gamma_DM, Gamma_C = np.zeros((4,xSteps))
    for i in range(xSteps):
        Gamma_LAnn0 = integ_LAnn_exact(xFirst[i], mL, mChi, mu, T, iSigma, mZp=mZp, **kwargs) *FdegLm*FdegLp
        Gamma_LScat0 = integ_LScat_exact(xFirst[i], mL, mChi, mu, T, iSigma, mZp=mZp, **kwargs) *FdegLm*FdegChi if (scat==2 or scat==4) else 0.
        Gamma_DMAnn0 = integ_DMAnn_exact(xFirst[i], mL, mChi, T, iSigma, mZp=mZp, **kwargs) *FdegChi**2 if (scat==4 or scat==3) else 0.
        Gamma_DMScat0 = 2*integ_DMScat_exact(xFirst[i], mChi, T, iSigma, mZp=mZp, **kwargs) *FdegChi**2 if scat==4 else 0.
        Gamma_C0= integ_Compt(iL, xFirst[i], mL, mChi, mu, T, iSigma, mZp=mZp, **kwargs) if iCompton==2 else 0.
        Gamma[i] = Gamma_LAnn0 + Gamma_LScat0 + Gamma_DMAnn0 + Gamma_DMScat0 + Gamma_C0
        
        Gamma_LAnn[i] = Gamma_LAnn0
        Gamma_LScat[i] = Gamma_LScat0
        Gamma_DM[i] = Gamma_DMAnn0 + Gamma_DMScat0
        Gamma_C[i] = Gamma_C0
    
    # calculate integrand from Gammas using interpolation
    GammaReg = 1e-100
    mask = np.array(Gamma<GammaReg)
    GammaMasked = Gamma.copy()
    GammaMasked[mask] = GammaReg #dont divide by 0
    lambd = (1-xChi**2/xFirst**2)**.5 / GammaMasked
    weighting = mfpweight(xFirst, xChi)
    integrand_vals = lambd * weighting
    intf = lambda x: np.exp(itp.interp1d(np.log(xFirst), np.log(integrand_vals), kind="linear")(np.log(x)))
 
    # calculate MFP integral
    @vegas.batchintegrand
    def intf2(x):
        x=x[...,0]
        integrand = intf(x)
        return integrand
    integ=vegas.Integrator([[xMin, xMax]])
    res=integ(intf2, nitn=10, neval=2000, alpha=.5).mean
    
    norm = mfpnorm(xChi)
    lambdaInv = norm/res

    if iCompton==1: # add Compton contribution (using the "inverse" approximation)
        lambdaInv_Compton = lambdaInv_Compt(iL, mL, mChi, mu, T, mZp=mZp, **kwargs)
        lambdaInv += lambdaInv_Compton

    return lambdaInv
