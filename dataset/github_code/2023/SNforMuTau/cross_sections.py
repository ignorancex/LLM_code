import numpy as np
import helper

'''
Listing of all cross-sections needed for the project
'''

### free-streaming
def y_s(xm, x1, x2, cosTh):
    ret=2*(xm**2 + x1*x2 - (x1**2-xm**2)**.5 * (x2**2-xm**2)**.5 * cosTh)
    return ret
def y_t(xL, xChi, x1, x2, cosTh): #symmetric in x1/x2 or xL/xChi
    ret=xL**2 + xChi**2 + 2*(x1*x2 - (x1**2-xL**2)**.5 * (x2**2-xChi**2)**.5 * cosTh)
    return ret

# s-channel cross-sections (without propagator)
def sigma0_Ann_S(epsL, epsDM, y1, LS, LP, DMS, DMP):
    sig= 1/(16*np.pi*y1) * (y1 - 4*epsDM)**.5/(y1-4*epsL)**.5 * (LP**2 * y1 + LS**2*(y1-4*epsL))* (y1 * (DMP**2+DMS**2) - 4*epsDM*DMS**2)
    return sig
def sigma0_Ann_SS(epsL, epsDM, y1):
    return sigma0_Ann_S(epsL, epsDM, y1, 1, 0, 1, 0)
def sigma0_Ann_SP(epsL, epsDM, y1):
    return sigma0_Ann_S(epsL, epsDM, y1, 1, 0, 0, 1)
def sigma0_Ann_PS(epsL, epsDM, y1):
    return sigma0_Ann_S(epsL, epsDM, y1, 0, 1, 1, 0)
def sigma0_Ann_PP(epsL, epsDM, y1):
    return sigma0_Ann_S(epsL, epsDM, y1, 0, 1, 0, 1)
def sigma0_Ann_V(epsL, epsDM, y1, LV, LA, DMV, DMA):
    sig= 1/(12*np.pi *y1) * (y1 - 4*epsDM)**.5/(y1-4*epsL)**.5 * (LA**2 *(4*epsL*(epsDM*(7*DMA**2 - 2*DMV**2) - y1 *(DMA**2 + DMV**2))
                                                                          + y1*(epsDM*(2*DMV**2-4*DMA**2) + y1 * (DMA**2+DMV**2)))+LV**2*(2*epsL+y1)*(epsDM*(2*DMV**2-4*DMA**2)+y1*(DMA**2+DMV**2)))
    return sig
def sigma0_Ann_RR(epsL, epsDM, y1): #LL LR RL all the same
    return sigma0_Ann_V(epsL, epsDM, y1, 1/2, 1/2, 1/2, 1/2)
def sigma0_Ann_VV(epsL, epsDM, y1):
    return sigma0_Ann_V(epsL, epsDM, y1, 1, 0, 1, 0)
def sigma0_Ann_VA(epsL, epsDM, y1):
    return sigma0_Ann_V(epsL, epsDM, y1, 1, 0, 0, 1)
def sigma0_Ann_AV(epsL, epsDM, y1):
    return sigma0_Ann_V(epsL, epsDM, y1, 0, 1, 1, 0)
def sigma0_Ann_AA(epsL, epsDM, y1):
    return sigma0_Ann_V(epsL, epsDM, y1, 0, 1, 0, 1)
def sigma0_Ann_LV(epsL, epsDM, y1):
    return sigma0_Ann_V(epsL, epsDM, y1, 1/2, -1/2, 1, 0)
def sigma0_Ann_T(epsL, epsDM, y1, DMT, DMAT):
    sig= 1/(6*np.pi*y1) * (y1-4*epsDM)**.5/(y1-4*epsL)**.5 * (epsL * (2*y1 * (DMAT**2 + DMT**2) -8*epsDM *(4*DMAT**2 - 5*DMT**2)) + y1 * (2*epsDM + y1)*(DMAT**2+DMT**2))
    return sig
def sigma0_Ann_TT(epsL, epsDM, y1):
    return sigma0_Ann_T(epsL, epsDM, y1, 1, 0)
def sigma0_Ann_TAT(epsL, epsDM, y1):
    return sigma0_Ann_T(epsL, epsDM, y1, 0, 1)
sigma0_Ann=np.array([sigma0_Ann_SS, sigma0_Ann_SP, sigma0_Ann_PS, sigma0_Ann_PP,
                    sigma0_Ann_VV, sigma0_Ann_VA, sigma0_Ann_AV, sigma0_Ann_AA,
                    sigma0_Ann_RR, sigma0_Ann_LV, sigma0_Ann_TT, sigma0_Ann_TAT])
sigmaName = ["SS", "SP", "PS", "PP", "VV", "VA", "AV", "AA", "RR", "LV", "TT", "TAT"]
nSig=len(sigma0_Ann)

# full s-channel cross-sections (including propagator)
def sigmaFS(mL, mChi, T, y1, iSigma, mZp=1, gL=1, gChi=1, Lambda=1, oneFermion=False, diff=None, limit="full", **kwargs):
    xL = mL/T
    xChi = mChi/T
    if limit=="full":
        if oneFermion:
            gamZp=helper.GamZpOne(mZp, mL, mChi, gL, gChi)
        else:
            gamZp=helper.GamZpMuTau(mZp, mL, mChi, gL, gChi)
        if diff is None:
            propagator=gL**2*gChi**2/( (y1*T**2 -mZp**2)**2 +mZp**2 * gamZp**2)
        else:
            propagator = gL**2*gChi**2 / ((diff * T**2)**2 + mZp**2 * gamZp**2)
    elif limit=="eft":
        propagator = 4/Lambda**4
    sigma = propagator * sigma0_Ann[iSigma](xL**2, xChi**2, y1) *T**2 #factor T^2 to get correct dimensions
    return sigma

### trapping
# t-channel processes (for general mZp)
def sigma0_Scat_V(epsL, epsDM, epsZ, y, LV, LA, DMV, DMA):
    sig=1/(8*np.pi) * \
        ( 1/(y*epsZ) /(y**2+y*(-2*epsL-2*epsDM+epsZ)+(epsL-epsDM)**2) *\
            ( LA**2*(2*y**3*(DMA**2+DMV**2)-y**2*(DMA**2+DMV**2)*(4*epsL+4*epsDM-3*epsZ)+2*y*(DMA**2*(epsL**2+10*epsL*epsDM-3*epsL*epsZ+epsDM**2+epsZ**2-3*epsDM*epsZ)+DMV**2*(epsL**2-epsDM*(2*epsL+epsZ)-3*epsL*epsZ+epsDM**2+epsZ**2)) + epsZ*(DMA**2+DMV**2)*(epsL-epsDM)**2) \
              + LV**2*(2*y**3*(DMA**2+DMV**2)-y**2*(DMA**2+DMV**2)*(4*epsL+4*epsDM-3*epsZ)+2*y*(DMA**2*(epsL**2-epsL*(2*epsDM+epsZ)+epsDM**2+epsZ**2-3*epsDM*epsZ)+DMV**2*(epsL**2+2*epsL*epsDM-epsL*epsZ+epsDM**2+epsZ**2-epsDM*epsZ))+epsZ*(DMA**2+DMV**2)*(epsL-epsDM)**2)) \
          -(2*np.log((y**2 -y*(2*epsL+2*epsDM-epsZ)+(epsL-epsDM)**2)/(y*epsZ)) /(y**2-2*y*(epsL+epsDM)+(epsL-epsDM)**2) \
          * (LA**2*(DMV**2*(y-2*epsL+epsZ)+DMA**2*(y-2*epsL-2*epsDM+epsZ))+LV**2*(DMV**2*(y+epsZ)+DMA**2*(y-2*epsDM+epsZ)))))
    return sig
def sigma0_Scat_VV(epsL, epsDM, epsZ, y1):
    return sigma0_Scat_V(epsL, epsDM, epsZ, y1, 1, 0, 1, 0)
def sigma0_Scat_LV(epsL, epsDM, epsZ, y1):
    return sigma0_Scat_V(epsL, epsDM, epsZ, y1, 1/2, -1/2, 1, 0)
sigma0_Scat=np.array([None, None, None, None,
                      sigma0_Scat_VV, None, None, None,
                      None, sigma0_Scat_LV, None, None]) #only need some combinations

# t-channel processes in EFT regime
def sigma0_Scat_S_EFT(epsL, epsDM, y1, LS, LP, DMS, DMP):
    sig= 1/(48*np.pi*y1**3) * (LP**2 *(-2*epsL*(epsDM+y1) +epsL**2 +(epsDM-y1)**2) *((DMP**2 + DMS**2)*(-2*epsL*(epsDM+y1) + epsL**2+(epsDM-y1)**2) + 6*epsDM * y1 * DMS**2)
                                 + LS**2*(6*epsL*y1*((DMP**2+DMS**2)*(-2*epsL*(epsDM+y1)+epsL**2+(epsDM-y1)**2)+8*epsDM*y1*DMS**2)
                                          +(-2*epsL *(epsDM + y1) +epsL**2 +(epsDM-y1)**2)*((DMP**2 + DMS**2)*(-2*epsL*(epsDM+y1) +epsL**2 +(epsDM-y1)**2)+6*epsDM*y1*DMS**2)))
    return sig
def sigma0_Scat_SS_EFT(epsL, epsDM, y1):
    return sigma0_Scat_S_EFT(epsL, epsDM, y1, 1, 0, 1, 0)
def sigma0_Scat_SP_EFT(epsL, epsDM, y1):
    return sigma0_Scat_S_EFT(epsL, epsDM, y1, 1, 0, 0, 1)
def sigma0_Scat_PS_EFT(epsL, epsDM, y1):
    return sigma0_Scat_S_EFT(epsL, epsDM, y1, 0, 1, 1, 0)
def sigma0_Scat_PP_EFT(epsL, epsDM, y1):
    return sigma0_Scat_S_EFT(epsL, epsDM, y1, 0, 1, 0, 1)
def sigma0_Scat_V_EFT(epsL, epsDM, y1, LV, LA, DMV, DMA):
    return 1/(24*np.pi *y1**3) * (LA**2*(epsL**2*(-2*epsDM*y1*(DMA**2+4*DMV**2)+6*epsDM**2*(DMA**2+DMV**2)-3*y1**2*(DMA**2+DMV**2)) \
                                         -2*epsL*(epsDM*y1**2*(7*DMV**2-23*DMA**2)+epsDM**2*y1*(DMA**2-5*DMV**2)+2*epsDM**3*(DMA**2+DMV**2)+2*y1**3*(DMA**2+DMV**2))\
                                         -2*epsL**3*(2*epsDM-y1)*(DMA**2+DMV**2)+epsL**4*(DMA**2+DMV**2)+(epsDM-y1)**2*(epsDM*y1*(4*DMA**2-2*DMV**2)+epsDM**2*(DMA**2+DMV**2)+4*y1**2*(DMA**2+DMV**2)))\
                                  +LV**2*(epsL**2*(2*epsDM*y1*(5*DMA**2+2*DMV**2)+6*epsDM**2*(DMA**2+DMV**2)+9*y1**2*(DMA**2+DMV**2))-2*epsL*(epsDM*y1**2*(7*DMA**2-11*DMV**2)+epsDM**2*y1*(4*DMA**2-2*DMV**2)\
                                                                                                                                              +2*epsDM**3*(DMA**2+DMV**2)+5*y1**3*(DMA**2+DMV**2))\
                                          -4*epsL**3*(epsDM+y1)*(DMA**2+DMV**2)+epsL**4*(DMA**2+DMV**2)+(epsDM-y1)**2*(epsDM*y1*(4*DMA**2-2*DMV**2)+epsDM**2*(DMA**2+DMV**2)+4*y1**2*(DMA**2+DMV**2))))
def sigma0_Scat_RR_EFT(epsL, epsDM, y1): #LL LR RL all the same
    return sigma0_Scat_V_EFT(epsL, epsDM, y1, 1/2, 1/2, 1/2, 1/2)
def sigma0_Scat_VV_EFT(epsL, epsDM, y1):
    return sigma0_Scat_V_EFT(epsL, epsDM, y1, 1, 0, 1, 0)
def sigma0_Scat_VA_EFT(epsL, epsDM, y1):
    return sigma0_Scat_V_EFT(epsL, epsDM, y1, 1, 0, 0, 1)
def sigma0_Scat_AV_EFT(epsL, epsDM, y1):
    return sigma0_Scat_V_EFT(epsL, epsDM, y1, 0, 1, 1, 0)
def sigma0_Scat_AA_EFT(epsL, epsDM, y1):
    return sigma0_Scat_V_EFT(epsL, epsDM, y1, 0, 1, 0, 1)
def sigma0_Scat_LV_EFT(epsL, epsDM, y1):
    return sigma0_Scat_V_EFT(epsL, epsDM, y1, 1/2, -1/2, 1, 0)
def sigma0_Scat_T_EFT(epsL, epsDM, y1, DMT, DMAT):
    return 1/(6*np.pi*y1**3) * (epsL**2*(epsDM*y1 + 6*epsDM**2+6*y1**2)*(DMT**2+DMAT**2) +epsL*(4*epsDM*y1**2*(13*DMT**2-5*DMAT**2)+epsDM**2*y1*(DMT**2+DMAT**2)-4*epsDM**3*(DMT**2+DMAT**2)-13*y1**3*(DMT**2+DMAT**2))-epsL**3*(4*epsDM+y1)*(DMT**2+DMAT**2)+epsL**4*(DMT**2+DMAT**2)+(epsDM-y1)**2*(epsDM*y1+epsDM**2+7*y1**2)*(DMT**2+DMAT**2))
def sigma0_Scat_TT_EFT(epsL, epsDM, y1):
    return sigma0_Scat_T_EFT(epsL, epsDM, y1, 1, 0)
def sigma0_Scat_TAT_EFT(epsL, epsDM, y1):
    return sigma0_Scat_T_EFT(epsL, epsDM, y1, 0, 1)
sigma0_Scat_EFT=np.array([sigma0_Scat_SS_EFT, sigma0_Scat_SP_EFT, sigma0_Scat_PS_EFT, sigma0_Scat_PP_EFT,
                          sigma0_Scat_VV_EFT, sigma0_Scat_VA_EFT, sigma0_Scat_AV_EFT, sigma0_Scat_AA_EFT,
                          sigma0_Scat_RR_EFT, sigma0_Scat_LV_EFT, sigma0_Scat_TT_EFT, sigma0_Scat_TAT_EFT]) #use symmetry

# ready-to-use cross-sections for trapping processes
def sigmaTR_t(mL, mChi, T, y1, iSigma, mZp=1, gL=1, gChi=1, oneFermion=False, limit="full", Lambda=1):
    xL = mL/T
    xChi = mChi/T
    xZp = mZp/T
    if limit=="full":
        sigma=sigma0_Scat[iSigma](xL**2, xChi**2, xZp**2, y1) *gL**2*gChi**2 /T**2 #factor T^2 for correct dimensions
    elif limit=="eft":
        sigma=sigma0_Scat_EFT[iSigma](xL**2, xChi**2, y1) /Lambda**4 * T**2
    mask = np.array(y1 < (xL + xChi)**2)
    sigma[mask]=0.
    return sigma

def sigmaTR_s(mL, mChi, T, y1, iSigma, **kwargs):
    sigma = sigmaFS(mL, mChi, T, y1, iSigma, **kwargs) #(mL, mChi) -> (mChi, mL)
    sigma *= (y1 -4*(mL/T)**2)/(y1-4*(mChi/T)**2) #just the different fluxes, everything else is the same
    return sigma

def sigmaTR_DMself_s(mChi, T, y1, iSigma, **kwargs):
    sigma = sigmaFS(mChi, mChi, T, y1, iSigma, **kwargs) #(mL, mChi) -> (mChi, mChi) (xChi and xChi flux are the same)
    return sigma

def sigmaTR_DMself_t(mChi, T, y1, iSigma, gChi=1, gL=1, mZp=1, **kwargs):
    sigma = sigmaTR_t(mChi, mChi, T, y1, iSigma, mZp=mZp, gChi=gChi, gL=gChi, **kwargs)
    return sigma

# resonant Compton processes
def sigmaFS_Compton_resonant(shat, mZp, mL, mChi, gChi, gL, oneFermion=False, **kwargs):
    xP = mZp**2 / mL**2
    fac1 = np.pi * helper.alphaEM * (gChi**2/4/np.pi) / mL**2
    fac2 = 1/shat**2 / (shat-1)**3
    GammaZpChi = helper.gamZp_i(mZp, mChi, gChi)
    GammaZp = helper.GamZpOne(mZp, mL, mChi, gL, gChi) if oneFermion else helper.GamZpMuTau(mZp, mL, mChi, gL, gChi)
    BRchi = GammaZpChi / GammaZp
    R = (shat**2 - 2*shat*(xP+1) + (xP-1)**2)**.5
    fac4 = (shat*(shat*(shat+7*xP+15) + 2*xP-1) -xP +1)*R
    fac5 = 4*shat**2*(shat**2-2*shat*(xP+3)+2*xP*(xP+1)-3)*np.arctanh(R/(shat-xP+1))
    return fac1 * fac2 * BRchi * (fac4 + fac5)
