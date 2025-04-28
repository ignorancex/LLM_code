#===========================StatisticFuncs.py===================================#
# Created by Bing-Long Zhang, 2023

# Contains functions for performing calculations of the distinction plot

#==============================================================================#
# import
import numpy as np
from scipy.optimize import differential_evolution
from numba import jit, float64
import time, multiprocessing, pickle
from U1Funcs import NuRates, dRdEU1FI, dRdEU1KineticMxing, DMvRatesGen, DMvRatesAMGen
from WIMPFuncs import C_SI
from Params import *
import matplotlib.pyplot as plt
from scipy import stats, interpolate
from joblib import Parallel, delayed
from tqdm import tqdm
#============================== log-likelihood ============================================#
@jit(nopython=True)
def globalFunc(datObs, RNuSM, RWIMP, muDM, RNuDiff, muNP, nuiList, b0, Uncs, exposure):
    temp = nuiList@RNuSM + (nuiList@RNuDiff)*muNP + RWIMP*muDM + b0
    return - exposure*np.sum(datObs*(np.log(exposure)+np.log(temp))-temp) + 0.5*np.sum(((1-nuiList)/Uncs)**2)

#==========================================================================================#
#====================== obtain parameters of the distribution =============================#
class StatisticClass(object):
    
    def __init__(self, SpectrumGenClass, modeHypothesis):
        self._Spectrum = SpectrumGenClass # get the information of spectra
        self._mode = modeHypothesis
        
        self._dict = dict(zip(['SMvsDMNP', 'DMvsNP'], [0,1])) # set the statistic scenario
        SMvsDMNP_thetaPrimeList = np.array([np.array([0.,0.]),np.array([1.,1.])])
        DMvsNP_thetaPrimeList = np.array([np.array([0.,1.]),np.array([1.,0.])])
        if self._mode == 'SMvsDMNP':
            self._thetaPrimeList = SMvsDMNP_thetaPrimeList
        elif self._mode == 'DMvsNP':
            self._thetaPrimeList = DMvsNP_thetaPrimeList
        return None
    
    def setModel(self, modelPara):
        return self._Spectrum.setModel(modelPara)
    
    def setDMMass(self, mDM):
        return self._Spectrum.setDMMass(mDM)
    
    def setMode(self, modeHypothesis):
        self._mode = modeHypothesis
    
    def plotSpectrum(self, DMMass, modelPara, sigma0, yrange=[1e-4,1e2]):
        return self._Spectrum.plotSpectrum(DMMass, modelPara, sigma0, yrange)
    
#========================== distinction ====================================================#
    def vsParaGen(self, exposure, i):
        # generate the parameters with some prepared materials
        # output: mu: mean, sig: standard derivation
        # ldd: matrix F, HMat: matrix H, varMat: matrix V
        # Fdelta: F.delta, delta: thetaPrime-muList, muList: theta0 or theta1
        thetaPrime = self._thetaPrimeList[i]
        lddTemp, varGauTerm = self._vsTempMatList[i]
        
        muList1, muList2 = self._thetaPrimeList
        
        ldd = exposure*lddTemp + np.diag(np.append([0.,0.], 1/(self._Spectrum._NuUnc**2)))
        varMat = lddTemp + exposure*varGauTerm
        G3 = ldd.copy()[2:,2:]
        nb = len(self._Spectrum._NuUnc)
        HMat = np.block([[np.zeros((2,2)), np.zeros((2,nb))],[np.zeros((nb,2)), np.linalg.inv(G3)]])
        
        lddInv = np.linalg.inv(ldd)
        varMatOneHalf = np.sqrt(exposure)*matOneHalfGen(varMat)
        Fdelta1 = ldd@np.append(thetaPrime-muList1, np.zeros(nb))
        Fdelta2 = ldd@np.append(thetaPrime-muList2, np.zeros(nb))
        
        sig = 2*((Fdelta1-Fdelta2)@(lddInv-HMat)@varMatOneHalf)
        mu = Fdelta1@(lddInv-HMat)@Fdelta1 - Fdelta2@(lddInv-HMat)@Fdelta2
        return [mu, np.sqrt(np.sum(sig*sig))]
    
    def CL(self, sigma0, exposure, b0Level, modeAsimov=1):
        # generate C.L.
        temp = [self.vsParaGen(exposure, i) for i in range(2)]
        if modeAsimov == 0:
            [[mu1, sig1], [mu2, sig2]] = temp
        else:
            temp2 = [self.vsAsimovGen(sigma0, exposure, b0Level, self._thetaPrimeList[i]) for i in range(2)]
            mu1, mu2 = temp2
            sig1, sig2 = temp[0][1], temp[1][1]
        return CLGen(mu1, mu2, sig1, sig2)
    
    def vsTempMatSet(self, sigma0, b0Level):
        # generate useful materials for obtaining parameters of distributions with fixed sigma0 and b0Level
        RNuSM, NuUncs, RNuNP = self._Spectrum._RNuSM, self._Spectrum._NuUnc, self._Spectrum._RNuNP
        RNuDiff = RNuNP-RNuSM
        RWIMP = self._Spectrum._RWIMP*10**(sigma0+45.)
        b0 = self._Spectrum._b0Flat*b0Level
        
        s1, s2, b, sigmaTheta = RWIMP, RNuDiff, RNuSM, NuUncs
        self._vsTempMatList = [vsMatGen(self._thetaPrimeList[i], s1, s2, b, sigmaTheta, b0) for i in range(2)]
        return 

    def vsAsimovGen(self, sigma0, exposure, b0Level, thetaPrime):
        # generate the mean by Asimov dataset
        RNuSM, NuUncs, RNuNP = self._Spectrum._RNuSM, self._Spectrum._NuUnc, self._Spectrum._RNuNP
        RNuDiff = RNuNP-RNuSM
        RWIMP = self._Spectrum._RWIMP*10**(sigma0+45.)
        b0 = self._Spectrum._b0Flat*b0Level
        
        datObs = thetaPrime[0]*RWIMP + thetaPrime[1]*np.sum(RNuDiff, axis=0) \
        + np.sum(RNuSM, axis=0) + b0
       
        mean1 = self._AsimovGen(datObs, RNuSM, RNuDiff, RWIMP, b0, NuUncs, exposure)
        return mean1
    
    def _AsimovGen(self, datObs, RNuSM, RNuDiff, RWIMP, b0, NuUncs, exposure):
        n = np.shape(NuUncs)[0]
        # To accelerate the minimum searching, memory assignment of Mat needs to be changed
        RNuSM = np.transpose(np.ascontiguousarray(np.transpose(RNuSM)))
        RNuDiff = np.transpose(np.ascontiguousarray(np.transpose(RNuDiff)))
        
        thetaPrimeList = self._thetaPrimeList
        H0Gen = lambda x: globalFunc(datObs, RNuSM, RWIMP, thetaPrimeList[0][0], RNuDiff, thetaPrimeList[0][1], x, b0, NuUncs, exposure)
        H1Gen = lambda x: globalFunc(datObs, RNuSM, RWIMP, thetaPrimeList[1][0], RNuDiff, thetaPrimeList[1][1], x, b0, NuUncs, exposure)

        boundsH0 = np.transpose([np.zeros(n)+1.0e-5, np.ones(n)*3.])
        likeNume = differential_evolution(H0Gen, boundsH0, tol=0, atol=0.01)
        likeDeno = differential_evolution(H1Gen, boundsH0, tol=0, atol=0.01)
        return 2*(likeNume.fun-likeDeno.fun)
    
#==========================DM/NP vs SM=========================================#
    def NPDiscoveryParaGen(self, exposure, b0Level):
        # WARNING: this function is not completed!!!!!!!
        RNuSM, NuUncs, RNuNP = self._Spectrum._RNuSM, self._Spectrum._NuUnc, self._Spectrum._RNuNP
        RNuDiff = RNuNP-RNuSM
        sigma0= -45.
        RWIMP = self._Spectrum._RWIMP*10**(sigma0+45.)
        b0 = self._Spectrum._b0Flat*b0Level
        
        s1, s2, b, sigmaTheta = RWIMP, RNuDiff, RNuSM, NuUncs
        try:
            res = phiGen2(np.sum(s2,axis=0), b, sigmaTheta, exposure, b0)
        except Expectation:
            raise ValueError('phiGen2 fails.')
        return res
    
    def DMDiscoveryParaGen(self, sigma0, exposure, b0Level):
        RNuSM, NuUncs, RNuNP = self._Spectrum._RNuSM, self._Spectrum._NuUnc, self._Spectrum._RNuNP
        RNuDiff = RNuNP-RNuSM
        RWIMP = self._Spectrum._RWIMP*10**(sigma0+45.)
        b0 = self._Spectrum._b0Flat*b0Level
        
        s1, s2, b, sigmaTheta = RWIMP, RNuDiff, RNuSM, NuUncs
        try:
            res = phiGen2(s1, b, sigmaTheta, exposure, b0)
        except Expectation:
            raise ValueError('phiGen2 fails.')
        return res
    
# Quasi-Asimov Dataset Method, like Asimov dataset but faster, see 2304.13665 for deteails
@jit([(float64[:], float64[:,:], float64[:], float64, float64[:])], nopython=True)
def phiGen2(s, b, sigmaTheta, exposure, b0):
    n = len(b) + 1
    sb = np.zeros((n,len(s)))
    sb[0] = s
    sb[1:] = b
    temp = np.sum(sb, axis=0)
    ijList = [[exposure*np.sum(sb[i]*sb[j]/temp) for j in range(i+1, n)] for i in range(n-1)]
    ldd = np.zeros((n,n))
    for i in range(0, n-1):
        ldd[i,i+1:] = ijList[i]
        ldd[i+1:, i] = ijList[i]
    diagTerm = np.append([0.],1/(sigmaTheta*sigmaTheta)) + exposure*np.array([np.sum(sb[i]*sb[i]/temp) for i in range(n)])
    ldd = ldd+np.diag(diagTerm)
    HMat = np.zeros((n,n))
    HMat[1:,1:] = np.linalg.inv(ldd[1:,1:])
    niuList = (-(np.append([1.],np.zeros(n-1)) - HMat@ldd@np.append([1.],np.zeros(n-1))))[1:] + 1.
    datObs = s + np.sum(b,axis=0)
    temp = np.sum(np.transpose(b)*niuList,axis=1)
    phi = 2*(-exposure*np.sum(datObs*(np.log(temp)-np.log(datObs))-(temp-datObs))+ 0.5*np.sum(((1-niuList)/sigmaTheta)**2))
    return phi

#@jit([(float64[:,:], float64[:,:], float64[:], float64, float64[:])], nopython=True)
def phiGenNP(s2, b, sigmaTheta, exposure, b0):
    s2Total = np.sum(s2,axis=0)
    sigmaTheta2 = sigmaTheta**2
    n = len(b) + 1
    nBin = len(s2[0])
    thetaNP = 1
    
    deriList = np.zeros((n,nBin))
    deriList[0] = s2Total
    deriList[1:] = thetaNP*s2+b
    deriNuListT = np.transpose(deriList[1:])
    
    vList = thetaNP*s2Total + np.sum(b, axis=0) + b0
    vInvList = 1/vList
    
    myMat = np.zeros((n,n))
    for i in range(0, n-1):
        tempIJ = np.array([np.sum(deriList[i]*deriList[j]*vInvList) for j in range(i+1,n)])
        myMat[i, i+1:] = tempIJ
        myMat[i+1:, i] = tempIJ
    diagTerms = np.array([np.sum(deriList[i]*deriList[i]*vInvList) for i in range(0,n)])
    myMat = myMat + np.diag(diagTerms)
    ldd = myMat
    
    HMat = np.zeros((n,n))
    HMat[1:,1:] = np.linalg.inv(ldd[1:,1:])
    niuList = (-(np.append([1.],np.zeros(n-1)) - HMat@ldd@np.append([1.],np.zeros(n-1))))[1:] + 1.
    
    datObs = thetaNP*s2Total + np.sum(b,axis=0)
    temp = np.sum(np.transpose(b)*niuList,axis=1)
    phi = 2*(-exposure*np.sum(datObs*(np.log(temp)-np.log(datObs))-(temp-datObs))+ 0.5*np.sum(((1-niuList)/sigmaTheta)**2))
    return phi

def matOneHalfGen(mat):
    try:
        va, ve = np.linalg.eigh(mat)
    except Exception as e:
        print('matOneHalfGen: ',e)
        raise ValueError('Failed to inverse a matrix.')
    if np.any(va<0.):
        raise ValueError('Failed to inverse a matrix.')
    return ve@np.diag(np.sqrt(va))@np.transpose(ve)

@jit([(float64[:], float64[:], float64[:,:], float64[:,:], float64[:], float64[:])], nopython=True)
def vsMatGen(thetaPrime, s1, s2, b, sigmaTheta, b0):
    # critical code for calculating temp mat
    s2Total = np.sum(s2,axis=0)
    sigmaTheta2 = sigmaTheta**2
    n = len(b) + 2
    nBin = len(s1)
    
    deriList = np.zeros((n,nBin))
    deriList[0] = s1
    deriList[1] = s2Total
    deriList[2:] = thetaPrime[1]*s2+b
    deriNuListT = np.transpose(deriList[2:])
    
    vList = thetaPrime[0]*s1 + thetaPrime[1]*s2Total + np.sum(b, axis=0) + b0
    vInvList = 1/vList
    
    myMat = np.zeros((n,n))
    for i in range(0, n-1):
        tempIJ = np.array([np.sum(deriList[i]*deriList[j]*vInvList) for j in range(i+1,n)])
        myMat[i, i+1:] = tempIJ
        myMat[i+1:, i] = tempIJ
    diagTerms = np.array([np.sum(deriList[i]*deriList[i]*vInvList) for i in range(0,n)])
    myMat = myMat + np.diag(diagTerms)
    lddTemp = myMat
    
    
    ijTerms = [[np.sum(deriNuListT[i]*deriNuListT[j]*sigmaTheta2) for j in range(i+1,nBin)] for i in range(0,nBin-1)]
    diagTerms = [np.sum(deriNuListT[i]*deriNuListT[i]*sigmaTheta2) for i in range(0,nBin)]
    myMat = np.zeros((nBin,nBin))
    for i in range(0, nBin-1):
        tempIJ = np.array([np.sum(deriNuListT[i]*deriNuListT[j]*sigmaTheta2) for j in range(i+1,nBin)])
        myMat[i, i+1:] = tempIJ
        myMat[i+1:, i] = tempIJ
    diagTerms = np.array([np.sum(deriNuListT[i]*deriNuListT[i]*sigmaTheta2) for i in range(0,nBin)])
    myMat = myMat + np.diag(diagTerms)
    
    varGauTermConst = np.outer(vInvList, vInvList)*myMat
    varGauTermIJ = [[np.sum(np.outer(deriList[i], deriList[j])*varGauTermConst)\
                             for j in range(i+1,n)] for i in range(0,n-1)]
    varGauTermDiag = [np.sum(np.outer(deriList[i], deriList[i])*varGauTermConst)\
                             for i in range(0,n)]
    
    myMat = np.zeros((n,n))
    for i in range(0, n-1):
        tempIJ = np.array([np.sum(np.outer(deriList[i], deriList[j])*varGauTermConst) for j in range(i+1,n)])
        myMat[i, i+1:] = tempIJ
        myMat[i+1:, i] = tempIJ
    diagTerms = np.array([np.sum(np.outer(deriList[i], deriList[i])*varGauTermConst) for i in range(0,n)])
    myMat = myMat + np.diag(diagTerms)
    varGauTerm = myMat
    return [lddTemp, varGauTerm]

#==========================================================================================#
#========================== useful functions =========================================#
def CLGen(mu1, mu2, sig1, sig2):
    # generate CL from distributions
    # mu1 should be less than mu2
    qobs = mu1+1.28*sig1
    #qobs = mu1+3*sig1
    return 1-stats.norm.cdf(qobs,loc=mu2,scale=sig2)

def biSearch(step0, n, func, paraSet, para0, flag):
    # find the critical point quickly for mono-functions
    [temp,para] = [step0,para0]
    tempRes = func(paraSet, para)
    recordPara = np.array([para])
    recordRes = np.array([tempRes[1]])
    if tempRes[0]==flag:
        [sign, label] = [1,0]
    else:
        [sign, label] = [-1,1]
    para = para+sign*temp;
    boundPara = para0+sign*100*step0
    
    for i in range(0,n):
        for j in range(0,100):
            if sign > 0 and (para>boundPara or abs(para-boundPara)<1e-5):
                para = para - 0.5*sign*temp
                break
            elif sign < 0 and (para<boundPara or abs(para-boundPara)<1e-5):
                para = para - 0.5*sign*temp
                break
            tempRes = func(paraSet, para)
            recordPara = np.append(recordPara, para)
            recordRes = np.append(recordRes, tempRes[1])
            if tempRes[0] != label:
                boundPara = para
                para = para - 0.5*sign*temp
                break
            para = para + sign*temp
        temp = 0.5*temp
    return [para, temp, np.array([recordPara,recordRes])]

def myFindRoot(dat):
    # obstain the root based on the last two elements
    x1, x2, y1, y2 = dat[0,-2], dat[0,-1], dat[1,-2], dat[1,-1]
    a = (y2-y1)/(x2-x1)
    b = (y1*x2-y2*x1)/(x2-x1)
    if a==0:
        res = (x1+x2)/2
    else:
        res = -b/a
    return res
#==========================================================================================#
#============= obtain the distinction or discovery limit sigma and exposure  ==============#
class Fog(object):
    
    def __init__(self):
        return
    
    def _expoVSCL(self, paraSet, expoLog):
        CL, b0Level, sigma0Log = paraSet
        temp = (self._CLGen(sigma0Log, 10**expoLog, b0Level, modeAsimov=1)-CL)
        if temp<0:
            return [0, temp]
        else:
            return [1, temp]
    
    def sigmaExpoListVSGen(self, paraGenClass, sigmaList, CL, b0Level, expoInit):
        # searching distiction limit cross section from large to small
        self._CLGen = paraGenClass.CL
            
        step0 = 0.1*0.5
        paraGenClass.vsTempMatSet(sigmaList[0], b0Level)
        [expoLog, temp, recordDat] = biSearch(step0, 4, self._expoVSCL, [CL, b0Level, sigmaList[0]], expoInit, 0)
        recordDatList = [recordDat]
        expoList = np.array([expoLog])
        iEnd = len(sigmaList)
        for i in range(1, len(sigmaList)):
            if expoLog<-10.:
                iEnd = i
                break
            paraGenClass.vsTempMatSet(sigmaList[i], b0Level)
            [expoLog, temp, recordDat] = biSearch(step0, 4, self._expoVSCL, [CL, b0Level, sigmaList[i]], expoLog, 0)
            expoList = np.append(expoList, expoLog)
            recordDatList.append(recordDat)
        def monoCheck(dat):
            d1, d2 = dat
            sortIndex = np.argsort(d1)
            d2 = d2[sortIndex]
            if np.all(np.diff(d2)>0) or np.all(np.diff(d2)<0):
                return True
            else:
                return False
        if not (np.all([monoCheck(dat) for dat in recordDatList])):
            print("No mono")
        return [sigmaList[:iEnd], expoList, recordDatList]
    
    def _expoDMCL(self, paraSet, expoLog):
        CL, b0Level, sigma0Log = paraSet
        #print([sigma0Log,expoLog])
        temp=self._CLGen(sigma0Log, 10**expoLog, b0Level)
        # (1-stats.ncx2.cdf(7.74,df=1,nc=temp[1]) achieve 3 sigma sensitivity
        temp = ((1-stats.ncx2.cdf(7.74,df=1,nc=temp))-CL)
        if temp<0:
            return [0, temp]
        else:
            return [1, temp]
        
    def sigma0InitGen(self, paraGenClass, expoLog, CL, b0Level, stopSigma=-30):
        self._CLGen = paraGenClass.DMDiscoveryParaGen
        def tempFunc(paraSet, sigma):
            return self._expoDMCL([paraSet[0], b0Level, sigma], expoLog)
        [sigma0, temp, recordDat] = biSearch(1., 4, tempFunc, [CL], -47., 0)
        #print(recordDat)
        if sigma0>stopSigma:
            raise ValueError("Wrong DM Mass Input")
        return sigma0
    
    def sigmaExpoListDMDiscoveryGen(self, paraGenClass, exposureEndLog, dExpoLog, CL, b0Level, exposureInitLog=-1.):
        # searching discovery limit cross section from large to small
        try:
            sigma0 = self.sigma0InitGen(paraGenClass, exposureInitLog, CL, b0Level)
        except ValueError:
            return 1
        
        step0 = 0.5**3
        [expoLog0, temp, recordDat] = biSearch(1., 6, self._expoDMCL, [CL, b0Level,sigma0], exposureInitLog, 0)
        recordDatList = [recordDat]
        expoList = np.array([expoLog0])
        sigmaList = np.array([sigma0])
        
        sigmaNext = sigma0-0.05
        [expoLog, temp, recordDat] = biSearch(step0, 3, self._expoDMCL, [CL, b0Level,sigmaNext], expoLog0, 0)
        expoList = np.append(expoList, expoLog)
        sigmaList = np.append(sigmaList, sigmaNext)
        recordDatList.append(recordDat)
        
        # use the gradient to predict the next discovery cross section with exposure increase of 0.1
        n = 200
        for i in range(2, n):
            #print([sigma0,sigmaNext,(sigmaNext-sigma0)/(expoLog-expoLog0)])
            sigma0Temp = sigmaNext
            sigmaNext = sigma0Temp+(sigmaNext-sigma0)/(expoLog-expoLog0)*dExpoLog
            
            temp = self._expoDMCL([CL, b0Level,sigmaNext], expoLog+dExpoLog)[0]
            
            if temp < 0:
                sigmaNext = sigma0Temp+(sigmaNext-sigma0Temp)/4
            sigma0 = sigma0Temp
            expoLog0 = expoLog
            [expoLog, temp, recordDat] = biSearch(step0, 3, self._expoDMCL, [CL, b0Level,sigmaNext], expoLog+dExpoLog, 0)
                
            if (sigmaNext>sigma0) or (expoLog<expoLog0) or (expoLog>exposureEndLog):
                break
            
            if check(recordDat):
                expoList = np.append(expoList, expoLog)
                sigmaList = np.append(sigmaList, sigmaNext)
                recordDatList.append(recordDat)
            else:
                break
            if expoLog>exposureEndLog:
                break
        return [sigmaList, expoList, recordDatList]
    
def check(dat):
    return np.any(dat>0.)&np.any(dat<0.)

#=================================================================================#
#========================== useful functions =====================================#
def findNPDiscoveryExposure(paraSet, expoLog):
    paraGenClass, CL, b0Level = paraSet
    temp=paraGenClass.NPDiscoveryParaGen(10**expoLog, b0Level)
    # (1-stats.ncx2.cdf(7.74,df=1,nc=temp[1]) achieve 3 sigma sensitivity
    temp = ((1-stats.ncx2.cdf(7.74,df=1,nc=temp))-CL)
    if temp<0:
        return [0, temp]
    else:
        return [1, temp]

# generate C.L. for some parameters
def vsCLGen(paraGenClass, sigma0List, exposureListLog, b0Level):
    CLList = []
    exposureList = 10**(exposureListLog)
    iEnd = len(exposureList)
    for i in range(len(sigma0List)):
        try:
            paraGenClass.vsTempMatSet(sigma0List[i], b0Level)
            temp = paraGenClass.CL(sigma0List[i], exposureList[i], b0Level,0)
            CLList.append(temp)
        except ValueError as e:
                #print(exposureListLog[i])
                iEnd = i
                break
    return np.array([exposureListLog[:iEnd],CLList])

def subFindroot(xnew,x,y):
    sortIndices = np.argsort(x)
    x = x[sortIndices]
    y = y[sortIndices]
    return np.interp(xnew,x,y)

# generate the bounds of exposure for n>1
def vsExpoGen(mDM, dat, CL):
    x, y = dat[0], dat[1]-CL
    transitionIndices = np.where(np.diff(np.sign(y))!=0)[0]
    if len(transitionIndices)==2:
        res = [subFindroot(0.,y[i:i+2],x[i:i+2]) for i in transitionIndices]
        return res, transitionIndices

    if len(transitionIndices)==3:
        #print("Numerical warning in myMul3 len>2: ", mDM, dat)
        transitionIndices = transitionIndices[-2:]
        res = [subFindroot(0.,y[i:i+2],x[i:i+2]) for i in transitionIndices]
        return res, transitionIndices
    elif len(transitionIndices)==1:
        return [], 1
    elif len(transitionIndices)==0:
        return [], 1
    else:
        print("? in vsExpoGen2")

# main function for generating data
def myMul(mDM, paraGenClass, FogGenClass, paraList):
    exposureEndLog, dExpoLog, CL, b0Level, NPDiscoveryExposureLog = paraList
    paraGenClass.setDMMass(mDM)
    
    exposureInitLog=np.log10(0.1/np.sum(paraGenClass._Spectrum._RNuSM))
    res1 = FogGenClass.sigmaExpoListDMDiscoveryGen(paraGenClass, exposureEndLog, dExpoLog, CL, b0Level, exposureInitLog=exposureInitLog)
    if not isinstance(res1,list):
        return 1
    exposureLogList, sigma0LogList = np.array([np.array([myFindRoot(d) for d in res1[2]]),res1[0]])
    if np.any(np.isnan(np.array([exposureLogList,sigma0LogList]))):
        return 1
    
    res3 = vsCLGen(paraGenClass, sigma0LogList, exposureLogList, 0.)
    
    NPDiscoverySigmaLog = subFindroot(NPDiscoveryExposureLog,exposureLogList,sigma0LogList)
    
    DistinctionExposureLog, transitionIndices = vsExpoGen(mDM, res3, CL)
    if len(DistinctionExposureLog)==0:
        DistinctionDat = [sigma0LogList, exposureLogList]
        return [NPDiscoverySigmaLog, DistinctionDat]
    
    DistinctionSigmaLog = subFindroot(DistinctionExposureLog,exposureLogList,sigma0LogList)
    #print(mDM, DistinctionExposureLog, transitionIndices)
    selIndices = np.zeros_like(sigma0LogList,dtype=bool)
    selIndices[transitionIndices[0]-2:transitionIndices[1]+4] = True
    
    vsSigma0LogList = sigma0LogList[selIndices]
    vsExpoInitLog = exposureLogList[selIndices][0]
    res4 = FogGenClass.sigmaExpoListVSGen(paraGenClass, vsSigma0LogList, CL, b0Level, vsExpoInitLog)
    vsExposureLogList, vsSigma0LogList = np.array([np.array([myFindRoot(d) for d in res4[2]]),res4[0]])
    
    y = 10**(vsExposureLogList)/(10**(exposureLogList[selIndices]))
    DistinctionDat = [sigma0LogList, exposureLogList, \
                    vsSigma0LogList, vsExposureLogList, y, DistinctionSigmaLog]
    return [NPDiscoverySigmaLog, DistinctionDat]

def resGen(mDMList, paraGenClass, FogGenClass, paraList, n_jobs=20):
    time_start = time.perf_counter()

    # Define the function to be parallelized
    def parallel_function(mDM, paraGenClass, FogGenClass, paraList):
        return myMul(mDM, paraGenClass, FogGenClass, paraList)

    # Use joblib to parallelize the computation and visualize the progress with tqdm
    res1 = Parallel(n_jobs=n_jobs)(delayed(parallel_function)(mDM, paraGenClass, FogGenClass, paraList) for mDM in tqdm(mDMList))

    time_end = time.perf_counter()
    print("Time costed: {0} s.".format(time_end-time_start))
    
    resLabel = np.where(np.array([1 if isinstance(item, list) else 0 for item in res1])==1)
    mDMList = mDMList[resLabel]
    res1 = [item for item in res1 if isinstance(item, list)]
    return [mDMList, res1]

# def resGen(mDMList, paraGenClass, FogGenClass, paraList):
#     time_start = time.perf_counter()
#     pool = multiprocessing.Pool(4)
#     multiple_results = [pool.apply_async(\
#             myMul, [mDMList[i],paraGenClass, FogGenClass, paraList]) for i in range(len(mDMList))]
#     pool.close()
#     pool.join()
#     res1= [res.get() for res in multiple_results]
#     time_end = time.perf_counter()
#     print("Time costed: {0} s.".format(time_end-time_start))
    
#     resLabel = np.where(np.array([1 if isinstance(item, list) else 0 for item in res1])==1)
#     mDMList = mDMList[resLabel]
#     res1 = [item for item in res1 if isinstance(item, list)]
#     return [mDMList, res1]
    
#=========================================================================#
#========================== spectras =====================================#
# NR means nuclear recoil for DM-nucleon interaction
# ER means eletron recoil for DM-nucleon interaction

class SpectrumNR(object):
    
    def __init__(self, mDM, E_th, E_max, ne, Nuc,modelPara,selList,mode='FI'):
        nuGen = NuRates(E_th,E_max,ne,Nuc)
        RNuSM, NuUnc = nuGen.vRatesSM(selList)
        if mode=='FI':
            RNuNP = nuGen.vRatesFIGen(dRdEU1FI, modelPara, selList)
        elif mode=='FD':
            RNuNP = nuGen.vRatesFDGen(dRdEU1KineticMxing, modelPara, selList)
        else:
            raise ValueError('Error in SpectrumNR init')
    
        RWIMP = DMvRatesGen(np.array([mDM]), E_th, E_max, ne, Nuc, C_SI)[1][0]
        # if len(selList) == 0:
        #     selList = np.linspace(0,len(NuUnc)-1,len(NuUnc),dtype=int)
        self._selList = selList
        
        iEnd = np.max([len(x[x>1.0e-10]) for x in RNuSM])
        self._RNuSM, self._NuUnc = RNuSM[:,:iEnd], NuUnc
        self._RNuNP = RNuNP[:,:iEnd]
        self._RWIMP = RWIMP[:iEnd]
        
        self._b0Flat = np.ones(ne)*np.sum(self._RNuSM)/ne
        #self._thetaPrimeList = np.array([[0.,1.], [1.,0.]])
        self._temp = [E_th, E_max, ne, Nuc, C_SI, iEnd]
        
        self._nuGen = nuGen
        self._mode = mode
        return None
    
    def setModel(self, modelPara):
        if self._mode=='FI':
            RNuNP = self._nuGen.vRatesFIGen(dRdEU1FI, modelPara, self._selList)
        elif self._mode=='FD':
            RNuNP = self._nuGen.vRatesFDGen(dRdEU1KineticMxing, modelPara, self._selList)
        else:
            raise ValueError('Error in SpectrumNR setModel')
        iEnd = self._temp[-1]
        self._RNuNP = RNuNP[:,:iEnd]
        return
    
    def setDMMass(self, mDM):
        E_th, E_max, ne, Nuc, C_SI, iEnd = self._temp
        RWIMP = DMvRatesGen(np.array([mDM]), E_th, E_max, ne, Nuc, C_SI)[1][0]
        self._RWIMP = RWIMP[:iEnd]
        return 
    
    def plotSpectrum(self, DMMass, modelPara, sigma0, yrange=[1e-4,1e2]):
        self.setDMMass(DMMass)
        self.setModel(modelPara)
        E_th, E_max, ne, Nuc, C_SI, iEnd = self._temp
        RNuSM, RNuNP, RWIMP = self._RNuSM, self._RNuNP, self._RWIMP*10**(45.+sigma0)
        RNuSMTotal, RNuNPTotal = np.sum(RNuSM, axis=0), np.sum(RNuNP, axis=0)
        return plotSpectrumFunc(RNuSMTotal, RNuNPTotal, RWIMP, E_th, E_max)


class SpectrumER(object):
    
    def __init__(self, mDMList, DMSpectrum, neERFunc, E_th, E_max, ne, Nuc, modelPara, selList, thresholdBin=1,mode='FI'):
        nuGen = NuRates(E_th,E_max,ne,Nuc)
        RNuSM = nuGen.vRatesFIGenLow(dRdEU1FI, [], neERFunc, selList)
        NuUnc = nuGen.vRatesSM(selList)[1]  
        # if len(selList) == 0:
        #     selList = np.linspace(0,len(NuUnc)-1,len(NuUnc),dtype=int)
        self._selList = selList
        
        if mode=='FI':
            RNuNP = nuGen.vRatesFIGenLow(dRdEU1FI, modelPara, neERFunc, selList)
        elif mode=='FD':
            RNuNP = nuGen.vRatesFDGenLow(dRdEU1KineticMxing, modelPara, neERFunc, selList)
        else:
            raise ValueError('Error in SpectrumNR init')
            
        self._RNuSM, self._NuUnc = RNuSM, NuUnc
        self._RNuNP = RNuNP
        self._mDMList, self._DMSpectrum = mDMList, DMSpectrum[:,thresholdBin-1:]*10**(-45)
        
        #self._thetaPrimeList = np.array([[0.,1.], [1.,0.]])
        self._nuGen = nuGen
        self._mode = mode
        self._neERFunc = neERFunc
        self._temp = [E_th, E_max, ne, Nuc]
        
        self._RNuSM = self._RNuSM[:, thresholdBin-1:]
        self._RNuNP = self._RNuNP[:, thresholdBin-1:]
        self._thresholdBin = thresholdBin
        temp = np.sum(self._RNuSM,axis=0)
        self._b0Flat = (np.ones(len(temp))*np.sum(temp)/(10-thresholdBin+1))
        
        return None
    
    def setModel(self, modelPara):
        if self._mode=='FI':
            RNuNP = self._nuGen.vRatesFIGenLow(dRdEU1FI, modelPara, self._neERFunc, self._selList)
        elif self._mode=='FD':
            RNuNP = self._nuGen.vRatesFDGenLow(dRdEU1KineticMxing, modelPara, self._neERFunc, self._selList)
        else:
            raise ValueError('Error in SpectrumNR init')
        self._RNuNP = RNuNP[:, self._thresholdBin-1:]
        return
    
    def setDMMass(self, mDM):
        mi = np.argmin(np.abs(self._mDMList-mDM))
        self._RWIMP = self._DMSpectrum[mi]
        return 
    
    def plotSpectrum(self, DMMass, modelPara, sigma0, yrange=[1e-4,1e2]):
        self.setDMMass(DMMass)
        self.setModel(modelPara)
        E_th, E_max, ne, Nuc = self._temp
        RNuSM, RNuNP, RWIMP = self._RNuSM, self._RNuNP, self._RWIMP*10**(45+sigma0)
        RNuSMTotal, RNuNPTotal = np.sum(RNuSM, axis=0), np.sum(RNuNP, axis=0)
        return plotSpectrumFunc2(RNuSMTotal, RNuNPTotal, RWIMP)


def plotSpectrumFunc(RNuSMTotal, RNuNPTotal, RWIMP, xMin, xMax):
    fig, ax = plt.subplots(figsize=(6,6))
    nBins = len(RNuSMTotal)
    StatX = np.linspace(xMin,xMax,nBins+1)
    DatX = np.array(list(map(lambda x: [x,x], StatX[1:-1]))).flatten()
    DatX = np.append([StatX[0]], np.append(DatX, [StatX[-1]]))
    def datYGen(dat):
        StatY = dat
        DatY = np.array(list(map(lambda x: [x,x], StatY))).flatten()
        return DatY
    
    DatX = DatX*1e6
    ax.plot(DatX, datYGen(RNuSMTotal), color='r', linestyle='-', linewidth=2,label='v: SM')
    ax.plot(DatX, datYGen(RWIMP), color='black', linestyle='-.',linewidth=2, label='DM')
    ax.plot(DatX, datYGen(RNuNPTotal), color='orange', linestyle='--',linewidth=2, label='v: $U(1)_{B-L}$')
    ax.plot(DatX, datYGen(RNuNPTotal)-datYGen(RNuSMTotal), color='c', linestyle='--',linewidth=2, label='v: diff')
    #ax.plot(DatX, datYGen(RNuNPTotal-RNuSMTotal), color='c', linestyle='--', label='Diff')
    ax.set_yscale('log')
    
    lfs = 20
    ax.set_xlabel(r"Recoil Energy [keV]",fontsize=lfs)
    ax.set_ylabel(r'Event Rates [ton$\times$year$^{-1}$]',fontsize=lfs)
    
    #ax.set_xlim([xMin-0.5,xMax+0.5])
    ax.set_ylim([1e-4,1e1])

    ax.tick_params(which='major',direction='in',width=2,length=5,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=5)
    
    ax.legend()
    return fig

def plotSpectrumFunc2(RNuSMTotal, RNuNPTotal, RWIMP):
    fig, ax = plt.subplots(figsize=(6,6))
    nBins = len(RNuSMTotal)
    xMin, xMax = 10+0.5-nBins, 10+0.5
    StatX = np.linspace(xMin,xMax,nBins+1)
    DatX = np.array(list(map(lambda x: [x,x], StatX[1:-1]))).flatten()
    DatX = np.append([StatX[0]], np.append(DatX, [StatX[-1]]))
    def datYGen(dat):
        StatY = dat
        DatY = np.array(list(map(lambda x: [x,x], StatY))).flatten()
        return DatY
    
    ax.plot(DatX, datYGen(RNuSMTotal), color='r', linestyle='-', linewidth=2,label='v: SM')
    ax.plot(DatX, datYGen(RWIMP), color='black', linestyle='-.',linewidth=2, label='DM')
    ax.plot(DatX, datYGen(RNuNPTotal), color='orange', linestyle='--',linewidth=2, label='v: $U(1)_{B-L}$')
    ax.plot(DatX, datYGen(RNuNPTotal)-datYGen(RNuSMTotal), color='c', linestyle='--',linewidth=2, label='v: diff')
    #ax.plot(DatX, datYGen(RNuNPTotal-RNuSMTotal), color='c', linestyle='--', label='Diff')
    ax.set_yscale('log')
    
    lfs = 20
    ax.set_xlabel(r"Emitted Electron",fontsize=lfs)
    ax.set_ylabel(r'Event Rates [ton$\times$year$^{-1}$]',fontsize=lfs)
    
    #ax.set_xlim([xMin-0.5,xMax+0.5])
    ax.set_ylim([1e-4,1e1])

    ax.tick_params(which='major',direction='in',width=2,length=5,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=5)
    
    ax.legend()
    return fig
    