#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:01:31 2019

@author: remy.benmessaoud
"""

"""===========================================================================
------------------------ feature Extraction Functions  ---------------------------------
we want to build a function that takes in argument a N * L matrix :
N is the number of channels and L is the length of the temporal window

it returns a vector of length N * Nf where Nf is the number of monovariate features    

List of features in order :
    --------- Stats ------------------
    _ Mean : Average
    _ Crest : Maximum value
    _ Trough : Minimum value
    _ Peak : absolute maximum
    _ Var : Variance
    #_ Std : standard deviation  not necessary
    _ Skw : Skewness
    _ Kurt : Kurtosis 
    _ RMS : Root Mean Square
    _ PAPR : Peak-to-Average Power Ratio = Peak/RMS
    _ FFAct : Form Factor = RMS /Mean   # !!! removed because division by zero removed
    _ totVar : total Variation : normalized sum of succesive differences of voltage 
    _ DFA : Detrended Fluctuation Analysis
    _ HuExp : Hurst Exponent
    _ HMob : Hjorth mobility 
    _ HComp : Hjorth complexity
    
    _ FInfo : Fisher information : which parameters? Tau and EmDim
    -------- Fractal ---------- pyeeg
    _ Tau : decorrelation Time obtained from autocorr(Tau) < 1/e ( index )
    _ EmDim : Embeding dimension given by FFN
    _ PFD : petrosian fractal dimension
    _ HFD : Higuchi’s fractal dimension: Kmax = 6 , 10 see ref 
    _ maximum Lyapunov exponent : nolista
    ------ Entropy ----------- pyeeg
    _ sampEn : sample Entropy
    _ SVDEn : SVD Entropy
    _SEn : Spectral Entropy
    ------- spectral --------- pyeeg
    _ PSI delta (0.5 - 4 Hz)
    _ PSI theta (4 - 8 Hz)
    _ PSI alfa (8 - 13 Hz)
    _ PSI beta2 (13 - 20 Hz)
    _ PSI beta1 (20 - 30 Hz)
    _ PSI gamma (30 - 60 Hz)
    _ RIR delta (0.5 - 4 Hz)
    _ RIR theta (4 - 8 Hz)
    _ RIR alfa (8 - 13 Hz)
    _ RIR beta2 (13 - 20 Hz)
    _ RIR beta1 (20 - 30 Hz)
    _ RIR gamma (30 - 60 Hz)
    ------- Connectivity ---------
    _ Connectivity : get paiwise Pearson's coefficients between the channels
==============================================================================
    nFeatures = 37 monovariate + connectivity 
    Nconnect(n) = n(n-1)/2
    nFeatures(n) = 37 * n + n(n-1)/2
"""
#monoFeaturesDict={0 : "Mean" , 1 : "Crest" , 2 : "Trough" , 3 : "Peak", 4  : "Var" , 5 : "Skw",6 : "Kurt", \
#              7 : "RMS" , 8 : "PAPR" , 9 : "totVar" , 10 : "DFA" , 11 : "HurstExp", 12 : "HMob",\
#              13 : "HComp" , 14 : "FInfo" , 15 : "dCorrTime" , 16 : "EmbDim" , 17 : "PFD" , 18 : "HFD6" , \
#              19 : "HFD10" , 20 : "mLyapExp" , 21 : "sampEn" , 22 : "SVDEn" , 23 : "SpEn" , 24 : "PSI_delta" ,\
#              25 : "RIR_delta" , 26 : "PSI_theta" , 27 : "RIR_theta" , 28 : "PSI_alfa" , 29 : "RIR_alfa" ,\
#              30 : "PSI_beta2RIR_delta" , 31 : "RIR_beta2" , 32 : "PSI_beta1" , 33 : "RIR_beta1" , 34 : "PSI_gamma" ,\
#              35 : "RIR_gamma"}

monoFeaturesDict=["Mean" , "Crest" , "Trough" , "Var" , "Skw",  "Kurt", "DFA" , "HMob" , "HComp" ,\
                  "dCorrTime" , "PFD" , "HFD10" , "SpEn" ,  "PSI_delta" , "RIR_delta" , "PSI_theta" ,\
                  "RIR_theta" , "PSI_alfa" , "RIR_alfa" , "PSI_beta2" , "RIR_beta2" , "PSI_beta1" ,\
                   "RIR_beta1" ,  "PSI_gamma" , "RIR_gamma"]

import pyeeg
import numpy as np
import scipy.stats as sp
import scipy.linalg as linalg 
import pandas as pd
from nolitsa import delay , dimension , lyapunov # Non Linear TimeSeries Analysis

# EEG caracteristics
freqBins = [0.5 , 4 , 8 , 13 , 20 , 30 , 45]
Fs = 128 # sampling frequency 
maxTauLag = Fs # maximum embeding Time for feature extraction :: useful for finding Tau= embeding delay
corrThresh = np.exp(-1) # instead of taking first zero of Autocorr for Tau we take first decorrelated Time  r(Tau) < 1/e
dim = np.arange(1, 10 + 1)
fracThresh = 0.01 # 1% 
nMono = len(monoFeaturesDict) # number of mono-variate features    25

""" function that gives the indices of the corresponding features"""
def feats2inds(featsList):
    if featsList == 'all':
        return 'all' 
    else:
        nChan = 21
        allMono = nMono * nChan
        nFeatsTot = allMono + nChan*(nChan-1)/2
        indsList = list()
        for kMono in range(nMono):
            if monoFeaturesDict[kMono] in featsList:
                indsList.extend([nMono * kChan + kMono for kChan in range(nChan)])
        if 'Connectivity' in featsList:
            indsList.extend([k for k in range(allMono , nFeatsTot)])
        
        return np.asarray(indsList)

""" 
new feats extractor : with reduced amount of feats"""

def myFeaturesExtractor1(X , myM , myV , myMin , myMax): # X has to be a matrix where each row is a channel
    N = len(X) # number of channels
#    L = len(X[0])
    
    # get number of features 
    nFeatures = nMono * N + N*(N-1)/2
    # here we initialize the list of features // We will transform it to an array later
    featList = np.zeros((int(nFeatures)))
    # deal with monovariate features first
    for kChan in range(N):
        kFeat = 0
        mySig = X[kChan , :]
        #========== Stats ========================
        myMean = myM[kChan]
        featList[nMono*kChan + kFeat] = myMean
        kFeat += 1

        featList[nMono*kChan + kFeat] = myMax[kChan]
        kFeat += 1

        featList[nMono*kChan + kFeat] = myMin[kChan]
        kFeat += 1
        
        myVar = myV[kChan]
        featList[nMono*kChan + kFeat] = myVar
        kFeat += 1
        featList[nMono*kChan + kFeat] = sp.skew(mySig)
        kFeat += 1
        featList[nMono*kChan + kFeat] = sp.kurtosis(mySig)
        kFeat += 1

        featList[nMono*kChan + kFeat] = pyeeg.dfa(mySig)
        kFeat += 1
        
        hMob , hComp = pyeeg.hjorth(mySig )
        featList[nMono*kChan + kFeat] = hMob
        kFeat += 1
        featList[nMono*kChan + kFeat] = hComp
        kFeat += 1
        ## ======== fractal ========================
        # Now we need to get the embeding time lag Tau and embeding dmension
        ac=delay.acorr(mySig, maxtau=maxTauLag, norm=True, detrend=True)
        Tau = firstTrue(ac < corrThresh) # embeding delay

        featList[nMono*kChan + kFeat] = Tau
        kFeat += 1

        PFD = pyeeg.pfd(mySig, D=None)
        hfd10 = pyeeg.hfd(mySig , 10)
        
        featList[nMono*kChan + kFeat] = PFD
        kFeat += 1

        featList[nMono*kChan + kFeat] = hfd10
        kFeat += 1
               
        ## ======== Entropy ========================
        # here we compute bin power 
        power, power_Ratio = pyeeg.bin_power(mySig , freqBins , Fs )
        featList[nMono*kChan + kFeat] = pyeeg.spectral_entropy(mySig, freqBins, Fs, Power_Ratio=power_Ratio)
        kFeat += 1
        ## ======== Spectral ========================
        for kBin in range(len(freqBins)-1):
            featList[nMono*kChan + kFeat] = power[kBin]
            kFeat += 1
            featList[nMono*kChan + kFeat] = power_Ratio[kBin]
            kFeat += 1
            
    # deal with multivariate features first        
    #============ connectivity ==================
    corrList = connectome(X) 
    nConnect = len(corrList)
    if N*(N-1)/2 != nConnect:
        raise ValueError('incorrect number of correlation coeffs')
    
    for kC in range(nConnect):
        featList[-nConnect + kC] = corrList[kC]
        
    return featList



def myFeaturesExtractor(X , myM , myV): # X has to be a matrix where each row is a channel
    N = len(X) # number of channels
    L = len(X[0])
    maxtLyap = min(500 , L//2 + L//4)
    lyapLags = np.arange(maxtLyap)/Fs
    
    # get number of features 
    nFeatures = nMono * N + N*(N-1)/2
    # here we initialize the list of features // We will transform it to an array later
    featList = np.zeros((int(nFeatures)))
    # deal with monovariate features first
    for kChan in range(N):
        kFeat = 0
        mySig = X[kChan , :]
        #========== Stats ========================
        myMean = myM[kChan]
        featList[nMono*kChan + kFeat] = myMean
        kFeat += 1
        myMax = max(mySig)
        featList[nMono*kChan + kFeat] = myMax
        kFeat += 1
        myMin = min(mySig)
        featList[nMono*kChan + kFeat] = myMin
        kFeat += 1
        peak = max(abs(np.array([myMin , myMax])))
        featList[nMono*kChan + kFeat] = peak
        kFeat += 1
        myVar = myV[kChan]
        featList[nMono*kChan + kFeat] = myVar
        kFeat += 1
        featList[nMono*kChan + kFeat] = sp.skew(mySig)
        kFeat += 1
        featList[nMono*kChan + kFeat] = sp.kurtosis(mySig)
        kFeat += 1
        myRMS = rms(mySig)
        featList[nMono*kChan + kFeat] = myRMS
        kFeat += 1
        featList[nMono*kChan + kFeat] = peak/myRMS
        kFeat += 1
        
        featList[nMono*kChan + kFeat] = totVar(mySig)
        kFeat += 1
        featList[nMono*kChan + kFeat] = pyeeg.dfa(mySig)
        kFeat += 1
        featList[nMono*kChan + kFeat] = pyeeg.hurst(mySig)
        kFeat += 1
        hMob , hComp = pyeeg.hjorth(mySig )
        featList[nMono*kChan + kFeat] = hMob
        kFeat += 1
        featList[nMono*kChan + kFeat] = hComp
        kFeat += 1
        ## ======== fractal ========================
        # Now we need to get the embeding time lag Tau and embeding dmension
        ac=delay.acorr(mySig, maxtau=maxTauLag, norm=True, detrend=True)
        Tau = firstTrue(ac < corrThresh) # embeding delay
        
        f1 , f2 , f3 = dimension.fnn(mySig, dim=dim, tau=Tau, R=10.0, A=2.0, metric='euclidean',\
                                     window=10,maxnum=None, parallel=True)
        myEmDim = firstTrue(f3 < fracThresh)
        # Here we construct the Embeding Matrix Em
        Em = pyeeg.embed_seq(mySig, Tau, myEmDim)
        U, s, Vh = linalg.svd(Em)
        W = s/np.sum(s)  # list of singular values in decreasing order 
        FInfo = pyeeg.fisher_info(X, Tau, myEmDim , W=W)
        featList[nMono*kChan + kFeat] = FInfo
        kFeat += 1
        featList[nMono*kChan + kFeat] = Tau
        kFeat += 1
        featList[nMono*kChan + kFeat] = myEmDim
        kFeat += 1
        #========================================
        PFD = pyeeg.pfd(mySig, D=None)
        hfd6 = pyeeg.hfd(mySig , 6)
        hfd10 = pyeeg.hfd(mySig , 10)
        # Now we fit aline and get its slope to have Lyapunov exponent
        divAvg = lyapunov.mle(Em, maxt=maxtLyap, window= 3 * Tau, metric='euclidean', maxnum=None)
        poly = np.polyfit(lyapLags, divAvg, 1, rcond=None, full=False, w=None, cov=False)
        LyapExp = poly[0]
        
        featList[nMono*kChan + kFeat] = PFD
        kFeat += 1
        featList[nMono*kChan + kFeat] = hfd6
        kFeat += 1
        featList[nMono*kChan + kFeat] = hfd10
        kFeat += 1
        featList[nMono*kChan + kFeat] = LyapExp
        kFeat += 1
               
        ## ======== Entropy ========================
        tolerance = 1 / 4
        entropyDim = max([myEmDim , PFD])
        
        featList[nMono*kChan + kFeat] = pyeeg.samp_entropy(mySig , entropyDim , tolerance )
        kFeat += 1
        featList[nMono*kChan + kFeat] = pyeeg.svd_entropy(mySig, Tau, myEmDim , W=W) 
        kFeat += 1
        
        # here we compute bin power 
        power, power_Ratio = pyeeg.bin_power(mySig , freqBins , Fs )
        featList[nMono*kChan + kFeat] = pyeeg.spectral_entropy(mySig, freqBins, Fs, Power_Ratio=power_Ratio)
        kFeat += 1
        ## ======== Spectral ========================
        for kBin in range(len(freqBins)-1):
            featList[nMono*kChan + kFeat] = power[kBin]
            kFeat += 1
            featList[nMono*kChan + kFeat] = power_Ratio[kBin]
            kFeat += 1
            
    # deal with multivariate features first        
    #============ connectivity ==================
    corrList = connectome(X) 
    nConnect = len(corrList)
    if N*(N-1)/2 != nConnect:
        raise ValueError('incorrect number of correlation coeffs')
    
    for kC in range(nConnect):
        featList[-nConnect + kC] = corrList[kC]
        
    return featList
        
""" 
Here we define intermediate functions
"""
# total Variation of a channel

def totVar(x): 
    N = len(x)
    xmin = min(x)
    xmax = max(x)
    totV=0
    for k in range(N - 1):
        totV= totV + abs(x[k+1] - x[k])
    return totV/(N - 1)/(xmax - xmin)

# RMS

def rms(x): 
    N = len(x)
    SE=0
    for k in range(N):
        SE= SE + x[k] * x[k]
    return np.sqrt(SE/(N - 1))


# Connectivity : this function appends the connectivity coeffs to featList

def connectome(X): 
    corrList =list()
    N = len(X)
    df = pd.DataFrame(np.transpose(X))
    C = np.array(df.corr()) 
    for i in range(N - 1):
        for j in range(i + 1,N):
            corrList.append(C[i , j])
    return (corrList)


# find first True in logical Array
def firstTrue(log):
    N = len(log)
    res= 2 #### change to None
    for k in range(N):
        if log[k]:
            res=k
            break
    return res

# append one by one the elements of an array in myList
def appendArray2List(myList , myArray ):
    N = len(myArray)
    for k in range(N):
        myList.append(myArray[k] )

"""
# Discret Wavelet Transform

import pywt

def DWT( x ):

	resp = pywt.dwt(x, 'db4')

	return resp
"""