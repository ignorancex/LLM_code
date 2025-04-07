#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:36:03 2019

@author: remy.benmessaoud
"""

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
    _ Std : standard deviation 
    _ Skw : Skewness
    _ Kurt : Kurtosis 
    _ RMS : Root Mean Square
    _ PAPR : Peak-to-Average Power Ratio = Peak/RMS
    _ FFAct : Form Factor = RMS /Mean
    _ totVar : total Variation : normalized sum of succesive differences of voltage 
    _ DFA : Detrended Fluctuation Analysis
    _ HuExp : Hurst Exponent
    _ HMob : Hjorth mobility 
    _ HComp : Hjorth complexity
    _ Connectivity : get paiwise Pearson's coefficients between the channels
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
    
==============================================================================
"""

import pyeeg
import numpy as np
import scipy.stats as sp
import scipy.linalg as linalg 
import pandas as pd
from nolitsa import delay , dimension , lyapunov # Non Linear TimeSeries Analysis
import time 

# EEG caracteristics
freqBins = [0.5 , 4 , 8 , 13 , 20 , 30 , 60]
Fs = 256 # sampling frequency 
maxTauLag = Fs # maximum embeding Time for feature extraction :: useful for finding Tau= embeding delay
corrThresh = np.exp(-1) # instead of taking first zero of Autocorr for Tau we take first decorrelated Time  r(Tau) < 1/e
dim = np.arange(1, 10 + 1)
fracThresh = 0.01 # 1% 
maxtLyap = 500
lyapLags = np.arange(maxtLyap)/Fs


def myFeaturesExtractor(X): # X has to be a matrix where each row is a channel
    N = len(X)
    # L = len(X[0])
    # here we initialize the list of features // We will transform it to an array later
    featList = list()
    timeList =list ()
    featName =list()
    for kChan in range(1):
        mySig = X[kChan , :]
        if kChan == 0:
            start=time.perf_counter_ns()
            
        #========== Stats ========================
        myMean = np.mean(mySig)
        featList.append(myMean)
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append("mean")
            start=end
        featList.append(max(mySig))
        
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append(" max")
            start=end
        featList.append(min(mySig))
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append(" min")
            start=end            
        peak =max(abs(mySig))
        featList.append(peak)
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append(" peak")
            start=end            
        myVar = np.var(mySig)
        featList.append(myVar)
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append(" var")
            start=end
        myVar = np.var(mySig)    
        myStd = np.sqrt(myVar)
        featList.append(myStd)
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append(" std")
            start=end             
        featList.append(sp.skew(mySig))
        if kChan == 0:
            end=time.perf_counter_ns()
            timeList.append(end -start)
            featName.append(" skew")
            start=end

        featList.append(sp.kurtosis(mySig))
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" kurt")
            start=end
        myRMS = rms(mySig)
        featList.append(myRMS)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" rms")
            start=end
        myRMS = rms(mySig)    
        featList.append(peak/myRMS)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" fact")
            start=end
        myRMS = rms(mySig)    
        featList.append(myRMS/myMean)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" papr")
            start=end
        featList.append(totVar(mySig))
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" totVar")
            start=end
            
        featList.append(pyeeg.dfa(mySig))
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" dfa")
            start=end
        featList.append(pyeeg.hurst(mySig))
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" hurst")
            start=end
        hMob , hComp = pyeeg.hjorth(mySig )
        featList.append(hMob)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" Hmob")
            timeList.append(end -start)
            featName.append(" Hcomp")
            start=end
        
        featList.append(hComp)
            
            
        
#        ## ======== fractal ========================
#        # Now we need to get the embeding time lag Tau and embeding dmension
#        ac=delay.acorr(mySig, maxtau=maxTauLag, norm=True, detrend=True)
#        Tau = firstTrue(ac < corrThresh) # embeding delay
#        featList.append(Tau)
#        if kChan == 0:
#            end=time.perf_counter_ns()
#            
#            timeList.append(end -start)
#            featName.append(" dCorrTime")
#            start=end
#        f1 , f2 , f3 = dimension.fnn(mySig, dim=dim, tau=Tau, R=10.0, A=2.0, metric='chebyshev', window=10,maxnum=None, parallel=True)
#        myEmDim = firstTrue(f3 < fracThresh)
##        if kChan == 0:
##            end=time.perf_counter_ns()
##            timeList.append(end -start)
##            featName.append(" embDim")
##            start=end
#        # Here we construct the Embeding Matrix Em
#        Em = pyeeg.embed_seq(mySig, Tau, myEmDim)
#        U, s, Vh = linalg.svd(Em)
#        W = s/np.sum(s)  # list of singular values in decreasing order 
#        
#        FInfo = pyeeg.fisher_info(X, Tau, myEmDim , W=W)
#        featList.append(FInfo)
#        if kChan == 0:
#            end=time.perf_counter_ns()
#            
#            timeList.append(end -start)
#            featName.append(" FInfo")
#            start=end
#
#        featList.append(myEmDim)
        
        
        PFD = pyeeg.pfd(mySig, D=None)
        featList.append(PFD)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" pfd")
            start=end
            
        hfd6 = pyeeg.hfd(mySig , 6)
        featList.append(hfd6)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" hfd6")
            start=end
        hfd10 = pyeeg.hfd(mySig , 10)
        featList.append(hfd10)
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" hfd10")
            start=end
        # Now we fit aline and get its slope to have Lyapunov exponent
#        divAvg = lyapunov.mle(Em, maxt=maxtLyap, window= 3 * Tau, metric='euclidean', maxnum=None)
#        poly = np.polyfit(lyapLags, divAvg, 1, rcond=None, full=False, w=None, cov=False)
#        LyapExp = poly[0]
#        featList.append(np.mean(LyapExp)) 
#        if kChan == 0:
#            end=time.perf_counter_ns()
#            
#            timeList.append(end -start)
#            featName.append("Lyapunov")
#            start=end
               
        ## ======== Entropy ========================
        
        # here we compute bin power 
        power, power_Ratio = pyeeg.bin_power(mySig , freqBins , Fs )
        
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append("Spectral")
            start=end
        featList.append( pyeeg.spectral_entropy(mySig, freqBins, Fs, Power_Ratio=power_Ratio))
        if kChan == 0:
            end=time.perf_counter_ns()
            
            timeList.append(end -start)
            featName.append(" specEn")
            start=end
            
#        tolerance = myStd / 4
#        entropyDim = max([myEmDim , PFD])
#        featList.append( pyeeg.samp_entropy(mySig , entropyDim , tolerance ) )
#        if kChan == 0:
#            end=time.perf_counter_ns()
#            
#            timeList.append(end -start)
#            featName.append(" sampEn")
#            start=end
#        featList.append( pyeeg.svd_entropy(mySig, Tau, myEmDim , W=W) )
#        if kChan == 0:
#            end=time.perf_counter_ns()
#            
#            timeList.append(end -start)
#            featName.append(" svdEn")
#            start=end
            
        ## ======== Spectral ========================
        appendArray2List(featList , power )
        appendArray2List(featList , power_Ratio )
    
    start=time.perf_counter_ns()
    connectome(X , featList)
    end=time.perf_counter_ns()
    timeList.append((end -start)/N/(N-1)*2)
    featName.append("connectivity")
            
    ll=list()
    ll.append(featName)
    ll.append(timeList)    
    return np.asarray(featList) , ll
        
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

def connectome(X , featList): 
    N = len(X)
    df = pd.DataFrame(np.transpose(X))
    C = np.array(df.corr()) 
    for i in range(N - 1):
        for j in range(i + 1,N):
            featList.append(C[i , j])


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