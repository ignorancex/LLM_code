#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:26:45 2019

@author: remy.benmessaoud
"""

import numpy as np 
from my_functions import evaluation_functions as myEval
from my_functions import postprocessing_functions as post
from scipy import signal
from my_functions import myfeature_extraction_functions as featsExtract
#++++++=================================================
sph = 5 * 60 # 90 min
sop = 30 * 60 #  min

sub = 3

nRuns = 3

postProcessWindow = 11
thresh = 9  # 0.82 * postProcessWindow
autorejcetFactor = 1.4999
sustainPoints = 7


filtOpt = True
freqProp = 0.03
filtOrder = 20
h = signal.firwin(filtOrder, freqProp)


feats2keep=["PFD" , "HFD10" , "PSI_beta1" ,\
                   "RIR_beta1" ,  "PSI_gamma" , "RIR_gamma"]

featsInds = featsExtract.feats2inds(feats2keep)


classifier = 'rfc'

timesAll , runScores ,  seizureTimes , badsAll = myEval.runExp(sub , nRuns , scoreType = 'contrast'  , classifier = classifier \
        , postProcessWindow = postProcessWindow , autorejcetFactor = autorejcetFactor  ,  transferLearning =  False , \
        others = 2 , selfFrac = 1 , tlFrac = 0.3 , preIctal = 10 , featsInds = featsInds)

leadingSeizureTimes , newInds = myEval.getLeadingSeizures(seizureTimes , timesAll , sph , sop)
nSeizures = len(leadingSeizureTimes)
times = timesAll[newInds]



senList = np.zeros((nRuns,))
fprList = np.zeros((nRuns,))
fprCorrectedList = np.zeros((nRuns,))
predTimesList = np.zeros((nRuns,))
pValList = np.zeros((nRuns,))
for kRun in range(nRuns):
    bads = badsAll[kRun][newInds]
    print('run {}'.format(kRun + 1))
    if classifier =='all':
        clfInd = 3
        preds = runScores[kRun][: , clfInd][newInds]
        
        countsList = list()
        for kClass in range(4):
            tempCounts = post.slidingWindow(times , preds[: , kClass] ,  seizureInfo = seizureTimes ,\
                                           postProcessWindow = postProcessWindow )
            nCounts = len(tempCounts)
            countsList.append(tempCounts.reshape((nCounts , 1)))
        scoresTemp = np.hstack(countsList)
    else:
        preds = runScores[kRun][newInds]
        
        scoresTemp = post.slidingWindow(times , preds ,  seizureInfo = seizureTimes ,\
                                           postProcessWindow = postProcessWindow )
    
    if filtOpt:
        scores = signal.convolve( scoresTemp , h , mode='same' ) 
    else:
        scores = scoresTemp 
        
    isDetected , falseWarnings , predictionTimes = myEval.getPerf(times , scores , leadingSeizureTimes , sph , sop , \
                                                                  thresh , sustain = False , bads = None , sustainPoints = sustainPoints)
    
    sn = np.mean(isDetected)
    numFalse = np.sum(falseWarnings)
    fpr = numFalse /(times[-1])*3600
    interIctalHours = (times[-1] - nSeizures*(sop + sph))/3600
    corrFPR = numFalse / interIctalHours
    
    meanPredTime = np.sum(predictionTimes[np.where(isDetected)]) / np.sum(isDetected)
    
    senList[kRun] = (sn)
    fprList[kRun] = (fpr)
    fprCorrectedList[kRun] = (corrFPR)
    predTimesList[kRun] = (meanPredTime)
    pValList[kRun] = myEval.getPValue(fpr , sop/3600 , np.sum(isDetected) , len(isDetected))
    
#####################################################################################
mSn = np.mean(senList)
mFPR = np.mean(fprList)
mCorrFPR = np.mean(fprCorrectedList)
mPredTime = np.mean(predTimesList)
mpVal = np.mean(pValList)

sSn = np.std(senList)
sFPR = np.std(fprList)
sCorrFPR = np.std(fprCorrectedList)
sPredTime = np.std(predTimesList)


print("=========== Results for subject {} ===========\nSensitivity = {:.2f}±{:.2f} %\nFPR = {:.4f}±{:.2f} (/h)\ncorrFPR = {:.4f}±{:.2f} (/h)\nprediction time = {:.2f}±{:.2f} min\npValue = {:.2e}\n==========================================".\
      format(sub , 100*mSn , 100*sSn , mFPR , sFPR , mCorrFPR , sCorrFPR , mPredTime / 60 , sPredTime/60 , mpVal))