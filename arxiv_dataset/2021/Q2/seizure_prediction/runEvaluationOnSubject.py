#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:56:26 2019

@author: remy.benmessaoud
"""
import numpy as np 
from my_functions import evaluation_functions as myEval
from my_functions import postprocessing_functions as post
import os.path
import pickle
#++++++=================================================
saveOpt = True
sph = 5 * 60 # 90 min
sop = 30 * 60 #  min
nRuns = 5
postProcessWindow = 13
thresh = 0.83 * postProcessWindow 
autorejcetFactor = 1.35
preIctal = 10
sustain = True
sustainPoints = 5
autoreject = True
tl = False
scoreType = 'contrast'


# deal with path
currentPath = os.getcwd()
cDir = os.path.basename(currentPath)
if not(cDir == 'neuroProject'):
    newPath = os.path.join(currentPath , 'neuroProject')
    if os.path.isdir(newPath):
        os.chdir(newPath)
    else:
        raise ValueError(' problem of path can not find neuroPpoject directory \nHere is the current path : '\
                         .format(currentPath))
#=============================================================================

classifiers = ['lda' , 'knn' , 'svc' , 'vote']
#classifiers = ['lda' , 'knn' ]

#classifiers = ['vote']
nClassifiers = len(classifiers)
tlOpt = [False , True]
#tlOpt = [True ]


""" here is the computation """
#=============================================================================
substr = os.getenv('SLURM_ARRAY_TASK_ID' , "value does not exist")
print("environment variable = {}".format(substr))

sub=int(substr)
print(sub)
kSub = sub - 1

#sub = 12

targetFile = os.path.join(\
currentPath ,"performanceEvaluationData", "sub_{}sop_{}_sph_{}_postProc_{:.2f}_{:.2f}_TL_{}_score_{}_sustain_{}_{}_autoreject_{}.pkl".format(\
                sub , int(sop/60) , int(sph/60) , thresh , postProcessWindow , tl , scoreType , sustain , sustainPoints , autoreject))

#=============================================================================


#=============================================================================

"""do the calculation only one time for all classifiers"""
classifier = 'all'

timesAll , runScores ,  seizureTimes , badsAll = myEval.runExp(sub , nRuns , scoreType = scoreType ,\
            classifier = classifier , autorejcetFactor = autorejcetFactor , thresh = thresh , \
            transferLearning =  tl , others = 2 , selfFrac = 1 , tlFrac = 0.5 , preIctal =  preIctal , currentPath = currentPath)
print("getting performance descriptors")

leadingSeizureTimes , newInds = myEval.getLeadingSeizures(seizureTimes , timesAll , sph , sop)

nOrigSeizures = len(seizureTimes)
nSeizures = len(leadingSeizureTimes)

times = timesAll[newInds]
interIctalHours = (times[-1] - nSeizures*(sop + sph))/3600


sensitivity = np.zeros((1  , nClassifiers))
predTimes = np.zeros((1 , nClassifiers))
FPR = np.zeros((1 , nClassifiers))
cFPR = np.zeros((1 , nClassifiers))
pVal = np.zeros((1 , nClassifiers))

sensitivityStd = np.zeros((1  , nClassifiers))
predTimesStd = np.zeros((1 , nClassifiers))
FPRStd = np.zeros((1 , nClassifiers))
cFPRStd = np.zeros((1 , nClassifiers))
    
    
for clfInd in range(len(classifiers)):
    senList = np.zeros((nRuns,))
    fprList = np.zeros((nRuns,))
    fprCorrectedList = np.zeros((nRuns,))
    predTimesList = np.zeros((nRuns,))
    badsList = np.zeros((nRuns,))
    for kRun in range(nRuns):
        print('run {}'.format(kRun + 1))
        preds = runScores[kRun]
        if autoreject:
            bads = badsAll[kRun][newInds]
        else:
            bads = None
        badsList[kRun] = 100 * np.mean(bads)   
            
        if classifier =='all':
            countsList = list()
            for kClass in range(4):
                tempCounts = post.slidingWindow(times , preds[: , clfInd][newInds] ,  seizureInfo = seizureTimes ,\
                                               postProcessWindow = postProcessWindow )
                nCounts = len(tempCounts)
                countsList.append(tempCounts.reshape((nCounts , 1)))
            scores = np.hstack(countsList)
        else:
            scores = post.slidingWindow(times , preds[newInds] ,  seizureInfo = seizureTimes ,\
                                               postProcessWindow = postProcessWindow )
        
        isDetected , falseWarnings , predictionTimes = myEval.getPerf(times , \
                        scores[: , clfInd] , leadingSeizureTimes , sph , sop , thresh , sustain = sustain , bads = bads , sustainPoints = sustainPoints)
        
        sn = np.mean(isDetected)
        numFalse = np.sum(falseWarnings)
        fpr = numFalse /(times[-1])*3600
        
        corrFPR = numFalse / interIctalHours
        
        meanPredTime = np.sum(predictionTimes[np.where(isDetected)]) / np.sum(isDetected)
        
        senList[kRun] = (sn)
        fprList[kRun] = (fpr)
        fprCorrectedList[kRun] = (corrFPR)
        predTimesList[kRun] = (meanPredTime)
        
    #####################################################################################
    mSn = np.mean(senList)
    mFPR = np.mean(fprList)
    mCorrFPR = np.mean(fprCorrectedList)
    mPredTime = np.mean(predTimesList)
    mpVal = myEval.getPValue(mFPR , sop/3600 , np.round(mSn * nSeizures)  , nSeizures)
    
    sSn = np.std(senList)
    sFPR = np.std(fprList)
    sCorrFPR = np.std(fprCorrectedList)
    sPredTime = np.std(predTimesList)
    
    sensitivity[0 , clfInd] = 100*mSn
    predTimes[0 , clfInd] = mPredTime/60
    FPR[0 , clfInd] = mFPR
    cFPR[0 , clfInd] = mCorrFPR
    pVal[0 , clfInd] = mpVal
    
    sensitivityStd[0 , clfInd] = 100*sSn
    predTimesStd[0 , clfInd] = sPredTime/60
    FPRStd[0 , clfInd] = sFPR
    cFPRStd[0 , clfInd] = sCorrFPR
    badsProp = np.mean(badsList)
    ####################################################################################

perfDict= {"patientName" : "Pat {}".format(int(sub)),\
"numberSeizures" : nSeizures,\
"numberSeizuresOrig" : nOrigSeizures,\
"interIctalHours" : interIctalHours,\
"sensitivity" : sensitivity,\
"predTimes" : predTimes,\
"FPR" : FPR,\
"cFPR" : cFPR,\
"pVal" : pVal,\
"sensitivityStd" : sensitivityStd,\
"predTimesStd" : predTimesStd,\
"FPRStd" : FPRStd,\
"cFPRStd" : cFPRStd , \
"badsProp" : badsProp}

f = open(targetFile,"wb")
pickle.dump(perfDict,f)
f.close()
print("perfDictinnary initialized")