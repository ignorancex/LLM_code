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
import os
import pickle
from scipy import signal
from my_functions import myfeature_extraction_functions as featsExtract
#++++++=================================================
saveOpt = True
sph = 5 * 60 # 90 min
sop = 30 * 60 #  min
nRuns = 3
postProcessWindow = 14
thresh = 0.83 * postProcessWindow 
autorejcetFactor = 1.45
preIctal = 10
sustain = True
sustainPoints = 4
autoreject = True
tl = False
scoreType = 'contrast'

filtOpt = True
freqProp = 0.03
filtOrder = 20
h = signal.firwin(filtOrder, freqProp)

"""       LDA SVC KNN """
weightsAbs = np.array([ 4 , 8 , 6])
weights = weightsAbs/np.sum(weightsAbs)
########################################
nHyper = 10
nThresh = 20
threshSpace = np.linspace(0.4 , 0.95 , num = nThresh )
postProcWinSpace = np.arange(7 , 7 + nHyper)
autorejcetFactorSpace = np.linspace(1.35 , 2 , num = nHyper )
sustainPointsSpace = np.arange(1 , 8)
alphaSpace = np.linspace(0.15 , 0.7 , num = nHyper )
n_estimatorsADASpace = np.array([50 , 60 , 70 , 80 , 90 , 100 , 110 , 120 , 130 , 140])
alphaMLPSpace = np.logspace(-5 , 1 , num = nHyper )

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

classifiers = ['rfc' ]
#classifiers = ['lda' , 'knn' ]

#classifiers = ['vote']
nClassifiers = len(classifiers)
tlOpt = [False , True]
#tlOpt = [True ]


""" here is the computation """
#=============================================================================
substr = os.getenv('PATIENT_ID' , "value does not exist")
print("environment variable Patient = {}".format(substr))

sub=int(substr)
print(sub)
kSub = sub - 1


threshIndstr = os.getenv('PARAMETER' , "value does not exist")
print("environment variable Parameter = {}".format(threshIndstr))

threshInd=int(threshIndstr)
print(threshInd)
autorejcetFactor = autorejcetFactorSpace[threshInd]


#alpha = 0.31
#n_estimatorsADA = n_estimatorsADASpace[threshInd]
#alphaMLP = alphaMLPSpace[threshInd]

#alpha = 0.15
#autorejcetFactor= 1.49
#sub = 8
#kSub = sub - 1

dataPath = os.path.join(currentPath ,"perfDataTest5feats","sub{}".format(sub))
subFileExists = os.path.exists(dataPath)
if not(subFileExists == 1):
    os.mkdir(dataPath)
#=============================================================================
"""do the calculation only one time for all classifiers"""
classifiers = ['lda' , 'knn' , 'svc' , 'vote']
classifiers = ['ada']
nClassifiers = len(classifiers)
classifier = 'rfc'

#feats2keep=["Mean" , "Crest" , "Trough" , "Var" , "Skw",  "Kurt", "DFA" , "HMob" , "HComp" ,\
#                  "dCorrTime" , "PFD" , "HFD10" , "SpEn" ,  "PSI_delta" , "RIR_delta" , "PSI_theta" ,\
#                  "RIR_theta" , "PSI_alfa" , "RIR_alfa" , "PSI_beta2" , "RIR_beta2" , "PSI_beta1" ,\
#                   "RIR_beta1" ,  "PSI_gamma" , "RIR_gamma"]

feats2keep=["PFD" , "HFD10" , "PSI_beta1" ,\
                   "RIR_beta1" ,  "PSI_gamma" , "RIR_gamma"]

featsInds = featsExtract.feats2inds(feats2keep)
#newWeights = np.array([ alpha , 1-alpha])

timesAll , runScores ,  seizureTimes , badsAll = myEval.runExp(sub , nRuns , scoreType = scoreType , \
 classifier = classifier , autorejcetFactor = autorejcetFactor  , transferLearning =  tl , others = 2\
                ,selfFrac = 1 , tlFrac = 0.5 , preIctal =  preIctal , currentPath = currentPath , featsInds = featsInds)
            
for postProcessWindow in postProcWinSpace:
    for sustainPoints in sustainPointsSpace:
        for threshProportion in threshSpace:
                
            thresh = threshProportion * postProcessWindow 
            
            
            targetFile = os.path.join(dataPath, \
    "sub_{}sop_{}_sph_{}_postProc_{:.2f}_{:.2f}_TL_{}_score_{}_sustain_{}_{}_autoreject_{}_RF5feats.pkl".format(\
               sub , int(sop/60) , int(sph/60) , thresh , postProcessWindow , tl , scoreType , sustain , \
               sustainPoints , autorejcetFactor ))
            
            #=============================================================================
            
            
            print("getting performance descriptors")
            
            leadingSeizureTimes , newInds = myEval.getLeadingSeizures(seizureTimes , timesAll , sph , sop)
            
            nOrigSeizures = len(seizureTimes)
            nSeizures = len(leadingSeizureTimes)
            
            times = timesAll[newInds]
            interIctalHours = (times[-1] - nSeizures*(sop + sph))/3600
            
            
            sensitivity = np.zeros((1  , nClassifiers))
            specificity = np.zeros((1  , nClassifiers))
            predTimes = np.zeros((1 , nClassifiers))
            FPR = np.zeros((1 , nClassifiers))
            cFPR = np.zeros((1 , nClassifiers))
            pVal = np.zeros((1 , nClassifiers))
            
            sensitivityStd = np.zeros((1  , nClassifiers))
            specificityStd = np.zeros((1  , nClassifiers))
            predTimesStd = np.zeros((1 , nClassifiers))
            FPRStd = np.zeros((1 , nClassifiers))
            cFPRStd = np.zeros((1 , nClassifiers))
                
                
            for clfInd in range(len(classifiers)):
                senList = np.zeros((nRuns,))
                specList = np.zeros((nRuns,))
                fprList = np.zeros((nRuns,))
                fprCorrectedList = np.zeros((nRuns,))
                predTimesList = np.zeros((nRuns,))
                badsList = np.zeros((nRuns,))
                for kRun in range(nRuns):
                    print('run {}'.format(kRun + 1))
                    preds = runScores[kRun]
                    if autoreject:
                        bads = badsAll[kRun][newInds]
                        badsList[kRun] = 100 * np.mean(bads) 
                    else:
                        bads = None
                        
                    if classifier in ['all' , 'svc+rfc']:

                        scoresTemp =    post.slidingWindow(times , preds[: , clfInd][newInds] ,  seizureInfo = seizureTimes ,\
                                                           postProcessWindow = postProcessWindow )
                    else:
                        scoresTemp = post.slidingWindow(times , preds[newInds] ,  seizureInfo = seizureTimes ,\
                                                           postProcessWindow = postProcessWindow )
                    
                    if filtOpt:
                        scores = signal.convolve( scoresTemp , h , mode='same' ) 
                    else:
                        scores = scoresTemp 
        
                    isDetected , falseWarnings , predictionTimes = myEval.getPerf(times , \
                                    scores , leadingSeizureTimes , sph , sop , thresh , sustain = sustain , bads = bads , sustainPoints = sustainPoints)
                    
                    sn = np.mean(isDetected)
                    numFalse = np.sum(falseWarnings)
                    fpr = numFalse /(times[-1])*3600
                    
                    corrFPR = numFalse / interIctalHours
                    spec = 1 - numFalse * (sop + sph)/60/interIctalHours
                    meanPredTime = np.sum(predictionTimes[np.where(isDetected)]) / np.sum(isDetected)
                    
                    senList[kRun] = (sn)
                    specList[kRun] = spec
                    fprList[kRun] = (fpr)
                    fprCorrectedList[kRun] = (corrFPR)
                    predTimesList[kRun] = (meanPredTime)
                    
                #####################################################################################
                mSn = np.mean(senList)
                mSpec = np.mean(specList)
                mFPR = np.mean(fprList)
                mCorrFPR = np.mean(fprCorrectedList)
                mPredTime = np.mean(predTimesList)
                mpVal = myEval.getPValue(mFPR , sop/3600 , np.round(mSn * nSeizures)  , nSeizures)
                
                sSn = np.std(senList)
                sSpec = np.std(specList)
                sFPR = np.std(fprList)
                sCorrFPR = np.std(fprCorrectedList)
                sPredTime = np.std(predTimesList)
                
                sensitivity[0 , clfInd] = 100*mSn
                specificity[0 , clfInd] = 100*mSpec
                predTimes[0 , clfInd] = mPredTime/60
                FPR[0 , clfInd] = mFPR
                cFPR[0 , clfInd] = mCorrFPR
                pVal[0 , clfInd] = mpVal
                
                sensitivityStd[0 , clfInd] = 100*sSn
                specificityStd[0 , clfInd] = 100*sSpec
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
            "specificity" : specificity,\
            "predTimes" : predTimes,\
            "FPR" : FPR,\
            "cFPR" : cFPR,\
            "pVal" : pVal,\
            "sensitivityStd" : sensitivityStd,\
            "specificityStd" : specificityStd,\
            "predTimesStd" : predTimesStd,\
            "FPRStd" : FPRStd,\
            "cFPRStd" : cFPRStd , \
            "badsProp" : badsProp}
            
            f = open(targetFile,"wb")
            pickle.dump(perfDict,f)
            f.close()
            print("perfDictinnary initialized")