#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:07:54 2020

@author: remy.benmessaoud
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from my_functions import evaluation_functions as myEval
import pandas as pd
import os.path
import os
from math import isnan
import xlsxwriter
import pickle
#++++++=================================================
saveOpt = True
sph = 5 * 60 # 90 min
sop = 30 * 60 #  min
nRuns = 5
postProcessWindow = 14
thresh = 0.83 * postProcessWindow 
autorejcetFactor = 1.45
preIctal = 10
sustain = True
sustainPoints = 4
autoreject = True
tl = False
scoreType = 'contrast'
"""                  LDA SVC KNN """
weightsAbs = np.array([ 4 , 8 , 6])
weights = weightsAbs/np.sum(weightsAbs)
########################################
nHyper = 10
nThresh = 20
threshSpace = np.linspace(0.4 , 0.95 , num = nThresh )
postProcWinSpace = np.arange(7 , 7 + nHyper)
autorejcetFactorSpace = np.linspace(1.35 , 2 , num = nHyper )
sustainPointsSpace = np.arange(1 , 8)

########################################


subjects2test = np.arange(1 , 25)
subjects2test = np.arange(1 , 3)
#subjects2test = np.array([1 , 2 , 3  , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24])
subjects2compareWith = np.array([1 , 2 , 3 , 4 , 8 , 9 , 12 , 13 , 17 , 18 , 19 , 20 , 22])
subjects2compareWithInds = (subjects2compareWith - np.ones(subjects2compareWith.shape)).astype(int)
#subjects2compareWithInds = np.array([0 , 1 , 2  , 4 , 8 , 9 , 11 , 12 , 16 , 17 , 18 , 19 , 21])
#subjects2test = [1 , 2 , 3 , 5 , 9 , 10 , 13 , 14 , 18 , 19 , 20 , 21 , 23]
#subjects2test = [1 , 2 ]
#subjects2test = [17 , 19 ]
nSubs=len(subjects2test)
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
#=============================================================================classifiers = ['lda' , 'knn' , 'svc' , 'vote']
#classifiers = ['lda' , 'knn' ]

#classifiers = ['vote']
classifiers = ['lda' , 'knn' , 'svc' , 'vote']
classifiers = ['rfc']
nClassifiers = len(classifiers)
#tlOpt = [False , True]
#tlOpt = [True ]
nWin = len(postProcWinSpace)
nThresh = len(threshSpace)
nSustain  = len(sustainPointsSpace)
#for kSub in range(nSubs):
for kSub in np.arange(0,24): #range(nSubs):   
    sub = subjects2test[kSub]       
    archiveSn = np.zeros((nHyper , nWin , nThresh , nSustain))
    archivecFPR = np.zeros((nHyper , nWin , nThresh , nSustain))
    
    archiveSnStd = np.zeros((nHyper , nWin , nThresh , nSustain))
    archivecFPRStd = np.zeros((nHyper , nWin , nThresh , nSustain))

    for kHyper in range(nHyper):
        autorejcetFactor = autorejcetFactorSpace[kHyper]
        for kWin in range(nWin):
            postProcessWindow = postProcWinSpace[kWin]
            for kSustain in range(nSustain):
                sustainPoints = sustainPointsSpace[kSustain]
                for kThresh in range(nThresh):
                    threshProportion = threshSpace[kThresh]
                    thresh = threshProportion * postProcessWindow     
                    
                    clfInd = 1
                    clf = classifiers[clfInd] 
                    
                    #print('sub = {}'.format(sub))
                    targetFile = os.path.join(currentPath \
                ,"perfDataTest5feats+RFC", 'sub{}'.format(sub),\
                "sub_{}sop_{}_sph_{}_postProc_{:.2f}_{:.2f}_TL_{}_score_{}_sustain_{}_{}_autoreject_{}_RF5feats.pkl".format(\
                    sub , int(sop/60) , int(sph/60) , thresh , postProcessWindow , tl , scoreType , sustain , sustainPoints ,\
                    autorejcetFactor ))
            
                    if os.path.exists(targetFile):
                        f = open(targetFile,"rb")
                        perfDict = pickle.load(f)
                        f.close()
                    else:
                        print(targetFile)
                        raise ValueError('data file not found ')
                        
                       
                    sensitivity = perfDict["sensitivity"]
                    sensStd = perfDict["sensitivityStd"]
                    predTimes = perfDict["predTimes"]
                    FPR = perfDict["FPR"]
                    cFPR = perfDict["cFPR"]
                    cFPRstd = perfDict["cFPRStd"]
                    
                    
                    
                    if clfInd == 1:
                        archiveSn[kHyper , kWin , kThresh , kSustain] = sensitivity[0 , clfInd]
                        archivecFPR[kHyper , kWin , kThresh , kSustain] = cFPR[0 , clfInd]
                        
                        archiveSnStd[kHyper , kWin , kThresh , kSustain] = sensStd[0 , clfInd]
                        archivecFPRStd[kHyper , kWin , kThresh , kSustain] = cFPRstd[0 , clfInd]
    
    strct2save = {"Sn" : archiveSn  , "FPR" : archivecFPR , "SnStd" : archiveSnStd , "FPRstd" : archivecFPRStd}
    structurePath = "Results/individualResults/archives_RF_Sn_FPR_sub_{}_5feats.pkl".format(sub)
    f = open(structurePath,"wb")
    pickle.dump(strct2save,f)
    f.close()
    print("archives saved")
    
    
    
    
