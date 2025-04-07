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
import matplotlib.pyplot as plt
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
threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
postProcWinSpace = np.arange(7 , 17)
autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
sustainPointsSpace = np.arange(1 , 7)

########################################


subjects2test = np.arange(1 , 25)
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
def fadeColor(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    assert len(c1)==len(c2)
    assert mix>=0 and mix<=1, 'mix='+str(mix)
    rgb1=np.array([int(c1[ii:ii+2],16) for ii in range(1,len(c1),2)])
    rgb2=np.array([int(c2[ii:ii+2],16) for ii in range(1,len(c2),2)])   
    rgb=((1-mix)*rgb1+mix*rgb2).astype(int)
    c='#'+''.join([hex(a)[2:] for a in rgb])
    return c
    
#classifiers = ['vote']
classifiers = ['lda' , 'knn' , 'svc' , 'vote']
nClassifiers = len(classifiers)
#tlOpt = [False , True]
#tlOpt = [True ]
nWin = len(postProcWinSpace)
nThresh = len(threshSpace)
nSustain  = len(sustainPointsSpace)
for kSub in [23]:
    structurePath = "Results/individualResults/archives_RF_Sn_FPR_sub_{}.pkl".format(kSub + 1)
    f = open(structurePath,"rb")
    perf = pickle.load(f)
    f.close()
    autoreject = 7
    
    newColors = ['tab:blue' , 'tab:orange' , 'tab:green' , 'tab:red' , 'tab:purple' , 'tab:brown' , 'tab:pink' , 'tab:gray','tab:olive','tab:cyan']
    for autoreject in np.arange(0 , 10):
        #autoreject = 7

        sn = perf["Sn"][autoreject,:,:,:]
        fpr = perf["FPR"][autoreject,:,:,:]    
        snStd = perf["SnStd"][autoreject,:,:,:]
        fprStd = perf["FPRstd"][autoreject,:,:,:]   
        
        wins = np.arange(0 , 10)
        nHyper = 10
        nThresh = 20
        threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
        postProcWinSpace = np.arange(7 , 17)
        autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
        sustainPointsSpace = np.arange(1 , 8)
        nWin = len(postProcWinSpace)
        nThresh = len(threshSpace)
        nSustain  = len(sustainPointsSpace)
        
        c1='#2C2255' #blue
        c2='#F7941E' #yellow
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
        nWins = len(wins)
        if autoreject ==0 :#True or 
            fig, axes = plt.subplots(nrows=1, ncols=1 , figsize=(6, 4))
            ax = axes
            
        #plt.figure()
        for kWin in range(nWins):
            win = wins[kWin]
            tempSn = sn[win , : , :].reshape((nThresh * nSustain,))
            tempFPR = fpr[win , : , :].reshape((nThresh * nSustain,))
            
            color=fadeColor(c1,c2,kWin/(nWins))
            color = newColors[autoreject]
            ax.scatter(tempFPR , tempSn , s = 4 , c=color  , alpha = 0.9)
            
        
        
    toto = 4
    totot = 4
    
    #    for kHyper in range(nHyper):
    #        autorejcetFactor = autorejcetFactorSpace[kHyper]
    #        for kWin in range(nWin):
    #            postProcessWindow = postProcWinSpace[kWin]
    #            for kSustain in range(nSustain):
    #                sustainPoints = sustainPointsSpace[kSustain]
    #                for kThresh in range(nThresh):
    #                    threshProportion = threshSpace[kThresh]
    #                    thresh = threshProportion * postProcessWindow     
                        
                        
        
        
        
