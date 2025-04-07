#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:08:38 2020

@author: remy.benmessaoud
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
#%%%%%%%%ù
alpha = 1
convFact = 1
structurePath = "archives_Sn_FPR_rfc_newGrid_cohort20.pkl"
f = open(structurePath,"rb")
archDict = pickle.load(f)
f.close()
for autoreject in [2]:
    #autoreject = 7
    sn = archDict["subSn"][autoreject , : , : , :]
    fpr = convFact*archDict["subFPR"][autoreject , : , : , :]
    
    wins = np.arange(0 , 3)
    nHyper = 10
    nThresh = 20
    threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
    postProcWinSpace = np.arange(7 , 17)
    autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
    sustainPointsSpace = np.arange(1 , 8)
    nWin = len(postProcWinSpace)
    nThresh = len(threshSpace)
    nSustain  = len(sustainPointsSpace)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
    nWins = len(wins)
    fig, axes = plt.subplots(nrows=1, ncols=1 , figsize=(4,3))
    ax = axes
    #plt.figure()
    for kWin in range(nWins):
        win = wins[kWin]
        tempSn = sn[win , : , :].reshape(((nThresh ) * nSustain,))
        tempFPR = fpr[win , : , :].reshape(((nThresh) * nSustain,))
        
        #color=fadeColor(c1,c2,kWin/(nWins))
        color =  'tab:blue'
        ss=ax.scatter(tempFPR , tempSn , s = 6 , c=color , alpha = alpha)
        if kWin == 0:
            ss.set_label('RF')
#%%
structurePath = "archives_Sn_FPR_mlp_newGrid_cohort20.pkl"
f = open(structurePath,"rb")
archDict = pickle.load(f)
f.close()
for autoreject in [5]:
    #autoreject = 7
    sn = archDict["subSn"][autoreject , : , : , :]
    fpr = convFact*archDict["subFPR"][autoreject , : , : , :]
    
    wins = np.arange(7 , 10)
    nHyper = 10
    nThresh = 20
    threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
    postProcWinSpace = np.arange(7 , 17)
    autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
    sustainPointsSpace = np.arange(1 , 8)
    nWin = len(postProcWinSpace)
    nThresh = len(threshSpace)
    nSustain  = len(sustainPointsSpace)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
    nWins = len(wins)

    #plt.figure()
    for kWin in range(nWins):
        win = wins[kWin]
        tempSn = sn[win , : , :].reshape((nThresh * nSustain,))
        tempFPR = fpr[win , : , :].reshape((nThresh * nSustain,))
        
        #color=fadeColor(c1,c2,kWin/(nWins))
        color =  'tab:green'     
        ss2 = ax.scatter(tempFPR , tempSn , s = 6 , c=color , alpha = alpha )
        if kWin == 0:
            ss2.set_label('MLP')
    #
    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #ax.set_xlabel('FPR (/h)')
    
    ax.set_xlabel('FPR (/h)' , fontweight = 'demibold' , fontsize = 14)
    ax.set_ylabel('Sensitivity (%)', fontweight = 'demibold' , fontsize = 14)
    
#%%
structurePath = "archives_Sn_FPR_knn_newGrid_cohort20.pkl"
f = open(structurePath,"rb")
archDict = pickle.load(f)
f.close()
for autoreject in [0]:
    #autoreject = 7
    sn = archDict["subSn"][autoreject , : , : , :]
    fpr =convFact* archDict["subFPR"][autoreject , : , : , :]
    
    wins = np.arange(7 , 10)
    nHyper = 10
    nThresh = 20
    threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
    postProcWinSpace = np.arange(7 , 17)
    autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
    sustainPointsSpace = np.arange(1 , 8)
    nWin = len(postProcWinSpace)
    nThresh = len(threshSpace)
    nSustain  = len(sustainPointsSpace)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
    nWins = len(wins)

    #plt.figure()
    for kWin in range(nWins):
        win = wins[kWin]
        tempSn = sn[win , : , :].reshape((nThresh * nSustain,))
        tempFPR = fpr[win , : , :].reshape((nThresh * nSustain,))
        
        #color=fadeColor(c1,c2,kWin/(nWins))
        color =  'tab:orange'
        ss3 = ax.scatter(tempFPR , tempSn , s = 6 , c=color  , alpha = alpha)
        if kWin == 0:
            ss3.set_label('KNN')
    #
    
    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #ax.set_xlabel('FPR (/h)')
    
    ax.set_xlabel('FPR (/h)' , fontweight = 'demibold' , fontsize = 14)
    ax.set_ylabel('Sensitivity (%)', fontweight = 'demibold' , fontsize = 14)

#%%
structurePath = "archives_Sn_FPR_svc_newGrid_cohort20.pkl"
f = open(structurePath,"rb")
archDict = pickle.load(f)
f.close()
for autoreject in [7]:
    #autoreject = 7
    sn = archDict["subSn"][autoreject , : , : , :]
    fpr = convFact*archDict["subFPR"][autoreject , : , : , :]
    
    wins = np.arange(7 , 10)
    nHyper = 10
    nThresh = 20
    threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
    postProcWinSpace = np.arange(7 , 17)
    autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
    sustainPointsSpace = np.arange(1 , 8)
    nWin = len(postProcWinSpace)
    nThresh = len(threshSpace)
    nSustain  = len(sustainPointsSpace)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
    nWins = len(wins)

    #plt.figure()
    for kWin in range(nWins):
        win = wins[kWin]
        tempSn = sn[win , : , :].reshape((nThresh * nSustain,))
        tempFPR = fpr[win , : , :].reshape((nThresh * nSustain,))
        
        #color=fadeColor(c1,c2,kWin/(nWins))
        color =  'tab:red'
        ss3 = ax.scatter(tempFPR , tempSn , s = 6 , c=color  , alpha = alpha)
        if kWin == 0:
            ss3.set_label('SVM')
    #
    
    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #ax.set_xlabel('FPR (/h)')
    
    ax.set_xlabel('FPR (/h)' , fontweight = 'demibold' , fontsize = 14)
    ax.set_ylabel('Sensitivity (%)', fontweight = 'demibold' , fontsize = 14)

#%%      , 'tab:purple'  
    
#%%
structurePath = "archives_Sn_FPR_ada_newGrid_cohort20.pkl"
f = open(structurePath,"rb")
archDict = pickle.load(f)
f.close()
for autoreject in [9]:
    #autoreject = 7
    sn = archDict["subSn"][autoreject , : , : , :]
    fpr =convFact* archDict["subFPR"][autoreject , : , : , :]
    
    wins = np.arange(7 , 10)
    nHyper = 10
    nThresh = 20
    threshSpace = np.linspace(0.6 , 0.9 , num = nThresh )
    postProcWinSpace = np.arange(7 , 17)
    autorejcetFactorSpace = np.linspace(1.25 , 1.75 , num = nHyper )
    sustainPointsSpace = np.arange(1 , 8)
    nWin = len(postProcWinSpace)
    nThresh = len(threshSpace)
    nSustain  = len(sustainPointsSpace)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
    nWins = len(wins)

    #plt.figure()
    for kWin in range(nWins):
        win = wins[kWin]
        tempSn = sn[win , : , :].reshape((nThresh * (nSustain),))
        tempFPR = fpr[win , : , :].reshape((nThresh * (nSustain),))
        
        #color=fadeColor(c1,c2,kWin/(nWins))
        color =  'tab:purple'
        ss4 = ax.scatter(tempFPR , tempSn , s = 6 , c=color  , alpha = 1)
        if kWin == 0:
            ss4.set_label('AdaBoost')
    #
    
    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #ax.set_xlabel('FPR (/h)')
    
    ax.set_xlabel('FPR (/h)' , fontweight = 'demibold' , fontsize = 14)
    ax.set_ylabel('Sensitivity (%)', fontweight = 'demibold' , fontsize = 14)
    
    ss5 = ax.plot(np.array([0,60/35]) , np.array([0,100]) , '--' , c = 'k' , label = 'Chance')
    #ss5.set_label('Chance')
    #plt.legend('RFC' , 'Aggregate' , 'KNN')
    ax.legend(loc = 4 , ncol = 2 , markerscale = 3 , frameon = False , fontsize = 'medium')
    plt.grid()
    plt.show() 
    plt.tight_layout()
    if False:
        plt.savefig('insights/Sn_FPR/autorejectFact_{}.png'.format(autorejcetFactorSpace[autoreject]))
