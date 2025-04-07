#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:48:14 2019

@author: remy.benmessaoud
"""

import os.path
import numpy as np
import pickle
import my_functions.pipeline_functions as pipe
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from kl import KLdivergence

#myPath = "/Users/remy.benmessaoud/Desktop/neuroProject/myEEGdata/formatted_data/features"
preIctal = 5
winSize = 15
bigMax = 781

""" function that takes the number of seizures and gives the amount of exterior data to add as a fraction of original data"""
def numSeizures2fraction(nSeizures):
    if nSeizures <= 3:
        res = 0.3
    elif nSeizures <= 5:
        res = 0.15
    elif nSeizures <= 7:
        res = 0.05
    else:
        res = 0
    return res
    
def getDataFromOthers(availableSubs , nSubs  , mySub , currentPath):
    myPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21")
    availableSubsList = list(availableSubs)
    subInd = np.where(availableSubs == mySub)[0]
    if len(subInd) > 0:
        availableSubsList.pop(subInd[0])
    allSubs = np.asarray(availableSubsList)
    indSubs = rnd.randint(0 , len(allSubs) , nSubs)
    others = allSubs[indSubs]
    
    feats2stack = list()
    labels2stack = list()
    for sub in others:
        ########## load data ####################
        fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , winSize , preIctal)
        filePath = os.path.join(myPath , fileSpec) 
        if os.path.exists(filePath) :
            with open(filePath, 'rb') as f:
                featsDict = pickle.load(f)  
        else:
            raise ValueError('{} does not exist'.format(filePath))
                    
        allfeats , alllabels , maxFeats = pipe.myDataWrapperForClassification(featsDict , [])
        feats , labels = pipe.seperateAndPermute(allfeats , alllabels)
        maxFeats = max(maxFeats , bigMax)
        
#        # now do the padding
#        temp = feats
#        l,m = temp.shape
#        if m < maxFeats:
#            pads = np.zeros((l , maxFeats - m))
#            feats = np.hstack((temp , pads))
        
        feats2stack.append(feats)
        labels2stack.append(labels)
        
    feats2return = np.vstack(feats2stack)  
    lbls2return = np.concatenate(labels2stack) 
    
    return feats2return , lbls2return  


def getDataFromOthers1(subjectsCluster , nSubs  , mySub , currentPath):
    myPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21")
    others = subjectsCluster[:nSubs]
    
    feats2stack = list()
    labels2stack = list()
    for sub in others:
        ########## load data ####################
        fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , winSize , preIctal)
        filePath = os.path.join(myPath , fileSpec) 
        if os.path.exists(filePath) :
            with open(filePath, 'rb') as f:
                featsDict = pickle.load(f)  
        else:
            raise ValueError('{} does not exist'.format(filePath))
                    
        allfeats , alllabels , maxFeats = pipe.myDataWrapperForClassification(featsDict )
        feats , labels = pipe.seperateAndPermute(allfeats , alllabels)
        maxFeats = max(maxFeats , bigMax)
        
#        # now do the padding
#        temp = feats
#        l,m = temp.shape
#        if m < maxFeats:
#            pads = np.zeros((l , maxFeats - m))
#            feats = np.hstack((temp , pads))
        
        feats2stack.append(feats)
        labels2stack.append(labels)
        
    feats2return = np.vstack(feats2stack)  
    lbls2return = np.concatenate(labels2stack) 
    
    return feats2return , lbls2return  

""" function that gets only preIctal samples from other subjects"""
  
def getpreIctalFromOthers(availableSubs , nSubs  , mySub , currentPath , preIctal , winSize):
    myPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21_allTimes_{}s".format(winSize))
    availableSubsList = list(availableSubs)
    subInd = np.where(availableSubs == mySub)[0]
    if len(subInd) > 0:
        availableSubsList.pop(subInd[0])
    allSubs = np.asarray(availableSubsList)
    indSubs = rnd.randint(0 , len(allSubs) , nSubs)
    others = allSubs[indSubs]
    
    feats2stack = list()
    labels2stack = list()
    for sub in others:
        ########## load data ####################
        fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , winSize , preIctal)
        filePath = os.path.join(myPath , fileSpec) 
        if os.path.exists(filePath) :
            with open(filePath, 'rb') as f:
                featsDict = pickle.load(f)  
        else:
            raise ValueError('{} does not exist'.format(filePath))
                    
        allfeats , alllabels , maxFeats = pipe.myPreIctalDataWrapper(featsDict , [])
        preIctalInd = np.where(alllabels == 1)[0]
        feats = allfeats[preIctalInd]
        labels = alllabels[preIctalInd]
        
        maxFeats = max(maxFeats , bigMax)
        
#        # now do the padding
#        temp = feats
#        l,m = temp.shape
#        if m < maxFeats:
#            pads = np.zeros((l , maxFeats - m))
#            feats = np.hstack((temp , pads))
        
        feats2stack.append(feats)
        labels2stack.append(labels)
        
    feats2return = np.vstack(feats2stack)  
    lbls2return = np.concatenate(labels2stack) 
    
    return feats2return , lbls2return  

""" function that gets only preIctal samples from other subjects"""
  
def getpreIctalFromOthers1(subjectsCluster , nSubs  , mySub , currentPath , winSize , preIctal):
    myPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21_allTimes_{}s".format(winSize))
    
    others = subjectsCluster[:nSubs]
    
    feats2stack = list()
    labels2stack = list()
    for sub in others:
        ########## load data ####################
        fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , winSize , preIctal)
        filePath = os.path.join(myPath , fileSpec) 
        if os.path.exists(filePath) :
            with open(filePath, 'rb') as f:
                featsDict = pickle.load(f)  
        else:
            raise ValueError('{} does not exist'.format(filePath))
                    
        allfeats , alllabels , maxFeats = pipe.myPreIctalDataWrapper(featsDict , [])
        preIctalInd = np.where(alllabels == 1)[0]
        feats = allfeats[preIctalInd]
        labels = alllabels[preIctalInd]
        
        maxFeats = max(maxFeats , bigMax)
        
#        # now do the padding
#        temp = feats
#        l,m = temp.shape
#        if m < maxFeats:
#            pads = np.zeros((l , maxFeats - m))
#            feats = np.hstack((temp , pads))
        
        feats2stack.append(feats)
        labels2stack.append(labels)
        
    feats2return = np.vstack(feats2stack)  
    lbls2return = np.concatenate(labels2stack) 
    
    return feats2return , lbls2return  
"""=========================================================================
function that gets the features Dictionnary and then wraps the data in the context of TL
It returns a simple n_sample x n_features+1  array ( n_features + the labels in first column)
"""
def myDataWrapperForTL(featsDict , tlFeats , tlLabels, seizure2keep = [] , clear = 'all' , post = 'all'):
    seizureData = featsDict["seizures_data"]
    feats2stack = list()
    labels2stack = list()
    nSeizures = featsDict["n_seizures" ]
    ####################################
    feats2stack.append(tlFeats)
    labels2stack.append(tlLabels)
    #############################
    for k in range(nSeizures):
        if not(k+1 in seizure2keep):
            
            allData = seizureData[k]["map"]
            allLabels = allData[: , 1]
            trainOpts = allData[: , 2]
            allFeats = allData[: , 3:]
            # find indices where trainOpt =1
            ind = np.nonzero(trainOpts)
            feats = allFeats[ind , :]
            labels = allLabels[ind]
            feats2stack.append(feats[0])
            labels2stack.append(labels)
    # now deal with seizureFree Files if they exist
    if clear == 'all' or len(clear) == 0:
        seizureFree = featsDict["seizurefree_data" ]
        nFiles = len(seizureFree)
        if  nFiles> 0:
            for k in range(nFiles):
                allData = seizureFree[k]["map"]
                allLabels = allData[: , 1]
                trainOpts = allData[: , 2]
                allFeats = allData[: , 3:]
                # find indices where trainOpt =1
                ind = np.nonzero(trainOpts)
                feats = allFeats[ind , :]
                labels = allLabels[ind]
                feats2stack.append(feats[0])
                labels2stack.append(labels)
    elif isinstance(clear ,list) and len(clear) > 0:
        # now deal with seizureFree Files if they exist
        seizureFree = featsDict["seizurefree_data" ]
        nFiles = len(seizureFree)
        if  nFiles> 0:
            for k in range(nFiles):
                if not(k in clear):
                    allData = seizureFree[k]["map"]
                    allLabels = allData[: , 1]
                    trainOpts = allData[: , 2]
                    allFeats = allData[: , 3:]
                    # find indices where trainOpt =1
                    ind = np.nonzero(trainOpts)
                    feats = allFeats[ind , :]
                    labels = allLabels[ind]
                    feats2stack.append(feats[0])
                    labels2stack.append(labels)
    if post == 'all':
        # now deal with seizureFree Files if they exist
        seizureFree = featsDict["post_seizure_data" ]
        nFiles = len(seizureFree)
        if  nFiles> 0:
            for k in range(nFiles):
                if not(k+1 in seizure2keep) and len(seizureFree[k]) > 0:
                    allData = seizureFree[k]["map"]
                    allLabels = allData[: , 1]
                    trainOpts = allData[: , 2]
                    allFeats = allData[: , 3:]
                    # find indices where trainOpt =1
                    ind = np.nonzero(trainOpts)
                    feats = allFeats[ind , :]
                    labels = allLabels[ind]
                    feats2stack.append(feats[0])
                    labels2stack.append(labels)
#     perform zeroFeats padding
#     get the biggest number of feats
#    N = len(feats2stack)
#    maxFeat = 0
#    for k in range(N):
#        l,m = feats2stack[k].shape
#        if m > maxFeat:
#            maxFeat = m
#    # now do the padding
#    for k in range(N):
#        temp = feats2stack[k]
#        l,m = temp.shape
#        if m < maxFeat:
#            pads = np.zeros((l , maxFeat - m))
#            feats2stack[k] = np.hstack((temp , pads))
            
    feats2return = np.vstack(feats2stack)  
    lbls2return = np.concatenate(labels2stack) 
    
    return feats2return , lbls2return        
        

def concatWithFractions(selfFeats , selfLabels , selfFrac , othersFeats \
                        , othersLabels , tlFrac, tlAll=False):
    Nself = len(selfLabels) 
    Nothers = len(othersLabels)
    nFracSelf = int(selfFrac * Nself)  
    nFracOthers = int(tlFrac * Nself) 
    
    if tlAll:
        nFracOthers = Nothers
        Nself = 1 
        
    selfInd = rnd.randint( Nself , size =  nFracSelf)
    tlInd = rnd.randint(Nothers ,  size =  nFracOthers)
    
    feats = np.vstack((selfFeats[selfInd , :] , othersFeats[tlInd , :]))
    labels = np.concatenate((selfLabels[selfInd] , othersLabels[tlInd]))
    
    nPnts = len(labels)
    randInds = rnd.randint(0 , nPnts , nPnts)
    
    return feats[randInds , :] , labels[randInds]


def getSimilarityOrder(sub , allfeats , myPath , shortCut):
    print("Getting similarity scores for subject : {}".format(sub))
    if shortCut:
        subjectClusterPath = "/Users/remy.benmessaoud/Desktop/neuroProject/myEEGdata/formatted_data/SubjectslikelinessOrders_onPreIctalOnly.pkl"
        if os.path.exists(subjectClusterPath) :
            with open(subjectClusterPath, 'rb') as f:
                subjectCluster = pickle.load(f)  
        else:
            raise ValueError('{} does not exist'.format(subjectClusterPath))
            
        cluster = subjectCluster[sub - 1]
        
    else:
        availableSubjects = np.arange(1 , 25)
        nSub = len(availableSubjects)
        feats_train_not_scaled = allfeats
        #####################################################################################
        scaleOpt = True
        if scaleOpt:
            # train scaler
            scaler = StandardScaler()
            feats_train = scaler.fit_transform(feats_train_not_scaled)
        else:
            feats_train = feats_train_not_scaled
        
        availableSubsList = list(availableSubjects).copy()
        subInd = np.where(availableSubjects == sub)[0]
        if len(subInd) > 0:
            availableSubsList.pop(subInd[0])
        others = np.asarray(availableSubsList)
        
        klMat = np.zeros((nSub,))
    
        for other in others:
    
            ########## load data ####################
    #        sub = availableSubjects[kSub]
            print("Doing subject : {}".format(other))
            fileSpec_other = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(other , winSize , preIctal)
            filePath = os.path.join(myPath , fileSpec_other) 
            if os.path.exists(filePath) :
                with open(filePath, 'rb') as f:
                    otherStruct = pickle.load(f)  
            else:
                raise ValueError('{} does not exist'.format(filePath))
                
            allfeats_other , alllabels_other , maxFeats = pipe.myDataWrapperForClassification(otherStruct )        
            #allfeats_other , alllabels_other , maxFeats = pipe.myPreIctalDataWrapper(otherStruct , [])
            feats_train_not_scaled_other , labels_train_other = pipe.seperateAndPermute(allfeats_other , alllabels_other)
            #feats_train_not_scaled_other = allfeats_other
            if scaleOpt:
    
                feats_train_other = scaler.transform(feats_train_not_scaled_other)
            else:
                feats_train_other = feats_train_not_scaled
            
            klMat[ other-1] = KLdivergence(feats_train , feats_train_other)
    
        # get orders per subject
    
        for k in range(nSub):
            div = klMat[k , :]
            sortInd = np.argsort(div)
            subs = sortInd + np.ones(sortInd.shape)
            
            orderedSubsList = list(subs).copy()
            subInd = np.where(subs == k+1)[0]
            if len(subInd) > 0:
                orderedSubsList.pop(subInd[0])
            cluster = np.asarray(orderedSubsList)

    return cluster


    
    
    
    