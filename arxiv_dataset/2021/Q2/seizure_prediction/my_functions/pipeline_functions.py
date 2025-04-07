#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:42:21 2019

@author: remy.benmessaoud
"""

""" Here we put all the intermediate functions necessary to the pipeline"""
import os.path
import numpy as np
import pickle
import  numpy.random as rnd
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import neighbors

dataPath = "/Users/remy.benmessaoud/Desktop/neuroProject/myEEGdata/formatted_data/features"
monoFeats=25

monoFeaturesDict=["Mean" , "Crest" , "Trough" , "Var" , "Skw",  "Kurt", "DFA" , "HMob" , "HComp" ,\
                  "dCorrTime" , "PFD" , "HFD10" , "SpEn" ,  "PSI_delta" , "RIR_delta" , "PSI_theta" ,\
                  "RIR_theta" , "PSI_alfa" , "RIR_alfa" , "PSI_beta2" , "RIR_beta2" , "PSI_beta1" ,\
                   "RIR_beta1" ,  "PSI_gamma" , "RIR_gamma"]
"""
Function that compares different classification algorithms
"""
#===========================================================================
def compareAlgo(data , labels , algos , scaleOpt = True , cv = 5):
    
    if scaleOpt:
        data = scale(data)
    res = dict()
    keys = list(algos)
    for key in keys:
        res.setdefault(key)
    
    for algo in keys:
        if algo == 'svc':
        #================== SVC ========================        
            if algos[algo]:
                clf = SVC(gamma='auto')
                scores = cross_validate(clf, data , labels , cv=cv, return_train_score=True)
                res[algo] = [100*np.mean(scores["test_score"]) , 100*np.mean(scores["train_score"])]
        if algo == 'tree':
        #================== tree ========================        
            if algos[algo]:
                clf = tree.DecisionTreeClassifier()
                scores = cross_validate(clf, data , labels , cv=cv, return_train_score=True)
                res[algo] = [100*np.mean(scores["test_score"]) , 100*np.mean(scores["train_score"])]
        if algo == 'lda':
        #================== lda ========================        
            if algos[algo]:
                clf = LDA()
                scores = cross_validate(clf, data , labels , cv=cv, return_train_score=True)
                res[algo] = [100*np.mean(scores["test_score"]) , 100*np.mean(scores["train_score"])]
        if algo == 'qda':
        #================== qda ========================        
            if algos[algo]:
                clf = QDA()
                scores = cross_validate(clf, data , labels , cv=cv, return_train_score=True)
                res[algo] = [100*np.mean(scores["test_score"]) , 100*np.mean(scores["train_score"])]
        if algo == 'knn':
        #================== knn ========================        
            if algos[algo]:
                clf = neighbors.KNeighborsClassifier(n_neighbors=7)
                scores = cross_validate(clf, data , labels , cv=cv, return_train_score=True)
                res[algo] = [100*np.mean(scores["test_score"]) , 100*np.mean(scores["train_score"])]
    return res

def printComparedAlgo(res):
    for algo in list(res):
        if not(res[algo]  is None):  
            print("====================== {} =========================\ntest score = {:.3f} %\ntrain score = {:.3f} %".\
                  format(algo , res[algo][0] , res[algo][1]))


#==============================================================================    
"""    nFeatures = 37 monovariate + connectivity 
    Nconnect(n) = n(n-1)/2
    nFeatures(n) = 37 * n + n(n-1)/2
"""
def getNchannfromFeatSize(featSize):
    # eq n2 -73n -2*featSize = 0
    #73= monofeats*2 -1
    rdelta = np.sqrt((monoFeats*2 -1)**2 + 8 * featSize)
    n = (rdelta - (monoFeats*2 -1))/2
    if n < 0 or n- int(n) !=0:
        raise ValueError(' number of features incorrect')
    return int(n)

#=============================================================================
def getFeatNames(nChan , featSize , withLabels = True):
    names = list()
    if withLabels:
        names.append( "target")
        
    for kchan in range(nChan):
        for kfeat in range(monoFeats):
           newName = "ch{}_".format(kchan + 1) + monoFeaturesDict[kfeat]
           names.append(newName)
    N = nChan
    for i in range(N - 1):
        for j in range(i + 1,N):
            newName = "corr({} , {})".format(i + 1 , j + 1)
            names.append(newName)
    return np.asarray(names)

#n = 5
#feats = n * monoFeats + n*(n-1)/2
#names = getFeatNames(n , feats)

"""function that gets the best feaures from the RFE ranks"""
def getBestFeatures(data , ranks , lastRank):
    #get ones indices
    indOnes = np.asarray(np.where(ranks <= lastRank))
    temp = data[: , indOnes] 
    l,s,d = temp.shape
    return    temp.reshape((l,d))

def getfeatsHistogram(nChan , featsRank , optNumber=None):
    mono=monoFeats * nChan 
    scores = np.zeros((monoFeats + 1,))
    nfeats = len(featsRank)
    if optNumber is None:
        stopBound = 2
    else:
        stopBound = int((optNumber + nfeats)/2)
    
    for k in range(nfeats):
        if k < mono:
            if featsRank[k] < stopBound:
                scores[k % monoFeats] +=1
        else:
            if featsRank[k] < stopBound:
                scores[-1] +=1
    normScores = np.zeros((monoFeats + 1,))
    normScores[0 : -1] = scores[0 : -1]/monoFeats
    normScores[-1] = scores[-1]/(nChan*(nChan-1)/2)
    return scores , normScores
 

def getChannelsHistogram(nChan , featsRank , optNumber=None):
    mono=monoFeats * nChan 
    scores = np.zeros((nChan ,))
    nfeats = len(featsRank)
    if optNumber is None:
        stopBound = 2
    else:
        stopBound = int((optNumber + nfeats)/2)
    
    for k in range(nfeats):
        if k < mono:
            if featsRank[k] < stopBound:
                scores[k % nChan] +=1
       
    return scores                
"""=========================================================================
function that gets the features Dictionnary and then wraps the data in the context of 
a leave one seizure out cross validation. 
It returns a simple n_sample x n_features+1  array ( n_features + the labels in first column)
This array is supposed to be given as input for the function new_seperate data

    featuresDict={ "n_seizures" : nSeizures , "file_names" : list() , "seizureInFile" : list(), "seizures_data" : list() , \
             "post_seizure_data" : list() , "seizurefree_data" : list() , "autoreject_threshold" : list()} 
"""
def myDataWrapperForClassification(featsDict , seizure2keep = [] , clear = 'all' , post = 'all'):
    seizureData = featsDict["seizures_data"]
    feats2stack = list()
    labels2stack = list()
    nSeizures = featsDict["n_seizures" ]
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
            
    if clear == 'all' or len(clear) == 0:
        # now deal with seizureFree Files if they exist
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
     
    # perform zeroFeats padding
    # get the biggest number of feats
    N = len(feats2stack)
    maxFeat = 0
    for k in range(N):
        l,m = feats2stack[k].shape
        if m > maxFeat:
            maxFeat = m
    # now do the padding
    for k in range(N):
        temp = feats2stack[k]
        l,m = temp.shape
        if m < maxFeat:
            pads = np.zeros((l , maxFeat - m))
            feats2stack[k] = np.hstack((temp , pads))
            
    feats2return = np.vstack(feats2stack)  
    lbls2return = np.concatenate(labels2stack) 
    
    return feats2return , lbls2return , maxFeat
                
"""=========================================================================
function that gives only preIctal samples
"""

def myPreIctalDataWrapper(featsDict , seizure2keep = []):
    seizureData = featsDict["seizures_data"]
    feats2stack = list()
    labels2stack = list()
    nSeizures = featsDict["n_seizures" ]
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
            
    # perform zeroFeats padding
    # get the biggest number of feats
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
    
    return feats2return , lbls2return , 0

                
"""=========================================================================
function that gets samples only from the seizure free files"
"""
def myInterIctalDataWrapper(featsDict ):
    feats2stack = list()
    labels2stack = list()
    # now deal with seizureFree Files if they exist
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
     
    # perform zeroFeats padding
    # get the biggest number of feats
#    N = len(feats2stack)
#    maxFeat = 0
#    for k in range(N):
#        l,m = feats2stack[k].shape
#        if m > maxFeat:
#    maxFeat = m
#    # now do the padding
#    for k in range(N):
#        temp = feats2stack[k]
#        l,m = temp.shape
#        if m < maxFeat:
#            pads = np.zeros((l , maxFeat - m))
#            feats2stack[k] = np.hstack((temp , pads))
    if len(feats2stack) > 0:           
        feats2return = np.vstack(feats2stack)  
        lbls2return = np.concatenate(labels2stack) 
    else:
        feats2return = None
        lbls2return = None 
    return feats2return , lbls2return , 0

"""=========================================================================
function that gets feats and labels from above function and then does the following steps:
    - seperate the data in 2 classes 
    _ gets equal number of samples fr both classes 
    _ stack and permute the samples
"""
def seperateAndPermute(feats , labels):
    ind0 = np.where(labels==0)[0]
    ind1 = np.where(labels==1)[0]
    labels0 = labels[ind0]
    labels1 = labels[ind1]    
    feats0 = feats[ind0 , :]
    feats1 = feats[ind1 , :]
    # find shortest array
    n0 =len(labels0)
    n1 =len(labels1)
    n = min(n0 , n1)
    rndind0 = rnd.randint(0 , n0 , n)
    rndind1 = rnd.randint(0 , n1 , n)
    
    # concatenate both classes 
    allLabels = np.concatenate((labels0[rndind0] , labels1[rndind1]))
    allFeats = np.vstack((feats0[rndind0 , :] , feats1[rndind1 , :]))
    N = len(allLabels)
    ind = rnd.randint(0 , N , N)
    
    return allFeats[ind , :] ,  allLabels[ind] 
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""" redo the same thing ( 2 functions wrap and permute ) """
def myEpochsWrapperForClassification(featsDict ): # here the features are the epochs
    seizureData = featsDict["seizures_data" ]
    feats2stack = list()
    labels2stack = list()
    nSeizures = featsDict["n_seizures" ]
    for k in range(nSeizures):
        allData = seizureData[k]["map"]
        allLabels = allData[: , 1]
        trainOpts = allData[: , 2]
        allFeats = seizureData[k]["epochs"]
        # find indices where trainOpt =1
        ind = np.nonzero(trainOpts)[0]
        feats = allFeats[ind]
        labels = allLabels[ind]
        feats2stack.extend(feats)
        labels2stack.append(labels)
    # now deal with seizureFree Files if they exist
    seizureFree = featsDict["seizurefree_data" ]
    nFiles = len(seizureFree)
    if  nFiles> 0:
        for k in range(nFiles):
            allData = seizureFree[k]["map"]
            allLabels = allData[: , 1]
            trainOpts = allData[: , 2]
            allFeats = seizureFree[k]["epochs"]
            # find indices where trainOpt =1
            ind = np.nonzero(trainOpts)[0]
            feats = allFeats[ind ]
            labels = allLabels[ind]
            feats2stack.extend(feats)
            labels2stack.append(labels)
        
    return feats2stack , np.concatenate(labels2stack)

#=============================================================================
"""function that gets the same number of samples for each class and permute them """
def getBadEpochs(feats , thresh):
    n , l = feats.shape
    nchans = getNchannfromFeatSize(l)
    
    #get peak2peak values for all samples
    peak2peak = np.zeros((n,))
    for k in range(n):
        crests = np.zeros((nchans,))
        troughs = np.zeros((nchans,))
        for kChan in range(nchans):
            crests[kChan] = feats[k , kChan * monoFeats + 1]
            troughs[kChan] = feats[k , kChan * monoFeats + 2]
        peak2peak[k] = max(crests - troughs)
    # look where peak2peak is greater than thresh
    return ( peak2peak > thresh ).astype(int)


"""=========================================================================
function that gets feats and labels from above function and then does the following steps:
    - seperate the data in 2 classes 
    _ gets equal number of samples fr both classes 
    _ stack and permute the samples
"""
def seperateAndPermuteEpochs(feats , labels):
    ind0 = np.where(labels==0)[0].astype(int)
    ind1 = np.where(labels==1)[0].astype(int)
    labels00 = labels[ind0]
    labels11 = labels[ind1]  
    
    feats00 = np.asarray(feats)[ind0 ]
    feats11 = np.asarray(feats)[ind1 ]
    
    # rearange feats1  : get rid of epochs with less channels
    n = len(feats11)
    l1 , m = feats11[0].shape
    l2 , m = feats11[-1].shape
    l = max(l1,l2)
    epochsList = list()
    lbls = labels11
    myLabels = list()
    for k in range(n):
        if feats11[k].shape == (l,m):
            epochsList.append( feats11[k] )
            myLabels.append(lbls[k])
    feats1 = np.stack(epochsList , 0) 
    labels1 = np.asarray(myLabels)
    # rearange feats1  : get rid of epochs with less channels
    n = len(feats00)
    l1 , m = feats00[0].shape
    l2 , m = feats00[-1].shape
    l = max(l1,l2)
    epochsList = list()
    lbls = labels00
    myLabels = list()
    for k in range(n):
        if feats00[k].shape == (l,m):
            epochsList.append( feats00[k] )
            myLabels.append(lbls[k])
    feats0 = np.stack(epochsList , 0) 
    labels0 = np.asarray(myLabels)
    
    # find shortest array
    n0 =len(labels0)
    n1 =len(labels1)
    n = min(n0 , n1)
    rndind0 = rnd.randint(0 , n0 , n)
    rndind1 = rnd.randint(0 , n1 , n)
    
    # concatenate both classes 
    allLabels = np.concatenate((labels0[rndind0] , labels1[rndind1]))
    allFeats = list()
    allFeats.extend(feats0[rndind0])
    allFeats.extend(feats1[rndind1])
    N = len(allLabels)
    ind = rnd.randint(0 , N , N)
    ffs = np.asarray(allFeats)[ind] 
    
    # rearrange the epochs in 3d array
    n = len(ffs)
    l , m = ffs[0].shape
    epochsList = list()
    lbls = allLabels[ind] 
    myLabels = list()
    for k in range(n):
        if ffs[k].shape == (l,m):
            epochsList.append( ffs[k] )
            myLabels.append(lbls[k])
    epochs = np.stack(epochsList , 0) 
    
    return epochs ,  np.asarray(myLabels)

""" ==================================
======================================
"""
def epochsWrapperForCrossVal(featsDict , seizure2keep , clear = 'all' , post = 'all' ): # here the features are the epochs
    seizureData = featsDict["seizures_data" ]
    feats2stack = list()
    labels2stack = list()
    nSeizures = featsDict["n_seizures" ]
    for k in range(nSeizures):
        if not(k+1 in seizure2keep):
            allData = seizureData[k]["map"]
            allLabels = allData[: , 1]
            trainOpts = allData[: , 2]
            allFeats = seizureData[k]["epochs"]
            # find indices where trainOpt =1
            ind = np.nonzero(trainOpts)[0]
            feats = allFeats[ind]
            labels = allLabels[ind]
            feats2stack.extend(feats)
            labels2stack.append(labels)
    if post == 'all':
        # now deal with seizureFree Files if they exist
        seizureFree = featsDict["post_seizure_data" ]
        nFiles = len(seizureFree)
        if  nFiles> 0:
            for k in range(nFiles):
                allData = seizureFree[k]["map"]
                allLabels = allData[: , 1]
                trainOpts = allData[: , 2]
                allFeats = seizureFree[k]["epochs"]
                # find indices where trainOpt =1
                ind = np.nonzero(trainOpts)[0]
                feats = allFeats[ind ]
                labels = allLabels[ind]
                feats2stack.extend(feats)
                labels2stack.append(labels)
                
    if clear == 'all':
        # now deal with seizureFree Files if they exist
        seizureFree = featsDict["seizurefree_data" ]
        nFiles = len(seizureFree)
        if  nFiles> 0:
            for k in range(nFiles):
                allData = seizureFree[k]["map"]
                allLabels = allData[: , 1]
                trainOpts = allData[: , 2]
                allFeats = seizureFree[k]["epochs"]
                # find indices where trainOpt =1
                ind = np.nonzero(trainOpts)[0]
                feats = allFeats[ind ]
                labels = allLabels[ind]
                feats2stack.extend(feats)
                labels2stack.append(labels)
        
    return feats2stack , np.concatenate(labels2stack)

""" function that takes S and R and gives S - R , i.e all the elements that are in S and not in R """
def setSubstraction(s , r):
    res = list()
    for k in s:
        if not(k in r):
            res.append(k)
    return np.asarray(res)

#=============================================================================
"""function that seperates samples for two classes"""
def seperateData(path , fileName , saveOpt = True , runAnyway = False):
    # check if seperate file already exists
    targetFile =os.path.join(path , "seperate_" + fileName + ".pkl")
    if os.path.exists(targetFile) and not(runAnyway):
        with open(targetFile, 'rb') as f:
            seperate = pickle.load(f)
        return seperate
    
    else:
        #load data
        fullPath = os.path.join(path , fileName + ".npy")
        if os.path.exists(fullPath):
            data = np.load(fullPath, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        else:
            raise ValueError('{} does not exist'.format(fullPath))
        labels = data[:,0]
        feats = data[: , 1:]
        
        labels0=list()
        feats0 = list()
        labels1=list()
        feats1 = list()
        n = len(labels)
        for k in range(n):
            lbl = labels[k]
            if lbl == 0:
                labels0.append(lbl)
                feats0.append(feats[k , :])
            else:
                labels1.append(lbl)
                feats1.append(feats[k , :])
        
        if len(labels0) + len(labels1) != n:
            raise ValueError(' lengthes not coinciding')
        # save result
        res = {"labels0" : np.asarray(labels0) , "features0": np.vstack(feats0) ,\
               "labels1" : np.asarray(labels1) , "features1" : np.vstack(feats1)}
        if saveOpt:
            targetFile ="seperate_" + fileName + ".pkl"
            targetPath = os.path.join(path , targetFile)
            f = open(targetPath,"wb")
            pickle.dump(res,f)
            f.close()
        return res 
    

#=============================================================================
"""function that gets the same number of samples for each class and permute them """
def getSamplesAndPermute(sub , winSize):
    fileSpec = "seperate_subject_{}_windowSize_{}".format(sub , winSize)
    filePath = os.path.join(dataPath , fileSpec + ".pkl") 
    if os.path.exists(filePath) :
        with open(filePath, 'rb') as f:
            seperate = pickle.load(f)
    else: 
        seperate = seperateData(dataPath , fileSpec)
    
    labels0 = seperate["labels0"]
    feats0 = seperate["features0"] 
    labels1 = seperate["labels1"]
    feats1 = seperate["features1"]
    n0 = len(labels0)
    n1 = len(labels1)
    n=min(n0 , n1)
    ind0 = rnd.randint(0 , n0 , n)
    ind1 = rnd.randint(0 , n1 , n)
    
    labels = np.concatenate((labels0[ind0] , labels1[ind1]))
    feats = np.vstack((feats0[ind0] , feats1[ind1]))
    n2 = len(labels)
    inds = rnd.randint(0 , n2 , n2)
    nSamples = len(labels)
    dataStruct = np.hstack((labels[inds].reshape((nSamples , 1)) , feats[inds , :]))
    return dataStruct

