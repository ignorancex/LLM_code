#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:44:12 2019

@author: remy.benmessaoud
"""
#from my_functions import mypreprocessing_for_epochs
from my_functions import pipeline_functions as pipe
from my_functions import postprocessing_functions as post
import numpy as np 
#import mne
import os.path
#import matplotlib.pyplot as plt
import os
import pickle

#from sklearn.pipeline import Pipeline
#from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection  import cross_val_score
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
import warnings
#import copy
from scipy.special import binom
import my_functions.my_transfer_learning_functions as tl
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#=============================================================================================
"""function that gets sensitivity and warning rate (specificity) from a single run
These metrics are measured reagrding this source :
    
The statistics of a practical seizure warning system
David E Snyder1, Javier Echauz2,4, David B Grimes1 and Brian Litt3,4
1 NeuroVista Corporation, 100 4th Avenue North, Suite 600, Seattle, WA 98109, USA
2 JE Research, Inc., 170 Wentworth Terrace, Alpharetta, GA 30022, USA
3 Department of Neurology, University of Pennsylvania, 3400 Spruce St, Philadelphia, PA 19104, USA
"""



def getPerf(times , scores , seizuresTimes , sph , sop , thresh , sustain = True , sustainPoints = 3 , \
            bads = None , badsHorizon = 0 , step = 5 ):
    
    epsilon = 10**-6
    if not(bads is None):
        autoreject = True
    else:
        autoreject = False
        
    N = len(seizuresTimes)
    lastTime = 0
    isDetected = np.zeros((N,))
    falseWarnings = np.zeros((N + 1,))
    warnings = (scores >= thresh).astype(int)
    """# now get the reduced warning vector"""
    nPnts = len(warnings)
#    allWarnings = copy.deepcopy(warnings)
    fileEnd = times[-1]
    nextSeizure = 0
    for k in range(nPnts - 1):
        # here we keep only the first sample that crosses the thresh
        t1 = times[k]
        # look if we reached a new seizure
        if nextSeizure < N:
            if t1 >= seizuresTimes[nextSeizure][0]:
                nextSeizure = nextSeizure + 1
            
        if warnings[k] == 1:
            l = 1
            t2 = times[k + l]
            ind2makezero = list()
            ind2BeingOne = list()
            if nextSeizure <= N - 1:
                onSet = seizuresTimes[nextSeizure][0]
            else:
                onSet = fileEnd
                
            stopCountingOnes = False    
            while k + l < nPnts - 1  and t2 - t1 < min(sph + sop , onSet - t1):
                ind2makezero.append(k + l)
                if warnings[k + l] == 1 and not(stopCountingOnes):
                    ind2BeingOne.append(k + l)
                else:
                    stopCountingOnes = True
                l = l + 1
                t2 = times[k + l]
            # now pass the useless warning
            nInds = len(ind2makezero)
            nIndsOne = len(ind2BeingOne)
                ##################################
            if not(autoreject) :
                if sustain :
                    if nIndsOne < sustainPoints - 1 : # Here we don't consider it as an alarm
                        warnings[np.asarray([k] , dtype = int)] = 0
                    else:# ALARM
                        warnings[np.asarray(ind2makezero , dtype = int)] = np.zeros((nInds ,))
                else:## Here it's an alarm
                    warnings[np.asarray(ind2makezero , dtype = int)] = np.zeros((nInds,))
             
                """ AUTOREJECT"""
            else: 
                if sustain :
                    if nIndsOne < sustainPoints - 1 :# not an alarm 
                        warnings[np.asarray([k] , dtype = int)] = 0
                    else: # the sustainibility condition is fullfilled 
                        badsHorizon = sustainPoints - 1
                        if np.sum(bads[k + sustainPoints - 1 - badsHorizon : k + sustainPoints ]) > badsHorizon/2 + epsilon: # In this case we have too many bad epochs in the considered window
                            warnings[np.asarray([k] , dtype = int)] = 0
                        else:# ALARM because not enough badEps to dicredit it
                            warnings[np.asarray(ind2makezero , dtype = int)] = np.zeros((nInds ,))
                else:
                    if np.sum(bads[k - badsHorizon : k + 1]) > badsHorizon/2 + epsilon: # it means it is a bad epoch
                        warnings[np.asarray([k] , dtype = int)] = 0
                    else:
                        warnings[np.asarray(ind2makezero , dtype = int)] = np.zeros((nInds,))
            
#    # now deal with the warnings happening at the end of the file
    endInd = np.where(np.logical_and(times > fileEnd - sop - sph , times > seizuresTimes[-1][1]))[0]
    warnings[endInd] = np.zeros(endInd.shape)
    #############################################
#    plt.figure()
#    plt.subplot(2 , 1 , 1)
#    plt.plot(times /60, allWarnings )#, marker = 'x' , linestyle = '')
#    plt.subplot(2 , 1 , 2)
#    plt.plot(times/60 , warnings )#, marker = 'x' , linestyle = '')  
#    for kSeiz in range(N):
#        start = seizuresTimes[kSeiz][0]
#        end = seizuresTimes[kSeiz][1]
#        xSeizure = np.array([start , start , end , end])/60
#        ySeizure = np.array([0 , 1 + 1 , 1 + 1 , 0])
#        plt.fill(xSeizure, ySeizure , 'red')   
        
    """ now look at the seizures """
    predictionTimes = -np.ones((N,))
    for kPart in range(N + 1):
        if kPart < N:
            onSet = seizuresTimes[kPart][0]
            inds = np.where(np.logical_and(times < onSet , times >= lastTime))[0]
            lastTime = seizuresTimes[kPart][1]
            predicted = False
            for i in inds:
                t = times[i]
                if warnings[i] == 1 :
                    if onSet < (t + sph + sop) and (onSet - t >= sph):
                        isDetected[kPart] = 1
                        if not(predicted):
                            if sustain:
                                predictionTimes[kPart] = onSet - t - step * (sustainPoints - 1)
                            else:
                                predictionTimes[kPart] = onSet - t 
                            predicted = True
                    else:
                        falseWarnings[kPart] = falseWarnings[kPart] + 1
                    
        else:
            inds = np.where(times > lastTime)[0]
            for i in inds:
                if warnings[i] == 1 :
                    falseWarnings[kPart] = falseWarnings[kPart] + 1
    
                
    return isDetected , falseWarnings , predictionTimes


""" function that keeps only leading Seizures """
def getLeadingSeizures(seizureTimes , times , sph , sop):
    minPredTime =  sop
    # first cluster the seizures
    nSeizures = len(seizureTimes)
    nextIsTooClose = np.zeros((nSeizures ,))
    for kSeiz in range(nSeizures - 1):
        mySeizureEnd = seizureTimes[kSeiz][1]
        nextSeizureOnset = seizureTimes[kSeiz + 1][0]
        if nextSeizureOnset - mySeizureEnd < minPredTime:
            nextIsTooClose[kSeiz] = 1
    
    # now we postProcess nextIsTooClose   
    seizureClusters = list()
    lastCluster = list()
    for k in range(nSeizures ):
        if not(k in lastCluster):
            currentCluster = list()
            currentCluster.append(k)
            nextIsClose = nextIsTooClose[k] == 1
            l = 1
            while nextIsClose and k + l < nSeizures - 1 :
                currentCluster.append(k+l)
                nextIsClose = nextIsTooClose[k + l] == 1
                l = l + 1
            seizureClusters.append(currentCluster) 
            lastCluster = currentCluster
    nLeading = len(seizureClusters)  
    leadingSeizuresTimes  = list()   
    
    avoidFirstSeizure = False
    for lead in range(nLeading):
        start = seizureTimes[seizureClusters[lead][0]][0]
        end = seizureTimes[seizureClusters[lead][-1]][1]
        if lead == 0 and start < minPredTime:
            avoidFirstSeizure = True
        leadingSeizuresTimes.append([ start , end ])
        
    # get new indeces
    newInds = list()
    lastTime = 0
    for kPart in range(nLeading + 1):
        
        if kPart < nLeading:
            if kPart == 0 and avoidFirstSeizure:
                inds = None
                lastTime = leadingSeizuresTimes[kPart][1] - 1
            else:
                onSet = leadingSeizuresTimes[kPart][0]
                inds = np.where(np.logical_and(times < onSet , times >= lastTime))[0]
                lastTime = leadingSeizuresTimes[kPart][1]
        else:
            inds = np.where(times >= lastTime)[0]
        if not(inds is None):
            newInds.append(inds.astype(int))  
        
    if avoidFirstSeizure:
        leadingSeizuresTimes.pop(0)
    return leadingSeizuresTimes  , np.concatenate(newInds)


"""function that gives the pValue for rejecting the hypothesis that the performance is better than chance"""
def getPValue(fpr , sop , m , M):
    poisson = 1 - np.exp(-sop * fpr)
    p = 0
    for i in np.arange(m , M + 1):
        p = p + binom(M , i) * (poisson**i) * (poisson**(M - i))
    if p == 0:
        p = 1
    return p





""" function that runs nRuns of the prediction experiment for one subject """
def runExp(sub , nRuns , windowSize = 15 , thresh = 17 , postProcessWindow = 20 , preIctal = 10 , scoreType = 'contrast', CregulSVC = 40,\
        classifier = 'vote' , autorejcetFactor = 1.4 , transferLearning = False , others = 2 , selfFrac = 1 , knnParam = 6,\
   tlFrac = 0.3 , currentPath = '/Users/remy.benmessaoud/Desktop/neuroProject' , weights = None , tlAll=False , \
   n_estimatorsRF = 107 , weightsSVC_RFC = np.array([0.2 , 0.8]) , n_estimatorsADA = 70 , alphaMLP = 0.0002 , featsInds = 'all'):
    
    runScores = list()
    runBads = list()
    numNeighbors = knnParam

    """ ==========================================================================
                            run Exp without transfer Learning
        =========================================================================="""
    if not(transferLearning):
        for kRun in range(nRuns):
            print("clf = {} , subject = {} , batch = {} ".format(classifier , sub , kRun + 1))
            #=============================================================================
            targetPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21_allTimes_{}s".format(int(windowSize)))
            fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , windowSize , preIctal)
            #fileSpec = "intra_{}_windowSize_{}_preIctal_{}_features.pkl".format(sub ,   , preIctal)
            filePath = os.path.join(targetPath , fileSpec) 
            #==============================  open subjects files  ===============================================
            fileExist = os.path.exists(filePath)
            if fileExist:
                subStruct = pickle.load( open( filePath , "rb" ) )
            else:
                raise ValueError(' problem with path :  file doesnt exist {}'.format(filePath))
            #print('data loaded')
    #        
    #        saveOpt = True
    #        filtOpt = True
            
            times2return , feats2return , seiz2return , filesInFig = post.mySerializer(sub , subStruct , windowSize) 
            
            nFigs = len(times2return)
            scaleOpt = True    
            globalautoReject = subStruct["autoreject_threshold"]
    #        fileNames = subStruct["file_names"]
            
            totalTimesList = list()
            totalPredsList = list()
            totalBads = list()
            seizureTimes = list()
            
            lastTimeInFig = 0
            lastInd = 0
            for fig in range(nFigs)  :
                #print('doing figure {}/{}'.format(fig + 1 , nFigs))
                """ here we ttrain on the rest of the windows """
                myTimes =   times2return[fig]
                tempFeats = feats2return[fig]
                figureFiles = filesInFig[fig]
                clear2keep , seiz2keep,= post.abs2clearAndSeiz(figureFiles , subStruct)
                mySeizInfo = seiz2return[fig]
                nSeizureInFig  = len(mySeizInfo)
                if nSeizureInFig > 0:
                    for toto in range(nSeizureInFig):
                        seizureTimes.append([lastTimeInFig + mySeizInfo[toto][0]  ,  lastTimeInFig + mySeizInfo[toto][1]])
                    
                if nSeizureInFig < 2:
                    """ in case there is less than one seizure per figure """
                    
                    ############### get training data ###################    
                    allfeats , alllabels , maxFeats = pipe.myDataWrapperForClassification(subStruct ,\
                                                                                seizure2keep = seiz2keep , clear = clear2keep)
                    feats_train_not_scaled , labels_train = pipe.seperateAndPermute(allfeats , alllabels)
                    
                    
                    if not(isinstance(featsInds, str) ):
                        feats_train_not_scaled = feats_train_not_scaled[: , featsInds]
                    
                    if scaleOpt:
                        # train scaler
                        scaler = StandardScaler()
                        feats_train = scaler.fit_transform(feats_train_not_scaled)
                    else:
                        feats_train = feats_train_not_scaled
                    
                    if classifier == 'vote' or classifier == 'all':
                        # Assemble a classifier
                        lda = LinearDiscriminantAnalysis()
                        svc = SVC(C = CregulSVC , kernel = "linear" ,  probability=True)
                        knn = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            lda.fit(feats_train, labels_train)
                        svc.fit(feats_train, labels_train)
                        knn.fit(feats_train, labels_train)
                    
                    elif classifier == 'svc+rfc':
                        svc = SVC(C = CregulSVC , kernel = "linear" ,  probability=True)
                        rfc = RandomForestClassifier(n_estimators=n_estimatorsRF)
                        
                        svc.fit(feats_train, labels_train)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            rfc.fit(feats_train, labels_train)
                        
                        
                    elif classifier == 'lda':
                        clf = LinearDiscriminantAnalysis()
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # fit classifier
                            #tempScore = cross_val_score(lda, feats_train, y= labels_train  , cv= 3 )
                            clf.fit(feats_train, labels_train)
                    elif classifier == 'svc':
                        clf = SVC(C = CregulSVC , kernel = "linear" ,  probability=True)
                        clf.fit(feats_train, labels_train)
                    elif classifier == 'knn':
                        clf = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                        clf.fit(feats_train, labels_train)
                    elif classifier == 'rfc':    
                        clf = RandomForestClassifier(n_estimators=n_estimatorsRF)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(feats_train, labels_train)
                    elif classifier == 'ada': 
                        clf = AdaBoostClassifier( n_estimators=n_estimatorsADA, learning_rate=1)
                        clf.fit(feats_train, labels_train)
                    
                    elif classifier == 'mlp': 
                        clf = MLPClassifier( alpha = alphaMLP)
                        clf.fit(feats_train, labels_train)
                    else:
                        raise ValueError(' Classifier not recognized')
                            
                    """delete training data"""
                    del allfeats , alllabels , maxFeats , feats_train_not_scaled , labels_train
                    # get the autoreject threshold
                    if len(figureFiles)==0 :
                        raise ValueError(' problem in finding the good file name')
                    else:
                        temp = globalautoReject.copy()  
                        figureFiles.reverse()
                        for kFile in figureFiles:
                            temp.pop(kFile)    
                            
                        autorejectThresh = np.mean(np.asarray(temp)) 
                        """ here we test on the given window """
                        badEps = pipe.getBadEpochs(tempFeats , autorejcetFactor * autorejectThresh)
                    
                    if not(isinstance(featsInds, str)):
                        tempFeats = tempFeats[: , featsInds]
                    
                    if scaleOpt:
                        myFeats = scaler.transform(tempFeats)
                    else:
                        myFeats = tempFeats
                    
                    del tempFeats
                    
                    nSamps = len(myTimes)
                    if scoreType == 'labels':
                        if classifier == 'vote':
                            allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                          knn.predict(myFeats).reshape(nSamps,1) ))
                            finalPred = np.around(np.average(allPreds , axis = 1 , weights = weights))
                        elif classifier == 'all':
                            allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                          knn.predict(myFeats).reshape(nSamps,1) ))
                            votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                            finalPred = np.hstack((allPreds , votePred))
                        else:
                            finalPred = clf.predict(myFeats).reshape(nSamps,1)
                    elif scoreType == 'proba':
                        if classifier == 'vote':
                            allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                      knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                            finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                        elif classifier == 'all':
                            allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                      knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                            votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                            finalPred = np.hstack((allPreds , votePred))
                        else:
                            finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            
                    elif scoreType == 'contrast':
                        if classifier in ['vote' , 'all']:
                            list2stack = list()
                            ###################################
                            preIctProb = lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = lda.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            preIctProb = svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = svc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            preIctProb = knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = knn.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            allPreds = np.hstack(list2stack)
                            if classifier == 'vote':
                                finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                            elif classifier == 'all':
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                                
                        elif classifier == 'svc+rfc':
                            list2stack = list()
                            ###################################
                            preIctProb = svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = svc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            preIctProb = rfc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = rfc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            allPreds = np.hstack(list2stack)
                            votePred = np.around(np.average(allPreds , axis = 1 , weights = weightsSVC_RFC)).reshape(nSamps,1) 
                            finalPred = np.hstack((allPreds , votePred))
                        
                        else:
                            finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                    else:
                        raise ValueError(' scoreType argument is not recognized')
                        
                    del myFeats
            #        counts = post.slidingWindow(myTimes , finalPred , seizureInfo = mySeizInfo, \
            #                                                postProcessWindow = postProcessWindow , alarmThresh = thresh)
                    
                    
                    totalTimesList.append(myTimes + lastTimeInFig * np.ones(myTimes.shape))
                    totalPredsList.append(finalPred)
                    totalBads.append(badEps)# + lastInd * np.ones(badEps.shape)) 
                    lastTimeInFig = lastTimeInFig + myTimes[-1]
                    lastInd = lastInd + len(myTimes)
                else:
                    """ in case there are more than one seizure per figure """
                    predsList = list()
                    timesList = list()
                    badEpsList = list()
                    
                    # get the autoreject threshold
                    if len(figureFiles)==0 :
                        raise ValueError(' problem in finding the good file name')
                    else:
                        temp = globalautoReject.copy()  
                        figureFiles.reverse()
                        for kFile in figureFiles:
                            temp.pop(kFile)    
                            
                        autorejectThresh = np.mean(np.asarray(temp)) 
                    # get number of parts sin figure    
                    nParts = nSeizureInFig
                    lastTime = -1
                    for kPart in range(nParts) :  
                        # get part indices
                        if kPart < nParts - 1:
                            inds = np.where(np.logical_and(myTimes < mySeizInfo[kPart][1] , myTimes > lastTime))[0]
                            lastTime = mySeizInfo[kPart][1] - 1
                        else:
                            inds = np.where(myTimes > lastTime)[0]
                       
                        tempFeatsPart = tempFeats[inds , :]
                        myTimesPart = myTimes[inds]
                        
                        ############### get training data ###################    
                        allfeats , alllabels , maxFeats = pipe.myDataWrapperForClassification(subStruct ,\
                                                                                              seizure2keep = [seiz2keep[kPart]] , clear = clear2keep)
                        feats_train_not_scaled , labels_train = pipe.seperateAndPermute(allfeats , alllabels)
                        
                        if not(isinstance(featsInds, str)):
                            feats_train_not_scaled = feats_train_not_scaled[: , featsInds]
                        
                        if scaleOpt:
                            # train scaler
                            scaler = StandardScaler()
                            feats_train = scaler.fit_transform(feats_train_not_scaled)
                        else:
                            feats_train = feats_train_not_scaled
                        if classifier == 'vote' or classifier == 'all':
                            # Assemble a classifier
                            lda = LinearDiscriminantAnalysis()
                            svc = SVC(C = CregulSVC , kernel = "linear" ,  probability=True)
                            knn = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                lda.fit(feats_train, labels_train)
                            svc.fit(feats_train, labels_train)
                            knn.fit(feats_train, labels_train)
                        
                        elif classifier == 'svc+rfc':
                            svc = SVC(C = CregulSVC , kernel = "linear" ,  probability=True)
                            rfc = RandomForestClassifier(n_estimators=n_estimatorsRF)
                            
                            svc.fit(feats_train, labels_train)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                rfc.fit(feats_train, labels_train) 
                            
                        elif classifier == 'lda':
                            clf = LinearDiscriminantAnalysis()
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                # fit classifier
                                #tempScore = cross_val_score(lda, feats_train, y= labels_train  , cv= 3 )
                                clf.fit(feats_train, labels_train)
                        elif classifier == 'svc':
                            clf = SVC(C = CregulSVC , kernel = "linear" ,  probability=True)
                            clf.fit(feats_train, labels_train)
                        elif classifier == 'knn':
                            clf = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                            clf.fit(feats_train, labels_train)
                        elif classifier == 'rfc':    
                            clf = RandomForestClassifier(n_estimators=n_estimatorsRF)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                clf.fit(feats_train, labels_train)
                        elif classifier == 'ada': 
                            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2) , n_estimators=600, learning_rate=1)
                            clf.fit(feats_train, labels_train)
                            
                        elif classifier == 'mlp': 
                            clf = MLPClassifier( alpha = alphaMLP)
                            clf.fit(feats_train, labels_train)
                        else:
                            raise ValueError(' Classifier not recognized')
                        """delete training data"""
                        del allfeats , alllabels , maxFeats , feats_train_not_scaled , labels_train
                        
                        """ here we test on the given window """
                        badEps = pipe.getBadEpochs(tempFeatsPart , autorejcetFactor * autorejectThresh)
                        
                        if not(isinstance(featsInds, str)):
                            tempFeatsPart = tempFeatsPart[: , featsInds]
                        
                        if scaleOpt:
                            myFeats = scaler.transform(tempFeatsPart)
                        else:
                            myFeats = tempFeatsPart
                        
                        nSamps = len(tempFeatsPart)
                        del tempFeatsPart
                        
                        if scoreType == 'labels':
                            if classifier == 'vote':
                                allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                              knn.predict(myFeats).reshape(nSamps,1) ))
                                finalPred = np.around(np.average(allPreds , axis = 1 , weights = weights))
                            elif classifier == 'all':
                                allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                              knn.predict(myFeats).reshape(nSamps,1) ))
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                            else:
                                finalPred = clf.predict(myFeats).reshape(nSamps,1)
                        elif scoreType == 'proba':
                            if classifier == 'vote':
                                allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                          knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                                finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                            elif classifier == 'all':
                                allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                          knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                            else:
                                finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                        elif scoreType == 'contrast':
                            if classifier in ['vote' , 'all']:
                                list2stack = list()
                                ###################################
                                preIctProb = lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                                interIctProb = lda.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                                list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                                ##################################
                                preIctProb = svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                                interIctProb = svc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                                list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                                ##################################
                                preIctProb = knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                                interIctProb = knn.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                                list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                                ##################################
                                allPreds = np.hstack(list2stack)
                                if classifier == 'vote':
                                    finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                                elif classifier == 'all':
                                    votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                    finalPred = np.hstack((allPreds , votePred))
                            
                            elif classifier == 'svc+rfc':
                                list2stack = list()
                                ###################################
                                preIctProb = svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                                interIctProb = svc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                                list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                                ##################################
                                preIctProb = rfc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                                interIctProb = rfc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                                list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                                ##################################
                                allPreds = np.hstack(list2stack)
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weightsSVC_RFC)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                        
                            
                            else:
                                finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            
                        else:
                            raise ValueError(' scoreType argument is not recognized')
                            
                        del myFeats
            
                        timesList.append(myTimesPart)
                        predsList.append(finalPred)
                        badEpsList.append(badEps)
                        
                    """ here is the plotting """
                    if classifier in ['all' , 'svc+rfc' ]:
                        totPreds = np.vstack(predsList)
                    else:
                        totPreds = np.concatenate(predsList)
        #            totTimes = np.concatenate(timesList)
                    badEps = np.concatenate(badEpsList)
            #        
            #        totcounts = post.slidingWindow(totTimes , totPreds ,  seizureInfo = mySeizInfo ,\
            #                                       postProcessWindow = postProcessWindow , alarmThresh = thresh)
                    
                    totalTimesList.append(myTimes + lastTimeInFig * np.ones(myTimes.shape))
                    totalPredsList.append(totPreds)
                    totalBads.append(badEps )#+ lastInd * np.ones(badEps.shape)) 
                    lastTimeInFig = lastTimeInFig + myTimes[-1]
                    lastInd = lastInd + len(myTimes) 
                    
            """ Here we evaluate the performances of each run """
            times = np.concatenate(totalTimesList)     
            
            if classifier in ['all' , 'svc+rfc' ]:
                preds = np.vstack(totalPredsList) 
            else:
                preds = np.concatenate(totalPredsList) 
             
            
            bads = np.concatenate(totalBads) 
#            # seizureTimes
#            # here we get the scores
#            if classifier =='all':
#                countsList = list()
#                for kClass in range(4):
#                    tempCounts = post.slidingWindow(times , preds[: , kClass] ,  seizureInfo = seizureTimes ,\
#                                                   postProcessWindow = postProcessWindow )
#                    nCounts = len(tempCounts)
#                    countsList.append(tempCounts.reshape((nCounts , 1)))
#                counts = np.hstack(countsList)
#            else:
#                counts = post.slidingWindow(times , preds ,  seizureInfo = seizureTimes ,\
#                                                   postProcessWindow = postProcessWindow )
        
            runScores.append(preds)
            runBads.append(bads)
    """ ==========================================================================
                            run Exp with transfer Learning
        =========================================================================="""
    if (transferLearning):
        subjectClusterPath = os.path.join(\
                    currentPath , "myEEGdata/formatted_data/SubjectslikelinessOrders_onPreIctalOnly.pkl" )
        if os.path.exists(subjectClusterPath) :
            with open(subjectClusterPath, 'rb') as f:
                subjectsCluster = pickle.load(f)  
        else:
            raise ValueError('{} does not exist'.format(subjectClusterPath))
        """we get only preictal samples from other subjects """    
        feats_others , labels_others = tl.getpreIctalFromOthers1(subjectsCluster[sub -1 ].astype(int) , others  , sub , \
                                                                 currentPath , windowSize , preIctal)
        #nOthers = len(labels_others)
        
        for kRun in range(nRuns):
            print("clf = {} , subject = {} , batch = {} ".format(classifier , sub , kRun + 1))
            #=============================================================================
            targetPath = os.path.join(currentPath , "myEEGdata/formatted_data/features_standard21_allTimes_{}s".format(int(windowSize)))
            fileSpec = "subject_{}_windowSize_{}_preIctal_{}_features_reduced.pkl".format(sub , windowSize , preIctal)
            #fileSpec = "intra_{}_windowSize_{}_preIctal_{}_features.pkl".format(sub , winSize , preIctal)
            filePath = os.path.join(targetPath , fileSpec) 
            #==============================  open subjects files  ===============================================
            fileExist = os.path.exists(filePath)
            if fileExist:
                subStruct = pickle.load( open( filePath , "rb" ) )
            else:
                raise ValueError(' problem with path :  file doesnt exist'.format(filePath))
            #print('data loaded')
    #        
    #        saveOpt = True
    #        filtOpt = True
            
            times2return , feats2return , seiz2return , filesInFig = post.mySerializer(sub , subStruct , windowSize) 
            
            nFigs = len(times2return)
            scaleOpt = True    
            globalautoReject = subStruct["autoreject_threshold"]
    #        fileNames = subStruct["file_names"]
            
            totalTimesList = list()
            totalPredsList = list()
            totalBads = list()
            seizureTimes = list()
            
            lastTimeInFig = 0
            lastInd = 0
            for fig in range(nFigs)  :
                #print('doing figure {}/{}'.format(fig + 1 , nFigs))
                """ here we ttrain on the rest of the windows """
                myTimes =   times2return[fig]
                tempFeats = feats2return[fig]
                figureFiles = filesInFig[fig]
                clear2keep , seiz2keep,= post.abs2clearAndSeiz(figureFiles , subStruct)
                mySeizInfo = seiz2return[fig]
                nSeizureInFig  = len(mySeizInfo)
                if nSeizureInFig > 0:
                    for toto in range(nSeizureInFig):
                        seizureTimes.append([lastTimeInFig + mySeizInfo[toto][0] , lastTimeInFig + mySeizInfo[toto][1]])
                    
                if nSeizureInFig < 2:
                    """ in case there is less than one seizure per figure """
                    
                    ############### get training data ###################    
                    allfeats , alllabels , maxFeats = pipe.myDataWrapperForClassification(subStruct ,\
                                                                                seizure2keep = seiz2keep , clear = clear2keep)
                    
                    feats_mixed_not_scaled , labels_mixed = tl.concatWithFractions(allfeats , alllabels \
                                            , selfFrac , feats_others , labels_others , tlFrac , tlAll=tlAll)
                    
                    feats_train_not_scaled , labels_train = pipe.seperateAndPermute(feats_mixed_not_scaled , labels_mixed)
                    
                    if scaleOpt:
                        # train scaler
                        scaler = StandardScaler()
                        feats_train = scaler.fit_transform(feats_train_not_scaled)
                    else:
                        feats_train = feats_train_not_scaled
                    
                    if classifier == 'vote' or classifier =='all':
                        # Assemble a classifier
                        lda = LinearDiscriminantAnalysis()
                        svc = SVC(kernel = "linear" ,  probability=True)
                        knn = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            lda.fit(feats_train, labels_train)
                        svc.fit(feats_train, labels_train)
                        knn.fit(feats_train, labels_train)
                        
                    elif classifier == 'lda':
                        clf = LinearDiscriminantAnalysis()
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # fit classifier
                            #tempScore = cross_val_score(lda, feats_train, y= labels_train  , cv= 3 )
                            clf.fit(feats_train, labels_train)
                    elif classifier == 'svc':
                        clf = SVC(kernel = "linear" ,  probability=True)
                        clf.fit(feats_train, labels_train)
                    elif classifier == 'knn':
                        clf = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                        clf.fit(feats_train, labels_train)
                    elif classifier == 'rfc':    
                        clf = RandomForestClassifier(n_estimators=n_estimatorsRF)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(feats_train, labels_train)
                    else:
                        raise ValueError(' Classifier not recognized')
                            
                    """delete training data"""
                    del allfeats , alllabels , maxFeats , feats_train_not_scaled , labels_train
                    # get the autoreject threshold
                    if len(figureFiles)==0 :
                        raise ValueError(' problem in finding the good file name')
                    else:
                        temp = globalautoReject.copy()  
                        figureFiles.reverse()
                        for kFile in figureFiles:
                            temp.pop(kFile)    
                            
                        autorejectThresh = np.mean(np.asarray(temp)) 
                        """ here we test on the given window """
                        badEps = pipe.getBadEpochs(tempFeats , autorejcetFactor * autorejectThresh)
                    
                    if scaleOpt:
                        myFeats = scaler.transform(tempFeats)
                    else:
                        myFeats = tempFeats
                    
                    del tempFeats
                    
                    nSamps = len(myTimes)
                    if scoreType == 'labels':
                        if classifier == 'vote':
                            allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                          knn.predict(myFeats).reshape(nSamps,1) ))
                            finalPred = np.around(np.average(allPreds , axis = 1 , weights = weights))
                        elif classifier == 'all':
                            allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                          knn.predict(myFeats).reshape(nSamps,1) ))
                            votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                            finalPred = np.hstack((allPreds , votePred))
                        else:
                            finalPred = clf.predict(myFeats).reshape(nSamps,1)
                    elif scoreType == 'proba':
                        if classifier == 'vote':
                            allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                      knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                            finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                        elif classifier == 'all':
                            allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                      knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                            votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                            finalPred = np.hstack((allPreds , votePred))
                        else:
                            finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                    
                    elif scoreType == 'contrast':
                        list2stack = list()
                        ###################################
                        preIctProb = lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                        interIctProb = lda.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                        list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                        ##################################
                        preIctProb = svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                        interIctProb = svc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                        list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                        ##################################
                        preIctProb = knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                        interIctProb = knn.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                        list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                        ##################################
                        allPreds = np.hstack(list2stack)
                        if classifier == 'vote':
                            finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                        elif classifier == 'all':
                            votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                            finalPred = np.hstack((allPreds , votePred))
                        else:
                            finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                    
                    else:
                        raise ValueError(' scoreType argument is not recognized')
                        
                    del myFeats
            #        counts = post.slidingWindow(myTimes , finalPred , seizureInfo = mySeizInfo, \
            #                                                postProcessWindow = postProcessWindow , alarmThresh = thresh)
                    
                    
                    totalTimesList.append(myTimes + lastTimeInFig * np.ones(myTimes.shape))
                    totalPredsList.append(finalPred)
                    totalBads.append(badEps )#+ lastInd * np.ones(badEps.shape)) 
                    lastTimeInFig = lastTimeInFig + myTimes[-1]
                    lastInd = lastInd + len(myTimes)
                else:
                    """ in case there are more than one seizure per figure """
                    predsList = list()
                    timesList = list()
                    badEpsList = list()
                    
                    # get the autoreject threshold
                    if len(figureFiles)==0 :
                        raise ValueError(' problem in finding the good file name')
                    else:
                        temp = globalautoReject.copy()  
                        figureFiles.reverse()
                        for kFile in figureFiles:
                            temp.pop(kFile)    
                            
                        autorejectThresh = np.mean(np.asarray(temp)) 
                    # get number of parts sin figure    
                    nParts = nSeizureInFig
                    lastTime = -1
                    for kPart in range(nParts) :  
                        # get part indices
                        if kPart < nParts - 1:
                            inds = np.where(np.logical_and(myTimes < mySeizInfo[kPart][1] , myTimes > lastTime))[0]
                            lastTime = mySeizInfo[kPart][1] - 1
                        else:
                            inds = np.where(myTimes > lastTime)[0]
                       
                        tempFeatsPart = tempFeats[inds , :]
                        myTimesPart = myTimes[inds]
                        
                        ############### get training data ###################    
                        allfeats , alllabels , maxFeats = pipe.myDataWrapperForClassification(subStruct ,\
                                                                                              seizure2keep = [seiz2keep[kPart]] , clear = clear2keep)
                        feats_mixed_not_scaled , labels_mixed = tl.concatWithFractions(allfeats , alllabels \
                                            , selfFrac , feats_others , labels_others , tlFrac)
                    
                        feats_train_not_scaled , labels_train = pipe.seperateAndPermute(feats_mixed_not_scaled , labels_mixed)
                        
                        if scaleOpt:
                            # train scaler
                            scaler = StandardScaler()
                            feats_train = scaler.fit_transform(feats_train_not_scaled)
                        else:
                            feats_train = feats_train_not_scaled
                        
                        if classifier == 'vote' or classifier == 'all':
                            # Assemble a classifier
                            lda = LinearDiscriminantAnalysis()
                            svc = SVC(kernel = "linear" ,  probability=True)
                            knn = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                lda.fit(feats_train, labels_train)
                            svc.fit(feats_train, labels_train)
                            knn.fit(feats_train, labels_train)
                            
                        elif classifier == 'lda':
                            clf = LinearDiscriminantAnalysis()
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                # fit classifier
                                #tempScore = cross_val_score(lda, feats_train, y= labels_train  , cv= 3 )
                                clf.fit(feats_train, labels_train)
                        elif classifier == 'svc':
                            clf = SVC(kernel = "linear" ,  probability=True)
                            clf.fit(feats_train, labels_train)
                        elif classifier == 'knn':
                            clf = neighbors.KNeighborsClassifier(numNeighbors , weights='uniform')
                            clf.fit(feats_train, labels_train)
                        elif classifier == 'rfc':    
                            clf = RandomForestClassifier(n_estimators=n_estimatorsRF)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                clf.fit(feats_train, labels_train)
                        else:
                            raise ValueError(' Classifier not recognized')
                        """delete training data"""
                        del allfeats , alllabels , maxFeats , feats_train_not_scaled , labels_train
                        
                        """ here we test on the given window """
                        badEps = pipe.getBadEpochs(tempFeatsPart , autorejcetFactor * autorejectThresh)
                        
                        if scaleOpt:
                            myFeats = scaler.transform(tempFeatsPart)
                        else:
                            myFeats = tempFeatsPart
                        
                        nSamps = len(tempFeatsPart)
                        del tempFeatsPart
                        if scoreType == 'labels':
                            if classifier == 'vote':
                                allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                              knn.predict(myFeats).reshape(nSamps,1) ))
                                finalPred = np.around(np.average(allPreds , axis = 1 , weights = weights))
                            elif classifier == 'all':
                                allPreds = np.hstack((lda.predict(myFeats).reshape(nSamps,1) , svc.predict(myFeats).reshape(nSamps,1) , \
                                              knn.predict(myFeats).reshape(nSamps,1) ))
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                            else:
                                finalPred = clf.predict(myFeats).reshape(nSamps,1)
                        elif scoreType == 'proba':
                            if classifier == 'vote':
                                allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                          knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                                finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                            elif classifier == 'all':
                                allPreds = np.hstack((lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1) , \
                                          knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1) ))
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                            else:
                                finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                        
                        elif scoreType == 'contrast':
                            list2stack = list()
                            ###################################
                            preIctProb = lda.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = lda.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            preIctProb = svc.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = svc.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            preIctProb = knn.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            interIctProb = knn.predict_proba(myFeats)[: , 0].reshape(nSamps,1)
                            list2stack.append(np.divide(np.positive(preIctProb - interIctProb) , preIctProb + interIctProb))
                            ##################################
                            allPreds = np.hstack(list2stack)
                            if classifier == 'vote':
                                finalPred = (np.average(allPreds , axis = 1 , weights = weights))
                            elif classifier == 'all':
                                votePred = np.around(np.average(allPreds , axis = 1 , weights = weights)).reshape(nSamps,1) 
                                finalPred = np.hstack((allPreds , votePred))
                            else:
                                finalPred = clf.predict_proba(myFeats)[: , 1].reshape(nSamps,1)
                            
                        else:
                            raise ValueError(' scoreType argument is not recognized')
                            
                            del myFeats
            
                        timesList.append(myTimesPart)
                        predsList.append(finalPred)
                        badEpsList.append(badEps)
                        
                    """ here is the plotting """
                    if classifier =='all':
                        totPreds = np.vstack(predsList)
                    else:
                        totPreds = np.concatenate(predsList)
        #            totTimes = np.concatenate(timesList)
                    badEps = np.concatenate(badEpsList)
            #        
            #        totcounts = post.slidingWindow(totTimes , totPreds ,  seizureInfo = mySeizInfo ,\
            #                                       postProcessWindow = postProcessWindow , alarmThresh = thresh)
                    
                    totalTimesList.append(myTimes + lastTimeInFig * np.ones(myTimes.shape))
                    totalPredsList.append(totPreds)
                    totalBads.append(badEps )#+ lastInd * np.ones(badEps.shape)) 
                    lastTimeInFig = lastTimeInFig + myTimes[-1]
                    lastInd = lastInd + len(myTimes) 
                    
            """ Here we evaluate the performances of each run """
            times = np.concatenate(totalTimesList) 
            if classifier =='all':
                preds = np.vstack(totalPredsList) 
            else:
                preds = np.concatenate(totalPredsList) 
            
            bads = np.concatenate(totalBads) 
#            # seizureTimes
#            # here we get the scores
#            if classifier =='all':
#                countsList = list()
#                for kClass in range(4):
#                    tempCounts = post.slidingWindow(times , preds[: , kClass] ,  seizureInfo = seizureTimes ,\
#                                                   postProcessWindow = postProcessWindow )
#                    nCounts = len(tempCounts)
#                    countsList.append(tempCounts.reshape((nCounts , 1)))
#                counts = np.hstack(countsList)
#            else:
#                counts = post.slidingWindow(times , preds ,  seizureInfo = seizureTimes ,\
#                                                   postProcessWindow = postProcessWindow )
        
            runScores.append(preds)
            runBads.append(bads)
    return times , runScores ,  seizureTimes , runBads
    
    
