#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:34:20 2019

@author: remy.benmessaoud
"""

import mne
from my_functions import myfeature_extraction_functions
import numpy as np
#import numpy.random as rnd
#from autoreject import get_rejection_threshold
from sklearn.preprocessing import scale
import time
import os.path
import pickle
Fs = 250 # Hz
hF = 45 #Hz cut off frequency for lowpass filter
lF =1
hFWidth = 5
lFWidth = 0.4
decim = 2 # downsampling factor 
myPath="/Users/remy.benmessaoud/Desktop/neuroProject"
standardChannelNum = 21

""" new main() """
def new_main_preprocessing(currentDir , sub , winSize , filtOpt , seizureFreeFiles , preIctal):
    labelsFolder = "labels_intra_winSize_{}s".format(int(winSize) )
    labelsPath = os.path.join(currentDir ,"myLabels", labelsFolder)
    dataDir = os.path.join(currentDir ,"myEEGdata", "EEG_intra" , 'intra{}'.format(sub))
    # check if seperate file already exists
    subject="intra{}".format(sub)
    targetFile =os.path.join(labelsPath , subject + "_preIctal_{}_labels_v1.pkl".format(preIctal))
    if os.path.exists(targetFile):
        with open(targetFile, 'rb') as f:
            subDict = pickle.load(f)
    else:
        raise ValueError(' labels file not found : {}'.format(targetFile))
            
            
    """ ===========first we preprocess the seizures Data ============"""
    nSeizures = subDict["n_seizures"]
    seizuresData = subDict["seizures_data"]
    featuresDict={ "n_seizures" : nSeizures , "file_names" : subDict["file_names"] , "fileLengths" : list() , \
                  "seizureInFile" : subDict["seizureInFile"], "seizures_data" : list() , "post_seizure_data" : list()\
                  , "seizurefree_data" : list() , "autoreject_threshold" : subDict["autoreject_threshold"]} 
    
    if "fileLengths" in subDict.keys():
        featuresDict["fileLengths"] = subDict["fileLengths"]
        
    for kSeizure in range(nSeizures): #""" here put nSeizures"""
        fileName =   seizuresData[kSeizure]["fileName"]      
        labelsMap =   seizuresData[kSeizure]["map"]   
        fileidx = subDict["file_names"].index(fileName)
        if not(labelsMap is None):
            if len(labelsMap)>0:
                epochs = new_makeMyEpochs(dataDir , fileName , labelsMap  , winSize , filtOpt)
                times = labelsMap[: , 0]
                labels = labelsMap[: , 1]
                trainOpts = labelsMap[: , 2]
                nEpochs = len(epochs)
                print(" Getting Thresholds")
                
                thresh = subDict["autoreject_threshold"][fileidx]
                epochstemp  = epochs.copy()
                
                reject = {"eeg" : thresh}
                epochstemp.drop_bad(reject=reject , verbose=False)
                
                newTimes = times.reshape((nEpochs,1))
                newLabels = labels.reshape((nEpochs,1))
                newTrainOpts = trainOpts.reshape((nEpochs,1))
                
                selection = epochstemp.selection
                for kEp in range(nEpochs):
                    if not(kEp in selection):
                        newTrainOpts[kEp] = 0
                
                
                print("rescaling epochs and extracting features")
                
                if len(newLabels) != nEpochs:
                    raise ValueError(' number of labels different from number of epochs')
                dataList = list()
                for k in range(nEpochs): #"""# here careful make it nEpochs"""
                    start=time.perf_counter_ns()
                    rawData=np.array(epochs[k].get_data()[0 , : , :])
                    # measure mean and variance before scaling 
                    myMean = np.mean(rawData , axis=1)
                    myVar = np.var(rawData , axis=1)
                    normData = scale(rawData , axis = 1)
                    myMin = np.min(rawData , axis=1)
                    myMax = np.max(rawData , axis=1)
                    print("Subject {}: Seizure: {}/{}: extracting features for epoch {} / {}".format(\
                          sub , kSeizure + 1,\
                          nSeizures , k+1 , nEpochs ))
                    
                    featVector = myfeature_extraction_functions.myFeaturesExtractor1(normData , myMean , myVar \
                                                                                     , myMin , myMax)
                    #featVector = np.zeros((1 , 21*37)) # for debugging 
                    dataList.append(featVector)
                    end=time.perf_counter_ns()
                    print("one feature vector calculation time : {:.2f} s".format((end-start)/10**9))
                feats = np.vstack(dataList)
                finalMap = np.hstack((newTimes , newLabels , newTrainOpts , feats))
                featuresDict["seizures_data"].append({ "fileName" : fileName , "map" : finalMap ,\
                            "seizureInfo" : seizuresData[kSeizure]["seizuresInfo"]} )
            # Here we decrease the number of seizures since the ones which occur too soon after a prior seizure 
            # are not considered in the analysis
            else:
                featuresDict["n_seizures"] = featuresDict["n_seizures"] -1
                featuresDict["seizureInFile"][fileidx] = featuresDict["seizureInFile"][fileidx] - 1
                
    
    """ ===========first we preprocess the post seizures Data ============"""
    
    post_seizure_data = subDict["post_seizure_data"]
    nPostSeizures = len(post_seizure_data)
    
    if "fileLengths" in subDict.keys():
        featuresDict["fileLengths"] = subDict["fileLengths"]
        
    for kSeizure in range(nPostSeizures): #""" here put nSeizures"""
        fileName =   post_seizure_data[kSeizure]["fileName"]      
        labelsMap =   post_seizure_data[kSeizure]["map"]   
        fileidx = subDict["file_names"].index(fileName)
        if not(labelsMap is None):
            if len(labelsMap)>0:
                epochs = new_makeMyEpochs(dataDir , fileName , labelsMap  , winSize , filtOpt)
                times = labelsMap[: , 0]
                labels = labelsMap[: , 1]
                trainOpts = labelsMap[: , 2]
                nEpochs = len(epochs)
                print(" Getting Thresholds")
                
                thresh = subDict["autoreject_threshold"][fileidx]
                epochstemp  = epochs.copy()
                
                reject = {"eeg" : thresh}
                epochstemp.drop_bad(reject=reject , verbose=False)
                
                newTimes = times.reshape((nEpochs,1))
                newLabels = labels.reshape((nEpochs,1))
                newTrainOpts = trainOpts.reshape((nEpochs,1))
                
                selection = epochstemp.selection
                for kEp in range(nEpochs):
                    if not(kEp in selection):
                        newTrainOpts[kEp] = 0
                
                
                print("rescaling epochs and extracting features")
                
                if len(newLabels) != nEpochs:
                    raise ValueError(' number of labels different from number of epochs')
                dataList = list()
                for k in range(nEpochs): #"""# here careful make it nEpochs"""
                    start=time.perf_counter_ns()
                    rawData=np.array(epochs[k].get_data()[0 , : , :])
                    # measure mean and variance before scaling 
                    myMean = np.mean(rawData , axis=1)
                    myVar = np.var(rawData , axis=1)
                    normData = scale(rawData , axis = 1)
                    myMin = np.min(rawData , axis=1)
                    myMax = np.max(rawData , axis=1)
                    print("Subject {}: Post_Seizure: {}/{}: extracting features for epoch {} / {}".format(\
                          sub , kSeizure + 1,\
                          nPostSeizures , k+1 , nEpochs ))
                    
                    featVector = myfeature_extraction_functions.myFeaturesExtractor1(normData , myMean , myVar \
                                                                                     , myMin , myMax)
                    #featVector = np.zeros((1 , 21*37)) # for debugging 
                    dataList.append(featVector)
                    end=time.perf_counter_ns()
                    print("one feature vector calculation time : {:.2f} s".format((end-start)/10**9))
                feats = np.vstack(dataList)
                finalMap = np.hstack((newTimes , newLabels , newTrainOpts , feats))
                featuresDict["post_seizure_data"].append({ "fileName" : fileName , "map" : finalMap ,\
                            "seizureInfo" : seizuresData[kSeizure]["seizuresInfo"]} )
            # Here we decrease the number of seizures since the ones which occur too soon after a prior seizure 
            # are not considered in the analysis
            else:
                featuresDict["post_seizure_data"].append([])
    """ =====================================================================
    ===================then  we preprocess the seizures free Data ============"""
    if len(seizureFreeFiles) > 0:
        if seizureFreeFiles[0]==-1:
            # here we take all the files
            all_seizurefree_data = subDict["seizurefree_data"]
            seizurefree_data = all_seizurefree_data
        else:
            # here it is more trick because we will preprocess only the selected files
            all_seizurefree_data = subDict["seizurefree_data"]
            # pick the concerned files
            seizureFreeFiles = ( seizureFreeFiles - np.ones(seizureFreeFiles.shape)).astype(int)
            files2keep = np.asarray(subDict["file_names"])[seizureFreeFiles]
            seizurefree_data = list()
            for kk in range(len(all_seizurefree_data)):
                fname = all_seizurefree_data[kk]["fileName"]
                if fname in files2keep:
                    seizurefree_data.append(all_seizurefree_data[kk])
        
        ## Now we do the preprocessing
        nFiles = len(seizurefree_data)
        
        for kFile in range(nFiles): #"""# here careful make it nFiles"""
            fileName =   seizurefree_data[kFile]["fileName"]      
            labelsMap =   seizurefree_data[kFile]["map"]  
            fileidx = subDict["file_names"].index(fileName)
            
            epochs = new_makeMyEpochs(dataDir , fileName , labelsMap  , winSize , filtOpt)
            times = labelsMap[: , 0]
            labels = labelsMap[: , 1]
            trainOpts = labelsMap[: , 2]
            nEpochs = len(epochs)
            print(" Getting Thresholds")
            
            thresh = subDict["autoreject_threshold"][fileidx]
            epochstemp  = epochs.copy()
            
            reject = {"eeg" : thresh}
            epochstemp.drop_bad(reject=reject , verbose=False)
            
            newTimes = times[:nEpochs].reshape((nEpochs,1))
            newLabels = labels[:nEpochs].reshape((nEpochs,1))
            newTrainOpts = trainOpts[:nEpochs].reshape((nEpochs,1))
            
            selection = epochstemp.selection
            for kEp in range(nEpochs):
                if not(kEp in selection):
                    newTrainOpts[kEp] = 0
                    
            print("rescaling epochs and extracting features")
            nEpochs = len(epochs)
            if len(newLabels) != nEpochs:
                raise ValueError(' number of labels different from number of epochs')
            dataList = list()
            for k in range(nEpochs): # here careful make it nEpochs
                start=time.perf_counter_ns()
                rawData=np.array(epochs[k].get_data()[0 , : , :])
                # measure mean and variance before scaling 
                myMean = np.mean(rawData , axis=1)
                myVar = np.var(rawData , axis=1)
                normData = scale(rawData , axis = 1)
                myMin = np.min(rawData , axis=1)
                myMax = np.max(rawData , axis=1)
                print("Subject {}: seizure free data: {}/{}: extracting features for epoch {} / {}".format(sub , kFile + 1,\
                      nFiles , k+1 , nEpochs ))
                
                featVector = myfeature_extraction_functions.myFeaturesExtractor1(normData , myMean , myVar , myMin , myMax)
                #featVector = np.zeros((1 , 21*37)) # for debugging
                dataList.append(featVector)
                end=time.perf_counter_ns()
                print("one feature vector calculation time : {:.2f} s".format((end-start) /10**9))
            feats = np.vstack(dataList)
            finalMap = np.hstack((newTimes , newLabels , newTrainOpts , feats))
            featuresDict["seizurefree_data"].append({ "fileName" : fileName , "map" : finalMap} )
            
    return featuresDict



""" 
Make epochs from corresponding time table of the corresponding file 
This excludes all the seizure segments from the analysis
Arguments : 
    subject Index : 1..24
    file Idx :
"""
def new_makeMyEpochs(currentDir , eegFile , myMap  , winSize , filtOpt):
    #nSamp = int(winSize * Fs)
    #first step : get EEG file Name and crresponding epoch times
    fileName =os.path.join(currentDir , eegFile)
    myTimes = myMap[: , 0]

    # read edf file 
    rawDummy = mne.io.read_raw_edf(fileName, eog=None, misc=None, stim_channel='auto',\
                          exclude=(), preload=True, verbose=None)
    
    raw = dropDummyChannels(rawDummy)
    if filtOpt:
        # low pass the raw before downsampling by factor 2
        raw.filter(None, hF, picks=None, filter_length='auto', l_trans_bandwidth='auto',\
                   h_trans_bandwidth= hFWidth , n_jobs=1, method='fir', iir_params=None, phase='zero',\
                   fir_window='hamming' , verbose=False)
        
        # High pass the raw before downsampling by factor 2
        raw.filter(lF, None , picks=None, filter_length='auto', l_trans_bandwidth='auto',\
                   h_trans_bandwidth= lFWidth , n_jobs=1, method='fir', iir_params=None, phase='zero',\
                   fir_window='hamming' , verbose=False)
        
    #data = np.array(raw.get_data())
    #nChan , nTimes = data.shape
    nEpochs = len(myTimes)
    
    #epochs = np.ndarray( (nEpochs , nChan , nSamp))
    events = np.zeros( (nEpochs , 3) , dtype=int )
    
    for kEpoch in range(nEpochs):
        start = int(myTimes[kEpoch] * Fs)
        events[kEpoch , 0] = start 
        events[kEpoch , 1] = 0
        events[kEpoch , 2] = 0
    
    # now deal with mne specifities
    myEpochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=winSize - 2/128 , preload=True , baseline = None,decim=decim)

    return myEpochs 


"""
Function that gets rid of the duplicates, dummy , collinear channels
"""
def fixChannels(raw):
    #exclude not EEG channels
    notEEGList = ['--' , 'VNS' , 'ECG' , 'EKG' , 'LUE' , 'ROC' , 'RA' , 'LOC' , '.']
    names = raw.ch_names
    nChans = len(names)
    notEEGChans = list()
    for k in range(nChans):
        chName = names[k]
        isNotEEG = False
        for sub in notEEGList:
            if chName.find(sub) !=-1:
                isNotEEG =True
        if isNotEEG:
            notEEGChans.append(chName)
    if len(notEEGChans)>0:
        raw.drop_channels(notEEGChans)
    
    names = raw.ch_names
    dummyChans = list()
    T7P7found = False
    for name in names:
        if name == 'T7-P7':
            T7P7found = True
        if name == 'P7-T7' and T7P7found:
            dummyChans.append(name)
        if name == 'T8-P8-1':
            dummyChans.append(name)
    if len(dummyChans)>0:
        raw.drop_channels(dummyChans)
        
    names = raw.ch_names
    nChans = len(names)
    
    if nChans != standardChannelNum:
        if nChans > standardChannelNum:
            n2remove = nChans - standardChannelNum
            raw.drop_channels(names[-n2remove:])
        if nChans < standardChannelNum:
            n2add = standardChannelNum - nChans 
            data = raw.get_data()
            addedNames = list()
            for k in range(n2add):
                addedNames.append('intrpolated-{}'.format(k+1))
            info = mne.create_info(addedNames, Fs)
            addedRaw = mne.io.RawArray(data[:n2add], info)
            raw.add_channels([addedRaw])
            raw.info['bads'].extend(addedNames) 
            raw.interpolate_bads()
    
        
    names = raw.ch_names
    nChans = len(names)        
    if nChans != standardChannelNum:
        raise ValueError(' number of channels not correct : nChannels = '.format(nChans))
    return raw


"""
Function that gets rid of the duplicates, dummy , collinear channels
"""
def dropDummyChannels(raw):
    #exclude not EEG channels
    notEEGList = ['Oz' ]
    names = raw.ch_names
    nChans = len(names)
    notEEGChans = list()
    for k in range(nChans):
        chName = names[k]
        isNotEEG = False
        for sub in notEEGList:
            if chName.find(sub) !=-1:
                isNotEEG =True
        if isNotEEG:
            notEEGChans.append(chName)
    if len(notEEGChans)>0:
        raw.drop_channels(notEEGChans)
    
#    # exclude collinear chans ( T8-P8 and P7-T7 mostly)
#    myData = raw.get_data(stop = 128)
#    C=np.absolute(np.corrcoef(myData))
#    n = C.shape[0]
#    coll = list()
#    names = raw.ch_names
#    for i in range(n - 1):
#        for j in range(i + 1,n):
#            if C[i , j] > 0.97:
#                coll.append(names[j])
#    if len(coll)>0:
#        raw.drop_channels(coll)
        
    return raw


"""
Function that concatenates all epochs for a single subject and gives the concatenated labels
"""
def getAllEpochsForSub(labelStruct , sub , winSize):
    subTable = labelStruct[sub - 1]
    del labelStruct
    nFiles = len(subTable)
    epochsList = list()
    labelsList = list()
    
    for k in range(nFiles): # careful here change to nFiles
        epochs , labels = makeMyEpochs(subTable , sub , k , winSize)
        labelsList.extend(labels)
        epochsList.append(epochs)
#    xx=2
#    yy=2
    return mne.concatenate_epochs(epochsList, add_offset=True) , np.asarray(labelsList)


"""
function to delete the labels corresponding to dropped epochs !!! not needed anymore
"""
def deleteDroppedLabels(labels , dropInfo):
    nInfo = len(dropInfo)
    nLabels = len(labels)
    if nLabels != nInfo:
        raise Exception('labels length different from dropping Info List')
    
    newLabels = list()
    for k in range(nInfo):
        if len(dropInfo[k]) == 0:
            newLabels.append(labels[k])
    
    return np.asarray(newLabels)



# get channel type from name
def chanName2Type(names):
    nChan = len(names)   
    types = list()
    
    for k in range(nChan):
        chan = names[k]
        isNotEEG = chan == 'VNS' or chan== 'ECG' or chan.find("EKG")!=-1 or \
        chan.find("LOC")!=-1 or chan.find("ROC")!=-1 or chan.find("LUE")!=-1 or chan.find("RA")!=-1
        if isNotEEG:
            types.append('misc')
        else:
            types.append('eeg')
    return types


#    return np.vstack(dataList) , newLabels