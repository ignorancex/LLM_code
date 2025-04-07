"""!/usr/bin/env python3
 -*- coding: utf-8 -*-

 =============================================================================
This scripts makes the labels for the following specifications
For each subject we have a dictionnar called "subDict", it contains the following keys:
    
    - "n_seizures" = total number of seizures
    - "file_names" = list of str , here we put always the complete relative path going from neuroProject
    -"seizures_data" = list of dictionnaries: with length = n_seizures
            each one having following keys : 
                {"fileName" = str ; "Map" = 3d array }  // here the 3 dimensions are: times , labels , trainOpt 
    - "post_seizure_data" = list ( { "fileName" = str ; "trainMap" = 2d array } )
    -"seizurefree_data" = list of dictionnaries { "fileName" = str ; "map" = 2d array}
    -"autoreject_threshold" = list of couples ( ( fileName , Threshold ) ...) 

 =============================================================================
"""
import numpy as np
import os.path
import mne
from my_functions import my_labeling_functions as myFunc
from my_functions.mypreprocessing import dropDummyChannels
from autoreject import get_rejection_threshold
import pickle
"""
 =============================================================================
function that makes event like object from times array 
 =============================================================================
"""
def times2events(times):
    nEpochs = len(times)
    #epochs = np.ndarray( (nEpochs , nChan , nSamp))
    events = np.zeros( (nEpochs , 3) , dtype=int )
    for kEpoch in range(nEpochs):
        start = int(times[kEpoch] * Fs)
        events[kEpoch , 0] = start 
        events[kEpoch , 1] =  1
        events[kEpoch , 2] =  1 
    return events
"""=========================================================================="""
####################### my Specifications ##############
Fs=256# Hz
windowLength= 15 #s
windowOverlap=2/3
winStep=int(windowLength*(1-windowOverlap)) # = 5 s
predHorizon = 600 #s 10 min 5 min ## in fact it is preIctal training period
postSeizureMargin= 600 #s 10 min
preIctalMargin=0
interIctalMargin=1800#s 30 min  time before next seizure 
endingFileMargin = 1200 #s 20 min  we reject the last 20 min of each file
#filesLengths=np.array([])
######################################################
hF = 45 #Hz cut off frequency for lowpass filter
lF =1
hFWidth = 5
lFWidth = 0.4
decim = 2
################################################
# if saveOpt = True we save labelStruct in myLabels
saveOpt = True
filtOpt = True
#saveOpt = False

myPath="/Users/remy.benmessaoud/Desktop/neuroProject"
dataPath=os.path.join(myPath,"myEEGdata","EEG_MIT")


#labelStruct=list() # list of subject specific label structure
fileNameMarker="File Name"
nSubjects= 24
firstSub = 1
subjectsList = np.arange(firstSub , nSubjects +1)
#subjectsList = [6]

for kSub in subjectsList:
    
    print("doing the labeling for subject : {}\n".format(kSub))
    # get subject name and summary file
    subject="chb{:02d}".format(kSub)
    labelsPath = os.path.join(myPath,"myLabels", "preIctal{}".format(int(predHorizon/60)) , \
        "labels_winSize_{}s".format(windowLength) , "{}_preIctal_{}_labels_v1.pkl".format(subject , int(predHorizon/60)))
    summaryFile="{}-summary.txt".format(subject)  
    
    ##open text file
    textFilePath=os.path.join(dataPath,subject,summaryFile)
    textFile=open(textFilePath, "r")
    txtLines=textFile.readlines()
    nLines=len(txtLines)
    """===========initialize subject specific dictionnar============"""
    subDict={ "n_seizures" : 0 , "file_names" : list() , "seizureInFile" : list(), "seizures_data" : list() , \
             "post_seizure_data" : list() , "seizurefree_data" : list() , "autoreject_threshold" : list(),\
             "nChannels" : list() , "fileLengths" : list()} 
    
    # we don't know the number of files so we set we use a boolean variable 
    kFile=1
    startInd=28 # line where file specifications begin
    kLine=startInd
    
    while kLine< nLines-3:
        
        #start scanning for fileName kFile
        lineFound=False
        while not(lineFound)  and kLine< nLines-3:
            
            line=txtLines[kLine]
            result = line.find(fileNameMarker)
            
            if result != -1: # it means we found the line of "File Name: chb24_01.edf"
                # get fileName
                fileName = myFunc.str2fileName(line)
                # here we do the labeling for this file
                fileStartInd=kLine
                # we get the length of the EEG file
                eegFile=os.path.join(dataPath,subject,fileName)
                relativeFilePath = os.path.join("myEEGdata","EEG_MIT"  ,subject,fileName)
                # we add the file name to the list
                
                
                raw=mne.io.read_raw_edf(eegFile, montage='deprecated', eog=None,misc=None, stim_channel='auto', \
                                        exclude=(), preload=True, verbose=None)
                fileLength=int(raw.n_times/Fs) # in s
                ignoreFile = fileLength < 6*60
                subDict["fileLengths"].append(fileLength)
                #times=raw.times
                if not(ignoreFile):    
                    subDict["file_names"].append(relativeFilePath)
                    # next step: get number of seizures in the file and if so the start and end times
                    seizureLineInd = kLine + 3
                    seizureLine=txtLines[seizureLineInd]
                    numSeizure=int(seizureLine[-2])
                    # increment number of seizures
                    subDict["n_seizures"] += numSeizure
                    subDict["seizureInFile"].append(numSeizure)
                    seizuresInfo=list()
                    
                    # ===========================filter the shit out of my signal==============================
                    nChannels = len(raw.ch_names)
                    subDict["nChannels"].append(nChannels)
                    
                    raw = dropDummyChannels(raw)
                    
                    if filtOpt:
                        # low pass the raw before downsampling by factor 2
                        raw.filter(None, hF, picks=None, filter_length='auto', l_trans_bandwidth='auto',\
                                   h_trans_bandwidth= hFWidth , n_jobs=1, method='fir', iir_params=None, phase='zero',\
                                   fir_window='hamming' , verbose=False)
            
                        # High pass the raw before downsampling by factor 2
                        raw.filter(lF, None , picks=None, filter_length='auto', l_trans_bandwidth='auto',\
                                   h_trans_bandwidth= lFWidth , n_jobs=1, method='fir', iir_params=None, phase='zero',\
                                   fir_window='hamming' , verbose=False)
                    #================================Now is the real thing ===========================================================
                    """====================  no seizures ====================="""
                    if numSeizure == 0:
                        # first :easy if no seizures we deal with it the old way and get the time2Labels Map
                        myMap , myValidMap = myFunc.getTimes2LabelsMap(windowLength , winStep , fileLength , numSeizure , seizuresInfo ,\
                                                          postSeizureMargin , predHorizon , preIctalMargin , \
                                                          interIctalMargin , endingFileMargin)
#                        n , d =myMap.shape
#                        myMap = np.hstack((myMap , np.ones((n ,1)) ))
                        finalMap = myFunc.getFinalMap(myValidMap , myMap)
                        subDict["seizurefree_data"].append({ "fileName" : relativeFilePath , "map" : finalMap})
                        # now we get the autoreject threshold
                        times = myValidMap[:,0]
                        events = times2events(times)
                        myEpochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=windowLength , preload=True ,\
                                              baseline = None,decim=decim)
                        reject = get_rejection_threshold(myEpochs)
                        # append rejecttion threshold 
                        subDict['autoreject_threshold'].append(reject["eeg"])
                    else:
                        """==================== one seizure or more ====================="""
                        # first we get the seizures info
                        for kSeizure in np.arange(numSeizure):
                            startTime = myFunc.getTimeFromString( txtLines[seizureLineInd + 2*kSeizure + 1] )
                            endTime = myFunc.getTimeFromString( txtLines[seizureLineInd + 2*kSeizure + 2] )
                            seizuresInfo.append(np.array([startTime , endTime]))
                        ############################################postSeizureValid
                        myMap , myvalidMap , postSeizure , postSeizureValid  = myFunc.getTimes2LabelsMap(windowLength ,\
                                        winStep , fileLength , numSeizure , seizuresInfo ,postSeizureMargin , \
                                        predHorizon ,preIctalMargin ,  interIctalMargin , endingFileMargin)
                        # now we do some organising of the data : 
                        # here for each seizure we have a map to use for train and one for validation 
                        # we will combine these two in one array with 3 atributes : times , labels , trainOpt 
                        if not(postSeizureValid is None):
                            postSeizureFinal = myFunc.getFinalMap(postSeizureValid , postSeizure)
                        else:
                            postSeizureFinal = []
                            
                        if len(myMap) != numSeizure:
                            raise ValueError(' array not consistent with seizures number')
                        allTimes = np.vstack(myvalidMap)[: , 0]
                        events = times2events(allTimes)
                        myEpochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=windowLength , preload=True ,\
                                              baseline = None,decim=decim)
                        reject = get_rejection_threshold(myEpochs)
                        # append rejecttion threshold 
                        subDict['autoreject_threshold'].append(reject["eeg"])
                        
                        for kSeizure in np.arange(numSeizure):
                            validMap = myvalidMap[kSeizure]
                            train = myMap[kSeizure]
                            finalMap = myFunc.getFinalMap(validMap , train)
                            subDict["seizures_data"].append( { "fileName" : relativeFilePath , "map" : finalMap , \
                                   "seizuresInfo" : seizuresInfo[kSeizure] } )
                        # just add the postSeizure data
                        subDict["post_seizure_data"].append( { "fileName" : relativeFilePath , "map" : postSeizureFinal} ) 
                    
                lineFound = True
                kFile +=1
            else:
                 kLine= kLine + 1
            print("one step")     
#        # here we check if there is an other file and edit the variable otherFile
#        # increase value of startInd to pass to the next file
        startInd= kLine + 5 # line begin where file specifications for next file
        kLine=startInd
        
    # here we finished with subject k so we append its tables to the structure  
    """here change to pickle"""
    if saveOpt:
        f = open(labelsPath,"wb")
        pickle.dump(subDict,f)
        f.close()
# Loop over subjects finished

    
    
