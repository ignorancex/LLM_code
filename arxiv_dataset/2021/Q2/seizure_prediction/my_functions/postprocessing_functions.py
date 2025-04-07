#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:14:19 2019

@author: remy.benmessaoud
"""
import numpy as np
fileLengths = [ 1 , 1 , 1 , 4 , 1 , 4 , 4 , 1 , 4 , 2 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 4 , 1]

def detectSeizure(times , classPred , windowSize = 15 , windowStep = 5 , postProcessWindow = 7 , alarmThresh = 3):
    predTimes = -1
    seizureTime = times[-1]
    nSamples = len(classPred)
    alarmTimes = list()
    alarms = np.zeros((nSamples - postProcessWindow +10,))
    temps =list()
    for k in np.arange(postProcessWindow , nSamples ):
        slidingWindow = np.arange(k - postProcessWindow , k ).astype(int)
        temp = sum(classPred[ slidingWindow ])
        
        temps.append(temp)
        
        if temp > alarmThresh:
            alarms[k - postProcessWindow] = 1
            alarmTimes.append(times[k] )
    nAlarms = len(alarmTimes)
    # get prediction times
    if nAlarms > 0:
       predTimes = np.zeros((nAlarms))
       for k in range(nAlarms):
           predTimes[k] = seizureTime - alarmTimes[k] 
    return predTimes , np.asarray(temps)

""" improvement of previous function to avoid high scores  just after a seizure"""
def slidingWindow(times , classPred ,  seizureInfo = [] , postProcessWindow = 7 ):
    parts = len(seizureInfo) + 1
    scoresList = list()
    lastTime = -1
    for kPart in range(parts):
        if kPart < parts - 1:
            inds = np.where(np.logical_and(times < seizureInfo[kPart][1] , times > lastTime))[0]
            lastTime = seizureInfo[kPart][1] - 1
        else:
            inds = np.where(times > lastTime)[0]
        
        tempTimes = times[inds]
        tempPreds = classPred[inds]
        tempScores = np.zeros(tempTimes.shape)
        for k in range(len(tempTimes)):
            if k < postProcessWindow - 1:
                tempScores[k] = np.sum(tempPreds[:k + 1])
            else:
                tempScores[k] = np.sum(tempPreds[k - postProcessWindow + 1 : k + 1])
            
        scoresList.append(tempScores)
    
    return np.concatenate(scoresList)
            

""" Serializer"""
def mySerializer(sub , subStruct , windowSize):
    seizureInFile = subStruct["seizureInFile"]
    fileNames = subStruct["file_names"]
    fileLengths = subStruct["fileLengths"]
    nFiles = len(seizureInFile)
    avgLen = int(np.round(np.mean(np.asarray(fileLengths)) / 3600) )# in hours
    nFilesperFigure  = 4 // avgLen
    
    
    time2stack = list()
    feats2stack = list()
    #ignored2stack = list()
    seizure2stack = list()
    kClear = 0
    kSeiz = 0
    kPostSeiz = 0
    postStruct =  subStruct["post_seizure_data"]
    nPostData = len(postStruct)
    for kFile in range(nFiles):
        
        nSeiz = seizureInFile[kFile]
        fileName = fileNames[kFile]
        #fileLen = fileLengths[kFile]
        
        if nSeiz == 0:
            fileStruct = subStruct["seizurefree_data"][kClear]
            Okay = (fileName == fileStruct["fileName"])
            if not(Okay):
                raise ValueError(' in mySerializer having problem to match file names')
            kClear = kClear  + 1
            myMap = fileStruct["map"]
            times = myMap[:,0]
            time2stack.append(times)
            feats2stack.append(myMap[:, 3:])
            seizure2stack.append([None , None])
        else:
            tempTimes = list()
            tempFeats= list()
            tempSeiz = list()
            for kSeizStruct in range(nSeiz): 
                fileStruct = subStruct["seizures_data"][kSeiz]
                Okay = (fileName == fileStruct["fileName"])
                if not(Okay):
                    raise ValueError(' in mySerializer having problem to match file names')    
                kSeiz = kSeiz + 1
                
                myMap = fileStruct["map"]
                times = myMap[:,0]
                tempTimes.append(times)
                tempFeats.append(myMap[:, 3:])
                tempSeiz.append(fileStruct["seizureInfo"])
                
                """ now we append the postSeizureData"""
            if kPostSeiz < nPostData :
                if len(postStruct[kPostSeiz]) > 0:
                    if  not(postStruct[kPostSeiz]["map"] is None):
                        fileStruct = postStruct[kPostSeiz]
                        Okay = (fileName == fileStruct["fileName"])
                        if not(Okay):
                            raise ValueError(' in mySerializer having problem to match file names')    
                        kPostSeiz = kPostSeiz + 1
                        
                        myMap = fileStruct["map"]
                        times = myMap[:,0]
                        
                        tempFeats.append(myMap[:, 3:])
                        tempTimes.append(times)
    
            time2stack.append(np.concatenate(tempTimes))
            feats2stack.append(np.vstack(tempFeats))
            seizure2stack.append(tempSeiz)
            
    """# Now we have to rearrange the lists to make clusters of nFilesperFigure"""
    if nFilesperFigure == 1:
        nFigures = len(time2stack)
        times2return = time2stack
        feats2return = feats2stack
        nSeiz = len(seizure2stack)
        seiz2return = list()
        for l in range(nSeiz):
            if not(seizure2stack[l][0] is None):
                seiz2return.append(seizure2stack[l])
            else:
                seiz2return.append([])
        """# now get the  filesInFig list"""
        filesInFig = list()
        for kFig in range(nFigures):
            filesInFig.append([kFig])

    else:
        times2return = list()
        feats2return = list()
        seiz2return = list()
        # find number of figures
        lastFile = nFiles % nFilesperFigure
        if lastFile == 0:
            nFigures = nFiles//nFilesperFigure
        else:
            nFigures = nFiles // nFilesperFigure + 1
        # now do the clustering
        for kFig in range(nFigures):
            tempTimes = list() 
            tempFeats= list()
            tempSeiz = list()
            lastTime = 0
            
            if kFig == nFigures - 1 and lastFile > 0:
                for kFile in range(lastFile):
                    times = time2stack[kFig * nFilesperFigure + kFile]
                    tempFeats.append(feats2stack[kFig * nFilesperFigure + kFile])
                    seizTemp = np.asarray( seizure2stack[kFig * nFilesperFigure + kFile])
                    if not(seizTemp[0] is None):
                        tempSeiz.append(seizTemp + lastTime * np.ones(seizTemp.shape))
                    finalTimes = times + lastTime * np.ones(times.shape)
                    tempTimes.append( finalTimes)
                    lastTime = finalTimes[-1]
            else:
                for kFile in range(nFilesperFigure):
                    times = time2stack[kFig * nFilesperFigure + kFile]
                    tempFeats.append(feats2stack[kFig * nFilesperFigure + kFile])
                    seizTemp = np.asarray( seizure2stack[kFig * nFilesperFigure + kFile])
                    if not(seizTemp[0] is None):
                        tempSeiz.append(seizTemp + lastTime * np.ones(seizTemp.shape))
                    finalTimes = times + lastTime * np.ones(times.shape)
                    tempTimes.append( finalTimes)
                    lastTime = finalTimes[-1]
            
            
            times2return.append(np.concatenate(tempTimes))
            feats2return.append(np.vstack(tempFeats))
            if len(tempSeiz)>0:
                seiz2return.append(np.vstack(tempSeiz))
            else:
                seiz2return.append([])
    
        """# now get the  filesInFig list"""
        filesInFig = list()
        for kFig in range(nFigures):
            files = list()
            if kFig == nFigures - 1 and lastFile > 0:
                for kFile in range(lastFile):
                    files.append(kFig * nFilesperFigure + kFile)
            else:
                for kFile in range(nFilesperFigure):
                    files.append(kFig * nFilesperFigure + kFile)
            filesInFig.append(files)
                
                
    return times2return , feats2return , seiz2return , filesInFig

"""function that takes a list of absolute index of files and gives the corresponding index in the sezurefree_data list """
def abs2clearAndSeiz(figureFiles , subStruct):
    clears = list()
    seizuresInds = list()
    fileNames = subStruct["file_names"]
    seizureFreeList = subStruct["seizurefree_data"]
    seizureList = subStruct["seizures_data"]
    seizuresInFiles = subStruct["seizureInFile"]
    
    for absInd in figureFiles:
        nSeizuresInFile = seizuresInFiles[absInd]
        isInFree = nSeizuresInFile == 0
        name = fileNames[absInd]
        """ look in seizureFree """
        if isInFree:
            found = False
            tempClear = None
            for kClear in range(len(seizureFreeList)):
                tempName = seizureFreeList[kClear]["fileName"]
                if name == tempName:
                    tempClear = kClear
                    found = True
            if not(found):
                raise ValueError(' corresponding file was not found in seizureFree Data structure')   
            else:
                clears.append(tempClear)
                
                """ look in seizureFree """
        else:
            for kSeiz in range(len(seizureList)):
                tempName = seizureList[kSeiz]["fileName"]
                if name == tempName:
                    seizuresInds.append(kSeiz)
            if len(seizuresInds) == 0:
                raise ValueError(' corresponding seziures of the figure were not found ')   

            
    return clears , seizuresInds
"""function that takes a list of absolute index of files and gives the corresponding index in the sezurefree_data list """


    