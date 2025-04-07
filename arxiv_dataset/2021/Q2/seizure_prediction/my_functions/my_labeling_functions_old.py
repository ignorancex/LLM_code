#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:20:14 2019

@author: remy.benmessaoud
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:28:59 2019

@author: remy.benmessaoud
"""

"""===========================================================================
------------------------ Labeling Functions  ---------------------------------
==============================================================================
"""
import numpy as np

def getTimeFromString( string ):
# Example :

# Seizure Start Time: 2451 seconds
# Seizure End Time: 2476 seconds

# we want to get the numbers    
    # fid index of : 
    startInd=string.find(":")
    endInd =string.find("seconds")
    if startInd == -1 or endInd == -1 :
        raise Exception('line not in good format\n{}'.format(string))
    else:
        return int(string[startInd +2 : endInd-1])
 
# =============================================================================
""" it takes 2 n,2 arrays and returns a n,3 array"""
def getFinalMap(valid , train):
    n , d = valid.shape
    trainTimes = train[ : ,0]
    lastTime = 0
    finalMaplist = list()
    for k in range(n):
        validTime = valid[k , 0] 
        sameSample = (validTime == lastTime)
        lastTime = validTime
        if not(sameSample):
            istrainSamp = validTime in trainTimes
            if istrainSamp:
                finalMaplist.append(np.array([validTime , valid[k , 1] , 1]))
            else:
                finalMaplist.append(np.array([validTime , valid[k , 1] , 0]))
    if len(finalMaplist)>0:
        res = np.vstack(finalMaplist)
    else:
        res=list()
    return res
    

"""
===========================================================================
# =============================================================================
# function that takes string and gives the file Name
# e.g : File Name: chb07_01.edf
"""
def str2fileName(myStr):
    # find the : character
    res = myStr.find(":")
    if res != -1:
        return myStr[res + 2 :-1 ]
    else:
        raise Exception('line not in good format\n{}'.format(myStr))
# =============================================================================

"""
# =============================================================================
# function that gives the map between the start time of each window and the labels
# arguments: size of window in seconds , windowStep in s, length of file , number of seizures, list of start/end times of each seizure
"""        
def getTimes2LabelsMap1(windowSize , windowStep , fileLength , numSeizures , timesList , \
                       postSeizureMargin , predHorizon , preIctalMargin , interIctalMargin , endingFileMargin):
      ###########################################################################
    ########################### numSeizure = 0 ################################
    
    if numSeizures == 0:
        # the simplest case where all the windows are considered as interictal
        # we just to remove the beginning and the ending 10 min
        windowStep = 2 * windowStep
        startInd = postSeizureMargin 
        endInd = fileLength - endingFileMargin - windowSize + windowStep
        myTimes = np.arange(startInd , endInd , windowStep)
         
        nPnts = len(myTimes)
        myTimesv=np.reshape(myTimes,(nPnts,1))
        myLabels = np.zeros((nPnts,1), dtype=int)
        myMap =  np.hstack((myTimesv , myLabels))
        
        ######### validation map ##############
        # find duration multiple of winSize
        validStart = 0
        validTimes = np.arange(validStart , fileLength , windowStep)
        validPnts = len(validTimes)
        validTimesv=np.reshape(validTimes,(validPnts,1))
        validLabels = np.zeros((validPnts,1), dtype=int)  ## here the label =1 for preIctal
        validMap=  np.hstack((validTimesv , validLabels))

        return myMap , validMap
#    """##########################################################################
#    ########################### numSeizure = 1 ################################'''
#    """
    # Now we move on to the case where numSeizure =1 because it is easy
    elif numSeizures == 1:
        ################# Preseizure Phase ####################################
        seizureStart = timesList[0][0]
        seizureEnd = timesList[0][1]
        mapsList=list()
        validationList = list()
        postSeizureList = list()
        if  seizureStart- interIctalMargin - windowSize < postSeizureMargin :
            # in this case the seizure happens too soon so we only take preIctal windows
            
            if predHorizon < seizureStart - preIctalMargin:
                ############### preIctal full period ############@
                startInd= seizureStart - preIctalMargin -predHorizon
                endInd = seizureStart- windowSize - preIctalMargin + windowStep
                myTimes = np.arange(startInd , endInd , windowStep)
            
                nPnts = len(myTimes)
                myTimesv=np.reshape(myTimes,(nPnts,1))
                myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                preIctalMap=  np.hstack((myTimesv , myLabels))
                mapsList.append( preIctalMap )
                
                ######### validation map ##############
                # find duration multiple of winSize
                validationDuration = windowStep * int(startInd/windowStep)
                validStart = startInd - validationDuration
                validTimes = np.arange(validStart , startInd , windowStep)
                validPnts = len(validTimes)
                validTimesv=np.reshape(validTimes,(validPnts,1))
                validLabels = np.zeros((validPnts,1), dtype=int)  ## here the label =1 for preIctal
                validMap=  np.hstack((validTimesv , validLabels))
                validationList.append( validMap )
                validationList.append( preIctalMap )
                
            else:
                ############### preIctal  ############@
                
                endInd = seizureStart- windowSize - preIctalMargin + windowStep
                # find the biggest preSeizure duration that is multiple of windowStep
                newPreIctalDuration = windowStep * int(endInd/windowStep)
                startInd= endInd - newPreIctalDuration
                myTimes = np.arange(startInd , endInd , windowStep)
            
                nPnts = len(myTimes)
                myTimesv=np.reshape(myTimes,(nPnts,1))
                myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                preIctalMap=  np.hstack((myTimesv , myLabels))
                mapsList.append( preIctalMap )
                validationList.append( preIctalMap )
                
        else:
            ####### interIctal ############
            startIndPre = postSeizureMargin 
            endIndPre = seizureStart - interIctalMargin - windowSize + windowStep
            myTimes = np.arange(startIndPre , endIndPre , windowStep)
            
            nPnts = len(myTimes)
            myTimesv=np.reshape(myTimes,(nPnts,1))
            myLabels = np.zeros((nPnts,1), dtype=int)
            myInterIctalMap =  np.hstack((myTimesv , myLabels))
            mapsList.append(myInterIctalMap)
            validationList.append(myInterIctalMap)
            #######  extend validaton times ##############
            
            validEnd = seizureStart - preIctalMargin -predHorizon
            validPeriod = windowStep * np.floor(validEnd/windowStep)
            validStart = endIndPre - validPeriod
            validTimes = np.arange(validStart , validEnd , windowStep)
            validnPnts = len(validTimes)
            validTimesv=np.reshape(validTimes,(validnPnts,1))
            myLabels = np.zeros((validnPnts,1), dtype=int)
            validMap =  np.hstack((validTimesv , myLabels))
            validationList.append(validMap)
            
            ############### preIctal  ############@
            startInd= seizureStart - preIctalMargin -predHorizon
            endInd = seizureStart- windowSize - preIctalMargin + windowStep
            myTimes  = np.arange(startInd , endInd , windowStep)
           
            nPnts = len(myTimes)
            myTimesv=np.reshape(myTimes,(nPnts,1))
            myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
            preIctalMap=  np.hstack((myTimesv , myLabels))
            mapsList.append( preIctalMap )
            validationList.append(preIctalMap)
            
        ################# Postseizure Phase  ######################################
        if  seizureEnd +  postSeizureMargin < fileLength - endingFileMargin - windowSize :
            
            startIndPost = seizureEnd +  postSeizureMargin 
            endIndPost = fileLength - endingFileMargin - windowSize+ windowStep
            myTimes  = np.arange(startIndPost , endIndPost , windowStep)
            nPnts = len(myTimes)
            myTimesv=np.reshape(myTimes,(nPnts,1))
            myLabels = np.zeros((nPnts,1), dtype=int)
            myPostMap =  np.hstack((myTimesv , myLabels))
            postSeizureList.append(myPostMap)
        ################# Here we concacetenate pre/post seizure maps
        myMap= list()
        myvalidMap= list()
        myMap.append(np.vstack(mapsList))
        myvalidMap.append( np.vstack(validationList) )
        if len(postSeizureList)>0 :
            postSeiz = np.vstack(postSeizureList)
        else:
            postSeiz = None
        return myMap , myvalidMap , postSeiz
    
    ##""" ###########################################################################
    ########################### numSeizure > 1 ################################"""
    elif numSeizures > 1:  
        myMap= list()
        myvalidMap= list()
        postSeizure = list()
        
        for kSeiz in np.arange(numSeizures):
            mapsList = list() # list of time2label maps that will be filled and vstacked
            validationList = list()
            
            """ =====================  First Seizure ================="""
            if kSeiz == 0:
                seizureStart = timesList[kSeiz][0]

                if  seizureStart- interIctalMargin - windowSize < postSeizureMargin :
                    # in this case the seizure happens too soon so we only take preIctal windows
                    if predHorizon < seizureStart - preIctalMargin:
                        ############### preIctal  ############@
                        startInd= seizureStart - preIctalMargin -predHorizon
                        endInd = seizureStart- windowSize - preIctalMargin + windowStep
                        myTimes = np.arange(startInd , endInd , windowStep)
                    
                        nPnts = len(myTimes)
                        myTimesv=np.reshape(myTimes,(nPnts,1))
                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                        preIctalMap=  np.hstack((myTimesv , myLabels))
                        mapsList.append( preIctalMap )
                        ######### validation map ##############
                        # find duration multiple of winSize
                        validationDuration = windowStep * int(startInd/windowStep)
                        validStart = startInd - validationDuration
                        validTimes = np.arange(validStart , startInd , windowStep)
                        validPnts = len(validTimes)
                        validTimesv=np.reshape(validTimes,(validPnts,1))
                        validLabels = np.zeros((validPnts,1), dtype=int)  ## here the label =1 for preIctal
                        validMap=  np.hstack((validTimesv , validLabels))
                        validationList.append( validMap )
                        validationList.append( preIctalMap )
                        
                    else:
                        ############### preIctal  ############@
                        endInd = seizureStart- windowSize - preIctalMargin + windowStep
                        # find the biggest preSeizure duration that is multiple of windowStep
                        newPreIctalDuration = windowStep * int(endInd/windowStep)
                        startInd= endInd - newPreIctalDuration
                        
                        myTimes = np.arange(startInd , endInd , windowStep)
                        nPnts = len(myTimes)
                        myTimesv=np.reshape(myTimes,(nPnts,1))
                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                        preIctalMap=  np.hstack((myTimesv , myLabels))
                        mapsList.append( preIctalMap )
                        validationList.append( preIctalMap )
                else:
                    ####### interIctal ############
                    startIndPre = postSeizureMargin 
                    endIndPre = seizureStart - interIctalMargin - windowSize + windowStep
                    myTimes = np.arange(startIndPre , endIndPre , windowStep)
                    
                    nPnts = len(myTimes)
                    myTimesv=np.reshape(myTimes,(nPnts,1))
                    myLabels = np.zeros((nPnts,1), dtype=int)
                    myInterIctalMap =  np.hstack((myTimesv , myLabels))
                    mapsList.append(myInterIctalMap)
                    validationList.append(myInterIctalMap)
                    #######  extend validaton times ##############
                    validStart = endIndPre
                    validEnd = seizureStart- windowSize - predHorizon - preIctalMargin
                    validTimes = np.arange(validStart , validEnd , windowStep)
                    validnPnts = len(validTimes)
                    validTimesv=np.reshape(validTimes,(validnPnts,1))
                    myLabels = np.zeros((validnPnts,1), dtype=int)
                    validationMap =  np.hstack((validTimesv , myLabels))
                    validationList.append(validationMap)
                    
                    ############### preIctal  ############@
                    startInd= seizureStart- windowSize - predHorizon - preIctalMargin
                    endInd = seizureStart- windowSize -preIctalMargin + windowStep
                    myTimes  = np.arange(startInd , endInd , windowStep)
                   
                    nPnts = len(myTimes)
                    myTimesv=np.reshape(myTimes,(nPnts,1))
                    myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                    preIctalMap=  np.hstack((myTimesv , myLabels))
                    mapsList.append( preIctalMap )
                    validationList.append(preIctalMap)
                """ =========== now we store info about seizure 1  =============="""
                myMap.append(np.vstack(mapsList))
                myvalidMap.append( np.vstack(validationList))
                
            """ ================   Intermediate  Seizures  ================ """        
            if kSeiz > 0:
                seizureStart = timesList[kSeiz][0]
                lastSeizureEnd = timesList[kSeiz - 1][1]
                if  seizureStart- interIctalMargin - windowSize < lastSeizureEnd + postSeizureMargin :
                    # in this case the seizure happens too soon so we only take preIctal windows
                    if lastSeizureEnd + postSeizureMargin < seizureStart - predHorizon -preIctalMargin:
                        ############### preIctal  ############@
                        startInd= seizureStart- windowSize - predHorizon -preIctalMargin
                        endInd = seizureStart- windowSize -preIctalMargin + windowStep
                        myTimes = np.arange(startInd , endInd , windowStep)
                    
                        nPnts = len(myTimes)
                        myTimesv=np.reshape(myTimes,(nPnts,1))
                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                        preIctalMap=  np.hstack((myTimesv , myLabels))
                        mapsList.append( preIctalMap )
                        ######### validation map ##############
                        # find duration multiple of winSize
                        validationDuration = windowStep * int((startInd-(lastSeizureEnd+ postSeizureMargin))/windowStep)
                        validStart = startInd - validationDuration
                        validTimes = np.arange(validStart , startInd , windowStep)
                        validPnts = len(validTimes)
                        validTimesv=np.reshape(validTimes,(validPnts,1))
                        validLabels = np.zeros((validPnts,1), dtype=int)  ## here the label =1 for preIctal
                        validMap=  np.hstack((validTimesv , validLabels))
                        validationList.append( validMap )
                        validationList.append( preIctalMap )
                    
                    else:
                        ############### preIctal  ############@
                        absStartInd= lastSeizureEnd + postSeizureMargin
                        endInd = seizureStart- windowSize - preIctalMargin + windowStep
                        # find the biggest preSeizure duration that is multiple of windowStep
                        newPreIctalDuration = windowStep * int((endInd - absStartInd) / windowStep)
                        startInd= endInd - newPreIctalDuration
                        
                        myTimes = np.arange(startInd , endInd , windowStep)
                    
                        nPnts = len(myTimes)
                        myTimesv=np.reshape(myTimes,(nPnts,1))
                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                        preIctalMap=  np.hstack((myTimesv , myLabels))
                        mapsList.append( preIctalMap )
                        validationList.append( preIctalMap )
                else:
                    ####### interIctal ############
                    startIndPre = lastSeizureEnd + postSeizureMargin 
                    endIndPre = seizureStart - interIctalMargin - windowSize + windowStep
                    myTimes  = np.arange(startIndPre , endIndPre , windowStep)
                    
                    nPnts = len(myTimes)
                    myTimesv=np.reshape(myTimes,(nPnts,1))
                    myLabels = np.zeros((nPnts,1), dtype=int)
                    myInterIctalMap =  np.hstack((myTimesv , myLabels))
                    mapsList.append(myInterIctalMap)
                    validationList.append( myInterIctalMap )
                    
                    #######  extend validaton times ##############
                    validStart = endIndPre
                    validEnd = seizureStart- windowSize - predHorizon -preIctalMargin
                    validTimes = np.arange(validStart , validEnd , windowStep)
                    validnPnts = len(validTimes)
                    validTimesv=np.reshape(validTimes,(validnPnts,1))
                    myLabels = np.zeros((validnPnts,1), dtype=int)
                    validationMap =  np.hstack((validTimesv , myLabels))
                    validationList.append(validationMap)
                                        
                    ############### preIctal  ############@
                    startInd= seizureStart- windowSize - predHorizon -preIctalMargin
                    endInd = seizureStart- windowSize -preIctalMargin + windowStep
                    myTimes  = np.arange(startInd , endInd , windowStep)
                   
                    nPnts = len(myTimes)
                    myTimesv=np.reshape(myTimes,(nPnts,1))
                    myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
                    preIctalMap=  np.hstack((myTimesv , myLabels))
                    mapsList.append( preIctalMap )
                    validationList.append( preIctalMap )
                """ =========== now we store info about seizure 1  =============="""
                myMap.append(np.vstack(mapsList))
                myvalidMap.append( np.vstack(validationList))
                
        ############### we look after the last seizure #######################      
        seizureEnd = lastSeizureEnd = timesList[- 1][1]
        if  seizureEnd +  postSeizureMargin < fileLength - endingFileMargin - windowSize :
            
            startIndPost = seizureEnd +  postSeizureMargin 
            endIndPost = fileLength - endingFileMargin - windowSize + windowStep
            myTimes  = np.arange(startIndPost , endIndPost , windowStep)
           
            nPnts = len(myTimes)
            myTimesv=np.reshape(myTimes,(nPnts,1))
            myLabels = np.zeros((nPnts,1), dtype=int)
            myPostMap =  np.hstack((myTimesv , myLabels))
            postSeizure.append(myPostMap)
        
        if len(postSeizure)>0 :
            postSeiz = np.vstack(postSeizure)
        else:
            postSeiz = None
            
        return myMap , myvalidMap , postSeiz
# =============================================================================

""" old version of times to labels mapping """
"""
## =============================================================================
## function that gives the map between the start time of each window and the labels
## arguments: size of window in seconds , windowStep in s, length of file , number of seizures, list of start/end times of each seizure
#"""        
#def getTimes2LabelsMap(windowSize , windowStep , fileLength , numSeizures , timesList , \
#                       postSeizureMargin , predHorizon , preIctalMargin , interIctalMargin , endingFileMargin):
#      ###########################################################################
#    ########################### numSeizure = 0 ################################
#    
#    if numSeizures == 0:
#        # the simplest case where all the windows are considered as interictal
#        # we just to remove the beginning and the ending 10 min
#        startInd = postSeizureMargin 
#        endInd = fileLength - endingFileMargin - windowSize + windowStep
#        myTimes = np.arange(startInd , endInd , windowStep)
#         
#        nPnts = len(myTimes)
#        myTimesv=np.reshape(myTimes,(nPnts,1))
#        myLabels = np.zeros((nPnts,1), dtype=int)
#        myMap =  np.hstack((myTimesv , myLabels))
#        return myMap
##    """##########################################################################
##    ########################### numSeizure = 1 ################################'''
##    """
#    # Now we move on to the case where numSeizure =1 because it is easy
#    elif numSeizures == 1:
#        ################# Preseizure Phase ####################################
#        seizureStart = timesList[0][0]
#        seizureEnd = timesList[0][1]
#        mapsList=list()
#        validationList = list()
#        if  seizureStart- interIctalMargin - windowSize < postSeizureMargin :
#            # in this case the seizure happens too soon so we only take preIctal windows
#            if predHorizon < seizureStart - preIctalMargin:
#                ############### preIctal  ############@
#                startInd= seizureStart - preIctalMargin -predHorizon
#                endInd = seizureStart- windowSize - preIctalMargin + windowStep
#                myTimes = np.arange(startInd , endInd , windowStep)
#            
#                nPnts = len(myTimes)
#                myTimesv=np.reshape(myTimes,(nPnts,1))
#                myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                preIctalMap=  np.hstack((myTimesv , myLabels))
#                mapsList.append( preIctalMap )
#            else:
#                ############### preIctal  ############@
#                
#                endInd = seizureStart- windowSize - preIctalMargin + windowStep
#                # find the biggest preSeizure duration that is multiple of windowStep
#                newPreIctalDuration = windowStep * int(endInd/windowStep)
#                startInd= endInd - newPreIctalDuration
#                myTimes = np.arange(startInd , endInd , windowStep)
#            
#                nPnts = len(myTimes)
#                myTimesv=np.reshape(myTimes,(nPnts,1))
#                myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                preIctalMap=  np.hstack((myTimesv , myLabels))
#                mapsList.append( preIctalMap )
#                
#        else:
#            ####### interIctal ############
#            startIndPre = postSeizureMargin 
#            endIndPre = seizureStart - interIctalMargin - windowSize + windowStep
#            myTimes = np.arange(startIndPre , endIndPre , windowStep)
#            
#            nPnts = len(myTimes)
#            myTimesv=np.reshape(myTimes,(nPnts,1))
#            myLabels = np.zeros((nPnts,1), dtype=int)
#            myInterIctalMap =  np.hstack((myTimesv , myLabels))
#            mapsList.append(myInterIctalMap)
#            
#            ############### preIctal  ############@
#            startInd= seizureStart - preIctalMargin -predHorizon
#            endInd = seizureStart- windowSize - preIctalMargin + windowStep
#            myTimes  = np.arange(startInd , endInd , windowStep)
#           
#            nPnts = len(myTimes)
#            myTimesv=np.reshape(myTimes,(nPnts,1))
#            myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#            preIctalMap=  np.hstack((myTimesv , myLabels))
#            mapsList.append( preIctalMap )
#            
#        ################# Postseizure Phase  ######################################
#        if  seizureEnd +  postSeizureMargin < fileLength - endingFileMargin - windowSize :
#            
#            startIndPost = seizureEnd +  postSeizureMargin 
#            endIndPost = fileLength - endingFileMargin - windowSize+ windowStep
#            myTimes  = np.arange(startIndPost , endIndPost , windowStep)
#           
#            nPnts = len(myTimes)
#            myTimesv=np.reshape(myTimes,(nPnts,1))
#            myLabels = np.zeros((nPnts,1), dtype=int)
#            myPostMap =  np.hstack((myTimesv , myLabels))
#            mapsList.append(myPostMap)
#            
#        ################# Here we concacetenate pre/post seizure maps
#        myMap= np.vstack(mapsList)
#        return myMap
#    ##""" ###########################################################################
#    ########################### numSeizure > 1 ################################"""
#    elif numSeizures > 1:
#        mapsList = list() # list of time2label maps that will be filled and vstacked
#        validationList = list()
#        for kSeiz in np.arange(numSeizures):
#            """ =====================  First Seizure ================="""
#            if kSeiz == 0:
#                seizureStart = timesList[kSeiz][0]
#
#                if  seizureStart- interIctalMargin - windowSize < postSeizureMargin :
#                    # in this case the seizure happens too soon so we only take preIctal windows
#                    if predHorizon < seizureStart - preIctalMargin:
#                        ############### preIctal  ############@
#                        startInd= seizureStart - preIctalMargin -predHorizon
#                        endInd = seizureStart- windowSize - preIctalMargin + windowStep
#                        myTimes = np.arange(startInd , endInd , windowStep)
#                    
#                        nPnts = len(myTimes)
#                        myTimesv=np.reshape(myTimes,(nPnts,1))
#                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                        preIctalMap=  np.hstack((myTimesv , myLabels))
#                        mapsList.append( preIctalMap )
#                    else:
#                        ############### preIctal  ############@
#                        endInd = seizureStart- windowSize - preIctalMargin + windowStep
#                        # find the biggest preSeizure duration that is multiple of windowStep
#                        newPreIctalDuration = windowStep * int(endInd/windowStep)
#                        startInd= endInd - newPreIctalDuration
#                        
#                        myTimes = np.arange(startInd , endInd , windowStep)
#                        nPnts = len(myTimes)
#                        myTimesv=np.reshape(myTimes,(nPnts,1))
#                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                        preIctalMap=  np.hstack((myTimesv , myLabels))
#                        mapsList.append( preIctalMap )
#                        
#                else:
#                    ####### interIctal ############
#                    startIndPre = postSeizureMargin 
#                    endIndPre = seizureStart - interIctalMargin - windowSize + windowStep
#                    myTimes = np.arange(startIndPre , endIndPre , windowStep)
#                    
#                    nPnts = len(myTimes)
#                    myTimesv=np.reshape(myTimes,(nPnts,1))
#                    myLabels = np.zeros((nPnts,1), dtype=int)
#                    myInterIctalMap =  np.hstack((myTimesv , myLabels))
#                    mapsList.append(myInterIctalMap)
#                    
#                    ############### preIctal  ############@
#                    startInd= seizureStart- windowSize - predHorizon - preIctalMargin
#                    endInd = seizureStart- windowSize -preIctalMargin + windowStep
#                    myTimes  = np.arange(startInd , endInd , windowStep)
#                   
#                    nPnts = len(myTimes)
#                    myTimesv=np.reshape(myTimes,(nPnts,1))
#                    myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                    preIctalMap=  np.hstack((myTimesv , myLabels))
#                    mapsList.append( preIctalMap )
#            """ ================   Intermediate  Seizures  ================ """        
#            if kSeiz > 0:
#                seizureStart = timesList[kSeiz][0]
#                lastSeizureEnd = timesList[kSeiz - 1][1]
#                if  seizureStart- interIctalMargin - windowSize < lastSeizureEnd + postSeizureMargin :
#                    # in this case the seizure happens too soon so we only take preIctal windows
#                    if lastSeizureEnd + postSeizureMargin < seizureStart - predHorizon -preIctalMargin:
#                        ############### preIctal  ############@
#                        startInd= seizureStart- windowSize - predHorizon -preIctalMargin
#                        endInd = seizureStart- windowSize -preIctalMargin + windowStep
#                        myTimes = np.arange(startInd , endInd , windowStep)
#                    
#                        nPnts = len(myTimes)
#                        myTimesv=np.reshape(myTimes,(nPnts,1))
#                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                        preIctalMap=  np.hstack((myTimesv , myLabels))
#                        mapsList.append( preIctalMap )
#                    
#                    else:
#                        ############### preIctal  ############@
#                        absStartInd= lastSeizureEnd + postSeizureMargin
#                        endInd = seizureStart- windowSize - preIctalMargin + windowStep
#                        # find the biggest preSeizure duration that is multiple of windowStep
#                        newPreIctalDuration = windowStep * int((endInd - absStartInd) / windowStep)
#                        startInd= endInd - newPreIctalDuration
#                        
#                        myTimes = np.arange(startInd , endInd , windowStep)
#                    
#                        nPnts = len(myTimes)
#                        myTimesv=np.reshape(myTimes,(nPnts,1))
#                        myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                        preIctalMap=  np.hstack((myTimesv , myLabels))
#                        mapsList.append( preIctalMap )
#                        
#                else:
#                    ####### interIctal ############
#                    startIndPre = lastSeizureEnd + postSeizureMargin 
#                    endIndPre = seizureStart - interIctalMargin - windowSize + windowStep
#                    myTimes  = np.arange(startIndPre , endIndPre , windowStep)
#                    
#                    nPnts = len(myTimes)
#                    myTimesv=np.reshape(myTimes,(nPnts,1))
#                    myLabels = np.zeros((nPnts,1), dtype=int)
#                    myInterIctalMap =  np.hstack((myTimesv , myLabels))
#                    mapsList.append(myInterIctalMap)
#                    
#                    ############### preIctal  ############@
#                    startInd= seizureStart- windowSize - predHorizon -preIctalMargin
#                    endInd = seizureStart- windowSize -preIctalMargin + windowStep
#                    myTimes  = np.arange(startInd , endInd , windowStep)
#                   
#                    nPnts = len(myTimes)
#                    myTimesv=np.reshape(myTimes,(nPnts,1))
#                    myLabels = np.ones((nPnts,1), dtype=int)  ## here the label =1 for preIctal
#                    preIctalMap=  np.hstack((myTimesv , myLabels))
#                    mapsList.append( preIctalMap )
#            
#        ############### we look after the last seizure #######################      
#        seizureEnd = lastSeizureEnd = timesList[- 1][1]
#        if  seizureEnd +  postSeizureMargin < fileLength - endingFileMargin - windowSize :
#            
#            startIndPost = seizureEnd +  postSeizureMargin 
#            endIndPost = fileLength - endingFileMargin - windowSize + windowStep
#            myTimes  = np.arange(startIndPost , endIndPost , windowStep)
#           
#            nPnts = len(myTimes)
#            myTimesv=np.reshape(myTimes,(nPnts,1))
#            myLabels = np.zeros((nPnts,1), dtype=int)
#            myPostMap =  np.hstack((myTimesv , myLabels))
#            mapsList.append(myPostMap)
#            
#        return np.vstack(mapsList) , np.vstack(validationList)