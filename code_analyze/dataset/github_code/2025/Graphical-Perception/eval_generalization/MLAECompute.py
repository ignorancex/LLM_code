import json
import os
import math
import argparse
import numpy as np
import util
import logging
import sys
import random
import util.pyutils as upy
from util.pyutils import multiImport
import Dataset.UtilIO as uio
from util.Config import ConfigObj
import util.shareCode
from util.shareCode import programInit, globalCatch

def mlaeLoss(test,true):
    return math.log2(abs(true-test)*100+0.125)

def absLoss(test,true):
    return abs(true-test)

def compareLoss(test,true,lossFunc=mlaeLoss,unmatchResult=False):
    #logging.info("%s |\t| %s"%(str(test),str(true)))
    
    testLen = len(test)
    trueLen = len(true)
    pubLen = min(testLen,trueLen)
    lossTotal=0
    lossCount=0
    unmatch=0
    for i in range(pubLen):
        lossTotal+=lossFunc(test[i],true[i])
        lossCount+=1
    if trueLen>testLen:
        for i in range(pubLen,trueLen):
            unmatch+=abs(test[i])
    elif trueLen<testLen:
        for i in range(pubLen,testLen):
            unmatch+=abs(test[i])
    if unmatchResult:
        return lossFunc(unmatch,0)
    else:
        return lossTotal/lossCount

def processTestResult(resultLossFile):
    result = []
    f = open(resultLossFile,"r")
    resultRaw = json.load(f)
    f.close()
    for r in resultRaw:
        if "label_l" in r.keys(): # transformer
            predr=r["pred_v"]
            result.append(predr)
            continue
        #elif "label" in r.keys(): # with cycle
        #    predr=r["pred_v"]
        #    predv=[]
        #    if len(predr)<=1:
        #        predv=predr[0]
        #        result.append(predv)
        #    else:
        #        predv.append(1)
        #        for i in range(len(predr)-1):
        #            predv.append(predv[i]/predr[i])
        #        maxv = max(predv)
        #        predv = [x/maxv for x in predv]
        #        result.append(predv)
        else:
            predr=None
            if "pred_v" in r.keys():
                predr=r["pred_v"]
            else:
                predr=r["pred_n"][0]
            result.append(predr)
    return result

def genStat(arr):


    sorted_arr=sorted(arr)
    quarter=len(sorted_arr)//4
    data=sorted_arr[quarter:-quarter]

    avg=np.mean(data)
    total_std=np.std(data)

    # total=0
    # for i in arr:
    #     total+=i
    # avg = total/len(arr)
    # total_std=0
    # for i in arr:
    #     total_std+=(i-avg)**2
    # total_std = (total_std/(len(arr)-1))**0.5

    dic={}
    dic["avg"]=avg
    dic["std"]=total_std
    return dic

def genLoss(result,target,dataMax):
    lossMLAE=[]
    lossMLAEun=[]
    lossABS=[]
    lossABSun=[]
    for i in range(len(result)):
        ri=[r/dataMax for r in result[i]]
        ti=[t/dataMax for t in target[i]]
        lossMLAE.append(compareLoss(ri,ti,mlaeLoss))
        lossMLAEun.append(compareLoss(ri,ti,mlaeLoss,True))
        lossABS.append(compareLoss(ri,ti,absLoss))
        lossABSun.append(compareLoss(ri,ti,absLoss,True))

    print(len(lossMLAE))
    dic={}
    dic["MLAE"]=genStat(lossMLAE)
    dic["MLAE_unmatch"]=genStat(lossMLAEun)
    dic["ABS"]=genStat(lossABS)
    dic["ABS_unmatch"]=genStat(lossABSun)
    dic["count"]=len(lossMLAE)
    return dic

def computeMLAE(config):


    ConfigObj.default(config,"mlae.iter.useIter",False)

    dataFileList = uio.load(config.mlae.dataListPath,"json")
    maxvPath = os.path.join(os.path.split(config.mlae.dataListPath)[0],"max_v.json")
    dataMax=1.0
    if os.path.exists(maxvPath):
        try:
            dataMax=uio.load(maxvPath,'json')
            dataMax = dataMax[0]
        except BaseException as e:
            logging.warning("Load max value failed %s"%maxvPath)
            logging.warning(e)
            dataMax=1.0
    else:
        logging.warning("Cannot find max value file, set 1.0 as default")
        
    dataFile = [os.path.join(config.mlae.dataTargetFolder,x[1]["num"]) for x in dataFileList]
    logging.info("Detect %d data"%len(dataFile))
    target=[]
    for d in dataFile:
        f = open(d,"r")
        obj = json.load(f)
        f.close()
        target.append(obj)    
    logging.info("Load target complete")

    finalResult=[]
    '''
        "path":"xxxx/xxx/xxx.json"
        "loss"
            "MLAE"
                "avg":2.3
                "std":1.2
            "ABS":
                "avg":2.3
                "std":1.2
            "count":2
    '''
    if config.mlae.iter.useIter:
        for Iter in range(*config.mlae.iter.iterRange):
            dic={}
            resultLossFile = config.mlae.iter.iterFile%Iter
            dic["path"]=resultLossFile
            try:
                result = processTestResult(resultLossFile)
                dic["loss"] = genLoss(result,target,dataMax)
                finalResult.append(dic)
            except BaseException as e:
                logging.warning("Error on Iter File %s > %s"%(resultLossFile,str(e)))
    else:
        for resultLossFile in config.mlae.computeList:
            dic={}
            dic["path"]=resultLossFile
            try:
                result = processTestResult(resultLossFile)
                dic["loss"] = genLoss(result,target,dataMax)
                finalResult.append(dic)
            except BaseException as e:
                logging.warning("Error on File List %s > %s"%(resultLossFile,str(e)))
    logging.info("Complete, %d results"%(len(finalResult)))
    logging.info("Save to %s"%(config.mlae.outputPath))
    uio.save(config.mlae.outputPath,finalResult,"json_format")
    return finalResult


def computeMLAE_mode(config,mode='test'):

    ConfigObj.default(config,"mlae.iter.useIter",False)

    dataFileList = uio.load(config.mlae.dataListPath,"json")
    maxvPath = os.path.join(os.path.split(config.mlae.dataListPath)[0],"max_v.json")
    dataMax=1.0
    if os.path.exists(maxvPath):
        try:
            dataMax=uio.load(maxvPath,"json")
            dataMax = dataMax[0]
        except BaseException as e:
            logging.warning("Load max value failed %s"%maxvPath)
            logging.warning(e)
            dataMax=1.0
    else:
        logging.warning("Cannot find max value file, set 1.0 as default")
        
    dataFile = [os.path.join(config.mlae.dataTargetFolder,x[1]["num"]) for x in dataFileList]
    logging.info("Detect %d data"%len(dataFile))
    target=[]
    for d in dataFile:
        f = open(d,"r")
        obj = json.load(f)
        f.close()
        target.append(obj)    
    logging.info("Load target complete")

    finalResult=[]
    '''
        "path":"xxxx/xxx/xxx.json"
        "loss"
            "MLAE"
                "avg":2.3
                "std":1.2
            "ABS":
                "avg":2.3
                "std":1.2
            "count":2
    '''
    if config.mlae.iter.useIter:
        for Iter in range(*config.mlae.iter.iterRange):
            dic={}
            resultLossFile = config.mlae.iter.iterFile%Iter
            dic["path"]=resultLossFile
            try:
                result = processTestResult(resultLossFile)
                dic["loss"] = genLoss(result,target,dataMax)
                finalResult.append(dic)
            except BaseException as e:
                logging.warning("Error on Iter File %s > %s"%(resultLossFile,str(e)))
    else:
        if mode=='test':

            for resultLossFile in config.mlae.computeList_test:
                dic={}
                dic["path"]=resultLossFile
                try:
                    result = processTestResult(resultLossFile)
                    dic["loss"] = genLoss(result,target,dataMax)
                    finalResult.append(dic)
                except BaseException as e:
                    logging.warning("Error on File List %s > %s"%(resultLossFile,str(e)))
            logging.info("Complete, %d results"%(len(finalResult)))
            logging.info("Save to %s"%(config.mlae.outputPath))
            uio.save(config.mlae.outputPath,finalResult,"json_format")
            # uio.save(config.mlae.outputPath+"_result",[result],"json_format")
            # uio.save(config.mlae.outputPath+"_target",[target],"json_format")

                # uio.save(config.mlae_valid.outputPath,finalResult,"json_format")
            return finalResult
        else:
            for resultLossFile in config.mlae.computeList_valid:
                dic={}
                dic["path"]=resultLossFile
                try:
                    result = processTestResult(resultLossFile)
                    dic["loss"] = genLoss(result,target,dataMax)
                    finalResult.append(dic)
                except BaseException as e:
                    logging.warning("Error on File List %s > %s"%(resultLossFile,str(e)))
            logging.info("Complete, %d results"%(len(finalResult)))
            logging.info("Save to %s"%(config.mlae.outputPath))
            # uio.save(config.mlae.outputPath,finalResult,"json_format")
            uio.save(config.mlae_valid.outputPath,finalResult,"json_format")
            # uio.save(config.mlae_valid.outputPath+"_result",[result],"json_format")
            # uio.save(config.mlae_valid.outputPath+"_target",[target],"json_format")
            return finalResult

    
def main():
    #parse params
    config = programInit()

    computeMLAE(config)
    logging.info("Complete, exit!")

if __name__ == '__main__':
    globalCatch(main)