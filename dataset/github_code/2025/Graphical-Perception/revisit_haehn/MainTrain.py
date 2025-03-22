import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import argparse
from util import LogConfig
from util import Config
from util.Config import ConfigObj
from Networks import *
from util.LogConfig import logConfig
import torchvision.transforms as transforms
import logging
import Dataset
import Dataset.UtilIO as uio
from visdom import Visdom
import sys
import util.parallelUtil as para
import random
import util
import util.pyutils as upy
from util.pyutils import multiImport
import time
import GenDataset
from GenDataset import GenDataset
import util.shareCode
from util.shareCode import programInit, globalCatch

class TrainInfo:

    def __init__(self):
        self.lossInfo={}
        self.curEpoch=0
        self.config=None
        self.curIters=0
        self.validResults={}
        self.testResults={}
        self.datasetInfo=None
        pass

def collate_fn_inner(data,dic):
    for k,v in data.items():
        if isinstance(v,dict):
            if k in dic.keys():
                collate_fn_inner(v,dic[k])
            else:
                subdic={}
                dic[k] = collate_fn_inner(v,subdic)
        elif isinstance(v,list):
            if k in dic.keys():
                dic[k]+=v
            else:
                dic[k]=v
        else: # tensor
            if k in dic.keys():
                dic[k] = torch.cat([dic[k],v.unsqueeze(0)],dim=0)
            else:
                dic[k]=v.unsqueeze(0)
    return dic

def collate_fn(all_data):
    dic={}
    for x in all_data:
        dic = collate_fn_inner(x,dic)
    return dic
    
class TimeTick:
    def __init__(self):
        self.t=time.time()
    
    def tick(self,tag=""):
        delta = self.t-time.time()
        logging.info("Time %s > %f s"%(tag,delta))
        self.t=time.time()
class TrainHelper:

    def __init__(self,device,config):
        #global config
        self.config=config
        self.device=device

        ConfigObj.default(self.config,"utility.debug",False)
        ConfigObj.default(self.config,"cuda.parallel",False)
        ConfigObj.default(self.config,"cuda.use_gpu",[0,1])
        ConfigObj.default(self.config,"continueTrain.enableContinue",False)
        ConfigObj.default(self.config,"continueTrain.autoContinue",False)
        ConfigObj.default(self.config,"test.displayImageIndex",0)
        ConfigObj.default(self.config,"test.storeImageIndex",[0])
        ConfigObj.default(self.config,"test.testResultIterOutputFolder","Iter_%d")
        ConfigObj.default(self.config,"utility.visdomPort",8097)
        ConfigObj.default(self.config,"utility.globalSeed",0)
        ConfigObj.default(self.config,"data.useThreads",1)
        ConfigObj.default(self.config,"data.trainSeed",100)
        ConfigObj.default(self.config,"data.validSeed",10000)
        ConfigObj.default(self.config,"data.inputFolder","input")
        ConfigObj.default(self.config,"data.outputFolder","target")
        ConfigObj.default(self.config,"data.orgFolder","org")
        ConfigObj.default(self.config,"modelOutput.saveEpoch",0)
        ConfigObj.default(self.config,"test.testEpoch",0)
        ConfigObj.default(self.config,"display.updateEpoch",0)
        ConfigObj.default(self.config,"earlyStop.tolerateTimes",10)
        ConfigObj.default(self.config,"earlyStop.minValue",0)
        ConfigObj.default(self.config,"earlyStop.enable",False)
        ConfigObj.default(self.config,"earlyStop.paramName","loss")
        


        # load model
        singleModelName = self.config.model.name.split(".")[-1]
        modelName = "Networks.%s.%s"%(self.config.model.name,singleModelName)
        logging.info("Create Model %s"%modelName)
        modelClass=multiImport(modelName)
        self.model=None
        try:
            logging.info("Try to pass dict")
            self.model = modelClass(**Config.obj2dic(config.model.param))
        except:
            logging.info("Direct pass params")
            self.model = modelClass(config.model.param)

        self.model.setConfig(config,device)

        self.setSeed(config.utility.globalSeed)
        self.info = TrainInfo()
        self.info.config = self.config
        self.info.curIters=0
        self.info.curEpoch=0

        
        self.enableVis = config.utility.visdomPort>0
        if self.enableVis:
            logging.info("Try to connect visdom, port %d"%(config.utility.visdomPort))
            self.viz=Visdom(port=config.utility.visdomPort)
        else:
            logging.info("Visdom is disabled!")

         # load data
        paths = uio.load(os.path.join(self.config.data.trainPath,"list"),"json")
        datasetManagerName = "Dataset.%s.%s"%(config.data.manageType,config.data.manageType)
        self._dataManagerClass = multiImport(datasetManagerName)
        
        self.trainData = self._dataManagerClass(config.data.trainPath,config,paths,"train")
        if self.info.datasetInfo is None:
            self.info.datasetInfo = self.trainData.onEpochComplete()
        else:
            self.trainData.onLoad(self.info.datasetInfo)

        paths = uio.load(os.path.join(self.config.data.validPath,"list"),"json")
        self.validData = self._dataManagerClass(config.data.validPath,config,paths,"valid")

        paths = uio.load(os.path.join(self.config.data.trainPath,"list_test"),"json")
        self.testData = self._dataManagerClass(config.data.trainPath,config,paths,"test")

        self.trainDataLen = len(self.trainData)
        self.validDataLen = len(self.validData)

        self.dataLoader = torch.utils.data.DataLoader(self.trainData,batch_size=config.trainParam.batchSize,shuffle=False,pin_memory=True,num_workers=config.trainParam.loadThread,collate_fn=collate_fn)
        self.validDataLoader = torch.utils.data.DataLoader(self.validData,batch_size=1,shuffle=False,pin_memory=True,num_workers=0,collate_fn=collate_fn)
        self.testDataLoader = torch.utils.data.DataLoader(self.testData,batch_size=1,shuffle=False,pin_memory=True,num_workers=0,collate_fn=collate_fn)

        self.debug = self.config.utility.debug

        trainLen = len(self.dataLoader)
        if self.config.modelOutput.saveEpoch!=0:
            self.config.modelOutput.saveIterInterval = int(self.config.modelOutput.saveEpoch*trainLen)
        if self.config.test.testEpoch!=0:
            self.config.test.testIterInterval = int(self.config.test.testEpoch*trainLen)
        if self.config.display.updateEpoch!=0:
            self.config.modelOutput.updateIterInterval = int(self.config.display.updateEpoch*trainLen)

        # load models
        if config.continueTrain.enableContinue:
            fromIter = self.config.continueTrain.fromEpoch*trainLen
            modelName = self.config.modelOutput.modelName % fromIter
            infoName = self.config.modelOutput.modelInfo % fromIter
            self.load(modelName,infoName)
            config = self.info.config
        elif config.continueTrain.autoContinue:
            maxIter = self.config.trainParam.maxEpoch*trainLen
            saveIter = self.config.modelOutput.saveIterInterval
            maxi=0
            detectCount=0
            for i in range(0,maxIter,saveIter):
                if os.path.exists(self.config.modelOutput.modelName%i) and os.path.exists(self.config.modelOutput.infoName%i):
                    maxi=max(maxi,i)
                    detectCount+=1
            if detectCount==0:
                logging.info("Cannot Detect model for continuing training")
            else:
                logging.info("Detect model for continuing training, Iter %d, Total Models %d"%(maxi,detectCount))
                modelName = self.config.modelOutput.modelName % maxi
                infoName = self.config.modelOutput.modelInfo % maxi
                self.load(modelName,infoName)
                config = self.info.config

        # generate data
        modelName = "Dataset.%s.%s"%(config.data.generator.genType,config.data.generator.genType)
        logging.info("Create dataset %s"%modelName)
        self.dataGenerator = multiImport(modelName)(config)
        self.genData()

        uio.mkdirsExceptFile(self.config.modelOutput.modelName)
        uio.mkdirsExceptFile(self.config.modelOutput.modelInfo)
        uio.mkdirsExceptFile(self.config.test.testResultOutputPath)

        if self.config.earlyStop.enable:
            logging.info("Early Stop is enabled, stop if validation cost %s not drop after %d validation times"% (self.config.earlyStop.paramName,self.config.earlyStop.tolerateTimes))
        self.needStop=False
        
       

    def setSeed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def genData(self):
        gend = GenDataset(self.config)
        gend.genData()

    def load(self, modelName, infoName):
        self.model = self.model.cpu()
        logging.info("Load model %s"%modelName)
        logging.info("Load model info %s"%infoName)

        self.model.load_state_dict(torch.load(modelName))
        self.model=self.model.to(self.device)
        raw=uio.load(infoName,"json")
        lossInfo = raw["lossInfo"]
        validResults = raw["validResults"]
        testResults = raw["testResults"]
        raw.pop("lossInfo")
        raw.pop("validResults")
        self.info = Config.toConfigObj(raw)
        self.info.lossInfo=lossInfo
        self.info.validResults=validResults
        self.info.testResults=testResults

        self.config=self.info.config
        self.model.setConfig(self.config,self.device)

        # draw 
        if self.enableVis:
            iters = [i for i in range(self.config.display.updateIterInterval,self.info.curIters,self.config.display.updateIterInterval)]
            iterx = np.array(iters)
            for k,v in self.info.lossInfo.items():
                if len(iterx)<len(v):
                    iters.append(self.info.curIters)
                    iterx = np.array(iters)
                self.viz.line(X=iterx,Y=np.array(v),win="Train",update='append',name=k)

            iters = [i for i in range(self.config.test.testIterInterval,self.info.curIters,self.config.test.testIterInterval)]
            iterx = np.array(iters)
            for k,v in self.info.validResults.items():
                if len(iterx)<len(v):
                    iters.append(self.info.curIters)
                    iterx = np.array(iters)
                self.viz.line(X=iterx,Y=np.array(v),win="valid",name=k,update='append')

            iters = [i for i in range(self.config.test.testIterInterval,self.info.curIters,self.config.test.testIterInterval)]
            iterx = np.array(iters)
            for k,v in self.info.testResults.items():
                if len(iterx)<len(v):
                    iters.append(self.info.curIters)
                    iterx = np.array(iters)
                self.viz.line(X=iterx,Y=np.array(v),win="test",name=k,update='append')
        

    def save(self):
        c=self.config.modelOutput
        modelFileName = c.modelName % self.info.curIters
        modelInfoFileName = c.modelInfo % self.info.curIters

        logging.info("Save model %s..."%modelFileName)
        torch.save(self.model.cpu().state_dict(),modelFileName)
        uio.save(modelInfoFileName,Config.obj2dic(self.info),"json")

        self.model.to(self.device)

    
    def train(self):
        logging.info("Begin to train, start epoch from %d"%(self.info.curEpoch))
        #torch.set_deterministic(True) #might cause crash
        while self.info.curEpoch<self.config.trainParam.maxEpoch:
            self.trainEpoch()
            if self.needStop:
                return
        pass


    def __debugDisplayInfo(self,dic,curName="x"):
        try:
            if dic is None:
                return
            logging.info(">> Part %s"%curName)
            if not isinstance(dic,dict):
                logging.info(str(dic))
                return
            for k,v in dic.items():
                name=curName+"."+k
                if isinstance(v,dict):
                    self.__debugDisplayInfo(v,name+"."+k)
                elif isinstance(v,list) or not torch.is_tensor(v):
                    logging.info(v)
                elif len(v.shape)==3 and v.shape[1]+v.shape[2]>50:
                    if self.enableVis:
                        self.viz.image(v.detach().cpu(),win="debug_%s"%name)
                elif len(v.shape)==4 and v.shape[1]<=4 and v.shape[2]+v.shape[3]>50:
                    if self.enableVis:
                        self.viz.image(v.detach().cpu(),win="debug_%s"%name)
                else:
                    logging.info("%s > %s"%(name,str(v.shape)))
                    if len(v.shape)>0:
                        logging.info(v.cpu().detach())
                    else:
                        logging.info(v.cpu().detach())
                    logging.info(" ")
        except BaseException as e:
            logging.warning(str(e))
            logging.info("> %s"%curName)
            logging.info(str(dic))
    '''
        Train one epoch,
        the model should implement method 
            .trainData(self,x)
        which return a dict include:
        loss
            a dict of loss values, visualize as a line chart
    '''
    def trainEpoch(self):
        logging.debug("Train Epoch %d"%self.info.curEpoch)
        # check iters
        skipIters = self.info.curIters - self.info.curEpoch * len(self.dataLoader)
        if skipIters>0:
            logging.info("Simulate Skip Iters %d -> %d"%(self.info.curIters-skipIters,self.info.curIters))
        self.model.train()
        #tt = TimeTick()
        for x in self.dataLoader:
            #tt.tick("data loader")
            if skipIters>0:
                skipIters-=1
                continue
            self.info.curIters+=1
            lossInfo = None
            debugInfo = None
            
            result = self.model.trainData(x)
            #tt.tick("train")
            if isinstance(result,tuple):
                lossInfo=result[0]
                debugInfo=result[1]
            else:
                lossInfo=result
            
            if self.info.curIters % self.config.display.updateIterInterval < 0.5:
                s = "Epoch %3d / %3d | Iters %6d | LR %.7f "%(self.info.curEpoch,self.config.trainParam.maxEpoch,self.info.curIters,self.model.getLR())
                numX=[self.info.curIters]
                for k,v in lossInfo.items():
                    if k not in self.info.lossInfo.keys():
                        self.info.lossInfo[k]=[]
                    vitem=v.cpu().detach().item()
                    self.info.lossInfo[k].append(vitem)
                    # visualize loss
                    numY=[vitem]
                    if self.enableVis:
                        self.viz.line(X=np.array(numX),Y=np.array(numY),win="Train",name=k,update='append')
                    s+= "| %s %8.5f "%(k,vitem)
                logging.info(s)

                if self.debug:
                    logging.info("--- debug info ---")
                    self.__debugDisplayInfo(x,"data")
                    self.__debugDisplayInfo(lossInfo,"loss")
                    self.__debugDisplayInfo(debugInfo,"debug")

            #tt.tick("update info")
            if self.info.curIters % self.config.test.testIterInterval < 0.5:
                self.test()
                self.valid()
                self.model.train()
                self.needStop = self.checkEarlyStop()
                if self.needStop:
                    return
            #tt.tick("test")
            if self.info.curIters % self.config.modelOutput.saveIterInterval < 0.5:
                self.save()
            #tt.tick("save")
        self.model.onEpochComplete(self.info.curEpoch)
        self.info.datasetInfo = self.trainData.onEpochComplete()
        self.info.curEpoch+=1

    def checkEarlyStop(self):
        if not self.config.earlyStop.enable:
            return False
        escfg=self.config.earlyStop
        try:
            if escfg.paramName not in self.info.validResults.keys():
                logging.warning("Unknown Parameter for early stop: "+escfg.paramName)
                return False
            v = self.info.validResults[escfg.paramName]
            if len(v)<escfg.tolerateTimes:
                return False
            minvInd=0
            minv=v[0]
            for i,singlev in enumerate(v):
                if singlev<minv:
                    minv = singlev
                    minvInd=i
            if minvInd<len(v)-escfg.tolerateTimes:
                logging.info("Decide to early stop, validation loss not drop")
                logging.info("Related value:"+str(v))
                return True
            elif minv<escfg.minValue:
                logging.info("Deicde to early stop, reach the minimum value %f"%escfg.minValue)
                return True
            else:
                logging.info("Min value %f, Min Index %d, Length %d, Tolerate %d"%(minv,minvInd,len(v),escfg.tolerateTimes))
        except BaseException as e:
            logging.warning("Exception while check results: "+str(e))
            return False
        return False
    '''
        the model should implement method 
            .test(self,x)
        which return a dict include:
        loss
            a dict of loss values, visualize as a line chart
        result
            a serializable (to json) object, direct store, will not visualize
        images
            a dict of images, visualize as a image,
            we only display a fixed index of image

        except inputs, all values will be stored to:
            testResultOutputPath+testResultIterOutputFolder
    '''
    def valid(self):
        self.model.eval()
        testIter=0
        lossInfo=[]
        lossTotalInfo={}
        resultInfo=[]
        imageInfo=[]
        testCount = len(self.validDataLoader)
        logging.info("Valid Begin! Num: %d"%testCount)
        for x in self.validDataLoader:
            uio.logProgress(testIter,testCount,"Valid Model")
            testIter+=1
            vv = self.model.test(x)
            # process loss
            vloss=vv["loss"]
            vloss2={}
            for k,v in vloss.items():
                vi=v.detach().cpu().item()
                vloss2[k]=vi
                if k not in lossTotalInfo:
                    lossTotalInfo[k]=vi
                else:
                    lossTotalInfo[k]+=vi
            lossInfo.append(vloss2)
            # process result
            resultDic={}
            for k,v in vv["result"].items():
                if v is not None:
                    resultDic[k]=v.detach().cpu().numpy().tolist()
                else:
                    resultDic[k]=None
            resultInfo.append(resultDic)

            if "images" in vv.keys():
                imageInfo.append(v["images"])
        #average
        for k in lossTotalInfo.keys():
            lossTotalInfo[k]/=testCount
            if k not in self.info.validResults.keys():
                self.info.validResults[k]=[lossTotalInfo[k]]
            else:
                self.info.validResults[k].append(lossTotalInfo[k])
        
        # visualize loss
        s="Valid Result |"
        for k,v in lossTotalInfo.items():
            numX=[self.info.curIters]
            numY=[v]
            if self.enableVis:
                self.viz.line(X=np.array(numX),Y=np.array(numY),win="valid",name=k,update='append')
            s+=" %s %8s |"%(k,str(v))
        logging.info(s+"    ")
        # visualize image
        if len(imageInfo)>0:
            imageInfoi = imageInfo[self.config.test.displayImageIndex]
            if self.enableVis:
                for k,v in imageInfoi.items():
                    self.viz.image(v.detach().cpu(),win="Valid_image_%s"%k)

        # store test content
        storePath = os.path.join(self.config.test.testResultOutputPath,self.config.test.testResultIterOutputFolder%self.info.curIters)
        logging.debug("Valid complete, store test results at %s"%storePath)
        os.makedirs(storePath,exist_ok=True)
        # store loss
        uio.save(os.path.join(storePath,"lossInfo"),lossInfo,"json")
        uio.save(os.path.join(storePath,"lossTotalInfo"),lossTotalInfo,"json")
        # store result data
        uio.save(os.path.join(storePath,"resultInfo"),resultInfo,"json")
        # store images
        for i in self.config.test.storeImageIndex:
            if i>=len(imageInfo):
                continue
            imInfo = imageInfo[i]
            for k,v in imInfo.items():
                imgStorePath = os.path.join(storePath,"valid_img_%5d_%s.png"%(i,k))
                transforms.ToPILImage()(v.detach().cpu()).save(imgStorePath)
        
    def test(self):
        self.model.eval()
        testIter=0
        lossInfo=[]
        lossTotalInfo={}
        resultInfo=[]
        imageInfo=[]
        testCount = len(self.testDataLoader)
        logging.info("Test Begin! Num: %d"%testCount)
        for x in self.testDataLoader:
            uio.logProgress(testIter,testCount,"Test Model")
            testIter+=1
            vv = self.model.test(x)
            # process loss
            vloss=vv["loss"]
            vloss2={}
            for k,v in vloss.items():
                vi=v.detach().cpu().item()
                vloss2[k]=vi
                if k not in lossTotalInfo:
                    lossTotalInfo[k]=vi
                else:
                    lossTotalInfo[k]+=vi
            lossInfo.append(vloss2)
            # process result
            resultDic={}
            for k,v in vv["result"].items():
                if v is not None:
                    resultDic[k]=v.detach().cpu().numpy().tolist()
                else:
                    resultDic[k]=None
            resultInfo.append(resultDic)

            if "images" in vv.keys():
                imageInfo.append(v["images"])
        #average
        for k in lossTotalInfo.keys():
            lossTotalInfo[k]/=testCount
            if k not in self.info.testResults.keys():
                self.info.testResults[k]=[lossTotalInfo[k]]
            else:
                self.info.testResults[k].append(lossTotalInfo[k])
        
        # visualize loss
        s="Test Result |"
        for k,v in lossTotalInfo.items():
            numX=[self.info.curIters]
            numY=[v]
            if self.enableVis:
                self.viz.line(X=np.array(numX),Y=np.array(numY),win="test",name=k,update='append')
            s+=" %s %8s |"%(k,str(v))
        logging.info(s+"    ")
        # visualize image
        if len(imageInfo)>0 and self.enableVis:
            imageInfoi = imageInfo[self.config.test.displayImageIndex]
            for k,v in imageInfoi.items():
                self.viz.image(v.detach().cpu(),win="Test_image_%s"%k)

        # store test content
        storePath = os.path.join(self.config.test.testResultOutputPath,self.config.test.testResultIterOutputFolder%self.info.curIters + "_test")
        logging.debug("Test complete, store test results at %s"%storePath)
        os.makedirs(storePath,exist_ok=True)
        # store loss
        uio.save(os.path.join(storePath,"lossInfo"),lossInfo,"json")
        uio.save(os.path.join(storePath,"lossTotalInfo"),lossTotalInfo,"json")
        # store result data
        uio.save(os.path.join(storePath,"resultInfo"),resultInfo,"json")
        # store images
        for i in self.config.test.storeImageIndex:
            if i>=len(imageInfo):
                continue
            imInfo = imageInfo[i]
            for k,v in imInfo.items():
                imgStorePath = os.path.join(storePath,"test_img_%5d_%s.png"%(i,k))
                transforms.ToPILImage()(v.detach().cpu()).save(imgStorePath)

def main():
    #parse params
    config = programInit()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda.detectableGPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("Detect GPU, Use gpu to train the model")
        device = torch.device("cuda")
    else:
        logging.warning("Cannot detect gpu, use cpu")

    th = TrainHelper(device,config)
    th.train()
    logging.info("Complete, save!")
    th.save()

if __name__ == '__main__':
    globalCatch(main)