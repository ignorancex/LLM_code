import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import argparse
from Networks import *
import torchvision.transforms as transforms
import logging
import Dataset
import Dataset.UtilIO as uio
import sys
import util.parallelUtil as para
import random
import util
import util.pyutils as upy
from util.pyutils import multiImport
from util import Config
from util.Config import ConfigObj
import util.shareCode
from util.shareCode import programInit, globalCatch
import GenDataset
import MLAECompute
import json

def collate_fn_inner(data,dic):
    for k,v in data.items():
        if isinstance(v,dict):
            if k in dic.keys():
                collate_fn_inner(v,dic[k])
            else:
                subdic={}
                collate_fn_inner(v,subdic)
                dic[k]=subdic
        elif isinstance(v,list):
            if k in dic.keys():
                dic[k]+=v
            else:
                dic[k]=v
        else: # tensor
            if k in dic.keys():
                torch.stack((dic[k],v.unsqueeze(0)))
            else:
                dic[k]=v.unsqueeze(0)

def collate_fn(all_data):
    dic={}
    for x in all_data:
        collate_fn_inner(x,dic)
    return dic
    
    #
class TestHelper:

    def __init__(self,device,config):
        #global config
        self.config=config
        self.device=device

        if len(config.refer.__dict__)>4:
            logging.info("Detect related config, borrow related config, except dataset config")
            self.config.model = config.refer.model
            self.config.testOption.model.basicPath = config.refer.modelOutput.modelName
            self.testResultOutputPath = config.refer.test.testResultOutputPath
            if self.config.utility.mainPath != 'result':
                self.config.testOption.model.basicPath = self.config.testOption.model.basicPath.replace(self.config.utility.mainPath,'result')
                self.testResultOutputPath = self.testResultOutputPath.replace(self.config.utility.mainPath,'result')

            self.config.trainParam = config.refer.trainParam
            self.name = config.refer.name


        ConfigObj.default(self.config,"utility.debug",False)
        ConfigObj.default(self.config,"cuda.parallel",False)
        ConfigObj.default(self.config,"cuda.use_gpu",[0,1])
        ConfigObj.default(self.config,"continueTrain.enableContinue",False)
        ConfigObj.default(self.config,"test.displayImageIndex",0)
        ConfigObj.default(self.config,"test.storeImageIndex",[0])
        ConfigObj.default(self.config,"utility.globalSeed",0)
        ConfigObj.default(self.config,"data.trainSeed",100)
        ConfigObj.default(self.config,"data.validSeed",10000)
        ConfigObj.default(self.config,"data.inputFolder","input")
        ConfigObj.default(self.config,"data.outputFolder","target")
        ConfigObj.default(self.config,"data.orgFolder","org")
        ConfigObj.default(self.config,"trainParam.batchSize",1)
        ConfigObj.default(self.config,"trainParam.learnType","Adam")
        ConfigObj.default(self.config,"trainParam.learnRate",0.001)
        ConfigObj.default(self.config,"trainParam.adam_weight_decay",0.0005)
        ConfigObj.default(self.config,"trainParam.learnRateMulti","1.0")
        ConfigObj.default(self.config,"trainParam.clipNorm",5)
        ConfigObj.default(self.config,"testOption.dataListPath","")
        ConfigObj.default(self.config,"testOption.outputResult",r"{utility.mainPath}/raw_result/{name}")

        # gener = GenDataset.GenDataset(self.config)
        # gener.genData()
        self.model=None
        

    def initLeft(self):
        config = self.config
        device = self.device
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
            # with open('a.json','w') as f:
            #     json.dump(config,f)
            print(str(config))

        self.model.setConfig(config,device)

        self.setSeed(config.utility.globalSeed)
        print("=============================================")
        # load data
        dataListPath = self.config.testOption.dataListPath
        print(dataListPath)
        if not isinstance(dataListPath,str) or len(dataListPath)==0:
            dataListPath = os.path.join(self.config.data.validPath,"list")
            logging.info("Data list path is None, use %s instead"%dataListPath)
        paths = uio.load(dataListPath,"json")

        paths = uio.load(os.path.join(self.config.data.validPath,"list"),"json")
        datasetManagerName = "Dataset.%s.%s"%(config.data.manageType,config.data.manageType)
        self._dataManagerClass = multiImport(datasetManagerName)
        
        self.testData = self._dataManagerClass(config.data.validPath,config,paths,"exp")
        print("=============================================")

        self.testDataLoader = torch.utils.data.DataLoader(self.testData,batch_size=1,shuffle=False,pin_memory=True,num_workers=0,collate_fn=collate_fn)

        self.debug = self.config.utility.debug

        self.setSeed(config.utility.globalSeed)


    def setSeed(self,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load(self, path):
        self.model = self.model.cpu()
        modelName = path#self.config.testOption.modelPath 
        logging.info("Load model %s"%modelName)

        self.model.load_state_dict(torch.load(modelName))
        self.model=self.model.to(self.device)
        print(modelName)

        self.model.setConfig(self.config,self.device)

    def findModels(self):
        pth = self.config.testOption.model.basicPath
        if pth is None or len(pth)==0:
            logging.error("basicPath is none")
        folder, filebasicName = os.path.split(pth)
        otherName = filebasicName.strip().replace(".pkl","").replace("%d","")
        flist=[]
        for root,dirs,files in os.walk(folder):
            for f in files:
                if f.endswith(".pkl"):
                    temp = f.replace(otherName,"").replace(".pkl","").strip()
                    iterValue = 0
                    try:
                        iterValue=int(temp)
                    except BaseException as e:
                        logging.warning("Ignore model file (not match) %s"%f)
                        continue
                    if iterValue<self.config.testOption.model.minIter:
                        logging.warning("Skip model file (low iter) %s"%f)
                        continue
                    realPath=os.path.join(root,f)
                    flist.append((iterValue,realPath))
                    logging.info("Detect model file %s"%realPath)
        logging.info("Discover %d models"%len(flist))
        if len(flist)==0:
            logging.warning("Cannot find any models, please check the configuration")
            logging.warning("path %s"%pth)
        return flist

    def test(self,storePath):
        os.makedirs(storePath,exist_ok=True)
        self.model.eval()
        lossInfo=[]
        lossTotalInfo={}
        resultInfo=[]
        imageInfo=[]
        testIter=0
        testCount = len(self.testDataLoader)
        logging.info("Test Begin! Num: %d"%testCount)
        for x in self.testDataLoader:
            testIter+=1
            uio.logProgress(testIter,testCount,"Test Model")
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
            # print(resultDic)
            resultInfo.append(resultDic)

            if "images" in vv.keys():
                imageInfo.append(v["images"])
        #average
        for k in lossTotalInfo.keys():
            lossTotalInfo[k]/=testCount
        
        # visualize loss
        s="Test Result |"
        for k,v in lossTotalInfo.items():
            s+=" %s %8s |"%(k,str(v))
        logging.info(s+"    ")

        # store test content
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
        
    


def runTest(config,device):
    th = TestHelper(device,config)
    models = th.findModels()
    sorted(models)
    
    config.mlae.computeList=[]

    if config.testOption.model.minIter==0:
        logging.info("Try to find model with minimum valid loss to test")
        minLoss=None
        minIter=0
        minIndex=0
        i=0
        for iters,modelPath in models:
            folder = os.path.join(th.testResultOutputPath,config.refer.test.testResultIterOutputFolder%iters)
            print(folder)
            f = os.path.join(folder,"lossTotalInfo")
            obj = uio.load(f,"json")
            maxLoss=-1.0
            for k,v in obj.items():
                maxLoss=max(v,maxLoss)
            logging.info("Model loss %f"%maxLoss)
            if minLoss is None or minLoss > maxLoss:
                minLoss=maxLoss
                minIter=iters
                minIndex=i
            i+=1

        logging.info("Decide to use Model (Iter %d), min loss %f"%(minIter,minLoss))

        newModels = [models[minIndex]]
        print(newModels)
        # for iters, modelPath in models:
        #     resultFile=os.path.join(config.testOption.outputResult%iters,"resultInfo.json")
        #     if os.path.exists(resultFile) and iters!=newModels[0][0]:
        #         logging.info("Iter %d is completed, decide to compute loss!"%iters)
        #         newModels.append((iters,modelPath))
        models=newModels[1:]
        models.append(newModels[0])
    print(models)
    init=False
    for iter,modelPath in models:
        resultFile=os.path.join(config.testOption.outputResult%iter,"resultInfo.json")
        # if os.path.exists(resultFile):
        #     logging.warning("Detect result file, skip test %s"%(config.testOption.outputResult%iter))
        #     logging.info("Related result file %s"%resultFile)
        # else:
        if not init:
            th.initLeft()
            init=True
        th.load(modelPath)
        print("=============================================")
        # print(th.model.network.layer4)
        th.test(config.testOption.outputResult%iter)
        config.mlae.computeList.append(resultFile)


def main():
    #parse params
    config = programInit()
    print(config.model)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda.detectableGPU
    device = torch.device("cpu")
    logging.info(config.cuda.detectableGPU)
    if torch.cuda.is_available():
        logging.info("Detect GPU, Use gpu to train the model")
        device = torch.device("cuda")

    runTest(config,device)

    



    logging.info("Begin to compute MLAE Score")

    if not isinstance(config.mlae.dataListPath,str):
        config.mlae.dataListPath = os.path.join(config.data.validPath,"list.json")
        config.mlae.dataTargetFolder = os.path.join(config.data.validPath,config.data.outputFolder)
        logging.info("Cannot find config.mlae.dataListPath, use %s instead"%config.mlae.dataListPath)
        logging.info("Use %s instead for targets"%config.mlae.dataTargetFolder)

    config.mlae.iter.useIter=False

    results = MLAECompute.computeMLAE(config)
    if len(results)<=0:
        logging.error("Not able to compute MLAE, no data found (test data or test model)")
        return
    logging.info("------- RESULTS -------") 

    minMlaeIndex=0
    minMlae=results[0]["loss"]["MLAE"]["avg"]

    minAbsIndex=0
    minAbs=results[0]["loss"]["ABS"]["avg"]

    for i,dic in enumerate(results):
        logging.info(dic["path"])
        loss=dic["loss"]
        mlae=loss["MLAE"]
        abs=loss["ABS"]
        if abs["avg"]<minAbs:
            minAbs=abs["avg"]
            minAbsIndex=i
        if mlae["avg"]<minMlae:
            minMlae=mlae["avg"]
            minMlaeIndex=i
        logging.info("MLAE %9.6f (%9.6f) | ABS %9.8f (%9.8f) "%(mlae["avg"],mlae["std"],abs["avg"],abs["std"]))
        logging.info("")

    logging.info("---- Min Value -----")
    logging.info("Min Abs")
    logging.info(results[minAbsIndex]["path"])
    mlae = results[minAbsIndex]["loss"]["MLAE"]
    abs = results[minAbsIndex]["loss"]["ABS"]
    logging.info("MLAE %9.6f (%9.6f) | ABS %9.8f (%9.8f) "%(mlae["avg"],mlae["std"],abs["avg"],abs["std"]))
    logging.info("")
    logging.info("")
    logging.info("Min MLAE")
    logging.info(results[minMlaeIndex]["path"])
    mlae = results[minMlaeIndex]["loss"]["MLAE"]
    abs = results[minMlaeIndex]["loss"]["ABS"]
    logging.info("MLAE %9.6f (%9.6f) | ABS %9.8f (%9.8f) "%(mlae["avg"],mlae["std"],abs["avg"],abs["std"]))


    logging.info("Complete, exit!")

if __name__ == '__main__':
    globalCatch(main)