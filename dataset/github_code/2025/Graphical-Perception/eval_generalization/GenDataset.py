import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import argparse
from Networks import *
import logging
import Dataset
import Dataset.UtilIO as uio
import sys
import util.parallelUtil as para
import random
import util
import util.pyutils as upy
from util.pyutils import multiImport
from util.Config import ConfigObj
import util.shareCode
from util.shareCode import programInit, globalCatch

class GenDataset:

    def __init__(self, config):
        self.config = config

        ConfigObj.default(self.config,"utility.debug",False)
        ConfigObj.default(self.config,"utility.globalSeed",0)
        ConfigObj.default(self.config,"data.trainSeed",100)
        ConfigObj.default(self.config,"data.validSeed",10000)
        ConfigObj.default(self.config,"data.inputFolder","input")
        ConfigObj.default(self.config,"data.outputFolder","target")
        ConfigObj.default(self.config,"data.orgFolder","org")
        ConfigObj.default(self.config,"data.useThreads",1)

        modelName = "Dataset.%s.%s"%(config.data.generator.genType,config.data.generator.genType)
        logging.info("Create dataset %s"%modelName)
        self.dataGenerator = multiImport(modelName)(config)
        pass

    def setSeed(self,seed):
        torch.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def genData(self):
        dataConfig=self.config.data
        
        os.makedirs(os.path.join(self.config.data.trainPath,self.config.data.inputFolder),exist_ok=True)
        os.makedirs(os.path.join(self.config.data.trainPath,self.config.data.outputFolder),exist_ok=True)
        os.makedirs(os.path.join(self.config.data.trainPath,self.config.data.orgFolder),exist_ok=True)
        os.makedirs(os.path.join(self.config.data.validPath,self.config.data.inputFolder),exist_ok=True)
        os.makedirs(os.path.join(self.config.data.validPath,self.config.data.outputFolder),exist_ok=True)
        os.makedirs(os.path.join(self.config.data.validPath,self.config.data.orgFolder),exist_ok=True)
        logging.info("Train path %s"%self.config.data.trainPath)
        logging.info("Valid path %s"%self.config.data.validPath)
        paths = []
        validPaths = []
        procs = []
        logging.info("Data Path %s"%dataConfig.trainPath)
        # if not os.path.exists(os.path.join(dataConfig.trainPath,"complete_tag.json")):
        self.dataGenerator.clear()
        logging.info("regen train data %d"%dataConfig.generator.trainPairCount)
        self.setSeed(dataConfig.trainSeed)
        if dataConfig.useThreads>1:
            procs = para.runParallelFor(self.dataGenerator.gen,0,dataConfig.generator.trainPairCount,dataConfig.useThreads,dataConfig.trainSeed)
        for i in range(dataConfig.generator.trainPairCount):
            self.setSeed(dataConfig.trainSeed + int(10e7) + i)
            uio.logProgress(i,dataConfig.generator.trainPairCount,"Gen Train Data List")
            if dataConfig.useThreads==1:
                self.dataGenerator.gen(i)
            v=self.dataGenerator.genFileList(i)
            paths.append(v)

        pathTest=[]
        indexes = [i for i in range(self.config.data.generator.trainPairCount)]
        random.shuffle(indexes)
        for i in range(self.config.data.generator.testPairCount):
            pathTest.append(paths[indexes[i]])
        uio.save(os.path.join(self.config.data.trainPath,"list_test"),pathTest,"json")
        uio.save(os.path.join(dataConfig.trainPath,"list"),paths,"json")
        # if not os.path.exists(os.path.join(self.config.data.validPath,"complete_tag.json")):
        self.dataGenerator.clear(False)
        logging.info("regen valid data %d"%self.config.data.generator.validPairCount)
        self.setSeed(dataConfig.validSeed)
        if self.config.data.useThreads>1:
            procs += para.runParallelFor(self.dataGenerator.genValidData,0,self.config.data.generator.validPairCount,self.config.data.useThreads,self.config.data.validSeed)
        for i in range(self.config.data.generator.validPairCount):
            self.setSeed(dataConfig.validSeed + i)
            uio.logProgress(i,self.config.data.generator.validPairCount,"Gen Valid Data List")
            if self.config.data.useThreads==1:
                self.dataGenerator.gen(i,False)
            validPaths.append(self.dataGenerator.genFileList(i,False))
        uio.save(os.path.join(self.config.data.validPath,"list"),validPaths,"json")
        para.waitProcs(procs)
        uio.save(os.path.join(dataConfig.trainPath,"max_v"),[self.dataGenerator.getMaxValue()],"json")
        uio.save(os.path.join(dataConfig.validPath,"max_v"),[self.dataGenerator.getMaxValue()],"json")
        uio.save(os.path.join(dataConfig.trainPath,"complete_tag"),[1],"json")
        uio.save(os.path.join(dataConfig.validPath,"complete_tag"),[1],"json")



def main():
    #parse params
    config = programInit()
    
    gener=GenDataset(config)
    gener.genData()
    logging.info("complete")

if __name__ == '__main__':
    globalCatch(main)

#--config_file TASK_2/genData.json --datasetName posLen_tp_1