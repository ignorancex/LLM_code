import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
import math
from math import radians, pi
from util.pyutils import multiImport
import copy
class MixGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config,False)
        gens = self.param.generators
        self.generators=[]
        self.invokeIndexTrain=[]
        self.invokeIndexValid=[]
        countTrain=0
        countValid=0
        for i,gen in enumerate(gens):
            trainNum = int(gen.ratio * self.config.data.generator.trainPairCount)
            testNum = int(gen.ratio * self.config.data.generator.testPairCount)
            validNum = int(gen.ratio * self.config.data.generator.validPairCount)
            className = gen.name.split(".")[-1]
            name="Dataset."+gen.name+"."+className
            classObj = multiImport(name)
            logging.info("Mix generator: %s"%name)
            nconfig = copy.deepcopy(self.config)
            nconfig.data.generator.trainPairCount = trainNum
            nconfig.data.generator.testPairCount = testNum
            nconfig.data.generator.validPairCount = validNum
            nconfig.data.generator.param.generators = []
            nconfig.data.generator.param = gen.param
            obj = classObj(nconfig)
            
            self.generators.append(obj)
            countTrain+=trainNum
            countValid+=validNum
            self.invokeIndexTrain.append(countTrain)
            self.invokeIndexValid.append(countValid)

    def _getGenerator(self,index,isTrainData=True):
        i=0
        if isTrainData:
            while self.invokeIndexTrain[i]<index: i+=1
        else:
            while self.invokeIndexValid[i]<index: i+=1
        return self.generators[i]

    def genFileList(self,index,isTrainData=True):
        return self._getGenerator(index,isTrainData).genFileList(index,isTrainData)

    def gen(self,index,isTrainData=True):
        return self._getGenerator(index,isTrainData).gen(index,isTrainData)


        

