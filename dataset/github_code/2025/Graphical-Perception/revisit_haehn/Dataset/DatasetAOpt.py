import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import PIL
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import random
import logging
import sys
from . import UtilIO as uio
from . import DatasetA
from util.Config import ConfigObj
import random
import time

'''
    A better dataset manager
        padding labels
        add noise
        use cache to boost speed
'''
class DatasetAOpt(DatasetA.DatasetA):

    def __init__(self,basicFolder,config,paths = [],tag_="default"):
        super().__init__(basicFolder,config,paths,tag=tag_)
        logging.info("DatasetA Optimize!")
        p=config.data.manageParam

        ConfigObj.default(p,"enableNoise",False)
        ConfigObj.default(p,"enablePaddingInputLabel",False)
        ConfigObj.default(p,"paddingTo",-1)
        ConfigObj.default(p,"enableDiskCache",False)
        ConfigObj.default(p,"enableRAMcache",False)
        ConfigObj.default(p,"enablePadding",False)
        ConfigObj.default(p,"enableImagePadding",False)

        self.cacheFolder=p.cacheFolder
        self.enablePadding=p.enablePaddingInputLabel
        if config.model.name not in ["trans.TrainDetr", "trans.TrainDetrOpt"] and self.enablePadding:
            logging.warning("DatasetAOpt: Not detr, but still use padding")
        if self.enablePadding and p.paddingTo<0:
            if isinstance(config.model.param.detr.num_queries,int):
                self.paddingTo = config.model.param.detr.num_queries
                logging.warning("DatasetAOpt: paddingTo < 0, use detr num_queries instead (%d)",self.paddingTo)
            elif isinstance(config.model.param.num_classes,int):
                self.paddingTo = config.model.param.num_classes
                logging.warning("DatasetAOpt: paddingTo < 0, use vgg/resnet num_classes instead (%d)",self.paddingTo)
            else:
                logging.warning("DatasetAOpt: paddingTo <0, but detr/vgg/resnet not exists, use 1000 instead")
                self.paddingTo=1000
        else:
            self.paddingTo=p.paddingTo
        self.useDiskCache = p.enableDiskCache
        self.useRamCache = p.enableRAMcache
        if self.useRamCache:
            logging.info("Use RAM Cache")
            self.ramCacheList=[]
        elif self.useDiskCache:
            logging.info("Use Disk Cache")
        else:
            logging.info("Realtime load and process")
        self.config=config
        self.param = p
        os.makedirs(self.cacheFolder,exist_ok=True)

        self.loadSeq=[i for i in range(len(self.pth))]
        # preprocess dataset!
        self._genCache()

    def onLoad(self,content):
        self.loadSeq=content

    def onEpochComplete(self):
        random.shuffle(self.loadSeq)
        return self.loadSeq

    def _loadCache(self,i):
        if self.useRamCache:
            return self.ramCacheList[i].clone()
        elif self.useDiskCache:
            return uio.load(os.path.join(self.cacheFolder,"%s_%d.npy"%(self.tag,i)))
        else:
            inputs = self.pth[i][0]
            outputs = self.pth[i][1]
            values = {"input":self.processDic(self.inputRoot,inputs),"target":self.processDic(self.outputRoot,outputs)}
            return values

    def _saveCache(self,i,v):
        if self.useRamCache:
            self.ramCacheList.append(v)
        elif self.useDiskCache:
            uio.save(os.path.join(self.cacheFolder,"%s_%d.npy"%(self.tag,i)),v)
        else:
            pass

    #preprocess the dataset, and store to cache folder or RAM
    def _genCache(self):
        imageKey=[]
        for k,v in self.pth[0][0].items():
            fix = v.split(".")[-1]
            if fix in ["png","jpg","jpeg","bmp"]:
                imageKey.append(k)
        
        self._imageKey=imageKey

        begin =time.time()
        if self.useDiskCache or self.useRamCache:
            completeTagPath=os.path.join(self.cacheFolder,"complete_tag_%s.json"%self.tag)
            if not os.path.exists(completeTagPath) or self.useRamCache:
                for i in range(len(self.pth)):
                    uio.logProgress(i,len(self.pth),"Generate Dataset Cache")
                    inputs = self.pth[i][0]
                    outputs = self.pth[i][1]
                    values = {"input":self.processDic(self.inputRoot,inputs),"target":self.processDic(self.outputRoot,outputs)}
                    self._saveCache(i,values)
            uio.save(completeTagPath,[1],"json")
        delta = time.time()-begin
        logging.info("Time Cost: "+str(delta))

    def __getitem__(self,index):
        loadIndex = self.loadSeq[index]
        v = self._loadCache(loadIndex)
        inputs = v["input"]
        if self.param.enableNoise:
            for k in self._imageKey:
                inputs[k]+=torch.randn_like(inputs[k])*self.param.noise
        return v

    def processImg(self,path):
        inputImage = Image.open(path,"r")
        if inputImage.size != self.sz:
            if self.param.enableImagePadding:
                pad = Image.new('RGB',self.sz,(255,255,255))
                pad.paste(inputImage, (0, 0, inputImage.size[0], inputImage.size[1]))
                inputImage = pad
            else:
                inputImage = inputImage.resize(self.sz,Image.ANTIALIAS)
        return self.trans(inputImage)

    def processJson(self,path):
        v = uio.load(path,"json")
        return torch.tensor(v) # retain type

    def processDic(self,root,dic):
        result={}
        for k,v in dic.items():
            inputPath = os.path.join(root,v)
            fix = v.split(".")[-1]
            if fix in ["png","jpg","jpeg","bmp"]:
                result[k]=self.processImg(inputPath)
            elif fix in ["json"]:
                if self.enablePadding:
                    v=uio.load(inputPath,"json")
                    if len(v)<self.paddingTo:
                        count = self.paddingTo-len(v)
                        if count<0:
                            logging.error("DatasetAOpt cannot do padding for %s, exceed the padding length %d"%(inputPath,-count))
                            count=0
                        v+=[0]*count # append zeros
                    result[k]=torch.tensor(v)
                else:
                    result[k]=self.processJson(inputPath) 
            else:
                logging.warning("Unresolved dataset file %s"%inputPath)
        return result