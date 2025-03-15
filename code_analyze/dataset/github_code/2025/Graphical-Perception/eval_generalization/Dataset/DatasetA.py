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
import random

class DatasetA(Dataset):

    def __init__(self,basicFolder,config,paths = [],tag="Default"):

        self.config=config
        cfgd=config.data
        self.sz=(cfgd.width,cfgd.height)
        self.trans = transforms.Compose([transforms.ToTensor()])
        self.inputRoot = os.path.join(basicFolder,cfgd.inputFolder)
        self.outputRoot = os.path.join(basicFolder,cfgd.outputFolder)
        logging.info("Dataset Input Folder %s"%self.inputRoot)
        logging.info("Dataset Target Folder %s"%self.outputRoot)
        self.tag=tag
        self.pth=paths
        self.basicFolder=basicFolder
        logging.info("DatasetA Input Size %d, tag %s"%(len(self.pth),tag))

    def onLoad(self,content):
        self.trainFileInputPath=content
        
    def onEpochComplete(self):
        random.shuffle(self.pth)
        return self.pth

    def __len__(self):
        return len(self.pth)

    def __getitem__(self,index):
        inputs = self.pth[index][0]
        outputs = self.pth[index][1]
        
        return {"input":self.processDic(self.inputRoot,inputs),"target":self.processDic(self.outputRoot,outputs)}

    def processImg(self,path):
        inputImage = Image.open(path,"r")
        if inputImage.size != self.sz:
            inputImage = inputImage.resize(self.sz,Image.ANTIALIAS)
        return self.trans(inputImage)

    def processJson(self,path):
        return [torch.tensor(uio.load(path,"json"))] # for varied 

    def processDic(self,root,dic):
        result={}
        for k,v in dic.items():
            inputPath = os.path.join(root,v)
            fix = v.split(".")[-1]
            if fix in ["png","jpg","jpeg","bmp"]:
                result[k]=self.processImg(inputPath)
            elif fix in ["json"]:
                result[k]=self.processJson(inputPath) 
            else:
                logging.warning("Unresolved dataset file %s"%inputPath)
        return result