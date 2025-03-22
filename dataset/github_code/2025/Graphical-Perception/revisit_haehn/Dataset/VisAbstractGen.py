import logging
import os
from . import UtilIO as uio
import numpy as np
import cv2
import random
from util.Config import ConfigObj

class VisAbstractGen:

    def __init__(self, config, default=True):
        self.config = config
        self.param = config.data.generator.param
        if default:
            self.setDefault(self.param)
        ConfigObj.default(self.param,"preprocess.enable",False)
        ConfigObj.default(self.param,"imgEnhance.enable",False)
        a,b,c = self._getFilePath(True)
        os.makedirs(a,exist_ok=True)
        os.makedirs(b,exist_ok=True)
        os.makedirs(c,exist_ok=True)
        a,b,c = self._getFilePath(False)
        os.makedirs(a,exist_ok=True)
        os.makedirs(b,exist_ok=True)
        os.makedirs(c,exist_ok=True)

        self.__genFix=[]
        self.__resetFix=False
        pass
    
    def setDefault(self,param):
        ConfigObj.default(param,"color.colorDiffLeast",20)
        ConfigObj.default(param,"color.useFixSettingChance",0.0)
        ConfigObj.default(param,"color.fillRectChance",1.0)
        ConfigObj.default(param,"color.fixColorBar.colorIsFixed",True)
        ConfigObj.default(param,"color.fixColorBar.color",[0,0,0])
        ConfigObj.default(param,"color.fixStroke.color",[0,0,0])
        ConfigObj.default(param,"color.fixStroke.colorIsFixed",True)
        ConfigObj.default(param,"color.background.colorIsFixed",True)
        ConfigObj.default(param,"color.background.color",[255,255,255])
        ConfigObj.default(param,"labelValue",1)
        ConfigObj.default(param,"values.enableTotalConstrain",False)
        ConfigObj.default(param,"values.pixelValue",False)
        ConfigObj.default(param,"values.valueDiffLeast",0)
        ConfigObj.default(param,"values.outputAverage",False)
        ConfigObj.default(param,"mark.dotColor",[0,0,0])
        ConfigObj.default(param,"mark.markMax",False)
        ConfigObj.default(param,"mark.markRandom",0)
        ConfigObj.default(param,"mark.ratio.ratioMarkOnly",False)
        ConfigObj.default(param,"mark.ratio.ratio2Only",False)
        ConfigObj.default(param,"mark.ratio.ratioNotMarkOnly",False)
        ConfigObj.default(param,"mark.fix",[])
        ConfigObj.default(param,"mark.genFix",0)
        ConfigObj.default(param,"preprocess.enable",False)
        ConfigObj.default(param,"imgEnhance.rotateL90Chance",0.0)
        ConfigObj.default(param,"imgEnhance.rotateR90Chance",0.0)
        ConfigObj.default(param,"values.useSpecialGen",False)
        ConfigObj.default(param,"mark.markAlwaysSameGroup",False)
        ConfigObj.default(param,"mark.markAlwaysDiffGroup",False)
        return param

    def getMaxValue(self): # the maximum value of outputs
        return 1

    def _getFilePath(self,isTrainData=True):
        filePath=None
        if isTrainData:
            filePath=self.config.data.trainPath
        else:
            filePath=self.config.data.validPath
        
        inputFilePath = os.path.join(filePath,self.config.data.inputFolder)
        outputFilePath = os.path.join(filePath,self.config.data.outputFolder)
        orgFilePath = os.path.join(filePath,self.config.data.orgFolder)
        return inputFilePath,outputFilePath,orgFilePath

    def getLength(self,isTrainData=True):
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)
        length0 = 0
        length1 = 0
        for root,dirs,files in os.walk(inputFilePath):
            length0 = len(files)
        for root,dirs,files in os.walk(outputFilePath):
            length1 = len(files)
        return (length0,length1)

    def clear(self,isTrainData=True):
        logging.info("Clear Data...")
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)
        for root,dirs,files in os.walk(inputFilePath):
            for f in files:
                os.remove(os.path.join(root,f))
            logging.info("Clear Files %d"%(len(files)))
        for root,dirs,files in os.walk(outputFilePath):
            for f in files:
                os.remove(os.path.join(root,f))
            logging.info("Clear Files %d"%(len(files)))
        pass

    def genFileList(self,index,isTrainData=True):
        inputs={}
        inputs["img"]=self.param.fileName%index + ".png"
        if self.param.preprocess.enable:
            inputs["dis"] = self.param.fileName%index + "_dis.png"
        outputs={}
        outputs["num"] = self.param.fileName%index +".json"
        outputs["ratio"] = self.param.fileName%index +"_r.json"
        outputs["label"] = self.param.fileName%index +"_l.json"
        outputs["label_l"] = self.param.fileName%index +"_ll.json"
        return inputs, outputs

    def genValidData(self,index):
        self.gen(index,False)

    # generate values, valueQuant --> decide the minimum gap between values
    def _genValues(self,count,valueQuant=0):
        pv = self.param.values

        values = [0]*count

        while True:
            for i in range(count):
                while True:
                    v = uio.fetchValue(pv.valueRange)
                    values[i]=v
                    if pv.valueDiffLeast>0:
                        flag=False
                        for j in range(i):
                            if abs(v-values[j])<pv.valueDiffLeast:
                                flag=True
                                break
                        if flag:
                            continue
                    break

            if pv.enableTotalConstrain and sum(values)!=pv.totalConstrain:
                continue
            else:
                break
        if valueQuant>0:
            values = [max(1,int(v*valueQuant))/valueQuant for v in values]
        return values
    
    # return a list of index, indicates which is required to be marked
    def _mark(self,values):
        pm = self.param.mark
        markList=[]
        if len(pm.fix)>0:
            markList+=pm.fix
        if pm.genFix>len(markList):
            lv = len(values)-1
            while True:
                tempv = random.randint(0,lv)
                if tempv not in pm.fix:
                    markList.append(tempv)
                    if pm.genFix<=len(markList):
                        break
                
        if pm.markMax:
            ind = 0
            maxv = values[0]
            for i,v in enumerate(values):
                if maxv<v:
                    maxv=v
                    ind=i
            markList.append(ind)
        if pm.markRandom>0:
            if pm.markRandomNeighbor:
                v = random.randint(0,len(values)-pm.markRandom)
                for i in range(v,v+pm.markRandom):
                    if i not in markList:
                        markList.append(i)
            else:
                indexes = [i for i in range(len(values)) if i not in markList]
                markList+=random.shuffle(indexes)[0:pm.markRandom]
        markList = sorted(markList)
        return markList

    def _processValues(self, valuesInput, markList):
        pm = self.param.mark.ratio
        if int(pm.ratioMarkOnly)+int(pm.ratioNotMarkOnly)+int(pm.ratio2Only)>1:
            logging.warning("Dataset generate: ratioMarkOnly, ratioNotMarkOnly, ratio2Only, more than one flags are setted to true")
        if pm.ratioMarkOnly: # return marked ratio, compare to maximum
            values = [valuesInput[i] for i in markList]
            maxv = max(valuesInput)
            valuesInput= [v/maxv for v in values]
        elif pm.ratioNotMarkOnly: # mark one, return other values, compare to maximum
            values = [valuesInput[i] for i in range(len(valuesInput)) if i not in markList]
            maxv = max(valuesInput)
            valuesInput= [v/maxv for v in values]
        elif pm.ratio2Only: # mark twice, return smaller/bigger 
            maxv = max(valuesInput[0], valuesInput[1])
            minv = min(valuesInput[0], valuesInput[1])
            valuesInput= [minv,maxv]
        elif pm.ratio2MarkOnly:
            values=[]
            for i in markList:
                values.append(valuesInput[i])
            maxv = max(values)
            minv = min(values)
            valuesInput= [minv/maxv]
        if self.param.values.outputAverage:
            return [sum(valuesInput)/len(valuesInput)]
        return valuesInput

    def _genRatio(self, values):
        # cycle consistency, default
        if len(values)==1:
            return values
        ratio = []
        for i in range(len(values)-1):
            ratio.append(values[i]/values[i+1])
        ratio.append(values[-1]/values[0])
        return ratio
        
    def _genColor(self,count):
        colorLists=[]
        backColor=[]
        strokeColor=[]
        cset = self.param.color
        if cset.background.colorIsFixed:
            backColor = cset.background.color
        else:
            backColor = uio.rdcolor()
        if uio.rd(cset.useFixSettingChance): # all bars assigned with fix color
            cset2 = cset.fixColorBar
            cset3 = cset.fixStroke
            if cset2.colorIsFixed:
                colorLists=[cset2.color]*count
            else:
                colorLists=[uio.rdcolorDiff(backColor,cset.colorDiffLeast)]*count
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            else:
                strokeColor=[uio.rdcolorDiff(backColor,cset.colorDiffLeast)]
        else: # each bar assigned with diff color
            colorLists.append(backColor)
            for i in range(count+1):
                colorLists.append(uio.rdcolorDiff(colorLists,cset.colorDiffLeast))
            strokeColor=[colorLists[-1]]
            cset3 = cset.fixStroke
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            colorLists=colorLists[1:-1]
            

        fill = uio.rd(cset.fillRectChance)
        return colorLists,backColor,fill,strokeColor

    def _preprocess(self,inputFilePath,image):
        if self.param.imgEnhance.enable:
            if self.param.imgEnhance.rotateL90Chance > random.random():
                image = np.rot90(image, -1)
                pass
            elif self.param.imgEnhance.rotateL90Chance > random.random():
                image = np.rot90(image, 1)
                pass
        os.makedirs(os.path.split(inputFilePath+".png")[0],exist_ok=True)
        if self.param.preprocess.enable:
            cv2.imwrite(inputFilePath+".png",image)
            img_gray = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2GRAY)
            ret,img_thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
            img_distance = cv2.distanceTransform(img_thresh, cv2.DIST_L2, 5).astype(np.uint8)
            cv2.imwrite(inputFilePath+"_dis.png",img_distance)
            #cv2.imwrite(orgFilePath+".png",image)
        else:
            cv2.imwrite(inputFilePath+".png",image)

