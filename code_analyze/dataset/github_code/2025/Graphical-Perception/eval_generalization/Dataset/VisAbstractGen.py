import logging
import os
from . import UtilIO as uio
import numpy as np
import cv2
import random
import math
from util.Config import ConfigObj
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from random_words import RandomWords
import sys
sys.path.append("..")
from util.color_pool import *

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
        ConfigObj.default(param,"color.colorDiffLeast",2)
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
        ConfigObj.default(param,"mark.ratio.MarkOnly",False)
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
                if not self.param.outdata:
                    while True:
                        v = uio.fetchValue(pv.valueRange)
                        values[i]=v
                        # print(v,end='')
                        if pv.valueDiffLeast>0:
                            flag=False
                            for j in range(i):
                                if abs(v-values[j])<pv.valueDiffLeast:
                                    flag=True
                                    break
                            if flag:
                                continue
                        break
                else:
                    v = uio.fetchValue(pv.valueRange)
                    values[i]=v

            if pv.enableTotalConstrain and sum(values)!=pv.totalConstrain:
                continue
            else:
                break
        if valueQuant>0:
            values = [max(1,int(v*valueQuant))/valueQuant for v in values]
        return values

    def normal_distribution(self,x,sig):
        y = np.exp(-(x) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
        return y
    def normal_distribution_list(self,xlist,sig):
        ylist=[]
        for x in xlist:
            y = np.exp(-(x) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
            ylist.append(y)
        return ylist
    def sort_list(self,ylist):
        a=sorted(ylist[0:len(ylist)//2])
        b=sorted(ylist[len(ylist)//2:])[::-1]
        return a+b
    
    def _genDistributionValues(self,count,valueQuant=0):
        typeOfData=self.param.DataDistribution
        randsig2=random.random()
        sig=math.sqrt(randsig2)
        xlist=[]
        xlist_neg=[]
        ylist=[]
        ylist_neg=[]
        maxvalue=self.normal_distribution(0,sig)
        while True:
            randx=random.uniform(0,3*sig)
            randy=self.normal_distribution(randx,sig)
            yvalue=int(randy/maxvalue*83+10)
            flag=True
            for y_ in ylist:
                if abs(yvalue-y_)<2:
                    flag=False
            if flag:
                ylist.append(yvalue)
            if len(ylist)==5:
                break

        while True:
            randx=random.uniform(-3*sig,0)
            randy=self.normal_distribution(randx,sig)
            yvalue=int(randy/maxvalue*83+10)
            flag=True
            for y_ in ylist:
                if abs(yvalue-y_)<2:
                    flag=False
            if flag:
                ylist.append(yvalue)
            if len(ylist)==10:
                break

        if typeOfData==1:
            values=self.sort_list(ylist)
        elif typeOfData==2:
            values=self.sort_list(ylist)
            values=[93+10-i for i in values]
        return values
    #     if typeOfData==1:
    #         rangeOfHeight=[[10,18],[18,26],[26,34],[34,42],[42,50],[50,58],[58,66],[66,74],[74,82],[82,93]]
    #         bias=[0,10]
    #     elif typeOfData==2:
    #         rangeOfHeight=[[10,18],[18,26],[26,34],[34,42],[42,50],[50,58],[58,66],[66,74],[74,82],[82,93]]
    #         rangeOfHeight=rangeOfHeight[::-1]
    #         bias=[0,10]
    #     elif typeOfData==3:
    #         rangeOfHeight=[[10,18],[18,26],[26,34],[34,42],[42,50],[50,58],[42,50],[34,42],[26,34],[18,26]]
    #         bias=[0,40]
    #     elif typeOfData==4:
    #         rangeOfHeight=[[85,93],[77,85],[69,77],[61,69],[53,61],[45,53],[53,61],[61,69],[69,77],[77,85]]
    # #         rangeOfHeight=[[100-i[0],100-i[1]] for i in rangeOfHeight]
    #         bias=[-40,0]
    #     values = [0]*count
    #     b=np.random.randint(bias[0],bias[1])
    #     for i in range(count):
    #         v = np.random.randint(rangeOfHeight[i][0],rangeOfHeight[i][1])
    #         values[i]=v+b
    #     return values

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
                # if tempv not in pm.fix:
                if tempv not in markList:
                    markList.append(tempv)

                    # in type1, the marked bar should be adjacent
                    if 'type1' in self.param.fileName:
                        mid_left_bar = self.param.barCount // 2 - 1
                        if tempv == 0 or tempv == mid_left_bar+1:
                            markList.append(tempv + 1)
                        elif tempv == lv or tempv == mid_left_bar:
                            markList.append(tempv - 1)
                        else:
                            diff = np.random.choice([-1,1])
                            markList.append(tempv + diff)
                    
                    if pm.markAdjancy:
                        if tempv!=4:
                            markList.append(tempv+1)
                        else:
                            markList.append(0)
                            
                    if pm.markStackedAdjancy:
                        if tempv==4:
                            markList.append(tempv-1)
                        else:
                            markList.append(tempv+1)

                    if 'type7' in self.param.fileName:
                        mid_left_bar = self.param.barCount // 2 - 1
                        if tempv < mid_left_bar+1:
                            tempv_right=random.randint(5,lv)
                            markList.append(tempv_right)
                        elif tempv > mid_left_bar:
                            tempv_left=random.randint(0,4)
                            markList.append(tempv_left)
                        else:
                            diff = np.random.choice([-1,1])
                            markList.append(tempv + diff)

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
                random.shuffle(indexes)
                markList=indexes[0:pm.markRandom]
                # markList+=random.shuffle(indexes)[0:pm.markRandom]
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
        elif pm.MarkOnly: # return marked ratio, compare to maximum
            values = [valuesInput[i] for i in markList]
            maxv = max(valuesInput)
            valuesInput= [v for v in values]
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

    def _getMaxValues(self, valuesInput, markList):

        values=[]
        for i in markList:
            values.append(valuesInput[i])
        # maxv = max(values)
        # minv = min(values)

        return values

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

    def _genColor_element_angle(self):
        rand=random.randint(0,len(bg_color)-1)
        # c = line_color_test[rand]
        c = bg_color[rand]
        backColor = c
        rand=random.randint(0,len(line_color)-1)
        # c = line_color_test[rand]
        c = line_color[rand]
        strokeColor = c
        # strokeColor=uio.bg_bar_colorDiff(backColor)
        dotColor=uio.linecolorDiff_bg_bar(backColor,strokeColor)
        return backColor,strokeColor,dotColor

    def _genTestColor_element_angle(self):
        bgcolor_index=random.randint(0,len(bg_color)-1)
        if self.param.bgcolor=='bright':
            Diff_bg_color_test_bright=getTestDiffbgcolor_bright()
            backColor = Diff_bg_color_test_bright[bgcolor_index]
            pass
        elif self.param.bgcolor=='dark':
            Diff_bg_color_test_dark=getTestDiffbgcolor()
            backColor = Diff_bg_color_test_dark[bgcolor_index]
        elif self.param.bgcolor=='white':
            Diff_bg_color_test_bright_15=getTestDiffbgcolor_bright_15()
            backColor = Diff_bg_color_test_bright_15[bgcolor_index]
        else:
            # backColor = bg_color[index%len(bg_color)]
            backColor = bg_color[bgcolor_index]
        # if self.param.bgcolor==
        # rand=random.randint(0,len(bg_color)-1)
        # # c = line_color_test[rand]
        # c = bg_color[rand]
        # backColor = c
        rand=random.randint(0,len(line_color)-1)
        c = line_color[rand]
        if self.param.linecolor=='color_pool':
            # strokeColor=[uio.linecolorDiff_bg_bar(backColor,barcolor)]
            strokeColor = c
        else:
            # strokeColor=[uio.linecolorDiff_bg_bar_bright(backColor,barcolor,self.param.linecolor,cset.colorDiffLeast)]
            # strokeColor=[uio.linecolorDiff_bg_bar_bright(bg_color[bgcolor_index],barcolor,self.param.linecolor,cset.colorDiffLeast)]
            strokeColor=uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],bg_color[bgcolor_index],self.param.linecolor)

        # strokeColor=uio.bg_bar_colorDiff(backColor)
        dotColor=uio.linecolorDiff_bg_bar(backColor,c)
        return backColor,strokeColor,dotColor

    def _genColor_element_area(self):
        rand=random.randint(0,len(bg_color)-1)
        # c = line_color_test[rand]
        c = bg_color[rand]
        backColor = c
        barColor=uio.bg_bar_colorDiff(backColor)
        lineColor=uio.linecolorDiff_bg_bar(backColor,barColor)
        fill = uio.rd(self.param.fillRectChance)
        return backColor,barColor,lineColor,fill

    def _genTestColor_element_area(self):
        bgcolor_index=random.randint(0,len(bg_color)-1)
        if self.param.bgcolor=='bright':
            Diff_bg_color_test_bright=getTestDiffbgcolor_bright()
            backColor = Diff_bg_color_test_bright[bgcolor_index]
            pass
        elif self.param.bgcolor=='dark':
            Diff_bg_color_test_dark=getTestDiffbgcolor()
            backColor = Diff_bg_color_test_dark[bgcolor_index]
        elif self.param.bgcolor=='white':
            Diff_bg_color_test_bright_15=getTestDiffbgcolor_bright_15()
            backColor = Diff_bg_color_test_bright_15[bgcolor_index]
        else:
            # backColor = bg_color[index%len(bg_color)]
            backColor = bg_color[bgcolor_index]
        barColor=uio.bg_bar_colorDiff(bg_color[bgcolor_index])
        a=barColor
        # barcolor=uio.bg_bar_colorDiff(bg_color[index%len(bg_color)],cset.colorDiffLeast)
        if self.param.barcolordark=='dark':
            color=[]
            for j in barColor:
                if j-50<0:
                    color.append(0)
                else:
                    color.append(j-50)
            barColor=tuple(color)
        # lineColor=uio.linecolorDiff_bg_bar(backColor,barColor)
        if self.param.linecolor=='color_pool':
            lineColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],a)]
        else:
            lineColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],a,self.param.linecolor)]
        fill = uio.rd(self.param.fillRectChance)
        return backColor,barColor,lineColor[0],fill

    def _genTrainColor(self,count,index):
        colorLists=[]
        backColor=[]
        strokeColor=[]
        cset = self.param.color
        if cset.background.colorIsFixed:
            backColor = cset.background.color
        else:
            # backColor = uio.rdcolor()
            rand=random.randint(0,len(bg_color)-1)
            # c = line_color_test[rand]
            c = bg_color[rand]
            backColor = c
            # backColor = bg_color[index%len(bg_color)]
        if uio.rd(cset.useFixSettingChance): # all bars assigned with fix color
            cset2 = cset.fixColorBar
            cset3 = cset.fixStroke
            if cset2.colorIsFixed:
                colorLists=[cset2.color]*count
            else:
                # colorLists=[uio.rdcolorDiff(backColor,cset.colorDiffLeast)]*count
                barcolor=uio.bg_bar_colorDiff(backColor,cset.colorDiffLeast)
                colorLists=[barcolor]*count
                pass
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            else:
                strokeColor=[uio.linecolorDiff_bg_bar(backColor,barcolor)]
        else: # each bar assigned with diff color
            colorLists.append(backColor)
            #each bar has different color
            for i in range(count+1):
                colorLists.append(uio.bg_bar_colorDiff(colorLists,cset.colorDiffLeast))
            # strokeColor=[colorLists[-1]]
            strokeColor=[uio.linecolorDiff_bg_bar(colorLists,backColor)]
            cset3 = cset.fixStroke
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            colorLists=colorLists[1:-1]
            

        fill = uio.rd(cset.fillRectChance)
        return colorLists,backColor,fill,strokeColor

    def _genTestColor(self,count,index):
        # bg_color_test=getTestbgcolor()
        Diff_bg_color_test_dark=getTestDiffbgcolor()
        Diff_bg_color_test_bright=getTestDiffbgcolor_bright()
        Diff_bg_color_test_bright_15=getTestDiffbgcolor_bright_15()
        # bar_color_test=getTestbarcolor()

        colorLists=[]
        backColor=[]
        strokeColor=[]
        cset = self.param.color

        bgcolor_index=random.randint(0,len(bg_color)-1)
        if cset.background.colorIsFixed:
            backColor = tuple(cset.background.color)
        else:
            # backColor = uio.rdcolor()
            if self.param.bgcolor=='bright':
                # backColor = bg_color_test[index%len(bg_color_test)]
                # backColor = Diff_bg_color_test_bright[index%len(Diff_bg_color_test_bright)]
                backColor = Diff_bg_color_test_bright[bgcolor_index]
                pass
            elif self.param.bgcolor=='dark':
                # backColor = Diff_bg_color_test_dark[index%len(Diff_bg_color_test_dark)]
                backColor = Diff_bg_color_test_dark[bgcolor_index]
            elif self.param.bgcolor=='white':
                backColor = Diff_bg_color_test_bright_15[bgcolor_index]
            # elif self.param.bgcolor=='pertubation':
            #     backColor = getTestbgcolor_perturbation(self.param.bgcolor_pertubation)[bgcolor_index]
            #     print(backColor)
            else:
                # backColor = bg_color[index%len(bg_color)]
                backColor = bg_color[bgcolor_index]
        # if uio.rd(cset.useFixSettingChance): # all bars assigned with fix color
        # print(self.param.barcolor)
        if self.param.barcolor=='same':
            cset2 = cset.fixColorBar
            cset3 = cset.fixStroke
            if cset2.colorIsFixed:
                barcolor=tuple(cset2.color)
                colorLists=[cset2.color]*count
            else:
                # colorLists=[uio.rdcolorDiff(backColor,cset.colorDiffLeast)]*count
                barcolor=uio.bg_bar_colorDiff(bg_color[bgcolor_index],cset.colorDiffLeast)
                # barcolor=uio.bg_bar_colorDiff(backColor,cset.colorDiffLeast,barcolor_pertubation=self.param.barcolor_pertubation)
                # barcolor=uio.bg_bar_colorDiff(bg_color[index%len(bg_color)],cset.colorDiffLeast)
                if self.param.barcolordark=='dark':
                    # color=[]
                    # for j in barcolor:
                    #     if j-50<0:
                    #         color.append(0)
                    #     else:
                    #         color.append(j-50)
                    # barcolor=tuple(color)
                    barcolor=Diff_bg_color_test_dark[bgcolor_index]
                colorLists=[barcolor]*count
                # print(colorLists)
                a=barcolor
                pass
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            else:
                if self.param.linecolor=='color_pool':
                    strokeColor=[uio.linecolorDiff_bg_bar(backColor,barcolor)]
                    # strokeColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],barcolor,strokecolor_pertubation=self.param.strokecolor_pertubation)]
                # elif self.param.linecolor=='bright':
                else:
                    # strokeColor=[uio.linecolorDiff_bg_bar_bright(backColor,barcolor,self.param.linecolor,cset.colorDiffLeast)]
                    # strokeColor=[uio.linecolorDiff_bg_bar_bright(bg_color[bgcolor_index],barcolor,self.param.linecolor,cset.colorDiffLeast)]
                    strokeColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],a,self.param.linecolor)]
                    # strokeColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],barcolor)]
                    # strokeColor=[(255,255,255)]
                    # print(strokeColor)
                #     pass
                # elif self.param.linecolor=='dark':
                #     pass
        elif self.param.barcolor=='Twogroup':
            colorLists.append(backColor)
            #each bar has different color
            for i in range(3):
                colorLists.append(uio.bg_bar_colorDiff(colorLists))
            # strokeColor=[colorLists[-1]]
            colorLists=colorLists[1:-1]
            # print(len(colorLists))
            colorLists=[colorLists[0]]*5+[colorLists[1]]*5
            if self.param.linecolor=='color_pool':
                    strokeColor=[uio.linecolorDiff_bg_bar(colorLists,backColor)]
                # elif self.param.linecolor=='bright':
            else:
                strokeColor=[uio.linecolorDiff_bg_bar_bright(colorLists,backColor,self.param.linecolor,cset.colorDiffLeast)]
            cset3 = cset.fixStroke
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            pass
        elif self.param.barcolor=='Fivegroup':
            colorLists.append(backColor)
            #each bar has different color
            for i in range(6):
                colorLists.append(uio.bg_bar_colorDiff(colorLists))
            # strokeColor=[colorLists[-1]]
            colorLists=colorLists[1:-1]
            # print(len(colorLists))
            colorLists=colorLists*2
            if self.param.linecolor=='color_pool':
                strokeColor=[uio.linecolorDiff_bg_bar(colorLists,backColor)]
                # elif self.param.linecolor=='bright':
            else:
                strokeColor=[uio.linecolorDiff_bg_bar_bright(colorLists,backColor,self.param.linecolor,cset.colorDiffLeast)]
            cset3 = cset.fixStroke
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            pass
        else: # each bar assigned with diff color
            colorLists.append(backColor)
            #each bar has different color
            for i in range(count+1):
                colorLists.append(uio.bg_bar_colorDiff(colorLists))
            # strokeColor=[colorLists[-1]]
            colorLists=colorLists[1:-1]
            if self.param.linecolor=='color_pool':
                    strokeColor=[uio.linecolorDiff_bg_bar(colorLists,backColor)]
                # elif self.param.linecolor=='bright':
            else:
                strokeColor=[uio.linecolorDiff_bg_bar_bright(colorLists,backColor,self.param.linecolor,cset.colorDiffLeast)]
            cset3 = cset.fixStroke
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            # colorLists=colorLists[1:-1]
            
        # print(colorLists)
        fill = uio.rd(cset.fillRectChance)
        return colorLists,backColor,fill,strokeColor

    def _genTestColor_pie(self,count,index):
        # bg_color_test=getTestbgcolor()
        Diff_bg_color_test_dark=getTestDiffbgcolor()
        Diff_bg_color_test_bright=getTestDiffbgcolor_bright()
        Diff_bg_color_test_bright_15=getTestDiffbgcolor_bright_15()
        # bar_color_test=getTestbarcolor()

        colorLists=[]
        backColor=[]
        strokeColor=[]
        cset = self.param.color

        bgcolor_index=random.randint(0,len(bg_color)-1)
        if cset.background.colorIsFixed:
            backColor = tuple(cset.background.color)
        else:
            # backColor = uio.rdcolor()
            if self.param.bgcolor=='bright':
                # backColor = bg_color_test[index%len(bg_color_test)]
                # backColor = Diff_bg_color_test_bright[index%len(Diff_bg_color_test_bright)]
                backColor = Diff_bg_color_test_bright[bgcolor_index]
                pass
            elif self.param.bgcolor=='dark':
                # backColor = Diff_bg_color_test_dark[index%len(Diff_bg_color_test_dark)]
                backColor = Diff_bg_color_test_dark[bgcolor_index]
            elif self.param.bgcolor=='white':
                backColor = Diff_bg_color_test_bright_15[bgcolor_index]
            # elif self.param.bgcolor=='pertubation':
            #     backColor = getTestbgcolor_perturbation(self.param.bgcolor_pertubation)[bgcolor_index]
            else:
                # backColor = bg_color[index%len(bg_color)]
                backColor = bg_color[bgcolor_index]
        if uio.rd(cset.useFixSettingChance): # all bars assigned with fix color
        # print(self.param.barcolor)
        # if self.param.barcolor=='same':
            cset2 = cset.fixColorBar
            cset3 = cset.fixStroke
            if cset2.colorIsFixed:
                colorLists=[cset2.color]*count
            else:
                # colorLists=[uio.rdcolorDiff(backColor,cset.colorDiffLeast)]*count
                barcolor=uio.bg_bar_colorDiff(bg_color[bgcolor_index],cset.colorDiffLeast)
                # barcolor=uio.bg_bar_colorDiff(bg_color[index%len(bg_color)],cset.colorDiffLeast)
                if self.param.barcolor=='dark':
                    color=[]
                    for j in barcolor:
                        if j-50<0:
                            color.append(0)
                        else:
                            color.append(j-50)
                    barcolor=tuple(color)
                colorLists=[barcolor]*count
                # print(colorLists)
                a=barcolor
                pass
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            else:
                if self.param.linecolor=='color_pool':
                    strokeColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],barcolor)]
                else:
                    strokeColor=[uio.linecolorDiff_bg_bar(bg_color[bgcolor_index],a,self.param.linecolor)]
        else: # each bar assigned with diff color
            colorLists.append(bg_color[bgcolor_index])
            #each bar has different color
            if self.param.barcolor=='dark':
                for i in range(count+1):
                    colorLists.append(uio.bg_bar_colorDiff_bardark(colorLists))
                pass
            else:
                for i in range(count+1):
                    colorLists.append(uio.bg_bar_colorDiff(colorLists,barcolor_pertubation=0))
                    # barcolor=uio.bg_bar_colorDiff(backColor,cset.colorDiffLeast,self.param.barcolor_pertubation)
            # strokeColor=[colorLists[-1]]
            colorLists=colorLists[1:-1]
            if self.param.linecolor=='color_pool':
                strokeColor=[uio.linecolorDiff_bg_bar(colorLists,backColor,strokecolor_pertubation=0)]
                # elif self.param.linecolor=='bright':
            else:
                strokeColor=[uio.linecolorDiff_bg_bar_bright(colorLists,backColor,self.param.linecolor,cset.colorDiffLeast)]
            cset3 = cset.fixStroke
            if cset3.colorIsFixed:
                strokeColor=[cset3.color]
            # colorLists=colorLists[1:-1]
            
        # print(colorLists)
        fill = uio.rd(cset.fillRectChance)
        return colorLists,backColor,fill,strokeColor


    def _preprocess(self,inputFilePath,image):
        if self.param.imgEnhance.enable:
            if self.param.imgEnhance.rotateL90Chance > random.random():
                image = np.rot90(image, -1).copy() 
                pass
            elif self.param.imgEnhance.rotateL90Chance > random.random():
                image = np.rot90(image, 1).copy() 
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
            # cv2.imwrite(inputFilePath+".png",image)
            image.save(inputFilePath+".png")

    def _preprocess_numpy(self,inputFilePath,image):
        if self.param.imgEnhance.enable:
            if self.param.imgEnhance.rotateL90Chance > random.random():
                image = np.rot90(image, -1).copy() 
                pass
            elif self.param.imgEnhance.rotateL90Chance > random.random():
                image = np.rot90(image, 1).copy() 
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
            # image.save(inputFilePath+".png")
            

