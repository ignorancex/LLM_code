import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj

class BarGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config)
        ConfigObj.default(self.param,"midGap",0)
        ConfigObj.default(self.param,"fixBarGap",-1)
        ConfigObj.default(self.param,"values.useSpecialGen",False)
        ConfigObj.default(self.param,"values.specialGen.count",10)
        ConfigObj.default(self.param,"values.specialGen.divScale",12)
        ConfigObj.default(self.param,"mark.fixPos",False)
        ConfigObj.default(self.param,"mark.fixPosY",9)
        ConfigObj.default(self.param,"mark.isFlag",True)

    def mark(self,image,center,dotColor):
        y=None
        if self.param.mark.fixPos:
            y = int(self.param.mark.fixPosY)
        else:
            y=int(center[1])
        x=int(center[0])
        image[y:y+1,x:x+1]=(dotColor[0],dotColor[1],dotColor[2])

    def gen(self,index,isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth,1)
        height = uio.fetchValue(self.param.outputHeight,1)
        barCount = uio.fetchValue(self.param.barCount)
        barWidth = uio.fetchValue(self.param.barWidth,1)
        lineThickness = uio.fetchValue(self.param.lineThickness,-1)
        spacePaddingLeft = uio.fetchValue(self.param.spacePaddingLeft)
        spacePaddingRight = uio.fetchValue(self.param.spacePaddingRight)
        spacePaddingTop = uio.fetchValue(self.param.spacePaddingTop)
        spacePaddingBottom = uio.fetchValue(self.param.spacePaddingBottom)
        midGap = uio.fetchValue(self.param.midGap)

        colorLists,backColor,fill,strokeColor = self._genColor(barCount)

        '''
        <       ><-------><        ><--------><          >
        padding  barWidth   empty    barWidth   padding
        '''

        horSpace = width - spacePaddingLeft - spacePaddingRight - midGap
        
        verSpace = height - spacePaddingTop - spacePaddingBottom
        if verSpace<=20:
            logging.error("Wrong Parameters! Vertical Padding is too large! Set 20 instead.")
            verSpace=20

        leftHorEmptySpace = horSpace - barWidth*barCount
        if lineThickness>0:
            leftHorEmptySpace-= barCount*lineThickness*4
        # avoid overlapping
        if leftHorEmptySpace<0:
            leftHorEmptySpace=0
            barWidth = int(horSpace/barCount)
            if lineThickness>0:
                barWidth-=lineThickness*2
                if barWidth<=2:
                    barWidth+=lineThickness*2-2
                    lineThickness=1
                    leftHorEmptySpace+=barCount*(lineThickness-1)*4
            if barWidth<=2:
                lineThickness=1
                barWidth=2
                emptyPadding = width-barWidth*barCount
                spacePaddingLeft = int((np.random.rand()*emptyPadding))
                spacePaddingRight = int(emptyPadding-spacePaddingLeft)
                leftHorEmptySpace = width-emptyPadding
        horEmptySpace = 0
        if barCount>1: 
            horEmptySpace = leftHorEmptySpace // (barCount-1)

        if lineThickness>0:
            horEmptySpace+=lineThickness*2

        if self.param.fixBarGap>0:
            horEmptySpace = self.param.fixBarGap

        barHeights = []
        maxBarHeight = 0


        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:] = backColor
        startOffsetX = int(spacePaddingLeft)
        startOffsetY = int(height-spacePaddingBottom)

        quant=verSpace
        if lineThickness>0:
            quant = verSpace//lineThickness
        if self.param.values.pixelValue:
            quant = 0
        values = self._genValues(barCount,quant)
        
        resetFix=False

        if self.param.values.useSpecialGen:
            v1=v2=0
            count1 = self.param.values.specialGen.count-1
            divScale = float(self.param.values.specialGen.divScale)
            while v1==v2:
                v1 = int(10*10**(random.randint(0,count1)/divScale))
                v2 = int(10*10**(random.randint(0,count1)/divScale))
            
            if len(self.param.mark.fix)==2:
                #logging.error("Bar: in useSpecialGen mode, must indicate 2 places to replace the in mark.fix")
                values[self.param.mark.fix[0]]=v1
                values[self.param.mark.fix[1]]=v2
            else:
                resetFix=True
                self.param.mark.fix=[]
                lv = len(values)-1
                ind1 = random.randint(0,lv)
                ind2 = random.randint(0,lv)
                while ind2==ind1:
                    ind2 = random.randint(0,lv)
                self.param.mark.fix=[ind1,ind2]
                values[self.param.mark.fix[0]]=v1
                values[self.param.mark.fix[1]]=v2

        markList = self._mark(values)
        valueMax = max(values)
        for i in range(barCount):
            if self.param.values.pixelValue:
                barHeight = max(1,int(values[i]))
            else:
                barHeight = max(1,int(verSpace*values[i]/valueMax))
            barHeights.append(barHeight)
            maxBarHeight = max(maxBarHeight,barHeight)
            cv2.rectangle(image,
                (startOffsetX,startOffsetY),
                (startOffsetX+barWidth,startOffsetY-barHeight),
                colorLists[i],
                -1 if fill else lineThickness
                )
            if lineThickness>0:
                cv2.rectangle(image,
                    (startOffsetX,startOffsetY),
                    (startOffsetX+barWidth,startOffsetY-barHeight),
                    strokeColor[0],
                    lineThickness
                )
            if i in markList:
                if self.param.mark.isFlag:
                    self.mark(image,(startOffsetX+barWidth*0.5,startOffsetY-5),self.param.mark.dotColor)
            startOffsetX += barWidth + horEmptySpace
            
            if i==barCount//2-1:
                startOffsetX+=midGap

        # if preprocess is enabled, preprocess input data
        
        # save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath = os.path.join(inputFilePath,fileName)
        outputFilePath = os.path.join(outputFilePath,fileName)
        orgFilePath = os.path.join(orgFilePath,fileName)

        self._preprocess(inputFilePath,image)

        barHeights = self._processValues(barHeights,markList)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath,[barHeights[0]/barHeights[1]],"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue],"json")
        else:
            uio.save(outputFilePath,barHeights,"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(barHeights),"json")

        
        ratio = self._genRatio(barHeights)
        uio.save(outputFilePath+"_r",ratio,"json")

        labels = [self.param.labelValue]*len(ratio)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath+"_l",[labels[0]],"json")
        else:
            uio.save(outputFilePath+"_l",labels,"json")

        if resetFix:
            self.param.mark.fix=[]

        

        


        

