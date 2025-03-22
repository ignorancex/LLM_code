import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj

class StackGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self,config):
        super().__init__(config)
        ConfigObj.default(self.param,"fixStackGap",0)
        ConfigObj.default(self.param,"values.useSpecialGen",False)
        ConfigObj.default(self.param,"values.specialGen.count",10)
        ConfigObj.default(self.param,"values.specialGen.divScale",12)
        ConfigObj.default(self.param,"mark.fix",[])
        ConfigObj.default(self.param,"values.maxGroupPixelHeight",0)

    def mark(self,image,center,dotColor):
        y=int(center[1])
        x=int(center[0])
        image[y:y+1,x:x+1]=(dotColor[0],dotColor[1],dotColor[2])

    def gen(self,index,isTrainData=True):
        width=uio.fetchValue(self.param.outputWidth,1)
        height=uio.fetchValue(self.param.outputWidth,1)
        stackCount = uio.fetchValue(self.param.stackCount) # each group has how many types?
        stackWidth = uio.fetchValue(self.param.stackWidth,1)
        stackGroup = uio.fetchValue(self.param.stackGroup) # how many groups?
        lineThickness = uio.fetchValue(self.param.lineThickness,-1)
        spacePaddingLeft = uio.fetchValue(self.param.spacePaddingLeft)
        spacePaddingRight = uio.fetchValue(self.param.spacePaddingRight)
        spacePaddingTop = uio.fetchValue(self.param.spacePaddingTop)
        spacePaddingBottom = uio.fetchValue(self.param.spacePaddingBottom)

        
        # process color for stackCount
        colorLists,backColor,fill,strokeColor = self._genColor(stackCount)

        '''
        <       ><-------><        ><--------><          >
        padding  barWidth   empty    barWidth   padding
        '''

        horSpace = width-spacePaddingLeft-spacePaddingRight
        verSpace = height-spacePaddingTop-spacePaddingBottom
        if verSpace<=20:
            logging.warning("Wrong Parameters! Vertical Padding is too large! Set 20 instead.")
            verSpace = 20

        leftHorEmptySpace = horSpace - stackWidth*stackGroup
        if lineThickness>0:
            leftHorEmptySpace-= stackGroup*lineThickness*2
        # avoid overlapping
        if leftHorEmptySpace<0:
            leftHorEmptySpace=0
            stackWidth = int(horSpace/stackGroup)
            if lineThickness>0:
                stackWidth-=lineThickness*2
                if stackWidth<=2:
                    stackWidth+=lineThickness*2-2
                    lineThickness=1
            if stackWidth<=2:
                lineThickness=1
                stackWidth=2
                emptyPadding = width-stackWidth*stackGroup
                spacePaddingLeft = int((np.random.rand()*emptyPadding))
                spacePaddingRight = int(emptyPadding-spacePaddingLeft)
                leftHorEmptySpace = width-emptyPadding

        horEmptySpace = 0
        if stackGroup>1:
            horEmptySpace=leftHorEmptySpace // (stackGroup-1)

        if lineThickness>0:
            horEmptySpace+=lineThickness*2

        if self.param.fixStackGap>0:
            horEmptySpace = self.param.fixStackGap

        stackHeights=[]
        maxStackBarHeight = 0



        image = np.ones(shape=(width,height,3),dtype=np.int8)*255
        image[:,:]=backColor
        startOffsetX=int(spacePaddingLeft)
        startOffsetY =int(height-spacePaddingBottom)

        quant=verSpace
        if lineThickness>0:
            quant = verSpace//lineThickness//2
        if self.param.values.pixelValue:
            quant = 0
        values = self._genValues(stackGroup*stackCount,quant)

        resetFix=False
        if self.param.values.useSpecialGen:
            count1 = self.param.values.specialGen.count-1
            divScale = float(self.param.values.specialGen.divScale)
            tryTimes=0
            while True:
                values = self._genValues(stackGroup*stackCount,quant)
                tryTimes+=1
                if tryTimes==233:
                    logging.warning("Stack Generator may suffer endless loop, please check your parameter settings (loop 233 times for special generator (position length task))")
                v1=v2=0
                if len(self.param.mark.fix)<2:
                    resetFix=True
                    self.param.mark.fix=[]
                    lv = len(values)-1
                    ind1 = random.randint(0,lv)
                    ind2 = random.randint(0,lv)
                    while ind2==ind1:
                        ind2 = random.randint(0,lv)
                    self.param.mark.fix=[ind1,ind2]
                while v1==v2:
                    v1 = int(10*10**(random.randint(0,count1)/divScale))
                    v2 = int(10*10**(random.randint(0,count1)/divScale))
                #if len(self.param.mark.fix)<2:
                #    logging.error("Stack: in useSpecialGen mode, must indicate 2 places to replace the in mark.fix")
                #elif len(values)%2==1:
                #    logging.error("Stack: require even values")
                values[self.param.mark.fix[0]]=v1
                values[self.param.mark.fix[1]]=v2
                fix1=self.param.mark.fix[0]
                fix2=self.param.mark.fix[1]
                vs=[v1,v2]
                halfLen=len(values)//2
                minValue=self.param.values.valueRange[0]
                totalMaxValue=self.param.values.valueRange[1]
                leftCount = halfLen
                leftQuant = totalMaxValue
                rightCount = halfLen
                rightQuant = totalMaxValue
                for i,fix in enumerate(self.param.mark.fix):
                    if fix in range(0,halfLen):
                        leftCount-=1
                        leftQuant-=vs[i]
                    else:
                        rightCount-=1
                        rightQuant-=vs[i]
                leftLimit=leftQuant//leftCount
                rightLimit=rightQuant//rightCount
                if leftLimit<minValue:
                    #logging.warning("Stack: left value is inadequate! %d -> %d"%(leftLimit, minValue))
                    if resetFix:
                        resetFix=False
                        self.param.mark.fix=[]
                    continue
                if rightLimit<minValue:
                    #logging.warning("Stack: right value is inadequate! %d -> %d"%(rightLimit, minValue))
                    if resetFix:
                        resetFix=False
                        self.param.mark.fix=[]
                    continue
                for i in range(0,halfLen):
                    values[i] = random.randint(minValue,leftLimit)
                for i in range(halfLen,len(values)):
                    values[i] = random.randint(minValue,rightLimit)
                sumLeft = sum(values[0:halfLen])
                sumRight = sum(values[halfLen:])
                if sumLeft>totalMaxValue or sumRight>totalMaxValue:
                    if resetFix:
                        resetFix=False
                        self.param.mark.fix=[]
                    continue
                values[fix1]=v1
                values[fix2]=v2
                break

        
        maxGroupValue = 0
        for i in range(stackGroup):
            sumv=0
            for j in range(stackCount):
                sumv+=values[i*stackCount+j]
            if sumv>maxGroupValue:
                maxGroupValue=sumv

        if not self.param.values.pixelValue:
            values = [v/maxGroupValue for v in values]
        else:
            if self.param.values.maxGroupPixelHeight>0 and sumv>self.param.values.maxGroupPixelHeight:
                values = [int(v/maxGroupValue*self.param.values.maxGroupPixelHeight) for v in values]

        markList=[]
        if self.param.mark.markAlwaysSameGroup:
            groupID = random.randint(0,stackGroup-1)
            startIndex = groupID*stackCount
            endIndex = startIndex+stackCount
            markList = self._mark(values[startIndex:endIndex])
            markList = [ i+startIndex for i in markList]
        elif self.param.mark.markAlwaysDiffGroup:
            indexList=[]
            for i in range(stackGroup):
                indPart = random.randint(0,stackCount-1)
                indexList.append(indPart+i*stackCount)
            filterValues = [values[i] for i in indexList]
            markList = self._mark(filterValues)
            markList = [indexList[i] for i in markList]
        else:
            markList = self._mark(values)


        for i in range(stackGroup):
            tmpOffsetY=startOffsetY
            stackBarHeight=0
            for j in range(stackCount):
                curIndex=i*stackCount+j
                stackHeight=0
                if self.param.values.pixelValue:
                    stackHeight = max(1,int(values[curIndex]))
                else:
                    stackHeight = max(1,int(values[curIndex] * verSpace))
                stackHeights.append(stackHeight)
                useColor=colorLists[j]
                useColor2 = strokeColor[0]
                if lineThickness>0:
                    useColor = colorLists[0]
                cv2.rectangle(image,
                    (startOffsetX, tmpOffsetY),
                    (startOffsetX + stackWidth, tmpOffsetY - stackHeight),
                    useColor,
                    -1 if fill else lineThickness
                )
                if lineThickness>0:
                    cv2.rectangle(image,
                        (startOffsetX, tmpOffsetY),
                        (startOffsetX + stackWidth, tmpOffsetY - stackHeight),
                        useColor2,
                        lineThickness
                    )
                if curIndex in markList:
                    self.mark(image,(startOffsetX+stackWidth*0.5,tmpOffsetY-stackHeight*0.5),self.param.mark.dotColor)
                tmpOffsetY -= stackHeight
                stackBarHeight+=stackHeight

            maxStackBarHeight=max(maxStackBarHeight,stackBarHeight)
            startOffsetX+=stackWidth+horEmptySpace

        # if preprocess is enabled, preprocess input data

        #save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath=os.path.join(inputFilePath,fileName)
        outputFilePath=os.path.join(outputFilePath,fileName)
        orgFilePath =os.path.join(orgFilePath,fileName)

        self._preprocess(inputFilePath,image)

        stackHeights = self._processValues(stackHeights,markList)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath,[stackHeights[0]/stackHeights[1]],"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue],"json")
        else:
            uio.save(outputFilePath,stackHeights,"json")
            uio.save(outputFilePath+"_ll",[self.param.labelValue]*len(stackHeights),"json")

        ratio = self._genRatio(stackHeights)
        uio.save(outputFilePath + "_r", ratio, "json")

        labels = [self.param.labelValue]*len(ratio)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath+"_l",[labels[0]],"json")
        else:
            uio.save(outputFilePath+"_l",labels,"json")

        if resetFix:
            self.param.mark.fix=[]