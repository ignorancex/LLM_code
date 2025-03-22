import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj
from PIL import Image, ImageFont, ImageDraw

class StackGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self,config):
        super().__init__(config)
        ConfigObj.default(self.param,"fixStackGap",0)
        ConfigObj.default(self.param,"values.useSpecialGen",False)
        ConfigObj.default(self.param,"values.specialGen.count",10)
        ConfigObj.default(self.param,"values.specialGen.divScale",12)
        ConfigObj.default(self.param,"mark.fix",[])
        ConfigObj.default(self.param,"mark.markAdjancy", False)
        ConfigObj.default(self.param,"mark.markStackedAdjancy", False)
        ConfigObj.default(self.param,"mark.markSize", 1)
        ConfigObj.default(self.param, "mark.dotDeviation", 0)
        ConfigObj.default(self.param,"values.maxGroupPixelHeight",0)
        ConfigObj.default(self.param, "AxisIntervals", 5)
        ConfigObj.default(self.param, "Axislabel", 'both')
        ConfigObj.default(self.param, "Yaxisvalue", 100)
        ConfigObj.default(self.param, "TitlePosition", 'mid')
        ConfigObj.default(self.param, "TitleFontSize", 12)
        ConfigObj.default(self.param, "TitleFontType", 'arial')
        ConfigObj.default(self.param, "Direction", 'vertical')
        ConfigObj.default(self.param, "bgcolor", 'color_pool')
        ConfigObj.default(self.param, "barcolor", 'same')
        ConfigObj.default(self.param, "barcolordark", 'no')
        ConfigObj.default(self.param, "linecolor", 'color_pool')
        ConfigObj.default(self.param, "xTickNumber", 'retain')
        ConfigObj.default(self.param, "TitleLength", 1)
        ConfigObj.default(self.param, "Legend", 'no')
        ConfigObj.default(self.param, "train", True)
        ConfigObj.default(self.param, "bgcolor_pertubation", 0)
        ConfigObj.default(self.param, "barcolor_pertubation", 0)
        ConfigObj.default(self.param, "strokecolor_pertubation", 0)
        ConfigObj.default(self.param, "lightness_pertubation", 0)
        ConfigObj.default(self.param, "randPos", False)
        ConfigObj.default(self.param, "outdata", False)
        ConfigObj.default(self.param, "imagePadding", 20)
        ConfigObj.default(self.param, "mask.isFlag", False)
        ConfigObj.default(self.param, "mask.type", "contour")
        ConfigObj.default(self.param, "changeTargetOnly", False)
        ConfigObj.default(self.param, "bgcolorL_perturbation", 0)
        ConfigObj.default(self.param, "bgcolorA_perturbation", 0)
        ConfigObj.default(self.param, "bgcolorB_perturbation", 0)
        ConfigObj.default(self.param, "LABperturbation", False)
        ConfigObj.default(self.param, "strokecolorL_perturbation", 0)
        ConfigObj.default(self.param, "strokecolorA_perturbation", 0)
        ConfigObj.default(self.param, "strokecolorB_perturbation", 0)
        ConfigObj.default(self.param, "strokeLABperturbation", False)
        ConfigObj.default(self.param, "barcolorL_perturbation", 0)
        ConfigObj.default(self.param, "barcolorA_perturbation", 0)
        ConfigObj.default(self.param, "barcolorB_perturbation", 0)
        ConfigObj.default(self.param, "barLABperturbation", False)

    def mark(self,image,center,dotColor):
        y=int(center[1])
        x=int(center[0]) + self.param.mark.dotDeviation
        # image[y:y+1,x:x+1]=(dotColor[0],dotColor[1],dotColor[2])
        # if self.param.mark.markSize==1:
        image[y:y+self.param.mark.markSize,x:x+self.param.mark.markSize]=(dotColor[0],dotColor[1],dotColor[2])

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
        # colorLists,backColor,fill,strokeColor = self._genColor(stackCount)
        if self.param.train:

            colorLists,backColor,fill,strokeColor = self._genTrainColor(stackCount,index)
        else:
            colorLists,backColor,fill,strokeColor = self._genTestColor_pie(stackCount,index)

        colorLists=uio.RGB2BGR(colorLists)
        backColor=uio.RGB2BGR(tuple(backColor))
        strokeColor=uio.RGB2BGR(strokeColor)
        
        lightness_pertubation=self.param.lightness_pertubation
        bgcolor_pertubation=self.param.bgcolor_pertubation
        barcolor_pertubation=self.param.barcolor_pertubation
        strokecolor_pertubation=self.param.strokecolor_pertubation
        backColor=np.array(backColor)-lightness_pertubation+bgcolor_pertubation
        backColor[backColor<0]=0
        backColor[backColor>255]=255
        backColor=tuple(backColor.tolist())
        if self.param.LABperturbation:
            temp_=uio.RGB2Lab(backColor)
            temp_[0]=temp_[0]+self.param.bgcolorL_perturbation
            temp_[1]=temp_[1]+self.param.bgcolorA_perturbation
            temp_[2]=temp_[2]+self.param.bgcolorB_perturbation
            backColor=uio.Lab2RGB(temp_)
        # fill=tuple((np.array(fill)-100).tolist())
        strokeColor=np.array(strokeColor)-lightness_pertubation+strokecolor_pertubation
        strokeColor[strokeColor<0]=0
        strokeColor[strokeColor>255]=255
        strokeColor=tuple(strokeColor.tolist())[0]
        if self.param.strokeLABperturbation:
            temp_=uio.RGB2Lab(strokeColor)
            temp_[0]=temp_[0]+self.param.strokecolorL_perturbation
            temp_[1]=temp_[1]+self.param.strokecolorA_perturbation
            temp_[2]=temp_[2]+self.param.strokecolorB_perturbation
            strokeColor=tuple([int(xxx) for xxx in uio.RGB2BGR(uio.Lab2RGB(temp_))])
            # strokeColor=uio.RGB2BGR(uio.Lab2RGB(temp_))
        for ccc in range(len(colorLists)):
            # colorLists[ccc]=tuple((np.array(colorLists[ccc])-lightness_pertubation+barcolor_pertubation).tolist())
            # colorLists[ccc]=np.array(colorLists[ccc])-lightness_pertubation+barcolor_pertubation
            # colorLists[ccc][colorLists[ccc]<0]=0
            # colorLists[ccc][colorLists[ccc]>255]=255
            # colorLists[ccc]=tuple(colorLists[ccc].tolist())
            if self.param.barLABperturbation:
                temp_=uio.RGB2Lab(colorLists[ccc])
                temp_[0]=temp_[0]+self.param.barcolorL_perturbation
                temp_[1]=temp_[1]+self.param.barcolorA_perturbation
                temp_[2]=temp_[2]+self.param.barcolorB_perturbation
                colorLists[ccc]=tuple([int(xxx) for xxx in uio.RGB2BGR(uio.Lab2RGB(temp_))])

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
        
        mask_image=Image.fromarray(np.zeros((100,100)))
        mask_draw = ImageDraw.ImageDraw(mask_image)
        
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
        
        if self.param.changeTargetOnly:
            values[markList[0]]=np.random.choice(self.param.changeTargetOnlyValue)
            values[markList[1]]=np.random.choice(self.param.changeTargetOnlyValue)
        
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
        
        bar_mid=[]
        self.param.mark.randPos=False
        # print(markList)
        for i in range(stackGroup):
            bar_mid.append(startOffsetX + int(stackWidth * 0.5))
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
                useColor2 = strokeColor
                # if lineThickness>0:
                #     useColor = colorLists[0]
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
                    mask_draw.rectangle(((startOffsetX, tmpOffsetY),(startOffsetX + stackWidth, tmpOffsetY - stackHeight)),fill=255,width=1)
                    if self.param.mark.randPos: 
                        if stackHeight==1:
                            randpos1 = 0
                        else:
                            randpos1 = np.random.randint(1, stackHeight)
                        randpos0 = np.random.randint(1, stackWidth)
                        
                        mark_x = startOffsetX + randpos0
                        mark_y = tmpOffsetY - randpos1
                        self.mark(image,(mark_x,mark_y),self.param.mark.dotColor)
                        pass
                    else:
                        if self.param.mask.isFlag:
                            pass
                        else:
                            self.mark(image,(startOffsetX+stackWidth*0.5,tmpOffsetY-stackHeight*0.5),self.param.mark.dotColor)
                tmpOffsetY -= stackHeight
                stackBarHeight+=stackHeight

            maxStackBarHeight=max(maxStackBarHeight,stackBarHeight)
            startOffsetX+=stackWidth+horEmptySpace

        tick_number=uio.fetchValue(self.param.AxisIntervals)
        ymax_value=uio.fetchValue(self.param.Yaxisvalue)
        TitlePosition=self.param.TitlePosition
        TitleFontSize=self.param.TitleFontSize
        TitleFontType=self.param.TitleFontType
        xTickNumber=self.param.xTickNumber
        TitleLength=self.param.TitleLength
        TitlePaddingLeft=self.param.TitlePaddingLeft

        Axislabel=self.param.Axislabel
        Imageim = Image.fromarray(np.uint8(image)).convert("RGB")
        # image=uio.add_title_and_axis(Imageim,tuple(backColor),bar_mid,tick_number,ymax_value,TitlePosition,TitleFontSize,TitleFontType,xTickNumber,TitleLength)
        
        
        image = uio.add_title_and_axis(Imageim,tuple(backColor),bar_mid,tick_number=tick_number,ymax_value=ymax_value,TitlePosition=TitlePosition,
                TitleFontSize=TitleFontSize,TitleFontType=TitleFontType,x_tick_number=xTickNumber,Axis=True,ImagePadding=self.param.imagePadding)

        
        image=uio.add_label(image,tuple(backColor),Axislabel)
        image=uio.add_title(image,tuple(backColor),TitlePosition=TitlePosition,
            TitleFontSize=TitleFontSize,TitleFontType=TitleFontType,TitleLength=TitleLength,TitlePaddingLeft=TitlePaddingLeft)
        
        if self.param.mask.isFlag:
            if self.param.mask.type=='contour':
                mask_image=np.array(mask_image)
                mask_image=np.uint8(mask_image)
                mask_image=np.pad(mask_image,((20,30),(35,15)),'constant',constant_values = (0,0))
                # image.paste(Image.fromarray(mask_image),(0,0))
                mask_image=np.expand_dims(mask_image,axis=2)
                image=np.array(image)
                image=np.concatenate((image, mask_image), axis=2)
                image=Image.fromarray(image)
        
        # image = uio.add_title(image,backColor,TitlePosition=TitlePosition,
        #     TitleFontSize=TitleFontSize,TitleFontType=TitleFontType,TitleLength=TitleLength,TitlePaddingLeft=TitlePaddingLeft,
        #     TitlePaddingTop=TitlePaddingTop,TitleLetterCount=TitleLetterCount,TitleLetter=TitleLetter,TitleLineCount=titlelinecount)
        
        image=np.array(image)
        # if preprocess is enabled, preprocess input data

        #save
        inputFilePath,outputFilePath,orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName%index
        inputFilePath=os.path.join(inputFilePath,fileName)
        outputFilePath=os.path.join(outputFilePath,fileName)
        orgFilePath =os.path.join(orgFilePath,fileName)

        self._preprocess_numpy(inputFilePath,image)

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