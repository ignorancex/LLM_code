import logging
import os
from . import UtilIO as uio
from . import VisAbstractGen
import numpy as np
import cv2
import random
from util.Config import ConfigObj
from PIL import Image, ImageFont, ImageDraw
import sys
sys.path.append("..")
from util.color_pool import *


class BarGenerator(VisAbstractGen.VisAbstractGen):

    def __init__(self, config):
        super().__init__(config)
        ConfigObj.default(self.param, "randDeleteBar", False)
        ConfigObj.default(self.param, "midGap", 0)
        ConfigObj.default(self.param, "fixBarGap", -1)
        ConfigObj.default(self.param, "values.useSpecialGen", False)
        ConfigObj.default(self.param, "values.specialGen.count", 10)
        ConfigObj.default(self.param, "values.specialGen.divScale", 12)
        ConfigObj.default(self.param, "mark.fixPos", False)
        ConfigObj.default(self.param, "mark.fixPosY", 9)
        ConfigObj.default(self.param, "mark.randPos", False)
        ConfigObj.default(self.param, "mark.kind", 'dot')
        ConfigObj.default(self.param, "mark.markAdjancy", False)
        ConfigObj.default(self.param, "mark.markStackedAdjancy", False)
        ConfigObj.default(self.param, "mark.markSize", 1)
        ConfigObj.default(self.param, "mark.bottom", False)
        ConfigObj.default(self.param, "mark.bottomValue", 5)
        ConfigObj.default(self.param, "mark.dotDeviation", 0)
        
        ConfigObj.default(self.param, "noise.addNoise", False)
        ConfigObj.default(self.param, "noise.noiseKind", None)
        ConfigObj.default(self.param, "noise.sigma", 50)
        ConfigObj.default(self.param, "pattern.addPattern", False)
        ConfigObj.default(self.param, "pattern.patternKind", None)
        # ConfigObj.default(self.param, "Axis.addAxis", False)
        ConfigObj.default(self.param, "Title.addTitle", False)
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
        ConfigObj.default(self.param, "DataDistribution", 0)
        ConfigObj.default(self.param, "TitlePaddingLeft", 0.5)
        ConfigObj.default(self.param, "TitlePaddingTop", 0)
        ConfigObj.default(self.param, "TitleLetter", False)
        ConfigObj.default(self.param, "TitleLetterCount", 5)
        ConfigObj.default(self.param, "XlabelDelta", 0)
        ConfigObj.default(self.param, "YlabelDelta", 0)
        ConfigObj.default(self.param, "bgcolor_pertubation", 0)
        ConfigObj.default(self.param, "barcolor_pertubation", 0)
        ConfigObj.default(self.param, "strokecolor_pertubation", 0)
        ConfigObj.default(self.param, "Axis", True)
        ConfigObj.default(self.param, "outdata", False)
        ConfigObj.default(self.param, "showBarCountExceptMarked", 8)
        ConfigObj.default(self.param, "lightness_pertubation", 0)
        ConfigObj.default(self.param, "mask.isFlag", False)
        ConfigObj.default(self.param, "mask.type", "contour")
        ConfigObj.default(self.param, "mask.maskLength", 1)
        ConfigObj.default(self.param, "mask.maskgap", 0)
        ConfigObj.default(self.param, "imagePadding", 20)
        ConfigObj.default(self.param, "titlelinecount", 1)
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


    def mark(self, image, center, dotColor,kind='dot'):
        y = None
        if self.param.mark.fixPos:
            y = int(self.param.mark.fixPosY)
        else:
            y = int(center[1])
        x = int(center[0])
        image[y:y + 1, x:x + 1] = (dotColor[0], dotColor[1], dotColor[2])

        if kind == 'slash':
            image[y+1:y + 2, x+1:x + 2] = (dotColor[0], dotColor[1], dotColor[2])
            image[y - 1:y , x -1 :x ] = (dotColor[0], dotColor[1], dotColor[2])



    def gen(self, index, isTrainData=True):
        width = uio.fetchValue(self.param.outputWidth, 1)
        height = uio.fetchValue(self.param.outputHeight, 1)
        barCount = uio.fetchValue(self.param.barCount)
        barWidth = uio.fetchValue(self.param.barWidth, 1)
        lineThickness = uio.fetchValue(self.param.lineThickness, -1)
        spacePaddingLeft = uio.fetchValue(self.param.spacePaddingLeft)
        spacePaddingRight = uio.fetchValue(self.param.spacePaddingRight)
        spacePaddingTop = uio.fetchValue(self.param.spacePaddingTop)
        spacePaddingBottom = uio.fetchValue(self.param.spacePaddingBottom)
        midGap = uio.fetchValue(self.param.midGap)

        tick_number=uio.fetchValue(self.param.AxisIntervals)
        ymax_value=uio.fetchValue(self.param.Yaxisvalue)
        TitlePosition=self.param.TitlePosition
        TitleFontSize=self.param.TitleFontSize
        TitleFontType=self.param.TitleFontType
        xTickNumber=self.param.xTickNumber
        TitleLength=self.param.TitleLength
        maskLength=self.param.mask.maskLength
        maskgap=self.param.mask.maskgap

        Axislabel=self.param.Axislabel
        Axis=self.param.Axis

        bgcolor=self.param.bgcolor
        barcolor=self.param.barcolor
        linecolor=self.param.linecolor

        if self.param.Direction=='Horizontal':
            isrotate=True
        else:
            isrotate=False

        '''
        <       ><-------><        ><--------><          >
        padding  barWidth   empty    barWidth   padding
        '''

        horSpace = width - spacePaddingLeft - spacePaddingRight - midGap

        verSpace = height - spacePaddingTop - spacePaddingBottom
        if verSpace <= 20:
            logging.error("Wrong Parameters! Vertical Padding is too large! Set 20 instead.")
            verSpace = 20

        leftHorEmptySpace = horSpace - barWidth * barCount
        if lineThickness > 0:
            #为什么乘以4而不是2
            leftHorEmptySpace -= barCount * lineThickness * 2
        # avoid overlapping
        if leftHorEmptySpace < 0:
            # print('ggg')
            leftHorEmptySpace = 0
            barWidth = int(horSpace / barCount)
            if lineThickness > 0:
                barWidth -= lineThickness * 2
                if barWidth <= 2:
                    barWidth += lineThickness * 2 - 2
                    lineThickness = 1
                    leftHorEmptySpace += barCount * (lineThickness - 1) * 4
            if barWidth <= 2:
                lineThickness = 1
                barWidth = 2
                emptyPadding = width - barWidth * barCount
                spacePaddingLeft = int((np.random.rand() * emptyPadding))
                spacePaddingRight = int(emptyPadding - spacePaddingLeft)
                leftHorEmptySpace = width - emptyPadding
        horEmptySpace = 0
        # print(barWidth)
        barWidth = uio.fetchValue(self.param.barWidth, 1)
        # print('leftHorEmptySpace',leftHorEmptySpace)
        if barCount > 1:
            # horEmptySpace = leftHorEmptySpace // (barCount - 1)
            # print(leftHorEmptySpace)
            horEmptySpace = leftHorEmptySpace // (barCount - 1)

        if lineThickness > 0:
            horEmptySpace += lineThickness * 2
            # horEmptySpace += 2

        if self.param.fixBarGap >= 0:
            horEmptySpace = self.param.fixBarGap
        # print(horEmptySpace)
        # horEmptySpace=0

        barHeights = []
        maxBarHeight = 0

        quant = verSpace
        if lineThickness > 0:
            quant = verSpace // lineThickness
        if self.param.values.pixelValue:
            quant = 0
        values = self._genValues(barCount, quant)
        # print(values)
        if self.param.DataDistribution != 0:
            values = self._genDistributionValues(barCount, quant)
        #print(values)


        resetFix = False

        if self.param.values.useSpecialGen:
            v1 = v2 = 0
            count1 = self.param.values.specialGen.count - 1
            divScale = float(self.param.values.specialGen.divScale)
            while v1 == v2:
                v1 = int(10 * 10 ** (random.randint(0, count1) / divScale))
                v2 = int(10 * 10 ** (random.randint(0, count1) / divScale))

            if len(self.param.mark.fix) == 2:
                # logging.error("Bar: in useSpecialGen mode, must indicate 2 places to replace the in mark.fix")
                values[self.param.mark.fix[0]] = v1
                values[self.param.mark.fix[1]] = v2
            else:
                resetFix = True
                self.param.mark.fix = []
                lv = len(values) - 1
                ind1 = random.randint(0, lv)
                ind2 = random.randint(0, lv)
                while ind2 == ind1:
                    ind2 = random.randint(0, lv)
                self.param.mark.fix = [ind1, ind2]
                values[self.param.mark.fix[0]] = v1
                values[self.param.mark.fix[1]] = v2

        markList = self._mark(values)
        
        if self.param.changeTargetOnly:
            # values[markList[0]]=np.random.choice(self.param.changeTargetOnlyValue)
            # values[markList[1]]=np.random.choice(self.param.changeTargetOnlyValue)
            # if len(self.param.changeTargetOnlyValue)==2:
            #     values[markList[0]]=np.random.randint(self.param.changeTargetOnlyValue[0],self.param.changeTargetOnlyValue[1])
            #     values[markList[1]]=np.random.randint(self.param.changeTargetOnlyValue[0],self.param.changeTargetOnlyValue[1])
            # else:
            values[markList[0]]=uio.fetchValue(self.param.changeTargetOnlyValue)
            values[markList[1]]=uio.fetchValue(self.param.changeTargetOnlyValue)

        showBarCountExceptMarked=self.param.showBarCountExceptMarked
        
        # showBarIdList=[]
        # while True:
        #     showbarid=np.random.randint(0,10)
        #     showBarIdList.append(showbarid)
        #     if showbarid in markList:
        #         showBarIdList.remove(showbarid)
        #     if len(list(set(showBarIdList)))==showBarCountExceptMarked:
        #         break

        # print(showBarIdList)
        bar_index = list(range(barCount))
        if self.param.randDeleteBar:
            for i in markList:
                bar_index.remove(i)
            select_num = np.random.randint(0,barCount-len(markList))
            bar_index = np.random.choice(bar_index,select_num,replace=False)
            bar_index = list(bar_index) + markList
            
        # if bgcolor=='color_pool':
        #     colorLists, backColor, fill, strokeColor = self._genTrainColor(barCount,index)
        # elif bgcolor=='bright':
        #     colorLists, backColor, fill, strokeColor = self._genTestColor(barCount,index)
        # elif bgcolor=='dark':
        #     colorLists, backColor, fill, strokeColor = self._genTestColor(barCount,index)
        # colorLists, backColor, fill, strokeColor = self._genTestColor(barCount,index)
        # print(colorLists)
        if self.param.train:
            colorLists, backColor, fill, strokeColor = self._genTrainColor(barCount,index)
        else:
            colorLists, backColor, fill, strokeColor = self._genTestColor(barCount,index)
        # colorLists, backColor, fill, strokeColor = self._genTrainColor(barCount,index)
        # colorLists, backColor, fill, strokeColor = self._genTestColor(barCount,index)

        # for i in range(5):
        #     colorLists.append([51, 153, 255])
        # print(colorLists)
        lightness_pertubation=self.param.lightness_pertubation
        bgcolor_pertubation=self.param.bgcolor_pertubation
        barcolor_pertubation=self.param.barcolor_pertubation
        strokecolor_pertubation=self.param.strokecolor_pertubation
        backColor=tuple((np.array(backColor)+lightness_pertubation+bgcolor_pertubation).tolist())
        if self.param.LABperturbation:
            temp_=uio.RGB2Lab(backColor)
            temp_[0]=temp_[0]+self.param.bgcolorL_perturbation
            temp_[1]=temp_[1]+self.param.bgcolorA_perturbation
            temp_[2]=temp_[2]+self.param.bgcolorB_perturbation
            backColor=uio.RGB2BGR(uio.Lab2RGB(temp_))
        # fill=tuple((np.array(fill)-100).tolist())
        strokeColor=tuple((np.array(strokeColor)-lightness_pertubation+strokecolor_pertubation).tolist())[0]
        if self.param.strokeLABperturbation:
            temp_=uio.RGB2Lab(strokeColor)
            temp_[0]=temp_[0]+self.param.strokecolorL_perturbation
            temp_[1]=temp_[1]+self.param.strokecolorA_perturbation
            temp_[2]=temp_[2]+self.param.strokecolorB_perturbation
            strokeColor=tuple([int(xxx) for xxx in uio.RGB2BGR(uio.Lab2RGB(temp_))])
            # strokeColor=uio.RGB2BGR(uio.Lab2RGB(temp_))
        for ccc in range(len(colorLists)):
            # colorLists[ccc]=tuple((np.array(colorLists[ccc])-lightness_pertubation+barcolor_pertubation).tolist())
            if self.param.barLABperturbation:
                temp_=uio.RGB2Lab(colorLists[ccc])
                temp_[0]=temp_[0]+self.param.barcolorL_perturbation
                temp_[1]=temp_[1]+self.param.barcolorA_perturbation
                temp_[2]=temp_[2]+self.param.barcolorB_perturbation
                colorLists[ccc]=tuple([int(xxx) for xxx in uio.RGB2BGR(uio.Lab2RGB(temp_))])
        # print(colorLists)
        # image = (np.ones(shape=(width, height, 3), dtype=np.int8) * 255).copy()
 
        # image[:, :] = backColor
        # image[:, :] = color[index]
        backColor=tuple(backColor)
        image = Image.new('RGB', (width,height), backColor)
        draw = ImageDraw.ImageDraw(image)
        
        mask_image=Image.fromarray(np.zeros((100,100)))
        mask_draw = ImageDraw.ImageDraw(mask_image)
        
        mask_gaussian_image=Image.fromarray(np.zeros((100,100)))
        mask_gaussian_draw = ImageDraw.ImageDraw(mask_gaussian_image)
        
        mask_point_image=Image.fromarray(np.zeros((100,100)))
        mask_point_draw = ImageDraw.ImageDraw(mask_point_image)
        
        mask_line_image=Image.fromarray(np.zeros((100,100)))
        mask_line_draw = ImageDraw.ImageDraw(mask_line_image)

        startOffsetX = int(spacePaddingLeft)
        startOffsetY = int(height - spacePaddingBottom)

        valueMax = max(values)
        marked_bar_area = []
        bar_mid=[]
        maskid=0
        for i in range(barCount):
            bar_mid.append(startOffsetX + barWidth * 0.5)
            if self.param.values.pixelValue:
                barHeight = max(1, int(values[i]))
            else:
                barHeight = max(1, int(verSpace * values[i] / valueMax))
            barHeights.append(barHeight)
            maxBarHeight = max(maxBarHeight, barHeight)

            if i not in bar_index:
                startOffsetX += barWidth + horEmptySpace
                if i == barCount // 2 - 1:
                    startOffsetX += midGap
                continue
            #if fill==true,fill the bar using colorLists[i],then if linethickness>0,again line the bar with strokecolor[0]
            #if fill==false,line the bar using colorlists[i],then if linethickness>0,again line the bar with strokecolor[0]
            # cv2.rectangle(image,
            #               (startOffsetX, startOffsetY),
            #               (startOffsetX + barWidth, startOffsetY - barHeight),
            #               colorLists[i],
            #               -1 if fill else lineThickness
            #               )
            
            draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=tuple(colorLists[i]),width=1)
            
            # if i in showBarIdList:
            #     draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=tuple(colorLists[i]),width=1)
            #     draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=None,outline=tuple(strokeColor[0]),width=lineThickness)
            
            # print(colorLists[i])
            # print(colorLists[i])
            if lineThickness > 0:
                draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=None,outline=tuple(strokeColor),width=lineThickness)
                pass
                # cv2.rectangle(image,
                #               (startOffsetX, startOffsetY),
                #               (startOffsetX + barWidth, startOffsetY - barHeight),
                #               strokeColor[0],
                #               lineThickness
                #               )
            if i in markList:
                # draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=tuple(colorLists[i]),width=1)
                # draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=None,outline=tuple(strokeColor[0]),width=lineThickness)
                # draw mask image
                # if maskid==0:
                #     maskid+=1
                #     mask_draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - int(barHeight*maskLength))),fill=255,width=1)
                # else:
                #     mask_draw.rectangle(((startOffsetX + maskgap, startOffsetY),(startOffsetX + barWidth + maskgap, startOffsetY - int(barHeight*maskLength))),fill=255,width=1)
                mask_draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - int(barHeight*maskLength))),fill=255,width=1)
                # mask_gaussian_draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=255,width=1)
                # mask_line_draw.rectangle(((startOffsetX, startOffsetY),(startOffsetX + barWidth, startOffsetY - barHeight)),fill=255,width=1)
                if self.param.mark.randPos:
                    randpos0 = np.random.randint(2, barWidth - 1)
                    randpos1 = np.random.randint(2, barHeight - 1)
                    mark_x = startOffsetX + randpos0
                    mark_y = startOffsetY - randpos1
                    dotColor=self.param.mark.dotColor
                    mask_point_draw.point([mark_x, mark_y], fill = 255)
                    draw.point([mark_x, mark_y], fill = (dotColor[0], dotColor[1], dotColor[2]))
                    # self.mark(image, (mark_x, mark_y),
                    #           self.param.mark.dotColor,
                    #           kind=self.param.mark.kind)
                elif self.param.mark.bottom:
                    mark_x = startOffsetX + barWidth * 0.5
                    mark_y = startOffsetY - self.param.mark.bottomValue
                    mask_gaussian_draw.rectangle(((mark_x-1,mark_y-1),(mark_x+1, mark_y+1)),fill=255,width=1)
                    dotColor=self.param.mark.dotColor
                    draw.point([mark_x, mark_y], fill = (dotColor[0], dotColor[1], dotColor[2]))
                    # mask_point_draw.rectangle(((mark_x-1,mark_y-1),(mark_x+1, mark_y+1)),fill=255,width=1)
                    mask_point_draw.point([mark_x, mark_y], fill = 255)
                    pass
                else:
                    mark_x = startOffsetX + barWidth * 0.5 
                    mark_y = startOffsetY - barHeight * 0.5 + self.param.mark.dotDeviation
                    mask_gaussian_draw.rectangle(((mark_x-1,mark_y-1),(mark_x+1, mark_y+1)),fill=255,width=1)
                    dotColor=self.param.mark.dotColor
                    if self.param.mask.isFlag:
                        # draw.rectangle(((mark_x-1,mark_y-1),(mark_x+1, mark_y+1)), fill = (dotColor[0], dotColor[1], dotColor[2]))
                        draw.point([mark_x, mark_y], fill = (dotColor[0], dotColor[1], dotColor[2]))
                        mask_point_draw.point([mark_x, mark_y], fill = 255)
                        mask_gaussian_draw.rectangle(((mark_x-1,mark_y-1),(mark_x+1, mark_y+1)),fill=255,width=1)
                        # randomoffsize=np.random.randint(0,barWidth)
                        # mask_line_draw.rectangle(((startOffsetX+randomoffsize, startOffsetY),(startOffsetX+randomoffsize + 1, startOffsetY - barHeight)),fill=255,width=1)
                        pass
                    else:
                        if self.param.mark.markSize==1:
                            draw.point([mark_x, mark_y], fill = (dotColor[0], dotColor[1], dotColor[2]))
                        elif self.param.mark.markSize==0:
                            pass
                        else:
                            draw.rectangle(((mark_x, mark_y),(mark_x+self.param.mark.markSize, mark_y+self.param.mark.markSize)), fill = (dotColor[0], dotColor[1], dotColor[2]))
                    # self.mark(image, (mark_x, mark_y),
                    #           self.param.mark.dotColor,
                    #           kind=self.param.mark.kind)
                marked_bar_area.append([(startOffsetX, startOffsetY - barHeight),
                                        (startOffsetX + barWidth, startOffsetY)])
            # print(startOffsetX)
            startOffsetX += barWidth + horEmptySpace
            if i == barCount // 2 - 1:
                startOffsetX += midGap


        # print(bar_mid)
        # if self.param.noise.addNoise:
        #     image,_ = uio.add_noise(image,sigma=self.param.noise.sigma,kind=self.param.noise.noiseKind)
            
        # if self.param.pattern.addPattern:
        #     image = uio.add_pattern(image,marked_bar_area,kind=self.param.pattern.patternKind)

        # only_add_axis=self.param.Axis.addAxis and (not self.param.Title.addTitle)
        # only_add_title=(not self.param.Axis.addAxis) and self.param.Title.addTitle
        # add_title_axis=self.param.Axis.addAxis and self.param.Title.addTitle

        # if only_add_axis:
        #     image = uio.add_axis(image,backColor)
        #     pass

        # if only_add_title:
        #     image = uio.add_title(image)
        #     pass

        # if add_title_axis:
        #     image = uio.add_title_axis(image,backColor)
        #     pass

        # if self.param.noise.addNoise:
        #     image,_ = uio.add_noise(image,sigma=self.param.noise.sigma,kind=self.param.noise.noiseKind)

        # bright_arr = np.array(image)-100
        # print(bright_arr)
        # # print(len(bright_arr))
        # for x in range(len(bright_arr)):
        #     for y in range(len(bright_arr[0])):
        #         for z in range(3):
        #             if bright_arr[x][y][z]<0:
        #                 bright_arr[x][y][z]=0
        # # print(bright_arr<0)
        # # bright_arr[bright_arr<0]=0
        # # bright_arr=(bright_arr + abs(bright_arr)) / 2
        # # bright_arr = bright_arr.astype('uint8') 
        # # np.where(bright_arr > 0, bright_arr, 0)
        # image = Image.fromarray(bright_arr)
        # backColor=tuple((np.array(backColor)-100).tolist())
        if isrotate:
            image = uio.add_title_and_axis_horizontal(image,backColor,bar_mid,tick_number,ymax_value,TitlePosition,TitleFontSize,TitleFontType)
            pass
        else:
            if Axis:
                image = uio.add_title_and_axis(image,backColor,bar_mid,tick_number=tick_number,ymax_value=ymax_value,TitlePosition=TitlePosition,
                                               TitleFontSize=TitleFontSize,TitleFontType=TitleFontType,x_tick_number=xTickNumber,Axis=Axis,ImagePadding=self.param.imagePadding)
            else:
                image = uio.add_title_and_axis(image,backColor,bar_mid,tick_number=tick_number,ymax_value=ymax_value,TitlePosition=TitlePosition,
                                               TitleFontSize=TitleFontSize,TitleFontType=TitleFontType,x_tick_number=xTickNumber,Axis=Axis)

        # if self.param.Axislabel !='either':
        # if self.param.TitlePaddingLeft:
        XlabelDelta=self.param.XlabelDelta
        YlabelDelta=self.param.YlabelDelta
        TitlePaddingLeft=self.param.TitlePaddingLeft
        TitlePaddingTop=self.param.TitlePaddingTop
        TitleLetter=self.param.TitleLetter
        TitleLetterCount=self.param.TitleLetterCount
        titlelinecount=self.param.titlelinecount

        # titlelinecount=np.random.randint(0,3)
        # TitlePaddingLeft=round(np.random.uniform(0.1,0.8),1)
        # if titlelinecount==2:
        #     TitleFontSize=10

        image = uio.add_label(image,backColor,Axislabel,XlabelDelta,YlabelDelta)
        image = uio.add_title(image,backColor,TitlePosition=TitlePosition,
            TitleFontSize=TitleFontSize,TitleFontType=TitleFontType,TitleLength=TitleLength,TitlePaddingLeft=TitlePaddingLeft,
            TitlePaddingTop=TitlePaddingTop,TitleLetterCount=TitleLetterCount,TitleLetter=TitleLetter,TitleLineCount=titlelinecount)
        
        if self.param.Legend=='yes':
            image = uio.add_legend(image,backColor,colorLists)

        # if preprocess is enabled, preprocess input data
        # image=np.array(image)
        # print(np.array(mask_image).shape)
        
        if self.param.mask.isFlag:
            # image.paste(mask_gaussian_image,(35,20))
            if self.param.mask.type=='gaussian':
                gauss_kernel = uio.gauss(3,0.1)
                gaussion_image = cv2.filter2D(np.array(mask_gaussian_image), -1, gauss_kernel)
                gaussion_image=np.uint8(gaussion_image)
                gaussion_image=np.pad(gaussion_image,((20,30),(35,15)),'constant',constant_values = (0,0))
                gaussion_image=np.expand_dims(gaussion_image,axis=2)
                image=np.array(image)
                image=np.concatenate((image, gaussion_image), axis=2)
                image=Image.fromarray(image)
            
            if self.param.mask.type=='point':
                mask_point_image=np.array(mask_point_image)
                mask_point_image=np.uint8(mask_point_image)
                mask_point_image=np.pad(mask_point_image,((20,30),(35,15)),'constant',constant_values = (0,0))
                # image.paste(Image.fromarray(mask_image),(0,0))
                mask_point_image=np.expand_dims(mask_point_image,axis=2)
                image=np.array(image)
                image=np.concatenate((image, mask_point_image), axis=2)
                image=Image.fromarray(image)
                # image.paste(Image.fromarray(gaussion_image),(35,20))
            # mask_image=np.zeros((150,150))
            if self.param.mask.type=='contour':
            
                mask_image=np.array(mask_image)
                mask_image=np.uint8(mask_image)
                mask_image=np.pad(mask_image,((20,30),(35,15)),'constant',constant_values = (0,0))
                # image.paste(Image.fromarray(mask_image),(0,0))
                mask_image=np.expand_dims(mask_image,axis=2)
                image=np.array(image)
                image=np.concatenate((image, mask_image), axis=2)
                image=Image.fromarray(image)
                
            if self.param.mask.type=='line':
                mask_line_image=np.array(mask_line_image)
                mask_line_image=np.uint8(mask_line_image)
                mask_line_image=np.pad(mask_line_image,((20,30),(35,15)),'constant',constant_values = (0,0))
                # image.paste(Image.fromarray(mask_image),(0,0))
                mask_line_image=np.expand_dims(mask_line_image,axis=2)
                image=np.array(image)
                image=np.concatenate((image, mask_line_image), axis=2)
                image=Image.fromarray(image)
            
        # print(type(image))
        # print(np.array(image)[50,50])
        

        # save
        inputFilePath, outputFilePath, orgFilePath = self._getFilePath(isTrainData)

        fileName = self.param.fileName % index
        inputFilePath = os.path.join(inputFilePath, fileName)
        outputFilePath = os.path.join(outputFilePath, fileName)
        orgFilePath = os.path.join(orgFilePath, fileName)

        self._preprocess(inputFilePath, image)
        barMaxHeight=self._getMaxValues(barHeights, markList)
        uio.save(outputFilePath+"_h", [int(barMaxHeight[0]*ymax_value/100)], "json")
        barHeights = self._processValues(barHeights, markList)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath, [barHeights[0] / barHeights[1]], "json")
            uio.save(outputFilePath + "_ll", [self.param.labelValue], "json")
        elif self.param.mark.genFix==1:
            uio.save(outputFilePath, [int(barMaxHeight[0]*ymax_value/100)], "json")
            uio.save(outputFilePath + "_ll", [self.param.labelValue], "json")
        else:
            uio.save(outputFilePath, barHeights, "json")
            
            uio.save(outputFilePath + "_ll", [self.param.labelValue] * len(barHeights), "json")

        ratio = self._genRatio(barHeights)
        uio.save(outputFilePath + "_r", ratio, "json")

        labels = [self.param.labelValue] * len(ratio)
        if self.param.mark.ratio.ratio2Only:
            uio.save(outputFilePath + "_l", [labels[0]], "json")
        else:
            uio.save(outputFilePath + "_l", labels, "json")

        if resetFix:
            self.param.mark.fix = []








