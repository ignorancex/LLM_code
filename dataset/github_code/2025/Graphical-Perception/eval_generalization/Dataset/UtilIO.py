import os
import time
import threading
import json
import pickle
import logging
import numpy as np
import random
import cv2
from PIL import Image, ImageFont, ImageDraw
from random_words import RandomWords
import sys
sys.path.append("..")
from util.color_pool import *

def __innerTask(cmd):
    logging.info("Execute command: "+cmd)
    os.system(cmd)
    logging.info("Command Finish: "+cmd)

def executeTask(cmd,waitForEnd=False):
    th = threading.Thread(target=__innerTask,args=(cmd,))
    th.start()
    if waitForEnd:
        th.join()
    return th

def suffix(s,suf):
    s = s.strip()
    if s.endswith(suf):
        return s
    else:
        return s+suf

def save(name,data,mode="npy"):
    #logging.trace("Save (%s): %s"%(mode,name))
    mkdirsExceptFile(name)
    if(mode=="json" and type(data) in [list,dict,set]):
        name=suffix(name,".json")
        with open(name,'w') as f:
            json.dump(data,f)
    elif(mode=="json_format" and type(data) in [list,dict,set]):
        name=suffix(name,".json")
        with open(name,'w') as f:
            json.dump(data,f,indent=4)
    elif mode=="npy":
        name=suffix(name,".npy")
        with open(name,'wb') as f:
            pickle.dump(data,f,True)
    elif mode=="txt":
        name=suffix(name,".txt")
        with open(name,'w') as f:
            f.write(str(data))
    return

def load(name,mode="npy",default=None):
    #logging.trace("Load (%s): %s"%(mode,name))
    data=None
    if(mode=="json"):
        name=suffix(name,".json")
        if not os.path.exists(name):
            logging.trace("Return default Value ")
            return default
        with open(name,'r') as f:
            data = json.load(f)
    elif mode=="npy":
        name=suffix(name,".npy")
        if not os.path.exists(name):
            #logging.trace("Return default Value ")
            return default
        with open(name,'rb') as f:
            data = pickle.load(f)
    elif mode=="txt":
        name=suffix(name,".txt")
        if not os.path.exists(name):
            #logging.trace("Return default Value ")
            return default
        with open(name,'r') as f:
            return f.readlines()
    return data

def mkdirsExceptFile(dir):
    os.makedirs(os.path.split(dir)[0],exist_ok=True)

__lastTime= time.time()
__lastValue = 0
def logProgress(curv,maxv,title="Process",printTimeInterval=20.0):
    global __lastTime
    curTime = time.time()
    if curTime-__lastTime<printTimeInterval and curv!=maxv:
        return 
    global __lastValue
    deltaTime=max(curTime-__lastTime,1e-5)
    deltav=max(float(curv-__lastValue),0.0)
    __lastValue = curv
    speed = deltav / deltaTime
    s1 = "%s > %7d | %7d ( %6.2f%% ) "%(title,curv,maxv,100.0*curv/maxv)
    s2 = "| Spd %8.3f /s | Left "%speed
    if speed>0:
        leftsec = (maxv-curv)/speed
        if leftsec<100:
            s2+="%6.2f s"%leftsec
        elif leftsec<6000: # 100min
            s2+="%6.2f m"%(leftsec/60.0)
        elif leftsec<3600*48: # 48h
            s2+="%6.2f h"%(leftsec/3600.0)
        else: # day
            s2+="%8.2f d"%(leftsec/86400.0)
    s=s1+s2
    logging.info(s)
    __lastTime = time.time()

'''
    Input can be a fixed value, or a range value
    Ex1:
        Input 123
        Output 123
    Ex2:
        Input [2.3,7.9]
        Output 4.6 or other random floats between [2.3,7.9)
    Ex3:
        Input [1,1000]
        Output 233 or other random ints between [1,1000)
    Ex4:
        Input [1,10,50,60]
        Output values [1,10) or [50,60)
    Ex5:
        Input [1,10,50,60,25,55]
        Output values [1,10) or [50,60) or [25,55)
'''
def fetchValue(attr,minimum=0,types=int):
    if isinstance(attr,(list,tuple)):
        if isinstance(attr[0],int) and isinstance(attr[1],int) and types==int:
            if len(attr)==2:
                v = np.random.randint(attr[0],attr[1])
                # print(v,end=' ')
            elif len(attr)>2:
                index = np.random.randint(len(attr)//2)*2
                v = np.random.randint(attr[index],attr[index+1])
        else:
            if len(attr)==2:
                v = np.random.rand()*(attr[1]-attr[0])+attr[0]
            elif len(attr)>2:
                index = np.random.randint(len(attr)//2)*2
                v = np.random.rand()*(attr[1+index]-attr[index])+attr[index]
        # print("%s -> %s"%(str(attr),str(v)))
    else:
        v = attr
    
    if v<minimum:
        logging.error("Wrong Attribute "+str(attr)+" | Expect Min: "+str(minimum))
        raise RuntimeError("Wrong attr")
    return v

def fetchMaxValue(attr):
    if isinstance(attr,(list,tuple)):
        if isinstance(attr[0],int) and isinstance(attr[1],int):
            return max(attr)-1
        else:
            return max(attr)
    else:
        return attr

def fetchMinValue(attr):
    if isinstance(attr,(list,tuple)):
         return min(attr)
    else:
        return attr



def rd(x):
    return random.random()<x

def rdcolor():
    r,g,b=random.random(),random.random(),random.random()
    r=int(r*255.999)
    g=int(g*255.999)
    b=int(b*255.999)
    return (r,g,b)

def _colorDiff(c1,c2):
    return ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**0.5

def rgb2xyz(x,y,z):
    x = x/255.0
    y = y/255.0
    z = z/255.0
    xnew = 0.4124564*x + 0.3575761*y + 0.1804375*z
    ynew = 0.2126729*x + 0.7151522*y + 0.0721750*z
    znew = 0.0193339*x + 0.1191920*y + 0.9503041*z
    return xnew, ynew, znew


def xyz2Lab(x,y,z):
    x = x/0.950456
    y = y/1.0
    z = z/1.088754
    
    if x>0.008856:
        xx = x**(1.0/3)
    else:
        xx = 7.787*x + 4.0/29
    
    if y>0.008856:
        yy = y**(1.0/3)
    else:
        yy = 7.787*y + 4.0/29
    
    if z>0.008856:
        zz = z**(1.0/3)
    else:
        zz = 7.787*z + 4.0/29
        
    L = 116.0*yy-16.0
    if L<0:L=0
    a = 500.0*(xx-yy)
    b = 200.0*(yy-zz)
    return L,a,b

# def calcDeltaE(r1,g1,b1,r2,g2,b2):
def calcDeltaE(c1,c2):
    r1=c1[0]
    g1=c1[1]
    b1=c1[2]
    r2=c2[0]
    g2=c2[1]
    b2=c2[2]
    x1, y1, z1 = rgb2xyz(r1,g1,b1)
    x2, y2, z2 = rgb2xyz(r2,g2,b2)
    
    L1, A1, B1 = xyz2Lab(x1, y1, z1)
    L2, A2, B2 = xyz2Lab(x2, y2, z2)
    
#     print(r1,g1,b1)
#     print(r2,g2,b2)
#     print(L1,A1,B1)
#     print(L2,A2,B2)
    
    # deltaE = ((L1-L2)**2 + (A1-A2)**2 + (B1-B2)**2)**0.5
    deltaL=L1-L2
    return abs(deltaL)

def rdcolorDiff(cmp,thres=20,count_times=100):
    count=0
    while True:
        count+=1
        c = rdcolor()
        if count>count_times:
            return c
        if isinstance(cmp,list):
            flag=False
            for cm in cmp:
                if _colorDiff(c,cm)<thres:
                    flag=True
                    break
            if flag:
                continue
            else:
                return c
        elif _colorDiff(c,cmp)>thres:
            return c
    return None

def bg_bar_colorDiff(cmp,thres=2,count_times=100,barcolor_pertubation=0):
    # bar_color_test=getTestbarcolor()
    count=0
    while True:
        count+=1
        # rand=random.randint(0,len(bar_color_test)-1)
        rand=random.randint(0,len(bar_color)-1)
        # c = bar_color_test[rand]
        c = bar_color[rand]
        # c = getTestbarcolor_perturbation(barcolor_pertubation)[rand]
        if count>count_times:
            # print("jjj")
            return c
        if isinstance(cmp,list):
            flag=False
            for cm in cmp:
                # if _colorDiff(c,cm)<thres:
                if calcDeltaE(c,cm)<thres:
                    flag=True
                    break
            if flag:
                continue
            else:
                return c
        # elif _colorDiff(c,cmp)>thres:
        elif calcDeltaE(c,cmp)>thres:
            # print(calcDeltaE(c,cmp))
            return c
    return None

def bg_bar_colorDiff_bardark(cmp,thres=2,count_times=100):
    bar_color=getTestbarcolor_dark()
    count=0
    while True:
        count+=1
        # rand=random.randint(0,len(bar_color_test)-1)
        rand=random.randint(0,len(bar_color)-1)
        # c = bar_color_test[rand]
        c = bar_color[rand]
        if count>count_times:
            print("jjj")
            return c
        if isinstance(cmp,list):
            flag=False
            for cm in cmp:
                # if _colorDiff(c,cm)<thres:
                if calcDeltaE(c,cm)<thres:
                    flag=True
                    break
            if flag:
                continue
            else:
                return c
        # elif _colorDiff(c,cmp)>thres:
        elif calcDeltaE(c,cmp)>thres:
            # print(calcDeltaE(c,cmp))
            return c
    return None

def linecolorDiff_bg_bar(cmp,barcolor,mode='color_pool',thres=2,count_times=100,strokecolor_pertubation=0):
    # line_color_test=getTestlinecolor()
    if mode=='bright':
        linecolor=getTestlinecolor_bright()
    elif mode=='dark':
        linecolor=getTestlinecolor_dark()
    elif mode=='black':
        return (0,0,0)
    else:
        # linecolor=line_color
        linecolor=getTeststrokecolor_perturbation(strokecolor_pertubation)
    count=0
    while True:
        count+=1
        # rand=random.randint(0,len(line_color_test)-1)
        rand=random.randint(0,len(linecolor)-1)
        # c = line_color_test[rand]
        c = linecolor[rand]
        if count>count_times:
            print("kkk")
            return c
        if isinstance(cmp,list):
            flag=False
            for cm in cmp:
                # if _colorDiff(c,cm)<thres:
                if calcDeltaE(c,cm)<thres:
                    flag=True
                    break
            if flag:
                continue
            else:
                return c
        # elif _colorDiff(c,cmp)>thres and _colorDiff(c,barcolor)>thres:
        elif calcDeltaE(c,cmp)>thres and calcDeltaE(c,barcolor)>thres:
            return c
    return None

def linecolorDiff_bg_bar_bright(cmp,barcolor,mode,thres=2,count_times=100):
    if mode=='bright':
        line_color_test=getTestlinecolor_bright()
    elif mode=='dark':
        line_color_test=getTestlinecolor_dark()
    elif mode=='black':
        return (0,0,0)
    count=0
    while True:
        count+=1
        rand=random.randint(0,len(line_color_test)-1)
        # rand=random.randint(0,len(line_color)-1)
        c = line_color_test[rand]
        # c = line_color[rand]
        if count>count_times:
            # print('jjj')
            return c
        if isinstance(cmp,list):
            flag=False
            for cm in cmp:
                # if _colorDiff(c,cm)<thres:
                if calcDeltaE(c,cm)<thres:
                    flag=True
                    break
            if flag:
                continue
            else:
                return c
        # elif _colorDiff(c,cmp)>thres and _colorDiff(c,barcolor)>thres:
        elif calcDeltaE(c,cmp)>thres and calcDeltaE(c,barcolor)>thres:
            return c
    return None

# def add_noise(img,sigma=20,kind='normal'):
#     if kind == 'normal':
#         # sigma=50
#         sigma=40
#     elif kind == 'lognormal':
#         sigma=10
#     elif kind == 'logistic':
#         sigma=20
#     # sigma=40
#     # print(kind)
#     i_shape = img.shape
#     sigma = float(sigma)
#     random_gennerator = getattr(np.random,kind)
#     noise = random_gennerator(size = i_shape)
#     # noise = np.random.normal(0,0.01,img.shape)
#     # noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise))
#     img = img.astype('float')
#     # img = np.where(img == 255,img - sigma*noise,img + sigma*noise)
#     img = np.where(img > 127, img - sigma * noise, img + sigma * noise)
#     return img.astype('int'), sigma*noise

# def add_noise(image,sigma=20,kind='normal'):

#     mean=0
#     var=0.01
#     image = np.array(image/255, dtype=float)
#     if kind=='normal':
#         noise = np.random.normal(mean, var ** 0.5, image.shape)
#     elif kind=='lognormal':
#         noise = np.random.lognormal(image.shape)
        
#     elif kind=='logistic':
#         noise = np.random.logistic(mean, var ** 0.5, image.shape) 
#     out = np.where(image > 0.5, image +  noise,0)
# #     out = image + noise
#     if out.min() < 0:
#         low_clip = -1.
#     else:
#         low_clip = 0.
#     out = np.clip(out, low_clip, 1.0)
#     out = np.uint8(out*255)
#     #cv.imshow("gasuss", out)
#     return out,''


def add_noise(image,sigma=10,kind='normal'):
    sigma=10
    mean=0
    var=sigma**2/255/255
    image = np.array(image/255, dtype=float)
    if kind=='normal':
        noise = np.random.normal(mean, var ** 0.5, image.shape)
    elif kind=='lognormal':
        noise = np.random.lognormal(mean, var ** 0.5,image.shape)
    elif kind=='logistic':
        noise = np.random.logistic(mean, var ** 0.5, image.shape) 
    out = np.where(image > 0.5, image +  noise,0)
#     out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out,''



def check_pos(pos, marked_bar_area):

    min_high = min(marked_bar_area[0][1][1], marked_bar_area[1][1][1])
    for area in marked_bar_area:
        if ((area[0][0] - 10) < pos[0] < (area[1][0] + 10)) and (pos[1] > min_high - 10):
            return False
    return True


def add_pattern(img, marked_bar_area, kind='rand'):
    if kind == 'rand':
        letters = list('abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*~<>?|/=+_-')
        letters_num = np.random.randint(1, 10)
        patterns = np.random.choice(letters, letters_num)

    pos = np.random.randint(0, 100, size=(letters_num, 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.UMat(img).get()
    
    for i, pattern in enumerate(patterns):
        if check_pos(pos[i], marked_bar_area):
            cv2.putText(img, pattern, tuple(pos[i]), font, 0.3, (0, 0, 0), 1)
    return img

def add_padding(image,backcolor):
    padding=25
    Imageim = Image.fromarray(np.uint8(image)).convert("RGB")
    width, height = Imageim.size
    bg = Image.new('RGB', (width+padding*2, height+padding*2), color=backcolor)
    bg.paste(Imageim, (padding, padding))
    return np.array(bg)

def add_title(image,backcolor,TitlePosition='mid',TitleFontSize=12,TitleFontType='arial',TitleLength=1,TitlePaddingLeft=0,TitlePaddingTop=0,TitleLetterCount=5,TitleLetter=False,TitleLineCount=1):
    width, height = image.size
    bg = Image.new('RGB', (width, height), color=backcolor)
    bg.paste(image, (0,0)) 
    draw = ImageDraw.Draw(bg)
    font_type="../util/%s.ttf"
    padding=20-TitlePaddingTop
    number=TitleLength
    title=''
    for ii in range(number):
        rw = RandomWords()
        word = rw.random_word()
        title+=word+' '
    title=title.strip()
    letters='abcdefghijklmnopqrstuvwxyz'
    if TitleLetter:
        title=''
        for letter in range(TitleLetterCount):
            title+=letters[random.randint(0,20)]
    # print(title)
    SimHei = font_type % (TitleFontType)

    font_title = ImageFont.truetype(SimHei, TitleFontSize)
    w, h = font_title.getsize(title)   #
    if TitlePosition=='left':
        padding_left=TitlePaddingLeft*image.size[1]
        # draw.text((padding_left, 0), title, fill="black", font=font_title)
        # draw.text((padding_left, 0+h//2), title, fill="black", font=font_title)
        if TitleLineCount==1:
            # draw.text((padding_left-w/2, (padding-h)//2), title, fill="black", font=font_title)
            draw.text((padding_left, (padding-h)//2), title, fill="black", font=font_title)
        elif TitleLineCount==2:
            draw.text((padding_left, -2), title, fill="black", font=font_title)
            draw.text((padding_left, h-6), title, fill="black", font=font_title)
        elif TitleLineCount==0:
            pass
        # draw.text((padding_left, (padding-h)//2), title, fill="black", font=font_title)
    elif TitlePosition=='right':
        draw.text((image.size[1]-w-10, (padding-h)//2), title, fill="black", font=font_title)
    else:
        # draw.text((padding_left, 0), title, fill="black", font=font_title)
        # draw.text((padding_left, 0+h//2), title, fill="black", font=font_title)
        draw.text(((image.size[1]-w)/2, (padding-h)//2), title, fill="black", font=font_title)
        # draw.text(((image.size[1]-w)/2, 0+h//2), title, fill="black", font=font_title)
    # draw.point((140,140), fill="black")
    # draw.point((145,145), fill="black")
    return bg

def add_axis(image,backcolor):
    padding=20
    img = np.ones(shape=(image.shape[0]+padding, image.shape[1]+padding, 3), dtype=np.int8) * 255
    img[:]=backcolor
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i][j+padding]=image[i][j]

    Imageim = Image.fromarray(np.uint8(img)).convert("RGB")
    draw = ImageDraw.Draw(Imageim)
    draw.line(((padding, 0), (padding, image.shape[0]-1)), fill='black')  # just here to create a visible box
    draw.line(((padding, image.shape[0]-1), (img.shape[1]-1, image.shape[0]-1)), fill='black') 
    # Draw x ticks
    [draw.line(((x,image.shape[0]),(x,image.shape[0]+5)),fill='black') for x in range(padding,img.shape[1],10)]
    # Draw x labels
    [draw.text((x,image.shape[0]+5),str(x-padding),fill='black') for x in range(padding,img.shape[1],50)]
    # Can do same for y...
    [draw.line(((padding,y),(padding-5,y)),fill='black') for y in range(0,image.shape[0],10)]
    # Draw y labels
    [draw.text((0,y),str(image.shape[0]-y),fill='black') for y in range(0,image.shape[0],50)]
    return np.array(Imageim)

def add_title_axis(image,backcolor,bar_mid):
    padding=20
    linethickness=1
    xtick_length=3
    ytick_length=5
    number=np.random.randint(1,2)
    title=''
    for ii in range(number):
        rw = RandomWords()
        word = rw.random_word()
        title+=word+' '
    title=title.strip()
    # img = Image.open("F:/graph_perception/test_result_3/dataset/posLen_tp_1_nonfixm_c_testdata_2/valid/input/bar_type1_0.png")
    img = np.ones(shape=(image.shape[0]+padding*2, image.shape[1]+padding*2, 3), dtype=np.int8) * 255
    img[:]=backcolor
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i+padding][j+padding]=image[i][j]

    Imageim = Image.fromarray(np.uint8(img)).convert("RGB")
    draw = ImageDraw.Draw(Imageim)
    # draw.rectangle([(0, 0), (109, 109)], outline='red') 
    # draw.rectangle([(padding, 0), (img.shape[1]-1, image.shape[0]-1)], outline='red')  # just here to create a visible box
    draw.line(((padding-1, padding-linethickness), (padding-1, image.shape[0]+padding-linethickness)), fill='black')  # y轴
    draw.line(((padding, image.shape[0]+padding-linethickness), (img.shape[1]-padding-linethickness, image.shape[0]+padding-linethickness)), fill='black') 
    # draw.rectangle([(10, 40), (100, 200)], fill='red', outline='red')
    SimHei = "../util/arial.ttf"    
    font = ImageFont.truetype(SimHei, 9)
    # Draw x ticks
    xxx=bar_mid
    xx=[i+linethickness/2 for i in xxx]
    xx_tick=[1,2,3,4,5,6,7,8,9,10]
    [draw.line(((x+padding,image.shape[0]+padding),(x+padding,image.shape[0]+padding+xtick_length)),fill='black') for x in xx]
    for i in range(len(xx)):
        x=xx[i]
        w, h = font.getsize(str(xx_tick[i]))
        draw.text((x+padding-w/2,image.shape[0]+padding+xtick_length),str(xx_tick[i]),fill='black',font=font)
    # [draw.line(((x,image.shape[0]+padding),(x,image.shape[0]+padding+xtick_length)),fill='black') for x in range(padding-1,image.shape[1]+padding+1,10)]
    # Draw x labels
    # for x in range(padding,img.shape[1]-padding+1,50):
    #     w, h = font.getsize(str(x-padding))
    #     draw.text((x-w/2,image.shape[0]+padding+xtick_length),str(x-padding),fill='black',font=font)
    # [draw.text((x,image.shape[0]+padding+xtick_length),str(x-padding),fill='black') for x in range(padding,img.shape[1]-padding+1,50)]
    # y ticks
    [draw.line(((padding-1,y),(padding-1-ytick_length,y)),fill='black') for y in range(padding-linethickness,image.shape[0]+padding+1,10)]
    # Draw y labels
    for y in range(0,image.shape[0]+1,50):
        w, h = font.getsize(str(image.shape[0]-y))
        draw.text((padding-ytick_length-w,y+padding-h/2-linethickness),str(image.shape[0]-y),fill='black',font=font)
    # [draw.text((0,y+padding),str(image.shape[0]-y),fill='black') for y in range(0,image.shape[0]+1,50)]
    font_title = ImageFont.truetype(SimHei, 12)
    w, h = font_title.getsize(title)   #
    # print(w)
    # print(Imageim.size)
    draw.text(((img.shape[1]-w)/2, 2), title, fill="black", font=font_title)

    array = np.array(Imageim)
    return array

'''
def add_title_and_axis(image,backcolor,bar_mid,tick_number,ymax_value,TitlePosition='mid'):
    tick_range_symbol=int(ymax_value/(tick_number-1))
    tick_range_true=int(100/(tick_number-1))
#     print(tick_range)
    padding=20
    linethickness=1
    xtick_length=3
    ytick_length=5
    number=np.random.randint(1,2)
    title=''
    for ii in range(number):
        rw = RandomWords()
        word = rw.random_word()
        title+=word+' '
    title=title.strip()
    # img = Image.open("F:/graph_perception/test_result_3/dataset/posLen_tp_1_nonfixm_c_testdata_2/valid/input/bar_type1_0.png")
    img = np.ones(shape=(image.shape[0]+padding*2, image.shape[1]+padding*2, 3), dtype=np.int8) * 255
    img[:]=backcolor
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i+padding][j+padding]=image[i][j]
    # pil_image 接收住这个图片对象
    # width 为图片的宽, height为图片的高
    Imageim = Image.fromarray(np.uint8(img)).convert("RGB")
    draw = ImageDraw.Draw(Imageim)
    # draw.rectangle([(0, 0), (109, 109)], outline='red') 
    # draw.rectangle([(padding, 0), (img.shape[1]-1, image.shape[0]-1)], outline='red')  # just here to create a visible box
    draw.line(((padding-1, padding-linethickness), (padding-1, image.shape[0]+padding-linethickness)), fill='black')  # y轴
    draw.line(((padding, image.shape[0]+padding-linethickness), (img.shape[1]-padding-linethickness, image.shape[0]+padding-linethickness)), fill='black') 
    # draw.rectangle([(10, 40), (100, 200)], fill='red', outline='red')
    SimHei = "C:/Windows/Fonts/arial.ttf"    # 一个字体文件
    font = ImageFont.truetype(SimHei, 9)  # 设置字体和大小
    # Draw x ticks
    xxx=bar_mid
    xx=[i+linethickness/2 for i in xxx]
    xx_tick=[1,2,3,4,5,6,7,8,9,10]
    [draw.line(((x+padding,image.shape[0]+padding),(x+padding,image.shape[0]+padding+xtick_length)),fill='black') for x in xx]
    for i in range(len(xx)):
        x=xx[i]
        w, h = font.getsize(str(xx_tick[i]))
        draw.text((x+padding-w/2,image.shape[0]+padding+xtick_length),str(xx_tick[i]),fill='black',font=font)
    # [draw.line(((x,image.shape[0]+padding),(x,image.shape[0]+padding+xtick_length)),fill='black') for x in range(padding-1,image.shape[1]+padding+1,10)]
    # Draw x labels
    # for x in range(padding,img.shape[1]-padding+1,50):
    #     w, h = font.getsize(str(x-padding))
    #     draw.text((x-w/2,image.shape[0]+padding+xtick_length),str(x-padding),fill='black',font=font)
    # [draw.text((x,image.shape[0]+padding+xtick_length),str(x-padding),fill='black') for x in range(padding,img.shape[1]-padding+1,50)]
    # y ticks
    [draw.line(((padding-1,y),(padding-1-ytick_length,y)),fill='black') for y in range(padding-linethickness,image.shape[0]+padding+1,tick_range_true)]
    # Draw y labels
    symbol_y=[y for y in range(0,ymax_value+1,tick_range_symbol)]
    true_y=[y for y in range(0,image.shape[0]+1,tick_range_true)]
    for i in range(len(true_y)):
        w, h = font.getsize(str(ymax_value-symbol_y[i]))
        draw.text((padding-ytick_length-w,true_y[i]+padding-h/2-linethickness),str(ymax_value-symbol_y[i]),fill='black',font=font)
#     for y in range(0,ymax_value+1,tick_range):
#         w, h = font.getsize(str(ymax_value-y))
#         print(str(ymax_value-y))
#         draw.text((padding-ytick_length-w,y+padding-h/2-linethickness),str(ymax_value-y),fill='black',font=font)
    # [draw.text((0,y+padding),str(image.shape[0]-y),fill='black') for y in range(0,image.shape[0]+1,50)]
    font_title = ImageFont.truetype(SimHei, 12)
    w, h = font_title.getsize(title)   #
    if TitlePosition=='else':
        choise=random.randint(0,1)
        if choise==1:
            draw.text((2, 2), title, fill="black", font=font_title)
    # elif TitlePosition=='right':
    #     print('kkkk',img.shape[1])
        else:
            draw.text((img.shape[1]-w-10, 2), title, fill="black", font=font_title)
    else:
        draw.text(((img.shape[1]-w)/2, 2), title, fill="black", font=font_title)
    # draw.text(((img.shape[1]-w)/2, 2), title, fill="black", font=font_title)
    type(Imageim)

    array = np.array(Imageim)
    return array


def add_title_and_axis_horizontal(image,backColor,bar_mid,tick_number=6,ymax_value=100,TitlePosition='mid'):
    
    image=np.rot90(image,3)
    
    tick_range_symbol=int(ymax_value/(tick_number-1))
    tick_range_true=int(100/(tick_number-1))
    padding=20
    linethickness=1
    xtick_length=3
    ytick_length=5
    number=np.random.randint(1,2)
    title=''
    for ii in range(number):
        rw = RandomWords()
        word = rw.random_word()
        title+=word+' '
    title=title.strip()
    # img = Image.open("F:/graph_perception/test_result_3/dataset/posLen_tp_1_nonfixm_c_testdata_2/valid/input/bar_type1_0.png")
    img = np.ones(shape=(image.shape[0]+padding*2, image.shape[1]+padding*2, 3), dtype=np.int8) * 255
    img[:]=backColor
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i+padding][j+padding]=image[i][j]
    # pil_image 接收住这个图片对象
    # width 为图片的宽, height为图片的高
    Imageim = Image.fromarray(np.uint8(img)).convert("RGB")
    draw = ImageDraw.Draw(Imageim)
    # draw.rectangle([(0, 0), (109, 109)], outline='red') 
    # draw.rectangle([(padding, 0), (img.shape[1]-1, image.shape[0]-1)], outline='red')  # just here to create a visible box
    draw.line(((padding, padding-linethickness), (padding, image.shape[0]+padding-linethickness)), fill='black')  # y轴
    draw.line(((padding, image.shape[0]+padding), (img.shape[1]-padding, image.shape[0]+padding)), fill='black') 
    # draw.rectangle([(10, 40), (100, 200)], fill='red', outline='red')

    SimHei = "C:/Windows/Fonts/arial.ttf"    # 一个字体文件
    font = ImageFont.truetype(SimHei, 9)  # 设置字体和大小
    # Draw x ticks
    xxx=bar_mid
    xx=[i+linethickness/2 for i in xxx]
    xx_tick=[1,2,3,4,5,6,7,8,9,10]
    # [draw.line(((padding-1,y),(padding-1-ytick_length,y)),fill='black') for y in range(padding-linethickness,image.shape[0]+padding+1,10)]
    [draw.line(((padding-1,x+padding),(padding-1-ytick_length,x+padding)),fill='black') for x in xx]
    for i in range(len(xx)):
        x=xx[i]
        w, h = font.getsize(str(xx_tick[i]))
    #     (padding-ytick_length-w,y+padding-h/2-linethickness)
        draw.text((padding-ytick_length-w,x+padding-h/2-linethickness),str(xx_tick[i]),fill='black',font=font)
    # [draw.line(((x,image.shape[0]+padding),(x,image.shape[0]+padding+xtick_length)),fill='black') for x in range(padding-1,image.shape[1]+padding+1,10)]
    # Draw x labels
    # for x in range(padding,img.shape[1]-padding+1,50):
    #     w, h = font.getsize(str(x-padding))
    #     draw.text((x-w/2,image.shape[0]+padding+xtick_length),str(x-padding),fill='black',font=font)
    # [draw.text((x,image.shape[0]+padding+xtick_length),str(x-padding),fill='black') for x in range(padding,img.shape[1]-padding+1,50)]
    # y ticks
    [draw.line(((y,image.shape[0]+padding),(y,image.shape[0]+padding+xtick_length)),fill='black') for y in range(padding,image.shape[0]+padding+1,tick_range_true)]
    # Draw y labels
    symbol_y=[y for y in range(0,ymax_value+1,tick_range_symbol)]
    true_y=[y for y in range(0,image.shape[0]+1,tick_range_true)]
    for i in range(len(true_y)):
        w, h = font.getsize(str(symbol_y[i]))
#         print(str(symbol_y[i]))
        draw.text((padding-w/2+true_y[i],image.shape[0]+padding+xtick_length),str(symbol_y[i]),fill='black',font=font)
#     for y in range(0,image.shape[0]+1,50):
#         w, h = font.getsize(str(y))
#         draw.text((padding-w/2+y,image.shape[0]+padding+xtick_length),str(y),fill='black',font=font)
    # [draw.text((0,y+padding),str(image.shape[0]-y),fill='black') for y in range(0,image.shape[0]+1,50)]

    font_title = ImageFont.truetype(SimHei, 12)
    w, h = font_title.getsize(title)   #
    if TitlePosition=='else':
        choise=random.randint(0,1)
        if choise==1:
            draw.text((2, 2), title, fill="black", font=font_title)
    # elif TitlePosition=='right':
    #     print('kkkk',img.shape[1])
        else:
            draw.text((img.shape[1]-w-10, 2), title, fill="black", font=font_title)
    else:
        draw.text(((img.shape[1]-w)/2, 2), title, fill="black", font=font_title)

    # draw.text(((img.shape[1]-w)/2, 2), title, fill="black", font=font_title)

    array = np.array(Imageim)
    return array
'''

def add_title_and_axis(image,backcolor,bar_mid,tick_number=5,ymax_value=100,TitlePosition='mid',TitleFontSize=12,TitleFontType='arial',x_tick_number='retain',Axis=True,ImagePadding=20):
    font_type="../util/%s.ttf"
    tick_range_symbol=int(ymax_value/(tick_number-1))
    tick_range_true=int(100/(tick_number-1))
#     print(tick_range)
    padding=ImagePadding
    linethickness=1
    xtick_length=3
    ytick_length=5
    # number=np.random.randint(1,2)
    # number=TitleLength
    # title=''
    # for ii in range(number):
    #     rw = RandomWords()
    #     word = rw.random_word()
    #     title+=word+' '
    # title=title.strip()
    # img = Image.open("F:/graph_perception/test_result_3/dataset/posLen_tp_1_nonfixm_c_testdata_2/valid/input/bar_type1_0.png")
#     img = np.ones(shape=(image.shape[0]+padding*2, image.shape[1]+padding*2, 3), dtype=np.int8) * 255
#     img[:]=backcolor
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             img[i+padding][j+padding]=image[i][j]
    
    Imageim=Image.new('RGB',(image.size[0]+padding*2, image.size[1]+padding*2),backcolor)
    Imageim.paste(image,(padding,padding))
    # pil_image 接收住这个图片对象
    # width 为图片的宽, height为图片的高
#     Imageim = Image.fromarray(np.uint8(img)).convert("RGB")
    if Axis:
        draw = ImageDraw.Draw(Imageim)
        # draw.rectangle([(0, 0), (109, 109)], outline='red') 
        # draw.rectangle([(padding, 0), (img.shape[1]-1, image.shape[0]-1)], outline='red')  # just here to create a visible box

        draw.line(((padding-1, padding-linethickness), (padding-1, image.size[0]+padding-linethickness)), fill='black')  # y轴
        draw.line(((padding, image.size[0]+padding-linethickness), (Imageim.size[1]-padding-linethickness, image.size[0]+padding-linethickness)), fill='black') 
        
        # draw.rectangle([(10, 40), (100, 200)], fill='red', outline='red')
        SimHei = "../util/arial.ttf"
        font = ImageFont.truetype(SimHei, 9)  
        # Draw x ticks
        xxx=bar_mid
        xx=[i+linethickness/2 for i in xxx]
        xx_tick=[1,2,3,4,5,6,7,8,9,10]
        for i in range(len(xx)):
            if x_tick_number=='odd_number':
                if i%2==0:
                    draw.line(((xx[i]+padding,image.size[0]+padding),(xx[i]+padding,image.size[0]+padding+xtick_length)),fill='black')
            elif x_tick_number=='even_number':
                if i%2!=0:
                    draw.line(((xx[i]+padding,image.size[0]+padding),(xx[i]+padding,image.size[0]+padding+xtick_length)),fill='black')
            else:
                draw.line(((xx[i]+padding,image.size[0]+padding),(xx[i]+padding,image.size[0]+padding+xtick_length)),fill='black')
                
        for i in range(len(xx)):
            if x_tick_number=='odd_number':
                if i%2==0:
                    x=xx[i]
                    w, h = font.getsize(str(xx_tick[i]))
                    draw.text((x+padding-w/2,image.size[0]+padding+xtick_length),str(xx_tick[i]),fill='black',font=font)
            elif x_tick_number=='even_number':
                if i%2!=0:
                    x=xx[i]
                    w, h = font.getsize(str(xx_tick[i]))
                    draw.text((x+padding-w/2,image.size[0]+padding+xtick_length),str(xx_tick[i]),fill='black',font=font)
            else:
                x=xx[i]
                w, h = font.getsize(str(xx_tick[i]))
                draw.text((x+padding-w/2,image.size[0]+padding+xtick_length),str(xx_tick[i]),fill='black',font=font)
        # [draw.line(((x+padding,image.size[0]+padding),(x+padding,image.size[0]+padding+xtick_length)),fill='black') for x in xx]
        # for i in range(len(xx)):
        #     x=xx[i]
        #     w, h = font.getsize(str(xx_tick[i]))
        #     draw.text((x+padding-w/2,image.size[0]+padding+xtick_length),str(xx_tick[i]),fill='black',font=font)
        # [draw.line(((x,image.shape[0]+padding),(x,image.shape[0]+padding+xtick_length)),fill='black') for x in range(padding-1,image.shape[1]+padding+1,10)]
        # Draw x labels
        # for x in range(padding,img.shape[1]-padding+1,50):
        #     w, h = font.getsize(str(x-padding))
        #     draw.text((x-w/2,image.shape[0]+padding+xtick_length),str(x-padding),fill='black',font=font)
        # [draw.text((x,image.shape[0]+padding+xtick_length),str(x-padding),fill='black') for x in range(padding,img.shape[1]-padding+1,50)]
        # y ticks
        [draw.line(((padding-1,y),(padding-1-ytick_length,y)),fill='black') for y in range(padding-linethickness,image.size[0]+padding+1,tick_range_true)]
        # Draw y labels
        symbol_y=[y for y in range(0,ymax_value+1,tick_range_symbol)]
        true_y=[y for y in range(0,image.size[0]+1,tick_range_true)]
        for i in range(len(true_y)):
            # if i != 0: 作用是去掉y轴上的 100 
            w, h = font.getsize(str(ymax_value-symbol_y[i]))
            draw.text((padding-ytick_length-w,true_y[i]+padding-h/2-linethickness),str(ymax_value-symbol_y[i]),fill='black',font=font)
#     for y in range(0,ymax_value+1,tick_range):
#         w, h = font.getsize(str(ymax_value-y))
#         print(str(ymax_value-y))
#         draw.text((padding-ytick_length-w,y+padding-h/2-linethickness),str(ymax_value-y),fill='black',font=font)
    # [draw.text((0,y+padding),str(image.shape[0]-y),fill='black') for y in range(0,image.shape[0]+1,50)]
    # SimHei = "F:/test/STXINGKA.TTF"    # 一个字体文件
    # SimHei = "C:/Windows/Fonts/arial.ttf" 
#     print(Imageim.size)
    # if TitlePosition=='left':
    #     draw.text((2, (padding-h)//2), title, fill="black", font=font_title)
    # elif TitlePosition=='right':
    #     draw.text((Imageim.size[1]-w-10, (padding-h)//2), title, fill="black", font=font_title)
    # else:
    #     draw.text(((Imageim.size[1]-w)/2, (padding-h)//2), title, fill="black", font=font_title)
#         draw.line(((0.1, (padding-h)//2),(140, (padding-h)//2)), fill="black")
#         draw.line(((0, (padding-h)//2+h),(140, (padding-h)//2+h)), fill="black")
#         draw.line(((0, padding),(140, padding)), fill="black")
    type(Imageim)
#     return array
    return Imageim

def add_title_and_axis_horizontal(image,backColor,bar_mid,tick_number=6,ymax_value=100,TitlePosition='mid',TitleFontSize=12,TitleFontType='arial'):
    font_type="../util/%s.ttf"
#     image=np.rot90(image,3)
    image=image.rotate(-90) 
    
    tick_range_symbol=int(ymax_value/(tick_number-1))
    tick_range_true=int(100/(tick_number-1))
    padding=20
    linethickness=1
    xtick_length=3
    ytick_length=5
    number=np.random.randint(1,2)
    title=''
    for ii in range(number):
        rw = RandomWords()
        word = rw.random_word()
        title+=word+' '
    title=title.strip()
    # img = Image.open("F:/graph_perception/test_result_3/dataset/posLen_tp_1_nonfixm_c_testdata_2/valid/input/bar_type1_0.png")
#     img = np.ones(shape=(image.shape[0]+padding*2, image.shape[1]+padding*2, 3), dtype=np.int8) * 255
#     img[:]=backColor

#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             img[i+padding][j+padding]=image[i][j]
    Imageim=Image.new('RGB',(image.size[0]+padding*2, image.size[1]+padding*2),backColor)
    Imageim.paste(image,(padding,padding))
    # pil_image 接收住这个图片对象
    # width 为图片的宽, height为图片的高
#     Imageim = Image.fromarray(np.uint8(img)).convert("RGB")
    draw = ImageDraw.Draw(Imageim)
    # draw.rectangle([(0, 0), (109, 109)], outline='red') 
    # draw.rectangle([(padding, 0), (img.shape[1]-1, image.shape[0]-1)], outline='red')  # just here to create a visible box
    draw.line(((padding, padding-linethickness), (padding, image.size[0]+padding-linethickness)), fill='black')  # y轴
    draw.line(((padding, image.size[0]+padding), (Imageim.size[1]-padding, image.size[0]+padding)), fill='black') 
    # draw.rectangle([(10, 40), (100, 200)], fill='red', outline='red')

    SimHei = "../util/arial.ttf"    # 一个字体文件
    font = ImageFont.truetype(SimHei, 9)  # 设置字体和大小
    # Draw x ticks
    xxx=bar_mid
    xx=[i+linethickness/2 for i in xxx]
    xx_tick=[1,2,3,4,5,6,7,8,9,10]
    # [draw.line(((padding-1,y),(padding-1-ytick_length,y)),fill='black') for y in range(padding-linethickness,image.shape[0]+padding+1,10)]
    [draw.line(((padding-1,x+padding),(padding-1-ytick_length,x+padding)),fill='black') for x in xx]
    for i in range(len(xx)):
        x=xx[i]
        w, h = font.getsize(str(xx_tick[i]))
    #     (padding-ytick_length-w,y+padding-h/2-linethickness)
        draw.text((padding-ytick_length-w,x+padding-h/2-linethickness),str(xx_tick[i]),fill='black',font=font)
    # [draw.line(((x,image.shape[0]+padding),(x,image.shape[0]+padding+xtick_length)),fill='black') for x in range(padding-1,image.shape[1]+padding+1,10)]
    # Draw x labels
    # for x in range(padding,img.shape[1]-padding+1,50):
    #     w, h = font.getsize(str(x-padding))
    #     draw.text((x-w/2,image.shape[0]+padding+xtick_length),str(x-padding),fill='black',font=font)
    # [draw.text((x,image.shape[0]+padding+xtick_length),str(x-padding),fill='black') for x in range(padding,img.shape[1]-padding+1,50)]
    # y ticks
    [draw.line(((y,image.size[0]+padding),(y,image.size[0]+padding+xtick_length)),fill='black') for y in range(padding,image.size[0]+padding+1,tick_range_true)]
    # Draw y labels
    symbol_y=[y for y in range(0,ymax_value+1,tick_range_symbol)]
    true_y=[y for y in range(0,image.size[0]+1,tick_range_true)]
    for i in range(len(true_y)):
        w, h = font.getsize(str(symbol_y[i]))
#         print(str(symbol_y[i]))
        draw.text((padding-w/2+true_y[i],image.size[0]+padding+xtick_length),str(symbol_y[i]),fill='black',font=font)
#     for y in range(0,image.shape[0]+1,50):
#         w, h = font.getsize(str(y))
#         draw.text((padding-w/2+y,image.shape[0]+padding+xtick_length),str(y),fill='black',font=font)
    # [draw.text((0,y+padding),str(image.shape[0]-y),fill='black') for y in range(0,image.shape[0]+1,50)]

    # SimHei = font_type % (TitleFontType) 

    # font_title = ImageFont.truetype(SimHei, TitleFontSize)

    # if TitlePosition=='left':
    #     draw.text((2, (padding-h)//2), title, fill="black", font=font_title)
    # elif TitlePosition=='right':
    #     draw.text((Imageim.size[1]-w-10, (padding-h)//2), title, fill="black", font=font_title)
    # else:
    #     draw.text(((Imageim.size[1]-w)/2, (padding-h)//2), title, fill="black", font=font_title)
#     draw.text(((img.shape[1]-w)/2, 2), title, fill="black", font=font_title)

#     array = np.array(Imageim)
#     return array
    return Imageim

def add_label(image,backcolor,Axislabel='both',XlabelDelta=0,YlabelDelta=0,YlabelPaddingLeft=0,XlabelPaddingTop=0):
    X=True
    Y=True
    if Axislabel=='both':
        X=True
        Y=True
    elif Axislabel=='rx':
        X=False
        Y=True
    elif Axislabel=='ry':
        X=True
        Y=False
    else:
        X=False
        Y=False
    
    img=Image.new('RGB',(150,150),backcolor)
#     img1=Image.fromarray(np.uint8(image)).convert("RGB")
    ylabel_space=15
    #     img1=Image.open('F:/test/demo6.jpg')
    img.paste(image,(ylabel_space,0))
    rw = RandomWords()
    SimHei = "../util/arial.ttf"    
    font_label = ImageFont.truetype(SimHei, 9)

    deltaX=XlabelDelta
    deltaY=YlabelDelta
    
    if X:
        xlabel = rw.random_word()
    #     xlabel='title'
        w, h = font_label.getsize(xlabel)   #
        draw = ImageDraw.Draw(img)
        # draw.text(((img.size[0]-w)/2+5,img.size[0]-15), xlabel, fill="black", font=font_title)
        draw.text((80-w/2+5+deltaX,img.size[0]-15), xlabel, fill="black", font=font_label)
    
    if Y:
        image1 = img.rotate(-90)
        ylabel=rw.random_word()
        w, h = font_label.getsize(ylabel)   #

        draw = ImageDraw.Draw(image1)
        draw.text(((image1.size[0]-w)/2+deltaY,3+YlabelPaddingLeft), ylabel, fill="black", font=font_label)
        img=image1.rotate(90)
    return img

def add_legend(image,backcolor,colorLists):
    img=Image.new('RGB',(170,170),backcolor)
    img.paste(image,(10,0))
    draw = ImageDraw.Draw(img)
    colorLists=list(set(colorLists))
    legend_number=len(colorLists)
    number=1
    padding=2
    rec_width=10
    gap=5
    title_list=[]
    length_list=[0]
    for i in range(legend_number):
        title=''
        for ii in range(number):
            rw = RandomWords()
            word = rw.random_word()
            title+=word+' '
        title=title.strip()
        title_list.append(title)
        SimHei = "../util/arial.TTF"    # 一个字体文件
        font_title = ImageFont.truetype(SimHei, 10)
        w, h = font_title.getsize(title)   #
        length_list.append(w+padding*2+rec_width+gap)
        
    img1=Image.new('RGB',(sum(length_list),20),backcolor)
    # print(img1.size)
    draw = ImageDraw.Draw(img1)
    for i in range(legend_number):
#         draw.rectangle(((padding,(20-rec_width)//2),(padding+rec_width,(20-rec_width)//2+rec_width)),fill=(255,0,0),width=1)
#         draw.text((padding+rec_width+gap,(20-h)//2), title, fill="black", font=font_title)
        draw.rectangle(((padding+sum(length_list[0:i+1]),(20-rec_width)//2),(padding+sum(length_list[0:i+1])+rec_width,(20-rec_width)//2+rec_width)),fill=colorLists[i],width=1)
        draw.text((padding+sum(length_list[0:i+1])+rec_width+gap,(20-h)//2), title_list[i], fill="black", font=font_title)
    img.paste(img1,(int(170-img1.size[0])//2,150))
#     draw.rectangle(((10,150),(20,160)),fill=(255,0,0),width=1)
    return img

def add_pie_legend(image,backcolor,colorLists):
    Imageim = Image.fromarray(np.uint8(image)).convert("RGB")
    width, height = Imageim.size
    bg = Image.new('RGB', (width, height), color=backcolor)
    bg.paste(Imageim, (0,0))  
    draw = ImageDraw.Draw(bg)
    colorLists=list(set(colorLists))
    legend_number=len(colorLists)
    number=1
    padding=2
    rec_height=10
    gap=5
    title_list=[]
    length_list=[0]
    width_list=[]
    for i in range(legend_number):
        title=''
        for ii in range(number):
            rw = RandomWords()
            word = rw.random_word()
            title+=word+' '
        title=title.strip()
        title_list.append(title[0])
        SimHei = "../util/arial.ttf"    
        font_title = ImageFont.truetype(SimHei, 10)
        w, h = font_title.getsize(title[0])   #
        length_list.append(h+padding*2+gap)
        width_list.append(w+padding*2+gap+rec_height)
        
    img1=Image.new('RGB',(max(width_list),sum(length_list)),backcolor)
    draw = ImageDraw.Draw(img1)
    for i in range(legend_number):
        draw.rectangle(((padding,padding+sum(length_list[0:i+1])),(padding+rec_height,padding+rec_height+sum(length_list[0:i+1]))),fill=colorLists[i],width=1)
        draw.text((padding+rec_height+gap,padding+sum(length_list[0:i+1])), title_list[i], fill="black", font=font_title)
    bg.paste(img1,(bg.size[0]-img1.size[0]-5,(bg.size[0]-img1.size[1])//2))
    return np.array(bg)

'''
def add_label(image,backcolor,Axislabel='both'):

    X=True
    Y=True
    if Axislabel=='both':
        X=True
        Y=True
    # elif Axislabel=='rx':
    #     X=False
    #     Y=True
    # elif Axislabel=='ry':
    #     X=True
    #     Y=False
    else:
        choise=random.randint(0,1)
        if choise==0:
            X=False
            Y=True
        else:
            X=True
            Y=False            

    img=Image.new('RGB',(150,150),backcolor)
    img1=Image.fromarray(np.uint8(image)).convert("RGB")
    ylabel_space=15
#     img1=Image.open('F:/test/demo6.jpg')
    img.paste(img1,(ylabel_space,0))
    SimHei = "C:/Windows/Fonts/arial.ttf"    # 一个字体文件
    font_label = ImageFont.truetype(SimHei, 9)
    rw = RandomWords()
    if X:
        xlabel = rw.random_word()
    #     xlabel='title'
        w, h = font_label.getsize(xlabel)   #
        # print(w,h)

        draw = ImageDraw.Draw(img)
        # draw.text(((img.size[0]-w)/2+5,img.size[0]-15), xlabel, fill="black", font=font_title)
        draw.text((80-w/2+5,img.size[0]-15), xlabel, fill="black", font=font_label)
    if Y:
        image1 = img.rotate(-90)
        ylabel=rw.random_word()
        w, h = font_label.getsize(ylabel)   #
        # print(w,h)

        draw = ImageDraw.Draw(image1)
        draw.text(((image1.size[0]-w)/2,3), ylabel, fill="black", font=font_label)
        img=image1.rotate(90)
    return np.array(img)
'''    

def RGB2BGR(attr):
    if isinstance(attr,list):
        colors=[]
        for color in attr:
            colors.append((color[2],color[1],color[0]))
        return colors
    elif isinstance(attr,tuple):
        return (attr[2],attr[1],attr[0])
    
    
def gauss(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8
    
    s = sigma**2
    sum_val =  0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += kernel[i, j]
    
    kernel = kernel/sum_val
    
    return kernel


M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])
 
 
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931
 
 
def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
 

def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB = np.array([gamma(c) for c in rgb])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)
 
 
def __xyz2lab__(xyz):

    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return [L, a, b]
 
 
def RGB2Lab(pixel):

    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab
 
 
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0
 
    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)
 
    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883
 
    return (x, y, z)
 
 
def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return tuple(rgb)
 
 
def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb
