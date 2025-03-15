import os
import time
import threading
import json
import pickle
import logging
import numpy as np
import random

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
            #logging.trace("Return default Value ")
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

        