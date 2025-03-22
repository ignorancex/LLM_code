import time
import random
from multiprocessing import Process
import logging

import torch
import random
import numpy as np
class ForRunImpl(Process):
    def __init__(self,iter,func,threadIndex):
        super(ForRunImpl,self).__init__()
        self.iter=iter
        self.func=func
        self.__lastTime= time.time()
        self.curTime = time.time()
        self.threadIndex = threadIndex

    def run(self):
        print("Run Subprocess %s"%str(self.iter))
        maxv=len(self.iter)
        curv=0
        for i in self.iter:
            self.func(i)
            curv+=1
            curTime = time.time()
            if curTime-self.__lastTime<10 and curv!=maxv:
                continue
            print("[%5d] MutiProc > %d | %d ( %6.2f%% )"%(self.threadIndex,curv,maxv,100.0*curv/maxv))
            self.__lastTime = time.time()
        print("Complete Subprocess %s"%str(self.iter))

def setSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

__initThreadCount=0
def runParallelFor(func,from_i,to_i,threadCount,seed=0):
    global __initThreadCount
    procs= []
    logging.info("Init process %d"%threadCount)
    for t in range(threadCount):
        iter = range(from_i+t,to_i,threadCount)
        setSeed(seed+__initThreadCount) # to avoid to generate same dataset with the same seed
        proc = ForRunImpl(iter,func,__initThreadCount)
        __initThreadCount+=1
        proc.start()
        procs.append(proc)
    return procs

def waitProcs(procs):
    logging.info("Wait process %d"%len(procs))
    for proc in procs:
        proc.join()
    logging.info("Wait complete")
