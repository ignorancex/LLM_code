# -*- coding: utf-8 -*-
"""
driver_.py

Driver code that calls the optimization models and writes results to log file.

Parameters
----------
n : int
    nxn landscape size 
cost : Array of int32
    Cost of each parcel in landscape.
tmax : int
    Total duration of simulated timesteps in RAMAS.
numGen : int
    Number of generations for ACO
modelType : str 
    "con" for constrained model and "uncon" for multiobjective model
"""

from constrainedModel_ import optimizeC, writeLogC
from unconstrainedModel_ import optimizeU, writeLogU

import numpy as np
import random
from timeit import default_timer as timer


random.seed(0)

n = 10 #nxn landscape size 
cost =  np.array([random.randint(1,10) for j in range(n*n)]) #cost for acquiring parcel
tmax = 100 #max number of timesteps in simulation 
numGen = 4 #number of ACO generations 
# constrained (con) or unconstrained/multiobjective (uncon)
modelType = "con" 

if modelType == "con":
    startCode = timer()
    my_prob, pop, algo, uda,B_met = optimizeC(n,cost,tmax,numGen) #run constrained model
    endCode = timer()
    totalTime = endCode-startCode #record time 
    writeLogC(n,cost,pop,B_met,tmax,uda,totalTime) #write output 
elif modelType == "uncon": 
    startCode = timer()
    my_prob, pop, algo, uda,B_met = optimizeU(n,cost,tmax,numGen) #run multiobjective model
    endCode = timer()
    totalTime = endCode-startCode #record time 
    writeLogU(n,cost,pop,B_met,tmax,uda,totalTime,my_prob) #write output 
else:
    raise ValueError("Input a valid model type")





