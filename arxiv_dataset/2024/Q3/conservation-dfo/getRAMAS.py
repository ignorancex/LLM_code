#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getRAMAS.py

Using the current reserve configuration, this code gets the necessary RAMAS files
from getMaps.R (by calling batch file getR.BAT) then invokes RAMAS and outputs 
the selected PVA metrics. 

There are a couple cases we've encountered when RAMAS wouldn't output PVA metrics:
    - no patches found (habitat threshold too high; no habitable parcels)
    - too many patches (this occurs typically with large problem sizes and 
                        we mitigate it by increasing the habitat threshold)
In these cases, we return a "bad" output so it has a low ranking in solutions. 

We only select a couple metrics for our model, but RAMAS will output 14 text files
Detailed descriptions are found in the README

    Parameters
    ----------
    n : int
        Problem size.
    config : Array of float64 (n*n,1)
        Current configuration X.
    i : int
        Current iteration.
    tmax : int
        Total duration of simulated timesteps in RAMAS.
        
    Returns
    -------
    risk : 
        PVA metrics. 


"""

import numpy as np
import subprocess
import os
from os import path
import re


def getObj(n,config,i,tmax):
    """ runs RAMAS to get the PVA metrics 
    Parameters
    ----------
    n : int
        Problem size.
    config : Array of float64 (n*n,1)
        Current configuration X.
    i : int
        Current iteration.
    tmax : int
        Total duration of simulated timesteps in RAMAS.
        
    Returns
    -------
    risk : 
        PVA metrics. If none are found, we return a "bad" output to not break code

    """
    #duration = 500
    basePath = os.getcwd()

    #creating directory for solutions from this iteration 
    iterPath = path.join(basePath,"data","n"+str(n),"iter"+str(i))
    if not os.path.exists(iterPath):
       os.makedirs(iterPath)
    
    #Save current configuration solution 
    fileName = "X"+str(n)+"_"+str(i)+".txt"
    np.savetxt(iterPath+"./"+fileName, config, delimiter=',',fmt = "%d")

    ## call r code to get .asc,.ptc, and .pdy files for X. 

    cmd_rcode = "runR.bat "+str(n) + " "+str(i)+" "+fileName
    subprocess.run(cmd_rcode, cwd = basePath, shell=True)
    
    #RUN SPATIAL DATA AND HABITAT DYNAMICS 
    subprocess.run("batch"+str(n)+"_"+str(i)+".BAT", cwd = iterPath, shell=True)
    
    mpFile = iterPath+"\\"+"Z"+str(n)+"_"+str(i)+".mp"
    if not os.path.isfile(mpFile):
        # if file doesn't exist, habitat dynamics didn't output an mp file. 
        print("No MP file was found.")
        return badProblem()
    

    outputMetaPath = path.join(iterPath,"output")
    
    #getting r results 
    risk = getResults(outputMetaPath,tmax)
    
    return risk

def getResults(outputMetaPath,tmax):
    """Getting using regex in the files saved in the directory

    Parameters
    ----------
    outputMetaPath : path
        Path to the output files.
    tmax : int
        Total duration of simulated timesteps in RAMAS.

    Returns
    -------
    risk : 
        PVA metrics. If none are found, we return a "bad" output to not break code

    """
    if not os.path.exists(outputMetaPath):
       os.makedirs(outputMetaPath)
    
    if not os.path.isfile(outputMetaPath+"\\"+"LocExtDur.txt"):
        # if there's no output, there are likely no patches found. 
        print("No metapop output files found.")
        return badProblem()
    
    # with open(outputMetaPath+"\\"+"Abund.txt",'r') as f:
    #     Abund = f.readlines()

    with open(outputMetaPath+"\\"+"IntExtRisk.txt",'r') as f:
        IntExtRisk = f.readlines()
        
    with open(outputMetaPath+"\\"+"TerExtRisk.txt",'r') as f:
        TerExtRisk = f.readlines()
        
    with open(outputMetaPath+"\\"+"QuasiExt.txt",'r') as f:
        QuasiExt = f.readlines()
        
    #medianFinalAbund = Abund[119].split()[3]
    #numPopulations = TerExtRisk[7].split()[0]
    riskTotExt  = TerExtRisk[14].split()[1] if TerExtRisk[14].split()[0] == "0" else 0
    # avgLocExtDur = [pop.split()[5] for pop in LocExtDur[15:]]
    medianQuasiExt = QuasiExt[13]
    expectedMinAbund = IntExtRisk[13].split()[-1]
    findNum = re.compile("(\d+(?:\.\d+)?)")
    findSign = re.compile("(=)")
    if findSign.search(QuasiExt[13]) is not None:
        medianQuasiExt = findNum.search(QuasiExt[13]).group()
    else:
        medianQuasiExt =  tmax+1 # max number of timesteps 
    print('Risk to Total Extinction: '+str(riskTotExt))
    # print('Average Local Extinction Duration: '+str(avgLocExtDur))
    print('Median quasi extinction: '+str(medianQuasiExt))
    print('Expected min abundance: '+str(expectedMinAbund))
    #print('Num Populations: '+str(numPopulations))
    #print('Median final abundance: '+str(medianFinalAbund))
    # risk = np.array([float(riskTotExt), -float(medianQuasiExt),-float(expectedMinAbund)])
    return float(riskTotExt), float(medianQuasiExt),float(expectedMinAbund)

def badProblem():
    """
    if no patches can be found, return this "bad" solution
    """
    print('BAD PROBLEM')
    riskTotExt = 1 #make this bad 
    medianQuasiExt = 0 # make this bad
    expectedMinAbund = 0 # make bad 
    #numPop = 10000 #make bad
    #medianFinalAbund = 0
    return float(riskTotExt), float(medianQuasiExt),float(expectedMinAbund)