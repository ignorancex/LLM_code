# -*- coding: utf-8 -*-
"""
unconstrainedModel_.py 

This code consists of all the functions necessary for solving a multi-objective
optimization problem with pygmo.  

getRAMAS is called in each iteration to obtain the PVA values and solves the problem:

Minimize Weighted Objective lambda*f(X)
s.t. 
Z = B * X
X is binary

where f(X) = [c^T*X (cost), risk(Z), time(Z), abundance(Z)]

In the paper, this is equivalent to equations (8).
"""

import numpy as np
import os
from os import path
import pygmo as pg
import re
from getRAMAS import getObj, getResults
from timeit import default_timer as timer
import subprocess




class unconModel:
    """
    User-defined problem using pygmo. 
    """
    def __init__(self, n, cost,B_met,tmax):
        """
        Parameters
        ----------
        n : int
            problem size.
        cost : Array of int32
            Cost of each parcel in landscape.
        B_met : Array of float64 size (3,1)
            Risk, time, and abundance for the base map B. Treating this as RHS for constraints.

        Returns
        -------
        None.

        """
        self.dim = n*n
        self.cost = cost
        self.B_met = B_met
        self.n = n
        self.tmax = tmax
        
    def fitness(self,x):
        """pygmo fitness function. See their documentation for details.
        
        Parameters
        ----------
        x : Array of float64 size (n*n,1)
            Current configuration of X.

        Returns
        -------
        list of objective value and constraints.

        """
        start = timer() #timing each iteration 
        self.x = x      
        i = getIter() #getting creative to track iterations so we can save each map
        i = int(i)
        risk,time,minAbund= getObj(self.n,x,i,self.tmax) #calling RAMAS
        obj1 =  sum(self.cost*x) # cost of preserving configuration X
        obj2 = risk  # minimize risk
        obj3 = -time # maximize time 
        obj4 = -minAbund #maximize min abundance 

        end = timer()
        totalTime = end-start #duration of iteration

        #writing this to text file 
        saveIterationOutput(obj1,obj2,obj3,obj4,self.n, i,totalTime,x)    
        return [obj1,obj2,obj3,obj4]

    def get_bounds(self):
        # lower and upper bounds for decision variables 
        return ([0]*self.dim,[1]*self.dim)
    def get_nic(self):
        # number of inequality constraints 
        return 0
    def get_nec(self):
        # number of equality constraints 
        return 0
    def get_nobj(self):
        # number of objectives 
        return 4
    def get_nix(self):
        # number of integer decision variable 
        return self.dim
    def _gradient(self, x):
        #estimate to gradient 
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

def getIter(i=[0]):  
    """Tracks the iteration count. Pygmo doesn't allow you to access its current iteration, hence this workaround. 

    Parameters
    ----------
    i :  optional
        Mutable variable gets evaluated once. The default is [0].

    Returns
    -------
        Number of times the function has been called (aka current iteration count).
    """
    i[0]+=1 # mutable variable get evaluated ONCE
    return int(i[0])


def saveIterationOutput(obj1,obj2,obj3,obj4,n, i,time,x):
    """Writes the objective/constraints for an iteration to a file "output.txt"

    Parameters
    ----------
    obj1 : float
        The value of 1st objective function.
    obj2 : float
        The value of 2nd objective function.
    obj3 : float
        The value of 3rd objective function.
    obj4 : float
        The value of 4th objective function.
    n : int
        Problem size. 
    i : int
        Current iteration.
    time : float
        Elapsed time to solve the problem in that iteration.
    x : Array of float64 of size (n*n,1)
        Current configuration

    Returns
    -------
    None.

    """

    log = ["\nIteration: "+str(i) + "\n",
     "x: "+ str(x)+"\n",
     "Cost: "+ str(obj1) +"\n",
     "Risk of Total Extinction: " + str(obj2) +"\n",
     "Median Time To Quasi Extinction: " +str(-obj3) +"\n",
     "Expected Minimum Abundance: "+str(-obj4)+"\n",
     "Elapsed Time: "+ str(time)+"\n\n"]
    print("".join(log)) #print to console as well 
    print("-"*30)
    path = "./data/"+"n"+str(n)+"/output"+"_n"+str(n)+"_uncon"+".txt"

    if i == 1:
        #for first iteration we need to write to file
        with open(path,"w") as out:
            out.write('{:*^90}'.format(str(i)))
            out.writelines(log)
            out.write("\n\n")
    else:
        #following iterations we can append to existing file 
        with open(path,"a") as out:
            out.write('{:*^90}'.format(str(i)))
            out.writelines(log)
            out.write("\n\n")
    return 


def getBMetrics(n,tmax):
    """Getting metrics from basemap B. This is used a baseline comparison to the X^* we obtain. 
    This is simply the scenario if we preserved everything. 

    Parameters
    ----------
    n : int
        Problem size. 
    tmax : int
        Total duration of simulated timesteps in RAMAS.

    Returns
    -------
    B_met : Array of float64 size (3,1)
        Risk, time, and abundance for the base map B.

    """
    #replace with your path 
    basePath = os.getcwd()
    nPath = path.join(basePath,"data","n"+str(n))
    fileName = "."
    cmd_rcode = "runR.bat "+str(n) + " "+str(0)+" "+fileName
    subprocess.run(cmd_rcode, cwd = basePath, shell=True)

    subprocess.run("batch"+str(n)+".BAT", cwd = nPath, shell=True)

    B_met = getResults(nPath,tmax) #getting metrics for this configuration 
    return B_met # (riskTotExt, medianQuasiExt, expectedMinAbund)

def optimizeU(n,cost,tmax,numGen):
    """ Calling pygmo to optimize code.
    

    Parameters
    ----------
    n : int
        problem size.
    cost : Array of int32
        Cost of each parcel in landscape.
    tmax : int
        Total duration of simulated timesteps in RAMAS.
    numGen : int
        Number of generations for ACO

    Returns
    -------
    my_prob : struct problem object of pygmo.core module 
        User defined problem.
    pop : struct population object of pygmo.core module 
        User defined population.
    algo : struct algorithm object of pygmo.core module 
        User defined algorithm.
    uda : struct 
        Output of algorithm, used to get log.
    B_met : Array of float64 size (3,1)
        Risk, time, and abundance for the base map B.

    """

    weights = [0.35,0.15,0.15,0.35] #weighig abundance and cost the highest since risk and time seem to be easier to satisfy

    B_met = getBMetrics(n,tmax) #get metrics if we preserve everything 
    my_prob = pg.problem(unconModel(n,cost,B_met,tmax)) #creating problem
    unconModel.gradient =unconModel._gradient #setting gradient 
    uncon_prob = pg.decompose(prob = my_prob, method = "weighted", weight = weights,z = [sum(cost)/3,B_met[0],B_met[1],B_met[2]])
    pop = pg.population(prob = uncon_prob,size = n*n+1,seed =0) #setting population
    # oracle param -> high if the solutions don't provide any feasible ones
    # kernal -> number of saved solutions
    # evalstop -> count the # of function evaluations without improvement, 
    algo = pg.algorithm(pg.gaco(gen = numGen,  ker = n,seed = 0,impstop = n*n+1,oracle = 1e6)) # ant colony opt
    algo.set_verbosity(1) #print log every generation 
    pop = algo.evolve(pop) 
    uda = algo.extract(pg.gaco)
    uda.get_log() 
    return my_prob, pop, algo, uda,B_met


def getConfigInd(x,n):
    """Finds the iteration of a given configuration by checking the log file output.txt

    Parameters
    ----------
    x : Array of float64 (n*n,1)
        Configuration.
    n : int
        problem size.

    Returns
    -------
    ind : int
        The index which had configuration x. 

    """
    outPath = "./data/"+"n"+str(n)+"/output"+"_n"+str(n)+"_uncon"+".txt"

    with open(outPath,'r') as f:
        log = f.read()
    x = str(x)[1:-1]
    pattern = "\d+(?=\n+x\W\W\W"+x+")" 
    ind = re.compile(pattern).search(log).group()
    return ind


def writeLogU(n,cost,pop,B_met,tmax,uda,totalTime, my_prob):
    """Writing log file, adding final results 
    
    Parameters
    ----------
    n : int
        Problem size.
    cost : Array of int32
        Cost of each parcel in landscape.
    pop : struct population object of pygmo.core module 
        User defined population.
    B_met : Array of float64 size (3,1)
        Risk, time, and abundance for the base map B.
    tmax : int
        Total duration of simulated timesteps in RAMAS.
    uda : struct 
        Output of algorithm, used to get log.
    totalTime : float
        Duration of running entire code.

    Returns
    -------
    None.

    """
    # find the indices of the optimal solutions 
    indexChamp = getConfigInd(pop.champion_x,n)
    index = getConfigInd(pop.get_x()[pop.best_idx()],n)
    weights = [0.35, 0.15, 0.15, 0.35]
    #getting the objectives    
    d = pg.decompose(prob = my_prob, method = "weighted", weight = weights,z = [sum(cost)/3,B_met[0],B_met[1],B_met[2]])

    costZ, riskZ, timeZ, abundZ= d.original_fitness(pop.champion_x)
    timeZ = -timeZ
    abundZ = -abundZ
    timeZ = ">"+str(tmax) if timeZ ==  tmax+1 else timeZ #if does not go extinct in tmax
    obj = pop.get_f()[pop.best_idx()]
    
    timeX = ">"+str(tmax) if B_met[1] == tmax+1 else B_met[1]
    costX = sum(cost*np.ones((n*n))) 
    
    #writing final results to logfile  
    outPath = "./data/"+"n"+str(n)+"/output"+"_n"+str(n)+"_uncon"+".txt"

    #outPath = "./data/"+"n"+str(n)+"/output.txt"
    with open(outPath,"a") as out:
        out.write('{:*^90}'.format('Population'))
        out.write("\n\n")
        out.write(str(pop))
        out.write("\n\n")
        out.write('{:*^90}'.format('Best Individual In Population'))
        out.write("\n\nPopulation Index: "+str(pop.best_idx())+"\n")
        out.write("Iteration Index: " + str(index)+"\n")
        out.write("Decision vector: "+str(pop.get_x()[pop.best_idx()])+"\n")
        out.write("Fitness vector: "+str(pop.get_f()[pop.best_idx()])+"\n\n")
        out.write('{:*^90}'.format('Best Individual That Ever Lived In Population (Champion)'))    
        out.write("\n\nIteration Index of Champion X*: " + str(indexChamp)+"\n")
        out.write("Decision vector: "+str(pop.champion_x)+"\n")
        out.write("Fitness vector: "+str(pop.champion_f)+"\n\n")
        out.write('{:*^90}'.format('LOG')+"\n")
        # assume that your data rows are tuples
        template = "{0:<10}|{1:<10}|{2:<10}|{3:<10}|{4:<12}|{5:<10}|{6:<10}\n" 
        out.write(template.format("Gen","Fevals","Best","Kernel","Oracle","dx","dp"))
        for log in uda.get_log():
            out.write(template.format(*log))
        out.write('{:*^90}'.format('END')+"\n")
        out.write("\n\nMetrics for Preserving Everything (B):\nCost: "+str(costX)+
                  "\nRisk of Total Extinction: " + str(B_met[0])+
                  "\nMedian Time to Quasi Extinction: " +str(timeX)+
                  "\nExpected Minimum Abundance: "+str(B_met[2])+
                  "\n\n")
        out.write("Metrics for Preserving Configuration X* (Z):\nCost: "+str(costZ)+ 
                  "\nRisk of Total Extinction: " + str(riskZ)+
                  "\nMedian Time to Quasi Extinction: " +str(timeZ)+
                  "\nExpected Minimum Abundance: "+str(abundZ)+
                  "\nIteration Index of Champion X*: " + str(indexChamp)+
                  "\n\n")
        out.write("\nTotal Elasped Time: "+ str(totalTime)+"\n")
    
    #printing just this to  console 
    print("Metrics for Preserving Everything (B):\nCost: "+str(costX)+
          "\nRisk of Total Extinction: " + str(B_met[0])+
          "\nMedian Time to Quasi Extinction: " +str(timeX)+
          "\nExpected Minimum Abundance: "+str(B_met[2])+
          "\n\n")
    print("Metrics for Preserving Configuration X* (Z):\nCost: "+str(costZ)+
          "\nRisk of Total Extinction: " + str(riskZ)+
          "\nMedian Time to Quasi Extinction: " +str(timeZ)+
          "\nExpected Minimum Abundance: "+str(abundZ)+
          "\nIteration Index of Champion X*: " + str(indexChamp)+
          "\n\n")
    print("Z^* Obj: "+str(obj))
    print("\nTotal Elasped Time: "+ str(totalTime))
    return