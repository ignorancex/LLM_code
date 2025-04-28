from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core_2 import Init, Iterate,  Observe, MeasureMutants, Plot, GraphStats, Escape_Absorbing

import time

import copy
from math import floor
import random
random.seed(1)

starttime = time.time()

#################################
###Argparse
#################################
from argparse import ArgumentParser
parser = ArgumentParser(description='Different F and Num')
parser.add_argument(
        '-C',
        '--C',
        type=float,
        required=True,
        help='The radius within which nodes are connected')


parser.add_argument(
        '-P',
        '--P',
        type=float,
        required=True,
        help='Proportion of Zealots')


parser.add_argument(
        '-F',
        '--F',
        type=float,
        required=True,
        help='M Fitness')

parser.add_argument(
        '-d',
        '--dir',
        type=str,
        required=True,
        help='directory save data in')



args = parser.parse_args()

N = n
C = float(args.C)
P = float(args.P)
z = P
F = float(args.F)

Z = int(z*N)

SaveDirName = str(args.dir)


SubSaveDirName = (SaveDirName +
    "/C_%0.3f"%(C))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for C",C)

print("Starting C: %0.3f"%(C))

#############################################################
#############################################################
#############################################################



#starttime = time.time()

HistoryMatrix = []
HistMatrix = []

#Repeated iterations:
for r in range(Repeats):
    #List denoting the opinion held:
        #0 - Strong
        #1 - Weak
        #2 - Zealot
    GraphList = np.zeros(N).astype(int)

    #Set zealots
    zealotsample = random.sample(range(0,N),k=Z)
    for i in zealotsample:
        GraphList[i] = 2

    print("C = %0.3f, rep = %d, Created GraphList"%(C,r))

    #List denoting the connections
    GraphNeighbourList = []
    for i in range(N):
        GraphNeighbourList.append(set())

    #Probability of connection
    p = C/N

    for i in range(len(GraphNeighbourList)):
        for j in range(i+1,len(GraphNeighbourList)):
            if random.random() < p:
                GraphNeighbourList[i].add(j)
                GraphNeighbourList[j].add(i)

    print("C = %0.3f, rep = %d, Created GraphNeighbourList"%(C,r))
    #Make a list of all nodes which are changeable:
    ChangeableList = []

    for i in range(len(GraphList)):
        for j in GraphNeighbourList[i]:
            if GraphList[i] != GraphList[j]:
                if GraphList[i] == 0 and (i not in ChangeableList) :
                    ChangeableList.append(i)

                if GraphList[j] == 0 and (j not in ChangeableList):
                    ChangeableList.append(j)

                continue

    ChangeableList = set(ChangeableList)

    #ActiveList: for each node, the number of active links it has
    #i.e the number of neighbours who disagree.
    ActiveList = np.zeros(len(GraphList))
    for i in range(len(GraphList)):
        for j in GraphNeighbourList[i]:
            if GraphList[i] != GraphList[j]:
                if GraphList[i] != 2:
                    ActiveList[i] += 1

                #if GraphList[j] != 2:
                #    ActiveList[j] += 1

    #print(ActiveList)

    ##############################################################################
    #The Process

    WEAKNUM = copy.copy(Z)

    Hist = np.zeros(N+1)
    History = []


    for t in range(T):
        #if t%(T/10) == 0:
        #    print(t)
            #print(ChangeableList)
        #print(len(ChangeableList))
        #Choose random node
        randnode = floor(random.random()*N)#random.randint(0, N-1)
        while GraphList[randnode] == 2:
            randnode = floor(random.random()*N)#random.randint(0,N-1)

        if randnode in ChangeableList:

            initialopinion = GraphList[randnode]

            #Iterate through neighbours, count number of weak:
            strongnum = 0
            for i in GraphNeighbourList[randnode]:
                if GraphList[i] == 0:
                    strongnum += 1

            #Evaluate prob of adopting strong position
            probstrong = strongnum / (strongnum + F*(len(GraphNeighbourList[randnode])-strongnum))

            if random.uniform(0,1) < probstrong:
                GraphList[randnode] = 0

            else:
                GraphList[randnode] = 1


            if initialopinion != GraphList[randnode]:
                #Update WEAKNUM:
                if GraphList[randnode] == 0:
                    WEAKNUM -= 1
                else:
                    WEAKNUM += 1


                #Search through the neighbourlists to adjust changeablelists
                checklist = copy.copy(GraphNeighbourList[randnode])
                """
                checklist.add(randnode)


                for i in checklist:
                    inlist = False
                    for j in GraphNeighbourList[i]:
                        if min(1,GraphList[i]) != min(1,GraphList[j]):
                            inlist = True
                            if GraphList[i] != 2 and (i not in ChangeableList):
                                ChangeableList.add(i)

                            if GraphList[j] != 2 and (j not in ChangeableList):
                                ChangeableList.add(j)

                            continue

                    if not inlist:
                        if GraphList[i] != 2:
                            ChangeableList.remove(i)
                        continue
                """
                ActiveList[randnode] = len(GraphNeighbourList[randnode]) - ActiveList[randnode]
                if ActiveList[randnode] == 0:
                    ChangeableList.remove(randnode)
                for i in checklist:
                    if GraphList[i] != 2:
                        if min(1,GraphList[randnode]) == min(1,GraphList[i]):
                            ActiveList[i] -= 1
                            if ActiveList[i] == 0:
                                try:
                                    ChangeableList.remove(i)
                                except:
                                    print("C=",C,"randnode",randnode,"type=",GraphList[randnode],"neighbornum = ", len(checklist),"removeable type:",GraphList[i],"WEAKNUM",WEAKNUM,"ChangeableList len",len(ChangeableList))
                                    sys.exit()
                                    
                        else:
                            if ActiveList[i] == 0:
                                ChangeableList.add(i)
                            ActiveList[i] += 1

        if len(ChangeableList) == 0:
            if SimInfinite and C>1/(1-z) and z/(1-F) < 1:
                #Look for a random non zealot weak opinion
                randnode = floor(random.random()*N)#random.randint(0, N-1)
                while GraphList[randnode] != 1:
                    randnode = floor(random.random()*N)#random.randint(0,N-1)


                GraphList[randnode] = 0
                WEAKNUM -= 1

                #Update changeable list
                ActiveList[randnode] = len(GraphNeighbourList[randnode]) - ActiveList[randnode]
                ChangeableList.add(randnode)
                if ActiveList[randnode] == 0:
                    ChangeableList.remove(randnode)
                for i in GraphNeighbourList[randnode]:
                    if GraphList[i] != 2:
                        if min(1,GraphList[randnode]) == min(1,GraphList[i]):
                            ActiveList[i] -= 1
                            if ActiveList[i] == 0:
                                ChangeableList.remove(i)
                        else:
                            if ActiveList[i] == 0:
                                ChangeableList.add(i)
                            ActiveList[i] += 1
            else:
                Hist = np.zeros(len(GraphList)+1)
                Hist[WEAKNUM] += int(T*(1-0.99))
                History = np.ones(int(T*(1-0.99))) * WEAKNUM
                print("Skip")
                break

        if t >= 0.99*T:
            Hist[WEAKNUM] += 1 
            History.append(WEAKNUM)

    #History = np.asarray(History)
    HistoryMatrix.append(History)
    HistMatrix.append(Hist)

"""
plt.figure()
for History in HistoryMatrix:
    plt.plot(np.arange(0,T),History/N)
plt.xscale('log')
plt.ylim(z,1)
plt.show()
"""
#History  = HistoryMatrix[0]
#for i in range(1,len(HistoryMatrix)):
#    History += HistoryMatrix[i]


#History = History.astype(float)
#History/= len(HistoryMatrix)


HistMatrix = np.asarray(HistMatrix)
HistoryMatrix = np.asarray(HistoryMatrix)


#Mind the mean of last 10% of each HistoryMatrix
Ratio = HistoryMatrix.astype(float)/N
Mean = np.mean(Ratio,axis=0)
Median = np.median(Ratio,axis=0)







"""
MNumMatrix = np.asarray(MNumMatrix)

Ratio = np.divide(MNumMatrix,GraphSizeList[:,np.newaxis])

Mean = np.mean(Ratio,axis=0)
Median = np.median(Ratio,axis=0)
"""

############################################################################
############################################################################
############################################################################

endtime = time.time()

timetaken = endtime-starttime

print("C = %0.3f, Time Taken:"%(C),timetaken)
############################################################################
###Saving###################################################################
############################################################################

OutputDatafilename = SubSaveDirName + '/datafile.npz'
np.savez(OutputDatafilename,
    n=n,
    C=C,
    Repeats=Repeats,
    T=T,
    P=P,
    F=F,
    #PNum=PNum,
    Mean=Mean,
    Median=Median,
    #Matrix=Matrix,
    #MNumMatrix=MNumMatrix,
    #GraphSizeList=GraphSizeList,
    #ZealotNumList=ZealotNumList,
    #deg_listList=deg_listList,
    #deg_cnt_listList=deg_cnt_listList,
    #MeanClusterCoeffList=MeanClusterCoeffList,
    #MeanDegreeList=MeanDegreeList,
    #DataPoints=DataPoints,
    timetaken=timetaken,
    HistMatrix=HistMatrix)




plt.figure()
for i in HistoryMatrix:
    plt.semilogx(np.arange(len(i)),i,'k',linewidth = 0.5,alpha=0.1)

plt.ylim(0,n)

plt.savefig(SubSaveDirName + '/History.png')
plt.close()





"""
plt.figure()
plt.plot(np.arange(0,len(MNumList)),MNumList/GraphSize)
for i in ZealotInvadedTime:
    plt.axvline(x=i,color='red',alpha = 0.1)

plt.plot([0,len(MNumList)],[Theory,Theory],color='black')

plt.savefig(os.path.abspath(SaveDirName) + "/MNum.png")
plt.close



#Correllations
ZTimegaplist = []
MZTimegaplist = []
for i in range(len(ZealotInvadedTime)-1):
    ZTimegaplist.append(ZealotInvadedTime[i+1]- ZealotInvadedTime[i])
    MZTimegaplist.append(MZealotInvadedTime[i+1]- MZealotInvadedTime[i])



plt.figure()
for i in range(len(ZTimegaplist)-1):
    plt.scatter([ZTimegaplist[i]],[ZTimegaplist[i+1]],color='blue')

plt.savefig(os.path.abspath(SaveDirName) + "/xgapCorrelations.png")
plt.close()


plt.figure()
for i in range(len(ZTimegaplist)-1):
    plt.scatter([ZTimegaplist[i]],[MZTimegaplist[i+1]],color='blue')

plt.savefig(os.path.abspath(SaveDirName) + "/xygapCorrelations.png")
plt.close()
"""
endtime = time.time()

timetaken = endtime-starttime

print("Time Taken:",timetaken)
