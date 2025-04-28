from Params import *

import sys
sys.path.insert(0,'../CoreFunctions')

from Core_2 import Init, Iterate,  Observe, MeasureMutants, Plot, GraphStats, Escape_Absorbing

import time




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

C = float(args.C)
P = float(args.P)
F = float(args.F)
SaveDirName = str(args.dir)


GraphDict["C"] = C
GraphDict["p"] = C/n
GraphDict["P"] = P
SubSaveDirName = (SaveDirName +
    "/C_%0.3f"%(C))

if not os.path.isdir(SubSaveDirName):
    os.mkdir(SubSaveDirName)
    print("Created Directory for C",C)

print("Starting C: %0.3f"%(C))

#############################################################
#############################################################
#############################################################



starttime = time.time()


#Lists to be saved
MNumMatrix = []
GraphSizeList = []
ZealotNumList = []

#HistMatrix
HistMatrix = np.zeros((Repeats,n+1))


#For plotting but not for saving, else too big
TempMNumMatrix = []

#Stats about the Graphs
deg_listList = []
deg_cnt_listList = []
MeanClusterCoeffList = []
MeanDegreeList = []

#Repeated iterations:

for R in range(Repeats):
    print("C",C,"Repeat",R)

    InitDict = Init(GraphDict)

    PNum = InitDict["PNum"]

    ParamsDict = {
            "Graph":InitDict["Graph"],
            "InactivePatchIDs":InitDict["InactivePatchIDs"],
            "MNum":InitDict["MNum"],
            "F":F,
            "ChangeableList":InitDict["ChangeableList"]}

    GraphSize = InitDict["NodeNum"]
    GraphSizeList.append(GraphSize)
    ZealotNumList.append(PNum)

    #Generate statdictionary and unpack
    StatDict = GraphStats(ParamsDict["Graph"])
    deg_listList.append(StatDict["deg_list"])
    deg_cnt_listList.append(StatDict["deg_cnt_list"])
    MeanClusterCoeffList.append(StatDict["MeanClusterCoeff"])
    MeanDegreeList.append(StatDict["MeanDegree"])    


    MNumList = []
    TempMNumList = []
    ZealotInvadedTime = []
    MZealotInvadedTime = []
    for t in range(T):
        print("C:",C,"Rep:",R,"t:",t)
        ParamsDict["t"] = t

        
        if (len(ParamsDict["ChangeableList"]) == 0):
            if (C>=1):
                #print("ESCAPE PROTOCAL")
                ParamsDict = Escape_Absorbing(ParamsDict)
                #print("C=",C,"t=",t,"len changelist=",len(ParamsDict["ChangeableList"]))
        

        if len(ParamsDict["ChangeableList"]) > 0:
            ParamsDict = Iterate(ParamsDict)

    
        if t > T-DataPoints:
            MNumList.append(ParamsDict["MNum"])

            HistMatrix[R][ParamsDict["MNum"]] += 1        

        TempMNumList.append(ParamsDict["MNum"])

        """
        if ParamsDict["InactivePatchActivated"]:
            ZealotInvadedTime.append(t)
            MZealotInvadedTime.append(MNumList[-1])
        """
    MNumMatrix.append(MNumList)
    TempMNumMatrix.append(TempMNumList)

MNumMatrix = np.asarray(MNumMatrix)
GraphSizeList = np.asarray(GraphSizeList)

Ratio = np.divide(MNumMatrix,GraphSizeList[:,np.newaxis])

Mean = np.mean(Ratio,axis=0)
Median = np.median(Ratio,axis=0)


############################################################################
############################################################################
############################################################################

endtime = time.time()

timetaken = endtime-starttime

print("Time Taken:",timetaken)
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
    PNum=PNum,
    Mean=Mean,
    Median=Median,
    #Matrix=Matrix,
    #MNumMatrix=MNumMatrix,
    GraphSizeList=GraphSizeList,
    ZealotNumList=ZealotNumList,
    #deg_listList=deg_listList,
    #deg_cnt_listList=deg_cnt_listList,
    MeanClusterCoeffList=MeanClusterCoeffList,
    MeanDegreeList=MeanDegreeList,
    DataPoints=DataPoints,
    timetaken=timetaken,
    HistMatrix=HistMatrix)




plt.figure()
for i in TempMNumMatrix:
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
