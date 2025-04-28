import os
import shutil

import numpy as np

import matplotlib.pyplot as plt

"""
Types:
    ER
    SmallWorld
    Geometric
"""


#Simulate an infinite graph by forcing a random node to flip if consensus is achieved.
#Turn off if the point is to study finite graphs
SimInfinite = False

Type = "ER"

#number of sites
n = 10000#1000#1000


#ER Stats
#Mean number of connections
minC = 0
maxC = 10
Cnum = 41
CList = np.linspace(minC,maxC,Cnum)
#Corresponding edge probability
#p = C/n

#Small World Stats
#Mean number of connections
k = 2
#rewiring probability
r = 0.2
#number of tries
t = 100

#Geometric Stats
#radius of the sense
minr = 0.#1/np.sqrt(n)
maxr = 0.1#05#np.sqrt(1/2)
rnum = 40
radiuslist = np.linspace(minr,maxr,rnum)


#General stats
Repeats =10#20#10#20
#Whether all the patches start infected or not
SingleActive = False

#Prob of Patch 
PList = np.asarray([0.3])#np.asarray([0.005,0.001])#np.linspace(0.1,1,10)

#Time taken for sim to run
T = int(1e9)#int(1e7)#6)#40000000#50000000#10000000#100000000

#Fitness of the mutant
FList = np.asarray([0.6,0.9])#np.asarray([0.9,0.95,0.99])#np.linspace(0.1,1,10)

#Whether I just use the largest component
LargestComponent = False

#PicTime is the time steps of the snapshorts of the system
PicTime = T/1000

#Number of points sampled at the end
DataPoints = int(T/10)






GraphDict = {
        "N":n,
        "Type":Type,
        "P":0,
        "SingleActive":SingleActive,
        "LargestComponent":LargestComponent
        }

if Type == "ER":
    GraphDict["C"] = 0
    GraphDict["p"] = 0

    SaveDirName= ("SaveFiles/Escape_Absorbing_ER_minC_%0.3f_maxC_%0.3f_Cnum_%d_NodeNum_%d_minZ_%0.5f_maxZ_%0.5f_Znum_%d_minF_%0.5f_maxF_%0.5f_Fnum_%d_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (minC,maxC,Cnum,n,min(PList),max(PList),len(PList),min(FList),max(FList),len(FList),T,SingleActive,Repeats,LargestComponent))

elif Type == "SmallWorld":
    GraphDict["k"] = k
    GraphDict["r"] = r
    GraphDict["t"] = t

    SaveDirName= ("SaveFiles/SW_k_%0.3f_r_%0.3f_t_%0.3f_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (k,r,t,n,P,F,T,SingleActive,Repeats,LargestComponent))

elif Type == "Geometric":
    GraphDict["radius"] = 0

    SaveDirName= ("SaveFiles/Geo_minr_%0.5f_maxr_%0.5f_rnum_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (minr,maxr,rnum,n,P,F,T,SingleActive,Repeats,LargestComponent))


elif Type == "Geometric_Torus":
    GraphDict["radius"] = 0

    SaveDirName= ("SaveFiles/GeoTorus_minr_%0.5f_maxr_%0.5f_rnum_%d_NodeNum_%d_ZealotProb_%0.5f_Fitness_%0.3f_Timesteps_%d_SingleActive_%r_Repeats_%d_LargestComponent_%r"%
                (minr,maxr,rnum,n,P,F,T,SingleActive,Repeats,LargestComponent))

else:
    raise Exception("Incorrect type of Graph")

SaveDirName += "_DataPoints_%d"%(DataPoints)


if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Savefile:",SaveDirName)

shutil.copyfile("Params.py", SaveDirName+"/Params.py")
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

