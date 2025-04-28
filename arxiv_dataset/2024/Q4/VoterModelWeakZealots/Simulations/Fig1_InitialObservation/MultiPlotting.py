import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

import numpy as np
import time

from scipy.stats import linregress

from scipy.optimize import curve_fit

import sys

import scipy.integrate as integrate
import scipy.special as special

from scipy.signal import argrelextrema

import os

import subprocess
import Plotting_NEW


starttime = time.time()

################################
##ArgParse######################
################################
import os.path

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The Directory %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

from argparse import ArgumentParser

parser = ArgumentParser(description='Plotting')
parser.add_argument('-d','--directory',help='The directory of the data')
args = parser.parse_args()

directory = str(args.directory)


#Find list of all the datafiles
tempdirlist = os.listdir(directory)
dirlist = []
for i in tempdirlist:
    if os.path.isdir(os.path.join(directory,i)):
        dirlist.append(os.path.join(directory,i))


PList = []
FList = []
CMatrix = []
EndMeanMatrix = []

for d in dirlist:
    print(d)
    try:
        P,F,CList,EndMean = Plotting_NEW.Single_Plot(d,directory,0)

        PList.append(P)
        FList.append(F)
        CMatrix.append(CList)
        EndMeanMatrix.append(EndMean)


    except Exception as e: print(e)

PList,FList,CMatrix,EndMeanMatrix= zip(*sorted(zip(PList,FList,CMatrix,EndMeanMatrix)))

#Heatmap of maximum values
P = []
F = []
C = []

for i in range(len(CMatrix)):
    EndMean = EndMeanMatrix[i]
    CList = CMatrix[i]

    maxindex = np.argmax(EndMean)
    max_C = CList[maxindex]

    C.append(max_C)
    P.append(PList[i])
    F.append(FList[i])

# Create a grid for the heatmap
P_unique = np.unique(P)
F_unique = np.unique(F)
heatmap_data = np.zeros((len(P_unique), len(F_unique)))

# Populate the heatmap data
for p, f, c in zip(P, F, C):
    p_index = np.where(P_unique == p)[0][0]
    f_index = np.where(F_unique == f)[0][0]
    heatmap_data[p_index, f_index] = np.log(c)

dF = F_unique[1] - F_unique[0]
dP = P_unique[1] - P_unique[0]

# Create the heatmap
plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower',
           extent=[F_unique[0]-dF/2, F_unique[-1]+dF/2, P_unique[0]-dP/2, P_unique[-1]+dP/2])
plt.colorbar(label='log(C) at max m')  # Add a colorbar with label

# Set axis labels
plt.xlabel('F')
plt.ylabel('P')

# Show the plot
plt.savefig(directory+"/Heatmap.png")
plt.close()



################################
#Heatmap of EndStates
P = []
F = []
E = []  #Measured Endstate

for i in range(len(CMatrix)):
    EndMean = EndMeanMatrix[i]
    CList = CMatrix[i]

    E.append(EndMean[-1])
    P.append(PList[i])
    F.append(FList[i])

# Create a grid for the heatmap
P_unique = np.unique(P)
F_unique = np.unique(F)
heatmap_data = np.zeros((len(P_unique), len(F_unique)))
theory_heatmap_data = np.zeros((len(P_unique), len(F_unique)))
# Populate the heatmap data
for p, f, e in zip(P, F, E):
    p_index = np.where(P_unique == p)[0][0]
    f_index = np.where(F_unique == f)[0][0]
    heatmap_data[p_index, f_index] = e
    theory_heatmap_data[p_index,f_index] = min(p/(1-f),1)


fig = plt.figure()
ax = fig.add_subplot(111)
# Create the heatmap
plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower',
           extent=[F_unique[0]-dF/2, F_unique[-1]+dF/2, P_unique[0]-dP/2, P_unique[-1]+dP/2])


#cbar = fig.colorbar(cp)
cbar = plt.colorbar()#label='EndState Mean')  # Add a colorbar with label
#cbar.set_label(label='endstate mean',size=0)
cbar.ax.tick_params(labelsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)

#plt.colorbar(label='EndState Mean')  # Add a colorbar with label

# Set axis labels
#plt.xlabel('F')
#plt.ylabel('P')

xticks = np.arange(0.2,1.2,0.2)
yticks = np.arange(0.1,1.1,0.2)


ax.set_xticks(xticks)
ax.set_yticks(yticks)

#plt.ylim(P,1)

plt.xticks(fontsize=30,fontname = "Arial")
plt.yticks(fontsize=30,fontname = "Arial")


# Show the plot
plt.savefig(directory+"/Heatmap_EndstateMean.png")
plt.savefig(directory+"/Heatmap_EndstateMean.pdf")


plt.close()







#Theory
# Create the heatmap
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(theory_heatmap_data, cmap='viridis', aspect='auto', origin='lower',
           extent=[F_unique[0]-dF/2, F_unique[-1]+dF/2, P_unique[0]-dP/2, P_unique[-1]+dP/2])


cbar = plt.colorbar()  # Add a colorbar with label
#cbar.set_label(label='endstate mean',size=0)
cbar.ax.tick_params(labelsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)


#Phase Sep
plt.plot([0.1,0.9],[0.9,0.1],color='k',linewidth=5)

#plt.colorbar(label='Endstate Theory')  # Add a colorbar with label

# Set axis labels
#plt.xlabel('F')
#plt.ylabel('P')

xticks = np.arange(0.2,1.2,0.2)
yticks = np.arange(0.1,1.1,0.2)


ax.set_xticks(xticks)
ax.set_yticks(yticks)

#plt.ylim(P,1)

plt.xticks(fontsize=30,fontname = "Arial")
plt.yticks(fontsize=30,fontname = "Arial")


# Show the plot
plt.savefig(directory+"/Heatmap_EndstateMean_Theory.png")
plt.savefig(directory+"/Heatmap_EndstateMean_Theory.pdf")


plt.close


endtime = time.time()

print("Time taken for plotting",endtime-starttime)

    
