import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as patches
#plt.rcParams['text.usetex'] = True

import numpy as np
import time

from scipy.stats import linregress

from scipy.optimize import curve_fit

import sys

import scipy.integrate as integrate
import scipy.special as special

from scipy.signal import argrelextrema

from scipy.signal import savgol_filter

from itertools import chain

from colour import Color

import os

import random
seed = 0
random.seed(seed)


sys.path.insert(0,'../CoreFunctions')
from Core_2 import CompleteDist


starttime = time.time()





Time = int(1e6)#2e8)

IGNORESTATIONARY = False

def P_Up(n,z,zs,F):
    return (1-n-z-zs)/(1-z-zs) * F*(n+z) / (1-n-z+F*(n+z))


def P_Down(n,z,zs,F):
    return n/(1-z-zs) * (1-n-z) / (1-n-z+F*(n+z))




#(N,F,z) sets
"""
Params = ([1000,0.99,1],
        [1000,0.99,10],
        [10000,0.99,1],
        [1000,0.998,1])#,
#        [10000,0.99,10])
"""

"""
Params = ([100,0.9,0.01,'k'],
        [1000,0.9,0.01,'m'],
        [4000,0.9,0.01,'y'],
        [100,0.99,0.09,'g'],
        [1000,0.99,0.09,'c'],
        [4000,0.99,0.09,'k'])
"""

Params = ([20,0.8,0.05,'k'],
        [200,0.8,0.05],
        [2000,0.8,0.05],
        [20,0.7,0.35,'k'],
        [200,0.7,0.35],
        [2000,0.7,0.35])


Data = []

fig = plt.figure(figsize=(85 / 25.4, 30 / 25.4))
ax = fig.add_subplot(111)


for i in Params:
    N = i[0]
    F = i[1]
    z = i[2]
    Z = round(z*N)
    #z = Z/N

    zs = 1/N
    ZS = 1


    print("N",N)
    print("F",F)
    print("z",z)

    #The number of resistant free nodes at first
    n = min(int((z/(1-F) - z) * N),int(N-N*z-1))


    #histlist = []

    hist = np.zeros(N)


    #########################################
    ###Simulation Of Complete################
    #########################################

    """
    for t in range(Time):
        if t%(Time/10) == 0:
            print("At time",t,"n=",n)
            #print(t)
        randval = random.uniform(0,1)

        upprob = (N-Z-ZS-n)/(N-Z-ZS) * F*(n+Z) / (N-Z-n + F*(n+Z))


        downprob = n/(N-Z-ZS) * (N-n-Z) / (N-n-Z+F*(n+Z))
        #upprob = (N-Z-n)/(N-Z) * F*(n+Z) / (N-Z-n-1 + F*(n+Z))
        
        #downprob = n/(N-Z) * (N-Z-n) / (N-Z-n + F*(n-1+Z))

        tot = upprob+downprob

        if tot==0:
            tot=1


        if not IGNORESTATIONARY:
            #Use this if allow stationary
            if randval < upprob:
                n += 1

            elif randval < upprob + downprob:
                n -= 1

        else:
            #Use this if we want to ignore stationary
            if randval < upprob/tot:
                n +=1

            else:
                n -= 1



        if n >= N-Z:
            n -=1


        #if t> 0.9*time:
        #histlist.append((n+Z)/N)
        hist[int(n + Z)] += 1
    """

    if N < 200:
        color = 'lightgrey'
        alpha=0.5
    elif N == 200:
        color = 'darkgrey'
    elif N == 2000:
        color = 'dimgrey'

    #plt.plot(np.linspace(0,1,N),hist/sum(hist)*N,color=i[3],linewidth=3,linestyle='dotted')

    n, Ana = CompleteDist(N,F,z*N)

    #print(n)
    #print(Ana)

    mean = integrate.simpson(y=Ana*(n+z),x=n)
    print("mean = ",mean)

    if abs(z/(1-F) -0.2) < 0.1:
        linestyle = 'solid'
    else:
        linestyle='dashed'
    plt.plot(n+z,Ana,color=color,linewidth=2,linestyle=linestyle)



plt.plot([0.25,0.25],[0,20],linestyle='solid',color='k')
plt.plot([1,1],[0,20],linestyle='dashed',color='k')



ax.set_xticks([0,0.2,0.4,0.6,0.8,1])#xticks)
ax.set_xticklabels([r'$0.0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'])

ax.set_yticks([0,10,20,30,40])
ax.set_yticklabels([ r'$0$',r'$10$',r'$20$',r'$30$',r'$40$'])

#plt.ylim(P,np.ceil(10*max(EndMedian))/10)#(P,1)
plt.ylim(0,20)
plt.xlim(0.1,1)

plt.xticks(fontsize=7,fontname = "Arial")
plt.yticks(fontsize=7,fontname = "Arial")







plt.savefig("fig.png",bbox_inches='tight',dpi=300)

plt.yscale('log')
plt.ylim(1e-5,1e3)
plt.savefig("fig_log.png",bbox_inches='tight',dpi=300)

plt.close()

endtime = time.time()

print("Time Taken:",endtime-starttime)
