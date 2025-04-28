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

starttime = time.time()

if __name__ == "__main__":
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
    parser.add_argument('-a','--all',type=int,help='Plot all of the sub-figures')
    args = parser.parse_args()

    directory = str(directory)



def Single_Plot(directory,Prev_directory,ALL):
    ###############################
    ##Extract Data#################
    ###############################
    #Find all the directories
    templist = os.listdir(directory)
    #print(templist)

    dirlist = []

    for i in range(len(templist)):
        if os.path.isdir(directory + '/' + templist[i]):
            print("Is directory!")
            npzlist = os.listdir(directory + '/' + templist[i])
            for j in range(len(npzlist)):
                if npzlist[j].endswith(".npz"):
                    dirlist.append(templist[i])


    MeanList = []
    MedianList = []
    CList = []

    MeanGraphSizeList = []
    MeanDegreeList = []

    AbsorbingStateProb = []

    #MeanList as above but disregard cases where we reach the absorbing state/
    MeanNoAbsorbList = []

    #Mean of the mean cluster coefficients
    MeanClusterCoeff = []

    for d in dirlist:
        filelist = os.listdir(directory + "/" + d)
        for names in filelist:
            if names.endswith(".npz"):
                filename = names
                print("Found File!")

        with np.load(os.path.join(directory,d,filename)) as data:
            Repeats = data['Repeats']
            nodenum = data['n']
            T = data['T']
            C = data['C']
            #p = data['p']
            F = data['F']
            P = data['P']
            MNumMatrix = data['MNumMatrix']
            GraphSizeList=data['GraphSizeList']
            ZealotNumList=data['ZealotNumList']
            #deg_listList=data['deg_listList']
            #deg_cnt_listList=data["deg_cnt_listList"]
            MeanClusterCoeffList=data["MeanClusterCoeffList"]
            MeanDegree=data["MeanDegreeList"]

            timetaken = data['timetaken']
            #print("Time Taken:",timetaken)

        MeanGraphSizeList.append(np.mean(GraphSizeList))
        MeanDegreeList.append(np.mean(MeanDegree))


        #Create Mean List
        Ratio = np.divide(MNumMatrix,GraphSizeList[:,np.newaxis])

        MeanList.append(np.mean(Ratio,axis=0))
        MedianList.append(np.median(Ratio,axis=0))

        #Mean Cluster Coeff
        MeanClusterCoeff.append(np.mean(MeanClusterCoeffList))

        #Absorbing State
        AbsorbedNum = 0
        for i in Ratio:
            if i[-1] == 1:
                AbsorbedNum += 1
        AbsorbingStateProb.append(AbsorbedNum/len(Ratio))

        masked_matrix = np.where(Ratio == 1, np.nan,Ratio)
        MeanNoAbsorbList.append(np.nanmean(masked_matrix,axis = 0))

        CList.append(C)

        CList = [float(value) if isinstance(value, np.ndarray) else value for value in CList]
        MeanList = [list(sublist) if isinstance(sublist, np.ndarray) else sublist for sublist in MeanList]
        MedianList = [list(sublist) if isinstance(sublist, np.ndarray) else sublist for sublist in MedianList]
        MeanNoAbsorbList = [list(sublist) if isinstance(sublist, np.ndarray) else sublist for sublist in MeanNoAbsorbList]
        #print(CList)
        #print(MeanList)

        #Plot each repeat for a C:
        print("Plot C = ",C)
        if ALL:
            print(args.ALL)

            #Figure for the number of mutants at each time point
            x = np.arange(len(MNumMatrix[0]))#MeanMNum))

            plt.figure()
            for i in range(len(MNumMatrix)):
                plt.plot(x,MNumMatrix[i]/GraphSizeList[i])

            plt.plot(x,MeanList[-1],color='black',linewidth=5)

            plt.plot(x,MeanNoAbsorbList[-1],color='orange',linewidth=5,label='No Absorb')

            Theory = P/(1-F)

            plt.legend(loc='upper left')

            plt.plot([min(x),max(x)],[Theory,Theory],'--r',linewidth=5)

            plt.ylim(P,1)

            plt.title("C=%0.3f"%(C))
            plt.savefig(str(directory) +'/'+str(d) +'/AllRepeats.png' )
            plt.savefig(str(directory) +'/C_%0.3f_AllRepeats.png'%(C) )

            plt.close()


            

    print("Finished all sub-plots")
    CList,MeanList,MedianList,MeanGraphSizeList,MeanDegreeList,AbsorbingStateProb,MeanNoAbsorbList,MeanClusterCoeff = zip(*sorted(zip(CList, MeanList, MedianList,MeanGraphSizeList,MeanDegreeList,AbsorbingStateProb,MeanNoAbsorbList,MeanClusterCoeff)))

    ##############################
    Theory = P/(1-F)
    print("Plotting Ednstates")


    print(CList)
    CList = np.asarray(CList)

    def NumTreesSize_s(N,s,C):

        """
        The number of trees of size s is T. Because of the large factorials
        we calculate log(T) and perform sums of log of the factorials. Then
        we convert back to T.
        """

        NList = np.arange(1,N +1)

        NsList = np.arange(1,N-s +1)

        sList = np.arange(1,s +1)

        term1 = 0#np.log(N)
        term2 = np.sum(np.log(NList))
        term3 = np.sum(np.log(NsList))
        term4 = np.sum(np.log(sList))
        term5 = (s-2)* np.log(s)
        term6 = (s-1) * np.log(C/N)
        term7 = (special.binom(s,2) - (s-1) + s*(N-s)) * np.log(1-C/N)

        tot = term1 + term2 - term3 - term4 + term5 + term6 + term7
        """

        term1 = np.log(2)
        term2 = (s-2)*np.log(s)
        term3 = (s-1)*np.log(C)
        term4 = -C*s
        term5 = -np.log(2-C)
        term6 = -np.sum(np.log(sList))

        tot = term1+term2+term3+term4+term5+term6
        """

        output = np.exp(tot)

        """
        term1 = -C*s
        term2 = (s-1)*np.log(C*s)
        term3 = -np.sum(np.log(sList))

        tot = term1 + term2 + term3 

        output = np.exp(tot)
        """
        """

        try:
            output = N * special.binom(N,s) * s**(s-2) * (C/N)**(s-1) * (1-C/N)**(special.binom(s,2) - (s-1) + s*(N-s))#N * (s**(s-2)*C**(s-1) * np.exp(-C*s))/special.factorial(s) * np.exp(C*(s**2 + 3*s - 2)/(2*N))
        except:
            output = np.zeros(len(C))
        print("sizeprob outpyt",output)
        """
        return output
    def MPerCluster(N,s,C,P):
        output2 = NumTreesSize_s(N,s,C) * s * (1-(1-P)**s)/N**2
        return output2
    def MRatio(N,C,P):
        mysum = np.zeros(len(CList),dtype=float)
        for s in range(1,N):
            a = np.array(MPerCluster(N,s,C,P),dtype='float64')
            #a[np.isnan(a)] = 0
            #a[np.isinf(a)] = 0
            print("s",s)
            print("a",a)
            #a=np.nan_to_num(a,posinf=0,neginf=0)
            #print("a",a)
            #a[np.isnan(a)] = 0
            #a[np.isinf(a)] = 0
            print("mysum",mysum)
            mysum = np.array(mysum,dtype='float64') + np.array(a,dtype='float64')

        return mysum
    #TheoryList = MRatio(nodenum,CList,P)
    TheoryList = []
    nodenumlist = []
    difflist = []
    slist = np.arange(1,nodenum)

    for C in CList:
        sproblist = []
        for s in slist:
            sproblist.append(NumTreesSize_s(nodenum,s,C))

        sproblist = np.asarray(sproblist)

        #sproblist /= sum(sproblist)**2

        print(C,sum(sproblist))

        diff = nodenum - sum(slist*np.asarray(sproblist))
        difflist.append(diff)
        mysum = 0
        for s in range(1,nodenum):
            a = sproblist[s-1]*s*(1-(1-P)**s)
            #print(C,a)
            mysum += a

        mysum += diff*(1-(1-P)**diff)
        TheoryList.append(mysum/nodenum)



    print("Starting high c prob")
    HighCTheoryList = []
    for i in range(len(CList)):
        C=CList[i]
        print("C",C)
        N = nodenum#difflist[i]
        p = C/N
        mysum = 0
        for k in range(1+1):
            mysum += special.binom(N-1,k)*p**k * (1-p)**(N-k-1)

        M = mysum + (1-mysum) * Theory
        HighCTheoryList.append(M)

    plt.figure()
    #Plot how the end-state mean evolves with C
    EndMean = []
    EndMedian = []

    EndNoAbsorbMean = []
    for i in range(len(MeanList)):
        EndMean.append(np.mean(MeanList[i][-int(T/10):]))
        EndMedian.append(np.mean(MedianList[i][-int(T/10):]))
        EndNoAbsorbMean.append(np.mean(MeanNoAbsorbList[i][-int(T/10):]))

    plt.scatter(CList,EndMean, label='Mean Endstate',marker='x')
    plt.scatter(CList,EndMedian, label='Median Endstate',marker='+')
    plt.scatter(CList,EndNoAbsorbMean, label='Mean Endstate No Absorb',marker='D')


    plt.plot(
            [min(CList),max(CList)],
            [Theory,Theory],
            '--k',
            linewidth=5,
            label='Complete Theory',
            alpha = 0.5)

    plt.plot(CList,TheoryList,label='Theory')
    plt.plot(CList,HighCTheoryList,label='High C Theory')

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("C")
    plt.ylabel("EndState Ratio of M")

    plt.legend(loc='lower right')

    plt.ylim(P,1)

    plt.grid(True)

    plt.savefig(str(directory) +'/EndMeanWithC.png')
    plt.savefig(str(Prev_directory)+"/P_%0.3f_F_%0.3f.png"%(P,F))
    plt.close()




    fig = plt.figure()
    ax = fig.add_subplot(111)

    #Plotting
    TheoryList[0] = P
    plt.scatter(CList,EndMean, label='Mean Endstate',marker='x',s=200)
    plt.plot(
            [min(CList),max(CList)],
            [Theory,Theory],
            '--k',
            linewidth=5,
            label='Complete Theory',
            alpha = 0.5)
    plt.plot(CList,TheoryList,'k',label='Theory',linewidth=5)

    xticks = np.linspace(0,1,6)

    if max(CList) ==10:
        xticks = np.linspace(0,10,6,dtype=int)

    if max(CList) == 2:
        xticks = np.linspace(0,2,6)

    yticks = np.linspace(0.8,1,6)


    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    plt.xticks(fontsize=30,fontname = "Arial")
    plt.yticks(fontsize=30,fontname = "Arial")
    plt.savefig(str(directory) +'/Fig_EndMeanWithC.png',bbox_inches='tight')
    plt.close()






    ############################################################

    plt.figure()
    #Plot how the end-state mean evolves with C
    EndMean = []
    EndMedian = []

    EndNoAbsorbMean = []
    for i in range(len(MeanList)):
        EndMean.append(np.mean(MeanList[i][-int(T/10):]))
        EndMedian.append(np.mean(MedianList[i][-int(T/10):]))
        EndNoAbsorbMean.append(np.mean(MeanNoAbsorbList[i][-int(T/10):]))

    plt.scatter(MeanDegreeList,EndMean, label='Mean Endstate',marker='x')
    plt.scatter(MeanDegreeList,EndMedian, label='Median Endstate',marker='+')
    plt.scatter(MeanDegreeList,EndNoAbsorbMean, label='Mean Endstate No Absorb',marker='D')


    plt.plot(
            [min(MeanDegreeList),max(MeanDegreeList)],
            [Theory,Theory],
            '--k',
            linewidth=5,
            label='Complete Theory',
            alpha = 0.5)

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("Mean Degree")
    plt.ylabel("EndState Ratio of M")

    plt.legend(loc='lower right')

    plt.ylim(P,1)

    plt.grid(True)

    plt.savefig(str(directory) +'/EndMeanWithDegree.png')
    plt.close()

    ############################################################


    ##############################
    plt.figure()
    plt.semilogy(CList,AbsorbingStateProb)

    plt.title("Fitness %0.3f, Zealot Ratio %0.3f"%(F,P))
    plt.xlabel("C")
    plt.ylabel("Prob of Absorbing State")

    plt.grid(True)

    plt.savefig(str(directory) + '/AbsorbingStateProb.png')
    plt.close()

    return P,F,CList,EndMean 
