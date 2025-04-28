#================================U1PlotFuncs.py==================================#
# Created by Bing-Long Zhang 2024

# Description:
# This file has many functions which are used to make the plots
#==============================================================================#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
from labellines import labelLine, labelLines
from scipy import interpolate
from StatisticFuncs import subFindroot
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pltdir = os.path.join(parent_dir, 'plots')
pltdir_png = os.path.join(pltdir, 'png')
def MySaveFig(fig,pltname,pngsave=True):
    fig.savefig(pltdir+pltname+'.pdf',bbox_inches='tight')
    if pngsave:
        fig.savefig(pltdir_png+pltname+'.png',bbox_inches='tight')

def line_background(lw,col):
    return [pe.Stroke(linewidth=lw, foreground=col), pe.Normal()]

def vsContourDatGen(massList, dat):
    indexList = np.array([True if len(d[1])>3 else False for d in dat])
    massList = massList[indexList]
    intDatList = [[d[1][2],d[1][4]] for d in dat if len(d[1])>3]
    intDatBoundsList = list(map(lambda x: [x[0].min(),x[0].max()],intDatList))
    intFuncList = list(map(lambda x: interpolate.interp1d(x[0],x[1]),intDatList))
    sigRange = np.linspace(-48.,-38,1000+1)
    def intFuncBounds(intFunc, var, bounds):
        if var>bounds[1]:
            return 0.
        elif var<bounds[0]:
            return 0.
        else:
            temp = intFunc(var)
            return temp
    def subFunc(i):
        nList = np.array(list(map(lambda x: intFuncBounds(intFuncList[i], x, intDatBoundsList[i]), sigRange)))
        return nList
    m, sig = np.meshgrid(massList, sigRange)
    n = np.transpose(list(map(subFunc, range(0,len(massList)))))
    return massList, 10**sigRange, n

def MakeLimitPlot_DMNSI(ax, xmin,xmax,ymax,ymin,\
                     facecolor=[0.0, 0.62, 0.38],edgecolor='darkgreen',edgecolor_collected='darkgray',\
                     alph=0.5,lfs=35,tfs=25):
    
    limitsPath = os.path.join(parent_dir, "data","WIMPLimits","SI")
    colorList = ["navy",'#4ff09d',[0.5, 0.0, 0.13],'c','r','orange','purple']
    experimentLimits = ['XENONnT','PandaX4TDMLimits','LZ2022','XENON1T8BLimits','PandaX4T8BLimits']
    labelList = [r"{\bf XENON-nT}", r"{\bf PandaX-4T}", r"{\bf LZ-2022}", \
                 r"{\bf XENON-1T} $^8B$", r"{\bf PandaX-4T} $^8B$"]
    lines = []
    for i in range(len(experimentLimits)):
        dat = np.loadtxt(os.path.join(limitsPath,experimentLimits[i]+".txt"))
        l1 = ax.plot(dat[:,0], dat[:,1],color=colorList[i],linewidth=3,label=labelList[i])
        lines.extend(l1)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    return ax, lines

def MakeLimitPlot_DMeConstant(ax, xmin,xmax,ymax,ymin,\
                     facecolor=[0.0, 0.62, 0.38],edgecolor='darkgreen',edgecolor_collected='darkgray',\
                     alph=0.5,lfs=35,tfs=25):
    
    limitsPath = os.path.join(parent_dir, "data","WIMPLimits","DM-e constant")
    colorList = ['c',"navy",'#4ff09d']
    experimentLimits = ['DarkSide-50','XENON-1T','PandaX-4T']
    labelList = [r"{\bf DarkSide-50}", r"{\bf XENON-1T}", r"{\bf PandaX-4T}"]
    lines = []
    for i in range(len(experimentLimits)):
        dat = np.loadtxt(os.path.join(limitsPath,experimentLimits[i]+" DM-e constant.txt"))
        l1 = ax.plot(1e3*dat[:,0], dat[:,1],color=colorList[i],linewidth=3,label=labelList[i])
        lines.extend(l1)
        
    ax.set_yticks(10.0**np.arange(-51,-30,1))
    
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    return ax, lines


def findroot(dat):
    x, y = dat[0], dat[1]
    transitionIndices = np.where(np.diff(np.sign(y))!=0)[0]
    if len(transitionIndices)==2:
        res = [subFindroot(0.,y[i:i+2],x[i:i+2]) for i in transitionIndices]
        return res
    else:
        print('error in findroot')
        return []
        
def datGet(resList, mDM):
    massList, resDat = resList[0], resList[1]
    resDat = resDat[np.argmin(np.abs(massList-mDM))][1]
    sigma0LogList, exposureLogList, vsSigma0LogList, \
            vsExposureLogList, y, DistinctionSigmaLog = resDat
    
    print(DistinctionSigmaLog)
    # DistinctionSigmaLog2 = findroot([vsSigma0LogList,y-1.])
    # print(DistinctionSigmaLog2)
    
    DistinctionExpoLog = np.interp(np.array(DistinctionSigmaLog), vsSigma0LogList[::-1], vsExposureLogList[::-1])
    print('Sigma: ', DistinctionSigmaLog, '(log), ', 10**DistinctionSigmaLog)
    print('Exposure: ', DistinctionExpoLog, '(log),  ', 10**DistinctionExpoLog)
    maxLabel = np.argmax(y)
    print('max n: ', 10**vsSigma0LogList[maxLabel], ' ratio: ', y[maxLabel], \
          'distinction exposure: ', 10**vsExposureLogList[maxLabel], 'DM discovery exposure ', \
           10**vsExposureLogList[maxLabel]/y[maxLabel])


def maxCheck(massList, dat):
    indexList = np.array([True if len(d[1])>3 else False for d in dat])
    massList = massList[indexList]
    intDatList = [[d[1][2],d[1][4]] for d in dat if len(d[1])>3]
    maxIndexList = [np.argmax(d[1]) for d in intDatList]
    temp = np.transpose([[d[0][i], d[1][i]] for d,i in zip(intDatList,maxIndexList)])
    res1 = np.transpose([massList,temp[0],temp[1]])
    res2 = res1[np.argmax(temp[1])]
    return np.array([res2[0], 10**res2[1], res2[2]])
