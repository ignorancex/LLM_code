"""
Produce plot and table from the measured data of transition times.
"""

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------
#Auxilliary functions and variables
#------------------------------------

#link of indices and polarization state labels
LCd_states = ('H','V','D','A','R','L') 

def readdata(fn):
    """
    Load measured data from npz file.
    Args:
        fn ... path + filename
    Returns:
        t ... array with timestamp of samples
        intensity ... voltage corresponding to optical intensity
        i1 ... index of transition start (500 ms from the the record start)
        i2 ... index of transition end
    """
    dat = np.load(fn)['valueList']
    intensity = dat[:,1]
    t = dat[:,0]
    S = 50
    mI1 = np.mean(intensity[0:S])
    mI2 = np.mean(intensity[-S:])    
    THR2 = 0.01 # 100%->1% transition
    thr2 = np.abs((intensity-mI2)/mI1)>THR2    
    thr1 = t>0.5
    i1 = np.where(thr1)[0][0]-1
    i2 = np.where(thr2)[0][-1]+2
    
    return t, intensity, i1, i2

#------------------------------------
#Main script
#------------------------------------

#Table with results
Table = np.zeros((6,6))

#Matplotlib settings
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 10
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', markersize = 2.5)
plt.rc('lines', markeredgewidth = 1)
plt.rc("errorbar", capsize=1)
plt.rc('lines', linewidth = 1.1)
plt.rc("text", usetex = True)

Toffset = 0.5 #where transition starts

#Produce transition plot and table
fig, ax = plt.subplots(6,6, figsize=(5,6), sharex=True, sharey=True)
for i, ini in enumerate(LCd_states):    
    for j, fin in enumerate(LCd_states):
        if i!=j:
            fn = f'Response_Data/20200525_17h35min_{ini}_to_{fin}_Time_response.npz'
            
            t, I, i1, i2 = readdata(fn) #load data
            I2 = I/np.mean(I[0:50]) #normalize them
            #... and plot them (and shift time 0 to start of transition in the graph)
            ax[i,j].plot(t-Toffset, I2, "-", c="k")
            ax[i,j].plot(t[i1:i2]-Toffset, I2[i1:i2], "-", c="r")

            trans = t[i2]-t[i1] #calculate transition time
            Table[i,j] = trans #and save it to table
            #graph labeling, formatting etc.
            ax[i,j].set_xlim(0.4-Toffset,1.2-Toffset)
            ax[i,j].text(0.95, 0.6, f'{trans*1000:.0f} ms\n{ini}$\\rightarrow${fin}', transform=ax[i,j].transAxes, horizontalalignment='right', fontsize=SMALL_SIZE)
            ax[i,j].vlines([0.5-Toffset],-0.1,1.1,colors='grey',linestyles='dotted')
            ax[i,j].hlines([1,0.5,0],0,5,linestyles='dotted',colors='grey')
        if i==5:
            ax[i,j].set_xlabel('$t$ [s]')
    ax[i,0].set_ylabel('$I(t)/I_{0}$')
    
plt.subplots_adjust(top=0.978,
bottom=0.093,
left=0.108,
right=0.988,
hspace=0.0,
wspace=0.0)
plt.savefig('tran_time.pdf')  
plt.show()

#Print transition duration table
print(np.round(Table*1000,0))
optimalorder = [0,4,5,3,2,1,0]
tot = sum([Table[i,j] for i, j in zip(optimalorder[0:-1],optimalorder[1:])])
print(tot)
print([(i,j) for i, j in zip(optimalorder[0:-1],optimalorder[1:])], sep='\n')
print(np.min(Table[Table>0]*1000))
print(np.max(Table*1000))