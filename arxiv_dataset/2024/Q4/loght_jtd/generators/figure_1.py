import matplotlib.pyplot as plt
import numpy as np
from math import *
import os.path
import sys

basepath = os.path.dirname(__file__)

# fundamental discriminants up to N
filepath = os.path.abspath(os.path.join(basepath, "..", "params.txt"))
Nexp = int(open(filepath).read().split()[1])
N = 10**int(open(filepath).read().split()[1])


#####################################
##   F i l e   F u n c t i o n s   ##
#####################################

# separate words from file
def splita( f):
	everything = f.read()
	return everything.split()
	
# take array from file (float)
def vectorize( vet):
	vector = []
	for x in vet: vector.append(float(x))
	return vector

	
#################
##   M a i n   ##
#################

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "1e" + str(Nexp), "negFDs_" + '1e' + str(Nexp) + ".txt"))
FDS = vectorize( splita(open(filepath)))

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "1e" + str(Nexp), "hjtds_" + '1e' + str(Nexp) + ".txt"))
HEI = vectorize( splita(open(filepath)))


### PLOT
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

params = {'axes.labelsize': 45,
         'axes.titlesize': 45,
         'xtick.labelsize': 45,
         'ytick.labelsize': 45,
         'ytick.right': True}

plt.rcParams.update(params)
plt.figure(figsize = (2100.0/96, 1220.0/96), dpi = 96)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.xlabel(r'$|D|$')
plt.ylabel(r'$\frac{\mathrm{ht}(j(\tau_{D}))}{3\log(|D|)}$')
plt.plot(FDS, HEI, marker=',', linestyle='', zorder=.5, color='tab:blue', alpha=1)

# red line
#plt.plot(range(10, int(FDS[len(FDS)-1])), [3 - 3*log(log(i))/log(i) for i in range(10, int(FDS[len(FDS)-1]))], linestyle='-', color='r')
#plt.plot(range(10, int(FDS[len(FDS)-1])), [3 + 6*log(log(i))/log(i) for i in range(10, int(FDS[len(FDS)-1]))], linestyle='-', color='r')


#line y = 3
#plt.plot(range(int(FDS[len(FDS)-1])), [3 for i in range(int(FDS[len(FDS)-1]))], linestyle='--', color='k', alpha=0.5)

plt.xlim(0, N)#FDS[len(FDS)-1])
#plt.yscale('log')
plt.ylim(0, 2)
plt.ylim(ymin=0)
plt.grid(zorder=2.5, color='k', alpha=.25)

#plt.show()
plt.savefig(os.path.abspath(os.path.join(basepath, "..", "imgs", "figure_" + "1e" + str(Nexp) + ".png")), dpi = 96)
