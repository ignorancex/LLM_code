import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import *
import os.path
import sys

basepath = os.path.dirname(__file__)

# residue classes
filepath = os.path.abspath(os.path.join(basepath, "params.txt"))
rCLS = [ [int(open(filepath).read().split()[5]), int(open(filepath).read().split()[6])],
		 [int(open(filepath).read().split()[7]), int(open(filepath).read().split()[8])],
		 [int(open(filepath).read().split()[9]), int(open(filepath).read().split()[10])] ]

if len(sys.argv) > 1: arg1 = int(sys.argv[1])
else: arg1 = 1

COLORS = [str(open(filepath).read().split()[12]),
		  str(open(filepath).read().split()[13 - 1/arg1]),
		  str(open(filepath).read().split()[14 - 2/arg1])]



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

sys.stdout.write("Sorting discriminants...")

filepath = os.path.abspath(os.path.join(basepath, "Ds.txt"))
DS = vectorize( splita(open(filepath)))
filepath = os.path.abspath(os.path.join(basepath, "lpl1chiD.txt"))
LPL = vectorize( splita(open(filepath)))

DD = [[] for i in range(len(rCLS))]
LP = [[] for i in range(len(rCLS))]

for i in range(len(DS)):
	for x in range(len(rCLS)):
		if ((-DS[i]) % rCLS[x][1] == rCLS[x][0]):
			DD[x].append(DS[i])
			LP[x].append(LPL[i])
			continue
	sys.stdout.write("\rSorting discriminants... %d" % -DS[i])
	sys.stdout.flush()

sys.stdout.write("\rSorting discriminants... OK.            \n")
sys.stdout.write("Plotting... ")


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
plt.ylabel(r'$-\frac{L^{\prime}(1,\chi_D)}{L(1,\chi_D)}$')

if arg1 == 1: ALF = [.5, .5, .5]
elif arg1 == 2: ALF = [.9, .45, .45]
elif arg1 == 3: ALF = [.9, .6, .3]

for x in range(len(rCLS)):
	plt.plot(DD[x], LP[x], marker='.', linestyle='', zorder=.5, color=COLORS[x], alpha=ALF[x])


# Legend

patCHs = [mpatches.Patch(color=COLORS[1], label=r'$D \equiv 0~(\mathrm{mod}~4)$'),
		  mpatches.Patch(color=COLORS[0], label=r'$D \equiv 1~(\mathrm{mod}~4)$'),
		  mpatches.Patch(color=COLORS[1], label=r'$D \equiv 8~(\mathrm{mod}~16)$'),
		  mpatches.Patch(color=COLORS[2], label=r'$D \equiv 12~(\mathrm{mod}~16)$')]

if arg1 == 2:
	plt.legend(handles=patCHs[1::-1], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0., fontsize=30)
elif arg1 == 3:
	plt.legend(handles=patCHs[1:], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0., fontsize=25)


#line y = 3
#plt.plot(range(int(FDS[len(FDS)-1])), [3 for i in range(int(FDS[len(FDS)-1]))], linestyle='--', color='k', alpha=0.5)

plt.xlim(0, DS[len(DS)-1])
#plt.yscale('log')
plt.ylim(-3.5, 3.5)
#plt.ylim(ymin=0)
plt.grid(zorder=2.5, color='k', alpha=.25)

#plt.show()
plt.savefig(os.path.abspath(os.path.join(basepath, "fig%d.png" % arg1)), dpi = 96)


# Finishes
sys.stdout.write("OK.\n")
