import matplotlib.pyplot as plt
from math import *
import os.path
import sys

basepath = os.path.dirname(__file__)

# goldbach's comet limit
filepath = os.path.abspath(os.path.join(basepath, "..", "params.txt"))
N = 10**int(open(filepath).read().split()[3])


#####################################
##   F i l e   F u n c t i o n s   ##
#####################################

# separate words from file
def splita( f):
	everything = f.read()
	return everything.split()
	
# take array from file
def vectorize( n, vet):
	vector = []
	for x in vet:
		if (int(x) <= n): vector.append(int(x))
		else: break
	return vector

	
#################
##   M a i n   ##
#################
	
filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "r2ns.txt"))
xRTWO = vectorize(N, splita(open(filepath)))
#print "xRTWO\n"

X = range(2, N)

# get rid of odd numbers
Y = []
for x in X:
	if x % 2 == 0:
		Y.append(x)
		sys.stdout.write("\r%d" % x)
		sys.stdout.flush()

#flush
sys.stdout.write("\r          ")
sys.stdout.flush()

# array to be ploted
xSIM = []
for n in Y:
	xSIM.append( xRTWO[n-2] )
	sys.stdout.write("\r%d" % n) #printing to keep track of progress
	sys.stdout.flush() 


# plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

params = {'axes.labelsize': 45,
         'axes.titlesize': 45,
         'xtick.labelsize': 45,
         'ytick.labelsize': 45}

plt.rcParams.update(params)
plt.figure(figsize = (2100.0/96, 1220.0/96), dpi = 96)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r"$n$ (even)")
plt.ylabel(r"$r_{{P},2}(n)$")

plt.plot(Y,xSIM, marker=',', linestyle='', color='royalblue')

plt.xlim(0, Y[len(Y)-1])
plt.ylim(ymin=0)

#plt.show()
plt.savefig(os.path.abspath(os.path.join(basepath, "..", "imgs", "figure_2.png")), dpi = 96)

sys.stdout.write("\r          \r")
sys.stdout.flush()
