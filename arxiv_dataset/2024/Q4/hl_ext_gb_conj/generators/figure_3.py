import scipy.integrate as integrate
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
from math import *
import os.path
import sys

basepath = os.path.dirname(__file__)

# Twin primes constant
C2 = 0.660161815846869573927812110014

# values up to N
filepath = os.path.abspath(os.path.join(basepath, "..", "params.txt"))
N = 10**int(open(filepath).read().split()[1])


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


#################################################
##   A r i t h m e t i c   F u n c t i o n s   ##
#################################################

# Hardy-Littlewood factor
def hlfactor(n):
	prod = 1
	for p in primes:
		if p > sqrt(n): break
		else:
			if (p > 2) and (n % p == 0): prod *= float(p-1)/(p-2)
	return prod

# integral[2..n-2] dt/(log t)(log n-t)
def int_factor(n):
	return integrate.quad(lambda t: (1/log(n-t))*(1/log(t)), 2, n-2)[0]


	
#################
##   M a i n   ##
#################
	
filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "primes.txt"))
primes = vectorize(N, splita(open(filepath)))

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "r2ns.txt"))
xRTWO = vectorize(N, splita(open(filepath)))

X = range(2,N)

# get rid of odd numbers
Y = []
for x in X:
	if x % 2 == 0:
		Y.append(x)
		if x % 100 == 0:
			sys.stdout.write("\r%d" % x)
			sys.stdout.flush()
	

#flush
sys.stdout.write("\r          ")
sys.stdout.flush()

# array to be ploted
xSIM = []
for n in Y:
	try:	
		xSIM.append( (xRTWO[n-2])/( 2 * C2 * hlfactor(n) * int_factor(n) ) )
	except (ZeroDivisionError, ValueError):
		xSIM.append(1)
	
	if n % 1000 == 0:
		sys.stdout.write("\r%d" % n)
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
plt.xlabel(r"$n$ (even)")
plt.ylabel("ratio")

plt.plot(Y,xSIM, marker=',', linestyle='', color='crimson')

plt.xlim(0, Y[len(Y)-1])
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.ylim((0,2))

#plt.show()
plt.savefig(os.path.abspath(os.path.join(basepath, "..", "imgs", "figure_3.png")), dpi = 96)

sys.stdout.write("\r          \r")
sys.stdout.flush()
