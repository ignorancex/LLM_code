import matplotlib.pyplot as plt
from math import *
import os.path
import sys

basepath = os.path.dirname(__file__)

# phis up to N
filepath = os.path.abspath(os.path.join(basepath, "..", "params.txt"))
N = 10**int(open(filepath).read().split()[5])


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

# primality test
def isprime(n):
	ret = 1
	for i in range(2, int(floor(sqrt(n))+1)):
		if (n % i == 0):
			ret = 0
			break
	return ret

# gcd
def gcd(a, b):
	if (a % b == 0): return b
	else: return gcd(b, a % b)

# euler totient
def phi(n):
	v_phi = 1
	Dn = []
	lim = sqrt(n)
	for p in primes:
		if n == 1: break
		elif p <= lim:
			while n % p == 0:
				Dn.append(p)
				n /= p
		else:
			Dn.append(n)
			break
	while len(Dn) > 0:
		p = Dn[0]
		exp = 0
		while p in Dn:
			exp += 1
			Dn.remove(p)
		v_phi *= p**exp - p**(exp-1)
	return v_phi

# distinct prime divisor function
def omega(n):
	cont = 0
	for p in primes:
		if p > sqrt(n): break
		if n % p == 0:
			cont += 1
	return cont

	
#################
##   M a i n   ##
#################

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "primes.txt"))
primes = vectorize(N, splita(open(filepath)))

X = range(N+1)

#array xPIE
xPIE = [0]
i = 0
for n in range(1, N+1):
	try:
		if n < primes[i]: xPIE.append( i)
		else:
			i += 1
			xPIE.append( i)
	except IndexError:
		xPIE.append( i)
sys.stdout.write("\r1/3")
sys.stdout.flush()

#array xPHI
xPHI = [0]
for n in range(1, N+1):
	xPHI.append( phi(n))
sys.stdout.write("\r2/3")
sys.stdout.flush()
		
#array xOMEG		
xOMEG = [0]
for n in range(1, N+1):
	xOMEG.append( omega(n))
sys.stdout.write("\r3/3")
sys.stdout.flush()


# get rid of odd numbers
Y = []
for x in X:
	if x % 2 == 0: Y.append(x)
X = Y

	
#arrays xPW e x2PHI2
xPW = []
for n in X:
	xPW.append( xPIE[n] - xOMEG[n])

x2PHI2 = []
for n in X:
	x2PHI2.append( xPHI[n]/2)


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
plt.xlabel(r"$2n$")
plt.text(50000, 19000, r"$\varphi(2n)/2$", fontsize=40, color='.5')
plt.text(75000, 4000, r"$\pi(2n)-\omega(2n)$", fontsize=40, color='0')

plt.plot(X,x2PHI2, marker='.', linestyle = '', color='.5')
plt.plot(X,xPW, linestyle = '-', color='0')

plt.xlim(0, X[len(X)-1])
plt.ylim(ymin=0)

#plt.show()
plt.savefig(os.path.abspath(os.path.join(basepath, "..", "imgs", "figure_1.png")), dpi = 96)

sys.stdout.write("\r          \r")
sys.stdout.flush()
