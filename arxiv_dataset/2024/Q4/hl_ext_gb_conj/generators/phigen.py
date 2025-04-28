from math import sqrt
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

# load primes.txt
def takeprimes( n, vet):
	primes = []
	for x in vet:
		if (int(x) <= n): primes.append(int(x))
		else: break
	return primes
	

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

# totient euler
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


#################
##   M a i n   ##
#################

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "primes.txt"))
primes = takeprimes(N, splita(open(filepath)))

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "phis.txt"))
f = open(filepath, 'w+')
f.truncate()

for n in range(1, N+1):
	f.write(str(phi(n)) + ' ')
	sys.stdout.write("\r%d" % n)
	sys.stdout.flush()

sys.stdout.write("\r          \r")
sys.stdout.flush()
