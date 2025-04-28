from math import *
from mpmath import *
import os.path
import sys

basepath = os.path.dirname(__file__)

# fundamental discriminants up to N
filepath = os.path.abspath(os.path.join(basepath, "..", "params.txt"))
Nexp = int(open(filepath).read().split()[1])
N = 10**int(open(filepath).read().split()[1])

# precision
PREC = int(open(filepath).read().split()[3])

#####################################
##   F i l e   F u n c t i o n s   ##
#####################################

# separate words from file
def splita( f):
	everything = f.read()
	return everything.split()

# load negFDs.txt
def takefds( n, vet):
	FDS = []
	for x in vet:
		if (int(x) <= n): FDS.append(int(x))
		else: break
	return FDS

# take last element
def takelast( vet):
	return vet[int(len(vet)-1)]

# compare vet sizes
def checklen( vet1, vet2):
	if ( len(vet1) == len(vet2)): return 0
	else: return 1


#################################################
##   A r i t h m e t i c   F u n c t i o n s   ##
#################################################

# primality test
def isprime(n):
	ret = 1
	if (n == 1): ret = 0
	else:
		for i in range(2, int(floor(sqrt(n))+1)):
			if (n % i == 0):
				ret = 0
				break
	return ret

# squarefreeness test
def issqfree(n):
	ret = 1
	for p in range(2, int(floor(sqrt(n))+1)):
		if (isprime(p) == 1):
			if (n % p == 0): n /= p
			if (n % p == 0):
				ret = 0
				break
	return ret

# roots of reduced bin quad forms of disc d
# -a < b <= a < c or 0 <= b <= a = c
# a <= sqrt(|D|/3)
# cf. Algorithm 5.3.5, \S5.3, p. 233 of COHEN "A course in computational AlgNT"
def redpts(D):
	RP = []
	b = int(D % 2)
	B = int(floor(sqrt(abs(D)/3)))
	
	while (b <= B):
		q = (b**2 - D)/4	#q = ac
		a = b
		while (a**2 <= q):
			if (a >= 1):
				if (q % a == 0):
					RP.append(float(-b)/(2*a) + (sqrt(abs(D))/(2*a))*j)
					if ((a != b) and (a**2 != q) and (b != 0)):
						RP.append(float(b)/(2*a) + (sqrt(abs(D))/(2*a))*j)
			a += 1
		b += 2	
	return RP

# absolute logarithmic Weil height of algebraic integer
# input: galois conjugates, output: real
def hei(A):
	soma = 0
	for a in A:
		soma += log(max(fabs(a), 1))
	return soma/len(A)


#################
##   M a i n   ##
#################

try:
	os.mkdir(os.path.join(basepath, "..", "lists", "1e" + str(Nexp)))
except (OSError):
	pass

filepath1 = os.path.abspath(os.path.join(basepath, "..", "lists", "1e" + str(Nexp), "negFDs_1e" + str(Nexp) + ".txt"))
f1 = open(filepath1, 'a+')
#f1.truncate()

filepath2 = os.path.abspath(os.path.join(basepath, "..", "lists", "1e" + str(Nexp), "hjtds_1e" + str(Nexp) + ".txt"))
f2 = open(filepath2, 'a+')
#f2.truncate()

try:	
	D = int(takelast( splita(open(filepath1))))
	lendiff = checklen( splita(open(filepath1)), splita(open(filepath2)))
	D += 1-lendiff
except (IndexError, IOError):
	D = 1

while D <= N:
	passer = 0
	
	if ((D % 4 == 3) and (issqfree(D) == 1)): passer = 1
	if ((D % 4 == 0) and ((D/4) % 4 == 1 or (D/4) % 4 == 2) and (issqfree(D/4) == 1)): passer = 1
	
	if (passer == 0):
		D += 1
		continue
	else:
		jGal = []
		RP = redpts(-D)
		for t in RP: jGal.append(1728 * kleinj(tau=t))
	
	try:
		if (lendiff == 0): f1.write(str(D) + ' ')
	except (NameError): 
		f1.write(str(D) + ' ')
	f2.write(nstr(hei(jGal)/(3*log(D)), PREC) + ' ')
	
	sys.stdout.write("\r%d" % -D)
	sys.stdout.flush()
	
	D += 1
	lendiff = 0
