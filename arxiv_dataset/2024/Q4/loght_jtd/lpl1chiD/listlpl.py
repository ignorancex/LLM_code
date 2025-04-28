from math import *
from mpmath import *
import os.path
import sys

import cypari2
pari = cypari2.Pari()
#pari.set_real_precision_bits(15)

basepath = os.path.dirname(__file__)

# fundamental discriminants up to N
filepath = os.path.abspath(os.path.join(basepath, "params.txt"))
N = 10**int(open(filepath).read().split()[1])
PREC = int(open(filepath).read().split()[3])

# precision, number of terms in deridiri
# UPTO = 10**3

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

#####
### Disclaimer: The next four functions below   ###
### are NOT actually used, in virtue of the way ###
### faster PARI/GP algorithm I found out about  ###
### in the middle of writing this algorithm :)  ###
#####

# calculate chi_D(n)
def chid(n, D):
	if (n == 1): return(1)
	elif (n == 0 or D%n == 0): return(0)
	elif (isprime(n) == 1):
		for k in range(n):
			if ((k**2) % n == (-D%n)): return(1)
		return(-1)
	else:
		for k in range(2,int(floor(sqrt(n))+1)):
			if (isprime(k) == 1 and n % k == 0):
				return chid(k,D)*chid(n/k,D)
			

# determine chi_D
def chicalc(D, prcs):
	if D == 4: return [0, 1, 0, -1]
	elif D == 8: return [0, 1, 0, 1, 0, -1, 0, -1]
	else:
		CHIS = []
		for n in range(min(D, prcs+1)):
			CHIS.append(chid(n,D))
		return CHIS

# von Mangoldt
def vonM(n):
	if (n == 1): return 0
	elif (isprime(n) == 1): return log(n)
	else:
		k = 2
		while (n % k != 0): k += 1
		if ((n/k) % k == 0): return vonM(n/k)
		else: return 0

# Dirichlet L-function
def deridiri(q, CHI, prcs):
	soma = 0
	for n in range(1,prcs+1):
		soma += -vonM(n)*CHI[n % q]/float(n)
	return soma


#################
##   M a i n   ##
#################

filepath1 = os.path.abspath(os.path.join(basepath, "Ds.txt"))
f1 = open(filepath1, 'a+')

filepath2 = os.path.abspath(os.path.join(basepath, "lpl1chiD.txt"))
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
#		chi = chicalc(D, D)
		lpl1 = -float(pari.lfun(-D,1,1)/pari.lfun(-D,1))
	
	try:
		if (lendiff == 0): f1.write(str(D) + ' ')
	except (NameError): 
		f1.write(str(D) + ' ')
	f2.write(nstr(lpl1, PREC) + ' ')
	
	sys.stdout.write("\r%d" % -D)
	sys.stdout.flush()
	
	D += 1
	lendiff = 0
