import os.path
import sys
from math import sqrt

basepath = os.path.dirname(__file__)

# primes up to N
filepath = os.path.abspath(os.path.join(basepath, "..", "params.txt"))
N = 10**int(open(filepath).read().split()[1])


#################
##   M a i n   ##
#################

filepath = os.path.abspath(os.path.join(basepath, "..", "lists", "primes.txt"))
f = open(filepath, 'w+')
f.truncate()

count = N  #variable to keep track of progress

# Eratosthenes sieve for p leq sqrt_n	
p = 2
X = range(N+1)
X[1] = 0


while p <= sqrt(N):
	sys.stdout.write("\r%d" % p)
	sys.stdout.flush()

	i = p
	while ((i+p) <= N):
		X[i+p] = 0
		i += p

	p += 1
	while X[p] == 0: p += 1

# Write nonzero terms
for x in X:
	if x != 0: f.write(str(x) + ' ')

sys.stdout.write("\r          \r")
sys.stdout.flush()
