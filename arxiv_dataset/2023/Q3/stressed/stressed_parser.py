import subprocess
import math
import time
import os
import sys

binomial = math.comb

data = {}

def runengine(n, l):
    with open(f"output/{n:02}_{l:02}", 'w') as f:
        start = time.time()
        subprocess.run(["./engine", str(n), str(l)], creationflags=8, stdout = f)
        end = time.time()
        delta = end-start
        print(f"Ran {n}, {l}; time: {delta:.3f} s")

def runall():
    for n in range(1, 57):
        runengine(n, 0)
    for n in range(57, 62):
        runengine(n, 2*n-93)
    
def loaddata(ddir = "output"):
    for fn in os.listdir(ddir):
        n = int(fn[:2])
        l = int(fn[3:5])
        with open(f"{ddir}/{fn}") as f:
            ds = f.read()
            d = [[int(a) for a in s.split()] for s in ds.strip().split("\n")]
            for i in range(l):
                d[i] = None
            data[n] = d

def count_stressed_n(n):
    finalcount = [0]*(3*n+1)
    for num1 in range((n+1)//2):
        for nposs3 in range(n):
            for j in range(nposs3 + 1):
                finalcount[2*n+1-num1+j] += data[n][num1][nposs3] * binomial(nposs3, j)
    return finalcount

def count_stressed_g(maxg):
    finalcount = [0]*(maxg+1)
    # a certain l yields max of 2+3l/2
    maxn = (2*maxg)//3-1 # 3->1, 4->1, 5->2, 6->3
    for n in range(1, maxn+1):
        for num1 in range((n+1)//2):
            if 2*n+1-num1 > maxg:
                continue
            elif 2*n+1-num1 == maxg:
                finalcount[maxg] += 2**num1*binomial((n-1)//2,num1)
                continue
            for nposs3 in range(n):
                for j in range(nposs3 + 1):
                    w = 2*n+1-num1+j
                    if w <= maxg:
                        finalcount[w] += data[n][num1][nposs3] * binomial(nposs3, j)
    return finalcount

def compute():
    loaddata()
    phi = (1+5**0.5)/2
    weights = [0]
    slength = [0]
    for n in range(1, 57):
        sn = count_stressed_n(n)
        slength.append(sum(sn))
        weights.append(sum(sn[i] * phi**(-i) for i in range(len(sn))))
    print(slength)
    print([sum(weights[:i+1]) for i in range(57)])
    print(count_stressed_g(95))
