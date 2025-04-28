#!/usr/bin/env python
import os
import numpy as np
import cantera as ct
import timeit
import argparse
import resource
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.sparse.linalg import eigs, svds
from scipy.linalg import eig
from scipy.special import factorial, binom
from scipy.integrate import ode, solve_ivp
import sys
import glob

#Command-line arguments
parser = argparse.ArgumentParser(description='Generate a sparse rate matrix from cantera model.')
parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for npy file output. Three files will be created for each reaction, storing rates, row indices, and column indices.')
parser.add_argument("--mechanism", type=str, required=False, default='mechanisms/h2o2.cti', dest='mechanism', help='Mechanism cti file. Default mechanisms/h2o2.cti.')
parser.add_argument("--refspecies", type=str, nargs='+', required=False, default=['H2', 'O2', 'OH', 'AR'], help="Reference multiindex for which the temperature and number of atoms are specified. Default ['H2', 'O2', 'OH', 'AR'].")
parser.add_argument("--refcounts", type=int, nargs='+', required=False, default=[8, 4, 1, 80], help='Reference multiindex for which the temperature and number of atoms are specified. Default [8, 4, 1, 80].')

parser.add_argument("--t0", type=float, required=False, default=1e-8, help='Initial integration time for propogating.')
parser.add_argument("--tmax", type=float, required=False, default=1e2, help='Final integration time for propogating.')
parser.add_argument("--Nt", type=int, required=False, default=101, help='Number of times to propogate.')

parser.add_argument("--seed", type=int, required=False, default=1, help='Random seed.')
parser.add_argument("--index", type=int, required=False, default=0, help='Initial state index.')

args = parser.parse_args()


#Main
filebase=args.filebase
mechanism=args.mechanism
np.random.seed(args.seed)
gas=ct.Solution(mechanism)
refmultiindex=np.zeros(len(gas.species_names),dtype=int)
if len(args.refspecies) != len(args.refcounts):
    print("refspecies and refcounts must be the same length")
for i in range(len(args.refspecies)):
    index=np.where([name ==  args.refspecies[i] for name in gas.species_names])[0][0]
    refmultiindex[index]=args.refcounts[i]

start=timeit.default_timer()
multiindices=np.load(filebase+"multiindices.npy")
dim=len(multiindices)
eigenvalues=np.load(filebase+"eigenvalues.npy")
eigenvectors=np.load(filebase+"eigenvectors.npy")
pinv=np.load(filebase+"pinv.npy")

tot=0
max=0
while np.abs(tot-1)>0.1 or np.abs(max-1)>0.1:
    index=np.random.randint(0,dim)
    if args.index != 0:
        index=args.index
    ic=np.zeros(dim)
    ic[index]=1
    alpha=pinv.dot(ic)
    tot=np.abs(np.sum(eigenvectors@alpha))
    max=np.max(np.abs(eigenvectors@alpha))
    print(index,tot,max)
print(multiindices[index])

times=[args.t0*(args.tmax/args.t0)**(n*1.0/(args.Nt-1)) for n in range(args.Nt)]
eigenvalues[-1]=0
states=np.real(np.array([np.dot(eigenvectors,np.exp(eigenvalues*t)*alpha) for t in times]))
np.save(filebase+"states_"+str(index)+".npy",states)
np.save(filebase+"times_"+str(index)+".npy",times)

runtime3=timeit.default_timer()-start

print("Runtime:", runtime3)
