#!/usr/bin/env python
import os
import numpy as np
import timeit
import argparse
import resource
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs, svds
from scipy.special import factorial, binom
from scipy.integrate import ode
import sys
import rlist
import glob

#Command-line arguments
parser = argparse.ArgumentParser(description='Generate a sparse rate matrix from cantera model.')
parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for npy file output. Three files will be created for each reaction, storing rates, row indices, and column indices.')
parser.add_argument("--species", type=int, required=False, default=9, help='Number of species.')
parser.add_argument("--reactions", type=int, required=False, default=10, help='Number of reactions.')
parser.add_argument("--molecules", type=int, required=False, default=10, help='Number of molecules.')
parser.add_argument("--seed", type=int, required=False, default=1, help='Random seed.')
parser.add_argument("--calculate", nargs=2, type=int, required=False, default=[0,-1], help='Rows to calculate in rate matrix. Takes starting index and ending index and save rate matrix corresponding to all reactions over these rows. Default [0 -1].')
parser.add_argument("--eigenvalues", type=int, required=False, default=-1, help='Flag to calculate  eigenvalues. If 1, then print args.temperature, args.pressure, total atoms, dimension, runtime, recursive calls, recursive levels, and save rate matrix, eigenvalues, and eigenvectors, then quit. Default 1.')
parser.add_argument("--accumulate", type=int, required=False, default=0, choices=[0,1], help='Flag to accumulate the multiindices from parallel runs.')
parser.add_argument("--propogate", type=int, required=False, default=0, choices=[0,1], help='Flag to propogate reference multiindex.')
parser.add_argument("--t0", type=float, required=False, default=1e-8, help='Initial integration time for propogating.')
parser.add_argument("--tmax", type=float, required=False, default=1, help='Final integration time for propogating.')
parser.add_argument("--Nt", type=int, required=False, default=100, help='Number of times to propogate.')
parser.add_argument("--print", type=int, required=False, default=0, choices=[0,1], help='Print runtimes.')
parser.add_argument("--reversible", type=int, required=False, default=0, choices=[0,1], help='Include reverse reactions.')
parser.add_argument("--scale", type=float, required=False, default=1, help='Exponential constant for rate constant sampling.')

args = parser.parse_args()

#Functions for relating multiindices to matrix indices
def get_multiindex(index):
    return multiindices[index]
def get_index(multiindex):
    return np.where(np.all(multiindices==multiindex,axis=1))[0][0]

def get_rate (multiindex, stoi, k):

    if np.all(multiindex>=stoi):
        return k*np.product(binom(multiindex, stoi)*factorial(stoi))
    else:
        return 0.

#We can parallelize this by calculating for each row, which should go relatively fast.
def calculate_sparse_elements_row(rstoi,pstoi,k,i):
    data=[]
    rows=[]
    columns=[]
    multiindex=get_multiindex(i)
    multiindex2=multiindex-rstoi+pstoi
    if np.all(multiindex2>=0.):
        rate=k*np.product(binom(multiindex, rstoi)*factorial(rstoi))
        j=get_index(multiindex2)
        data.append(rate)
        rows.append(i)
        columns.append(j)
        data.append(-rate)
        rows.append(i)
        columns.append(i)
    return data,rows,columns

#Main loop over rows to enumerate sparse data
def calculate_sparse_elements(rstoi, pstoi, k, startrow, endrow):
    data=[]
    rows=[]
    columns=[]
    for  i in range(startrow,endrow):
        rowdata, rowrows, rowcolumns = calculate_sparse_elements_row(rstoi, pstoi, k, i)
        data+=rowdata
        rows+=rowrows
        columns+=rowcolumns
    return data,rows,columns

#Main
filebase=args.filebase
ns=args.species
if args.reversible==1:
    nr=2*args.reactions
else:
    nr=args.reactions
nm=args.molecules
np.random.seed(args.seed)
refmultiindex=np.zeros(ns,dtype=int)
#initial state occupied randomly
for ind in np.random.choice(np.arange(ns),size=args.molecules,replace=True):
    refmultiindex[ind]+=1

#Calculate the space of possible states
if args.accumulate==0:
    start=timeit.default_timer()
    rvecs=np.zeros((nr,ns),dtype=int)

    #ensure that the reactions are distinct...
    if args.reversible==1:
        rates=np.random.exponential(args.scale, size=2*nr)
        for i in range(0,nr,2):
            reactants=np.random.choice(np.arange(ns),size=2,replace=False)
            products=np.random.choice(np.setdiff1d(np.arange(ns),reactants),size=2,replace=False)

            rvecs[i,reactants]=-1
            rvecs[i,products]=1
            rvecs[i+1,reactants]=1
            rvecs[i+1,products]=-1
    else:
        rates=np.random.exponential(args.scale, size=nr)
        for i in range(nr):
            reactants=np.random.choice(np.arange(ns),size=2,replace=False)
            products=np.random.choice(np.setdiff1d(np.arange(ns),reactants),size=2,replace=False)
            rvecs[i,reactants]=-1
            rvecs[i,products]=1
    print(rates)

    multiindices=rlist.list(refmultiindex, rvecs)
    sorted=np.argsort(np.sum((ns**np.arange(ns)*multiindices),axis=1))
    multiindices=multiindices[sorted]

    dim=len(multiindices)
    runtime=timeit.default_timer()-start

    np.save(filebase+"multiindices.npy",multiindices)
    out=open(filebase+"out.dat","w")

    print(dim, ns, runtime, file=out)
    print(*refmultiindex, file=out)
    out.close()
    if(args.print == 1):
        print("state space dimension: ", dim)
        print("state space runtime: ", runtime)
else:
    if os.path.isfile(filebase+"multiindices.npy"):
        multiindices=np.load(filebase+"multiindices.npy")

    file=open(filebase+"out.dat","r")
    instrings=file.readline().split()
    dim=int(instrings[0])
    runtime=float(instrings[1])
    instrings=file.readline().split()
    refmultiindex=np.array([int(float(i)) for i in instrings])

startrow, endrow=args.calculate
if endrow==-1 or endrow > dim:
    endrow=dim
if endrow>startrow:
    #Loop through each reaction index and calculate spase elements
    start=timeit.default_timer()

    data=[]
    rows=[]
    columns=[]
    for rind in range(nr):
        rstoi=np.zeros(ns,dtype=int)
        pstoi=np.zeros(ns,dtype=int)
        rindices=np.where(rvecs[rind]<0)
        pindices=np.where(rvecs[rind]>0)
        rstoi[rindices]=-rvecs[rind,rindices].astype(int)
        pstoi[pindices]=rvecs[rind,pindices].astype(int)
        reac_data,reac_rows,reac_columns=calculate_sparse_elements(rstoi, pstoi, rates[rind], startrow, endrow)
        data+=reac_data
        rows+=reac_rows
        columns+=reac_columns
    if not os.path.exists(filebase+"rows"):
        os.mkdir(filebase+"rows")
    if not os.path.exists(filebase+"columns"):
        os.mkdir(filebase+"columns")
    if not os.path.exists(filebase+"data"):
        os.mkdir(filebase+"data")
    np.save(filebase+"rows/%i_%i.npy"%(startrow,endrow),rows)
    np.save(filebase+"columns/%i_%i.npy"%(startrow,endrow),columns)
    np.save(filebase+"data/%i_%i.npy"%(startrow,endrow),data)
    runtime=timeit.default_timer()-start
    out=open(filebase+"_%icout.dat"%(start),"w")
    print("sparsity ", len(rows)/(dim*dim), runtime, file=out)
    out.close()
    if(args.print == 1):
        print("calculate runtime:", runtime)

#Calculate eigenvalues
if args.eigenvalues==-1:
    args.eigenvalues=dim

if args.accumulate==1:
    if  os.path.isfile(args.filebase+"rows.npy") and os.path.isfile(args.filebase+"columns.npy") and os.path.isfile(args.filebase+"data.npy"):
        rows=np.load(args.filebase+"rows.npy")
        columns=np.load(args.filebase+"columns.npy")
        data=np.load(args.filebase+"data.npy")
    else:
        files=glob.glob(args.filebase+"rows/*.npy")
        rows=[]
        for file in files:
            row=np.load(file).tolist()
            rows+=row
        files=glob.glob(args.filebase+"columns/*.npy")
        columns=[]
        for file in files:
            column=np.load(file).tolist()
            columns+=column
        files=glob.glob(args.filebase+"data/*.npy")
        data=[]
        for file in files:
            dat=np.load(file).tolist()
            data+=dat
np.save(filebase+"rows.npy",rows)
np.save(filebase+"columns.npy",columns)
np.save(filebase+"data.npy",data)
if args.eigenvalues>0:
    #accumulate rows, data, and columns
    start=timeit.default_timer()

    ratematrix=coo_matrix((np.array(data),(np.array(columns),np.array(rows))),(int(dim),int(dim)))

    if args.eigenvalues < ratematrix.shape[0]:
        eigenvalues,eigenvectors=eigs(ratematrix, args.eigenvalues, sigma=-1e-1, which='LM')
        svals=svds(ratematrix, args.eigenvalues, which='SM', return_singular_vectors=False)
    else:
        eigenvalues,eigenvectors=np.linalg.eig(ratematrix.toarray())
        svals=np.linalg.svd(ratematrix.toarray(),compute_uv=False)
    sorted=np.argsort(eigenvalues)
    np.save(filebase+"eigenvalues.npy",eigenvalues.astype(complex)[sorted])
    np.save(filebase+"singularvalues.npy",svals)
    np.save(filebase+"eigenvectors.npy",eigenvectors.astype(complex)[:,sorted])

    runtime=timeit.default_timer()-start
    out=open(filebase+"eout.dat","w")
    print(runtime, eigenvalues[sorted[0]], eigenvalues[sorted[-1]], file=out)
    out.close()
    if(args.print == 1):
        print("eigenvalues runtime:", runtime)

if args.propogate == 1:
    start=timeit.default_timer()

    if os.path.isfile(args.filebase+"rows.npy") and os.path.isfile(args.filebase+"columns.npy") and os.path.isfile(args.filebase+"data.npy"):
        rows=np.load(args.filebase+"rows.npy")
        columns=np.load(args.filebase+"columns.npy")
        data=np.load(args.filebase+"data.npy")
    else:
        files=glob.glob(args.filebase+"rows/*.npy")
        rows=[]
        for file in files:
            row=np.load(file).tolist()
            rows+=row
        files=glob.glob(args.filebase+"columns/*.npy")
        columns=[]
        for file in files:
            column=np.load(file).tolist()
            columns+=column
        files=glob.glob(args.filebase+"data/*.npy")
        data=[]
        for file in files:
            dat=np.load(file).tolist()
            data+=dat
        np.save(filebase+"rows.npy",rows)
        np.save(filebase+"columns.npy",columns)
        np.save(filebase+"data.npy",data)

    ratematrix=coo_matrix((np.array(data),(np.array(columns),np.array(rows))),(int(dim),int(dim)))

    def func(t,y):
        return ratematrix.dot(y)

    y0=np.zeros(dim)
    y0[get_index(refmultiindex)]=1.0
    times=[args.t0*(args.tmax/args.t0)**(n*1.0/args.Nt) for n in range(args.Nt)]
    sol=solve_ivp(func,[0,args.tmax], y0, method='BDF', t_eval=times, atol=1e-8, rtol=1e-6, first_step=args.t0/100, jac=ratematrix)
    vals=np.transpose(np.array(sol.y)).tolist()

    np.save(filebase+"times.npy",np.array(times))
    np.save(filebase+"propogate.npy",vals)

    runtime=timeit.default_timer()-start
    out=open(filebase+"rout.dat","w")
    print(runtime, file=out)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=out)
    print(dim, file=out)
    out.close()
    if(args.print == 1):
        print("propogate success:", sol.success)
        print("propogate runtime:", runtime)
if(args.print == 1):
    print("memory:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
