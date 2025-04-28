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

#Command-line arguments
parser = argparse.ArgumentParser(description='Generate a sparse rate matrix from cantera model.')
parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for npy file output. Three files will be created for each reaction, storing rates, row indices, and column indices.')
parser.add_argument("--mechanism", type=str, required=False, default='mechanisms/h2o2.cti', dest='mechanism', help='Mechanism cti file. Default mechanisms/h2o2.cti.')
parser.add_argument("--temperature", type=float, required=False, default=1000, help='Temperature in Kelvin. Default 1000.')
parser.add_argument("--adiabatic", type=int, choices=[0, 1, 2], required=False, default=1, help='Convert energy from reactions to heat. The values 0, 1, and 2 correspond to constant volume/temperature, constant volume/energy, and constant pressure/enthalpy, respectively. The temperature is specify the reference multiindix specified with --reference. ')
parser.add_argument("--refspecies", type=str, nargs='+', required=False, default=['H2', 'O2', 'OH', 'AR'], help="Reference multiindex for which the temperature and number of atoms are specified. Default ['H2', 'O2', 'OH', 'AR'].")
parser.add_argument("--refcounts", type=int, nargs='+', required=False, default=[8, 4, 1, 80], help='Reference multiindex for which the temperature and number of atoms are specified. Default [8, 4, 1, 80].')

parser.add_argument("--pressure", type=float, required=False, default=1, help='Pressure in atm. Default 1.')
parser.add_argument("--calculate", type=int, required=False, default=1, choices=[0,1], help='Flag to calculate  rate matrix. Default 1.')
parser.add_argument("--eigenvalues", type=int, required=False, default=-1, help='Flag to calculate  eigenvalues. If 1, then print args.temperature, args.pressure, total atoms, dimension, runtime, recursive calls, recursive levels, and save rate matrix, eigenvalues, and eigenvectors, then quit. Default 1.')
parser.add_argument("--propogate", type=int, required=False, default=0, choices=[0,1], help='Flag to propogate reference multiindex.')
parser.add_argument("--thrs", type=float, required=False, default=1e-3, help='Threshold for including reactions.')
parser.add_argument("--tau", type=float, required=False, default=1e-5, help='Time scale for shift invert.')
parser.add_argument("--t0", type=float, required=False, default=1e-8, help='Initial integration time for propogating.')
parser.add_argument("--tmax", type=float, required=False, default=1e2, help='Final integration time for propogating.')
parser.add_argument("--Tmin", type=float, required=False, default=200, help='Minimum temperature for states to enumerate.')
parser.add_argument("--Nt", type=int, required=False, default=101, help='Number of times to propogate.')
parser.add_argument("--print", type=int, required=False, default=1, choices=[0,1], help='Print runtimes.')
parser.add_argument("--csv", type=int, required=False, default=0, choices=[0,1], help='Save files to csv format.')

args = parser.parse_args()

#Functions for relating multiindices to matrix indices
def get_multiindex(index):
    return multiindices[index]
def get_index(multiindex):
    temp=np.where(np.all(multiindices==multiindex,axis=1))[0]
    if(len(temp)>0):
        return temp[0]
    else:
        return -1;

def get_rate (multiindex, stoi, k, vol):
    if np.all(multiindex>=stoi):
        return k*np.product(binom(multiindex, stoi))/(ct.avogadro*vol)**(np.sum(stoi)-1)
    else:
        return 0.

#Given that we already changed the gas to each state, we should have saved the rate constants and reused them here
def calculate_sparse_elements_row(rind,i):
    data=[]
    rows=[]
    columns=[]
    multiindex=get_multiindex(i)
    #forward reaction
    multiindex2=multiindex-rstois[rind]+pstois[rind]
    if np.all(multiindex2>=0):
        j=get_index(multiindex2)
        k=frateconstants[i,rind]
        if not np.isnan(k) and j>=0:
            rate=get_rate(multiindex,rstois[rind],k,refvol)
            data.append(rate)
            rows.append(i)
            columns.append(j)
    #reverse reaction
    multiindex2=multiindex+rstois[rind]-pstois[rind]
    if np.all(multiindex2>=0):
        k=rrateconstants[i,rind]
        j=get_index(multiindex2)
        if not np.isnan(k) and j>=0:
            rate=get_rate(multiindex,pstois[rind],k,refvol)
            data.append(rate)
            rows.append(i)
            columns.append(j)
    return data,rows,columns

#Main loop over rows to enumerate sparse data
def calculate_sparse_elements(rind, startrow, endrow):
    data=[]
    rows=[]
    columns=[]
    for  i in range(startrow,endrow):
        rowdata, rowrows, rowcolumns = calculate_sparse_elements_row(rind,i)
        data+=rowdata
        rows+=rowrows
        columns+=rowcolumns
    return data,rows,columns

def setstate(multiindex):
    T,P,X=gas.TPX
    if(args.adiabatic == 1):
        try:
            gas.UVX=refenergy/refmass,refvol/refmass,multiindex
            quant=ct.Quantity(gas, moles=np.sum(multiindex)/ct.avogadro)
            if(quant.T > args.Tmin):
                T=quant.T
                P=quant.P
            else:
                gas.TPX=T,P,X
                return 0,0
        except:
            gas.TPX=T,P,X
            return 0,0
    elif(args.adiabatic == 2):
        try:
            gas.HPX=(refenth+np.dot(multiindex-refmultiindex,refpotentials)/(1000*ct.avogadro))/refmass,args.pressure*ct.one_atm,multiindex
            quant=ct.Quantity(gas, moles=np.sum(multiindex)/ct.avogadro)
            if(quant.T > args.Tmin):
                T=quant.T
                P=quant.P
            else:
                gas.TPX=T,P,X
                return 0,0
        except:
            gas.TPX=T,P,X
            return 0,0
    return T,P/ct.one_atm

def list(current, threshold):
    stack=[current]
    previous=[str(current)]
    T,P=setstate(current)
    frateconstants=[gas.forward_rate_constants]
    rrateconstants=[gas.reverse_rate_constants]
    multiindices=[current]
    temperatures=[T]
    pressures=[P]

    while(len(stack)>0):
        current=stack.pop()
        for rind in range(nr):
            T,P=setstate(current)
            next=current-rstois[rind]+pstois[rind]
            if np.all(next>=0) and not str(next) in previous:
                k=gas.forward_rate_constants[rind]
                frate=get_rate(current,rstois[rind],k,refvol)
                T,P=setstate(next)
                if T!=0:
                    k=gas.reverse_rate_constants[rind]
                    rrate=get_rate(next,pstois[rind],k,refvol)
                    if frate>threshold*rrate:
                        stack.append(next)
                        multiindices.append(next)
                        temperatures.append(T)
                        pressures.append(P)
                        previous.append(str(next))
                        frateconstants.append(gas.forward_rate_constants)
                        rrateconstants.append(gas.reverse_rate_constants)
            if gas.is_reversible(rind):
                T,P=setstate(current)
                next=current+rstois[rind]-pstois[rind]
                if np.all(next>=0) and not str(next) in previous:
                    k=gas.reverse_rate_constants[rind]
                    frate=get_rate(current,pstois[rind],k,refvol)
                    T,P=setstate(next)
                    if T!=0:
                        k=gas.forward_rate_constants[rind]
                        rrate=get_rate(next,rstois[rind],k,refvol)
                        if frate>threshold*rrate:
                            stack.append(next)
                            multiindices.append(next)
                            temperatures.append(T)
                            pressures.append(P)
                            previous.append(str(next))
                            frateconstants.append(gas.forward_rate_constants)
                            rrateconstants.append(gas.reverse_rate_constants)

    return np.array(multiindices), np.array(temperatures), np.array(pressures), np.array(frateconstants), np.array(rrateconstants)

#Main
filebase=args.filebase
mechanism=args.mechanism
gas=ct.Solution(mechanism)
gas.TP=args.temperature,args.pressure*ct.one_atm
ns=gas.n_species
nr=gas.n_reactions
species=gas.species_names
elements=gas.element_names
refmultiindex=np.zeros(ns,dtype=int)
if len(args.refspecies) != len(args.refcounts):
    print("refspecies and refcounts must be the same length")
for i in range(len(args.refspecies)):
    index=np.where([name ==  args.refspecies[i] for name in gas.species_names])[0][0]
    refmultiindex[index]=args.refcounts[i]

sp_atoms=[]
for i in range(ns):
    sp_atoms.append(np.array([int(gas.species()[i].composition[el] if el in gas.species()[i].composition.keys() else 0) for el in elements]))
sp_atoms=np.array(sp_atoms)
atoms=np.zeros(len(elements),dtype=int)
for i in range(len(refmultiindex)):
    atoms += refmultiindex[i]*sp_atoms[i]

atoms=np.array(atoms)
gas.TPX=args.temperature,args.pressure*ct.one_atm,refmultiindex
refquant=ct.Quantity(gas,moles=np.sum(refmultiindex)/ct.avogadro)
refenth=refquant.enthalpy
refentr=refquant.entropy
refenergy=refquant.int_energy
refmass=refquant.mass
refvol=refquant.volume
refpotentials=refquant.chemical_potentials
rstois=np.array(np.transpose(gas.reactant_stoich_coeffs()),dtype=int)
pstois=np.array(np.transpose(gas.product_stoich_coeffs()),dtype=int)


runtime1=0
runtime2=0
runtime3=0
runtime4=0
#Calculate the space of possible states
if args.calculate == 1:
    start=timeit.default_timer()
    multiindices,temperatures,pressures,frateconstants,rrateconstants=list(refmultiindex,args.thrs)
    sorted=np.argsort(np.sum((ns**np.arange(ns)*multiindices),axis=1))
    multiindices=multiindices[sorted]
    temperatures=temperatures[sorted]
    pressures=pressures[sorted]
    frateconstants=frateconstants[sorted]
    rrateconstants=rrateconstants[sorted]
    dim=len(multiindices)

    runtime1=timeit.default_timer()-start
    np.save(filebase+"multiindices.npy",multiindices)
    np.save(filebase+"temperatures.npy",temperatures)
    np.save(filebase+"pressures.npy",pressures)

    if(args.print == 1):
        print("state space dimension: ", dim)
        print("state space runtime: ", runtime1)
        sys.stdout.flush()

    #Loop through each reaction index and calculate spase elements
    start=timeit.default_timer()

    gas=ct.Solution(mechanism)
    gas.TPX=args.temperature,args.pressure*ct.one_atm,refmultiindex
    refquant=ct.Quantity(gas,moles=np.sum(refmultiindex)/ct.avogadro)
    refenth=refquant.enthalpy
    refmass=refquant.mass
    refvol=refquant.volume
    quant=ct.Quantity(gas, moles=np.sum(refmultiindex)/ct.avogadro)

    data=[]
    rows=[]
    columns=[]
    for rind in range(nr):
        reac_data,reac_rows,reac_columns=calculate_sparse_elements(rind, 0, dim)
        data+=reac_data
        rows+=reac_rows
        columns+=reac_columns
    np.save(filebase+"spatoms.npy",sp_atoms)

    np.save(filebase+"rows.npy",rows)
    np.save(filebase+"columns.npy",columns)
    np.save(filebase+"data.npy",data)
    runtime2=timeit.default_timer()-start

    if(args.print == 1):
        print("calculate runtime:", runtime2)
        sys.stdout.flush()

multiindices=np.load(filebase+"multiindices.npy")
dim=len(multiindices)
rows=np.load(filebase+"rows.npy")
columns=np.load(filebase+"columns.npy")
data=np.load(filebase+"data.npy")

ratematrix=coo_matrix((np.array(data),(np.array(columns),np.array(rows))),(int(dim),int(dim)))

ratematrix=ratematrix.tolil()
ratematrix=ratematrix-np.diag(np.sum(ratematrix.toarray(),axis=0))

#Calculate eigenvalues
if args.eigenvalues==-1:
    args.eigenvalues=dim
if args.eigenvalues>0:
    start=timeit.default_timer()
    if args.eigenvalues < ratematrix.shape[0]:
        v0=np.zeros(dim)
        v0[get_index(refmultiindex)]=1
        eigenvalues,eigenvectors=eigs(ratematrix, args.eigenvalues, sigma=-1/args.tau, which='LM',v0=v0)
    else:
        eigenvalues,eigenvectors=eig(ratematrix)

    sorted=np.argsort(np.real(eigenvalues))
    eigenvalues=eigenvalues[sorted]
    eigenvectors=eigenvectors[:,sorted]

    np.save(filebase+"eigenvalues.npy",eigenvalues.astype(complex))
    np.save(filebase+"eigenvectors.npy",eigenvectors.astype(complex))

    index=np.where(np.all(refmultiindex==multiindices,axis=1))[0]
    ic=np.zeros(dim)
    ic[index]=1
    pinv=np.linalg.pinv(eigenvectors)
    alpha=pinv.dot(ic)
    np.save(filebase+"alpha.npy",alpha.astype(complex))
    np.save(filebase+"pinv.npy",pinv.astype(complex))

    times=[args.t0*(args.tmax/args.t0)**(n*1.0/(args.Nt-1)) for n in range(args.Nt)]
    eigenvalues[-1]=0
    states=np.real(np.array([np.dot(eigenvectors,np.exp(eigenvalues*t)*alpha) for t in times]))
    np.save(filebase+"states.npy",states)
    np.save(filebase+"times.npy",times)

    if args.csv:
        np.savetxt(filebase+"states.csv", states.tolist(), fmt='%.18e', delimiter=',')
        np.savetxt(filebase+"eigenvalues.csv", eigenvalues.tolist(), fmt='%.18e', delimiter=',')
        np.savetxt(filebase+"eigenvectors.csv", eigenvectors.tolist(), fmt='%.18e', delimiter=',')

    runtime3=timeit.default_timer()-start

    if(args.print == 1):
        print("eigenvalues runtime:", runtime3)
        sys.stdout.flush()

if args.propogate == 1:
    start=timeit.default_timer()

    def func(t,y):
        return ratematrix.dot(y)
    y0=np.zeros(dim)
    y0[get_index(refmultiindex)]=1.0
    times=[args.t0*(args.tmax/args.t0)**(n*1.0/(args.Nt-1)) for n in range(args.Nt)]

    sol=solve_ivp(func,[0,args.tmax], y0, method='BDF', t_eval=times, atol=1e-6, rtol=1e-6, first_step=args.t0/100, jac=ratematrix)
    vals=np.transpose(np.array(sol.y)).tolist()

    np.save(filebase+"times.npy",np.array(times))
    np.save(filebase+"propogate.npy",vals)

    runtime4=timeit.default_timer()-start

    if(args.print == 1):
        print("propogate success:", sol.success)
        print("propogate runtime:", runtime4)
        sys.stdout.flush()


out=open(filebase+"out.dat","w")
print(dim, ns, runtime1, runtime2, runtime3, runtime4, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=out)
print(*refmultiindex, file=out)
print(*elements, file=out)
out.close()
if(args.print == 1):
    print("memory:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    sys.stdout.flush()

if args.csv:
    np.savetxt(filebase+"w_rows.csv", rows, fmt='%.18e', delimiter=',')
    np.savetxt(filebase+"w_columns.csv", columns, fmt='%.18e', delimiter=',')
    np.savetxt(filebase+"w_data.csv", data, fmt='%.18e', delimiter=',')
    np.savetxt(filebase+"species.csv", np.array(multiindices,dytpe=int))
    head=''
    for el in gas.element_names:
        head = head + el + ','
    np.savetxt(filebase+"atoms.csv", np.array(sp_atoms, dtype=int), header=head)
