#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import cantera as ct
import timeit
import argparse


def runsim (times):
    if(pressureflag==0):
        if(energyflag==1):
            r = ct.IdealGasReactor(gas, name='R1')
        else:
            r = ct.IdealGasReactor(gas, name='R1', energy='off')
    else:
        if(energyflag==1):
            r = ct.IdealGasConstPressureReactor(gas, name='R1')
        else:
            r = ct.IdealGasConstPressureReactor(gas, name='R1', energy='off')
    sim = ct.ReactorNet([r])

    states = ct.SolutionArray(gas, extra=['t'])

    for t in times:
        sim.advance(t)
        norm=0

        states.append(r.thermo.state, t=t)

    return states

parser = argparse.ArgumentParser(description='Generate a sparse rate matrix from cantera model.')
parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for npy file output. Three files will be created for each reaction, storing rates, row indices, and column indices.')
parser.add_argument("--temperature", type=float, required=False, default=1000, help='Temperature in Kelvin. Default 1000.')
parser.add_argument("--adiabatic", type=int, choices=[0, 1, 2], required=False, default=1, help='Convert energy from reactions to heat. The values 0, 1, and 2 correspond to constant volume/temperature, constant volume/energy, and constant pressure/enthalpy, respectively. The temperature is specify the reference multiindix specified with --reference. ')
parser.add_argument("--refspecies", type=str, nargs='+', required=False, default=['H2', 'O2', 'OH', 'AR'], help="Reference multiindex for which the temperature and number of atoms are specified. Default ['H2', 'O2', 'OH', 'AR'].")
parser.add_argument("--refcounts", type=int, nargs='+', required=False, default=[8, 4, 0, 80], help='Reference multiindex for which the temperature and number of atoms are specified. Default [8, 4, 0, 80].')
parser.add_argument("--pressure", type=float, required=False, default=1, help='Pressure in atm. Default 1.')
parser.add_argument("--t0", type=float, required=False, default=1e-8, help='Initial integration time for propogating.')
parser.add_argument("--tmax", type=float, required=False, default=1e2, help='Final integration time for propogating.')
parser.add_argument("--Npoints", type=int, required=False, default=100, help='Number of times to propogate.')
parser.add_argument("--mechanism", type=str, required=False,default='h2o2.cti', help='Cantera mechanism file.')
args = parser.parse_args()

start=timeit.default_timer()
Npoints = args.Npoints
t0=args.t0
tmax=args.tmax
temperature=args.temperature
pressure=args.pressure*ct.one_atm

if args.adiabatic == 0:
    energyflag=0
    pressureflag=1
elif args.adiabatic == 1:
    energyflag=1
    pressureflag=0
else:
    energyflag=1
    pressureflag=1
out=args.filebase

#Change initial conditions here
gas = ct.Solution(args.mechanism)
species=gas.species_names
reactions = gas.reactions()
ns=gas.n_species
refmultiindex=np.zeros(ns,dtype=int)
if len(args.refspecies) != len(args.refcounts):
    print("refspecies and refcounts must be the same length")
for i in range(len(args.refspecies)):
    index=np.where([name ==  args.refspecies[i] for name in gas.species_names])[0][0]
    refmultiindex[index]=args.refcounts[i]
gas.TPX = temperature, pressure, refmultiindex
times=[t0*(tmax/t0)**(n*1.0/Npoints) for n in range(Npoints)]

observation=runsim(times)

rstois=np.array(np.transpose(gas.reactant_stoich_coeffs()),dtype=int)
pstois=np.array(np.transpose(gas.product_stoich_coeffs()),dtype=int)

np.save(args.filebase+"rstois.npy",rstois)
np.save(args.filebase+"pstois.npy",pstois)
np.save(args.filebase+"rates.npy",observation.net_rates_of_progress)
np.save(args.filebase+"times.npy", observation.t)
np.save(args.filebase+"concentrations.npy", observation.X)
np.save(args.filebase+"temperatures.npy", observation.T)
np.save(args.filebase+"pressures.npy", observation.P/ct.one_atm)


stop=timeit.default_timer()
print ('runtime: %f'%(stop-start))
sys.stdout.flush()
