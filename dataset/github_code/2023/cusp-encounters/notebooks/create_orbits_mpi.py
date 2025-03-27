# This file computes a set of orbits in the Milky Way potential
#
# If you use the recommended the default setup of the repository, and you
# have downloaded the precomputed caches with ../init.sh, then this code
# should simply read the precomputed results and return immediately
#
# However, you can also create your own set of orbits
# The results are automatically cached into ../caches/milkyway_orbits.hdf5
# To use with mpi parallelization on e.g. 8 processors:
#   mpirun -n 8 python -u create_orbits_mpi.py
# With the current parameter setup this will take a couple of hours. To
# first test that everything is working, you might wanna reduce e.g.
# the number of integration steps below, before executing at full length
#

import numpy as np
import time

import sys
sys.path.append("../adiabatic-tides")
sys.path.append("..")
import cusp_encounters.milkyway

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except:
    print("MPI import did not work. I will proceed without mpi4py")
    comm = None
    rank = 0

cachedir = "../caches"
mw = cusp_encounters.milkyway.MilkyWay(adiabatic_contraction=True, cachedir=cachedir, mode="cautun_2020")

t0 = time.time()

print("""Note: In create_orbits_mpi.py we are using subsamp=1000, to reduce storage / memory requirements.
         This still gives very good approximations to the plots in the paper, but if you want to exactly recreate the paper
         results, please change to subsamp=100 and recreate the cache by yourself (See create_orbits_mpi.py)""")
orbits = mw.create_dm_orbits(100000, nsteps=100000, rmax=500e3, addinfo=True, adaptive=True, subsamp=1000, mpicomm=comm) # subsamp=100
print("Task %d took %.1f seconds" % (rank, time.time() - t0))

#if rank == 0:
#    for key in orbits:
#        print("Shape of block ", key, " is ", orbits[key].shape)
