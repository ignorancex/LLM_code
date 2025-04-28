import stability_functions as sf
import numpy as np
import sys
import rebound
import corner
import pandas as pd
from multiprocessing import Pool
import os
import glob

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
system = systems[4]
sim_names_base = system + "/" + system

# n_workers = os.cpu_count()
n_workers = 8

def collision(reb_sim, col):
    reb_sim.contents._status = 5
    return 0

ns = [30000, 2000000]
name_adds = ["loge", "logm"]

for selection in range(2):

    n = ns[selection]
    name_add = name_adds[selection]
    sim_names = sim_names_base + "_" + name_add

    print("working on " + name_add)
    
    logm = name_add=="logm"
    loge = not logm

    def to_test(nsim):
        np.random.seed(nsim)
        sim = sf.build_Hadden_system(system, logm=logm, loge=loge)
        sim_copy = sim.copy()

        P1 = sim.particles[1].P
        maxorbs = 100000
        name = sim_names + "_start_%d.bin"%nsim

        try:
            sim.integrate(maxorbs * P1, exact_finish_time=0)
            sf.replace_snapshot(sim_copy, name)
            with open("jobs/" + system + "_" + name_add + "_nb_%d.pbs"%nsim, "w") as of:
                of.write("#!/bin/bash\n")
                of.write("#PBS -N " + name_add + "_stab  # name of job\n")
                of.write("#PBS -l nodes=1:ppn=1 # how many nodes and processors per node\n")
                of.write("#PBS -l walltime=10:00:00\n")
                of.write("#PBS -l pmem=4gb  # how much RAM is needed\n")
                of.write("#PBS -A cyberlamp  # which allocation to use (either cyberlamp, open, or ebf11_a_g_sc_default)\n")
                of.write("#PBS -j oe  # put outputs and error info in the same file\n\n")

                of.write("echo \"Starting job $PBS_JOBNAME\"\n")
                of.write("date\n")
                of.write("starttime=$SECONDS\n")
                of.write("echo \"Job id: $PBS_JOBID\"\n")
                of.write("echo \"About to change into $PBS_O_WORKDIR\"\n")
                of.write("cd $PBS_O_WORKDIR\n")
                of.write("echo \"Running code\"\n")
                of.write("python test_stability.py 4 %d %d\n"%(nsim,selection))
                of.write("date\n")
                of.write("echo \"took $((SECONDS - starttime)) seconds\"\n")
                of.write("echo \"done :)\"\n")
        except:
            if (sim.t/P1) > 10000 or nsim<10000:
                sf.replace_snapshot(sim_copy, name)

    pool = Pool(processes=n_workers)

    nsim_list = np.arange(0, n)
    res = pool.map(to_test, nsim_list)