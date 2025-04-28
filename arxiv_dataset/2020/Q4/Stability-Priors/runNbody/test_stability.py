import numpy as np
import pandas as pd
import sys
import rebound

def collision(reb_sim, col):
    reb_sim.contents._status = 5
    return 0

systems = ["HR858", "K431", "TOI270", "L98-59", "K23"]
system = systems[int(sys.argv[1])]
sim_names = system + "/" + system

nsim = int(sys.argv[2])
name_adds = ["loge", "logm"]
name_add = name_adds[int(sys.argv[3])]
sim = rebound.SimulationArchive(sim_names + "_" + name_add + "_start_%d.bin"%nsim)[0]
filename = sim_names + "_" + name_add + "_sa_%d.bin"%nsim

P1 = sim.particles[1].P
maxorbs = 1e9

sim.automateSimulationArchive(filename, interval=maxorbs*P1/100., deletefile=True)
try:
    sim.integrate(maxorbs * P1, exact_finish_time=0)
    print("stable")
except:
    sim.simulationarchive_snapshot(filename)  # save final snapshot if collision occurs
    print("unstable")
print(sim.t / P1)