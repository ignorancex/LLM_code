import os, sys
import numpy
import itertools
import random
import copy
import os, sys
import time

import swarm
import global_functions
import experiments

#make movie (.mp4) from max_time_steps frames (.png)
def mk_mov(dir,max_time_steps):
    os.chdir(dir)
    #for nt in range(max_time_steps):
    for nt in range(max_time_steps):
        if os.path.isfile("N" + str(3 * nt).zfill(3) + ".png"):
            command = "cp " + "N" + str(3 * nt).zfill(3) + ".png "
            command2 = "cp " + "N" + str(0).zfill(3) + ".png "
            copy_file1 = "N" + str(3 * nt + 1).zfill(3) + ".png"
            copy_file2 = "N" + str(3 * nt + 2).zfill(3) + ".png"
            os.system(command + copy_file1)  # "cp N000.png "
            os.system(command2 + copy_file2)
    os.system("ffmpeg -r 6 -i N%03d.png -vcodec mpeg4 -y signaling.mp4 -loglevel warning") #-aspect 4:3

############################# START ###############################################

#specify input on command line and in input file; python3 main input.

start_time = time.time()

if len(sys.argv) > 1:
    inp = list(open(str(sys.argv[1]), "r"))
    int_flag = True
    for line in inp:
        c = line.rsplit("#")
        if c[0] == "" and int(c[1].rsplit(".")[0]) != 1:
            int_flag = False

        elif c[0] != "":
            name = c[1].rsplit(";")[0]

            try:
                value = int(c[0])
            except:
                value = float(c[0])

            if int_flag == False and value == 0:
                value = False
            elif int_flag == False and value == 1:
                value = True

            locals()[name] = value

""" if not specified in input, assign values here: """
if "numagents" not in locals(): numagents = 4
if "max_time_steps" not in locals(): max_time_steps = 20
if "bits" not in locals(): bits = 1
if "size_input" not in locals(): size_input = 3
if "alpha" not in locals(): alpha = 5 #error rate in input transimission

model0 = swarm.network(numagents, bits, size_input, print_flag=True)

""" if not specified in input, assing values: """
#1. parameters for GA
if "n_pop" not in locals(): n_pop = 1 #51
if "n_ind" not in locals(): n_ind = 5 #25 #the number of possible topologies: 2**[n*(n-1)/2] with (n 2) possible activation functions
if "p_mutate" not in locals(): p_mutate=0.01 #mutation rate in GA
if "p_lambda" not in locals(): p_lambda=0.2 #if p_lambda == 0: select from full population if p_lambda=1: keep entire population
#2. parameters for network growth
if "p_annealing" not in locals(): p_annealing = 0.01
if "cost" not in locals(): cost = [0.0,0.0]

print("---Preliminary experiments---")
experiment = experiments.preliminary()

#First experiment:
experiment.evolutionary_perception(model0,max_time_steps,size_input,alpha,n_pop,n_ind,p_mutate,p_lambda)

#Second experiment:
#experiment.evolution_of_topology(model0,cost,p_annealing,max_time_steps,size_input,alpha,n_pop,n_ind,p_lambda,p_mutate)

#mk_mov("network_drawings/states/",max_time_steps)
print("Running time: %s seconds" % numpy.around(time.time() - start_time,2))