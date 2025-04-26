from itertools import product
from os import environ, system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from common import load_properties

print("Starting . . .\n")

# ensure project directory exists
proj         = argv[1].replace("/", "")
project_path = "path/RL/%s" % proj
code_dir     = dirname(abspath(argv[0]))
scripts_dir  = "%s/%s" % (dirname(abspath(proj)), proj)
SCRIPT_NAME  = 'e_%s' % proj[:4]
assert exists(project_path)


# load and parse project properties
p = load_properties(project_path)

seeds 		= int(p['seeds'])
dirnames 	= sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))
sizes 		= range(2, len(dirnames) + 1)
exits           = [0]
algos           = ['Q']
start_states    = ['0']
task            = 10
max_num_clsf    = len(dirnames) * seeds
sizes           = range(10, max_num_clsf + 1, 10)

RULES           = ['L2'] #'AVG', 'PROD', 'CON', 'WA']
epsilons        = ['0.01'] #, '0.10', '0.25', '0.50']

ages            = [500000]
conv_iters      = [0]
strategies      = ['pessimisticrnd', 'pessimeuclid', 'pessimcos', 'pessimcorr', 'pessimyule', 'pessimkappa'] 
all_params1 = list(product(sizes, [code_dir], [project_path], range(seeds), epsilons, RULES, strategies, exits, algos, ages, conv_iters, start_states))

ages            = [0]
conv_iters      = [10]
strategies      = ['greedyrnd'] #, 'diversityrnd', 'euclideanrnd', 'correlationrnd', 'yulernd', 'kapparnd', 'gsernd']
all_params2 = list(product(sizes, [code_dir], [project_path], range(seeds), epsilons, RULES, strategies, exits, algos, ages, conv_iters, start_states))

##########################################

all_parameters = all_params2 + all_params1



total = 0

jobs    = "%s.jobs" % SCRIPT_NAME
jobs_fn = "%s/%s" % (scripts_dir, jobs)
with open(jobs_fn, "w+") as jobs_file:
    for parameters in all_parameters:
        size, working_dir, project_path, seed, epsilon, RULE, strategy, exit, algo, age, conv, start = parameters
        #RL_bp10_seed0_epsilon0.01_pre0_conv5_exit0_greedy_WA_Q_start-0.fmax
        filename = "%s/RL_RESULTS/ORDER%s/RL_bp%s_seed%s_epsilon%s_pre%i_conv%i_exit%i_%s_%s_%s_start-%s.fmax" % (project_path, seed, size, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)
        if not exists(filename):

            cmd = 'python -W ignore %s/rl_ens.py %s %s %s %s %s %s %s %s %s %s %s' % (code_dir, proj, size, age, seed, epsilon, RULE, strategy, start, exit, conv, algo)
            total += 1
            jobs_file.write("%s\n" % cmd)
jobs_file.close()


if (total > 0):
    lsf    = "%s.lsf" % SCRIPT_NAME
    lsf_fn = "%s/%s" % (scripts_dir, lsf)
    with open(lsf_fn, "w+") as lsf_file:
        lsf_file.write("#!/bin/bash\n"
                   "#BSUB -P account\n"

                   #"#BSUB -q expressalloc\n" ### Use expressalloc if < 2h
		   "#BSUB -q premium\n"
                   "#BSUB -W 00:10\n"

                   #"#BSUB -q low\n"
                   #"#BSUB -W 05:00\n"

                   "#BSUB -J %s\n"
                   "#BSUB -n %i\n"

                   #"#BSUB -R rusage[mem=500]\n" 

                   #"#BSUB -m manda\n"
                   "#BSUB -o %%J.stdout\n"
                   "#BSUB -eo %%J.stderr\n"
                   "#BSUB -L /bin/bash\n\n"
		   #"source pathsoft/miniconda/envs/ana/bin/activate ana\n\n"
                   #"#export PATH=${PATH}:/hpc/packages/minerva-common/python/2.7.6/bin\n"
                   #"#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/hpc/packages/minerva-common/python/2.7.6/lib\n"
                   #"#export PYTHONPATH=${PYTHONPATH}:/hpc/packages/minerva-common/python/2.7.6/lib/python2.7/site-packages\n\n"


                   "module load python\n"
                   "module load py_packages\n"
                   "module load selfsched\n\n"
                   "cd %s\n"

                   #"mpirun selfsched < %s\n\n" % (SCRIPT_NAME, (total+1), scripts_dir, jobs)) #need one extra core to monitor the jobs
                   "mpirun selfsched < %s\n\n" % (SCRIPT_NAME, 10, scripts_dir, jobs)) #need one extra core to monitor the jobs

                   # # # # # # # # # #
                   # OR RUN THREADS: #
                   # # # # # # # # # #
                   #"python pathRL/code/doParallelJobs.py --ifile %s/%s --task %s\n"
                   #"\n" % (JOB_NAME, (math.ceil(len(all_parameters)/task)+1), project_path, jobs, task))

        lsf_file.close()


    print("Total jobs: %i/%i" % (total, len(all_parameters)))
    print("<%s> written out at %s" % (jobs, jobs_fn))
    print("<%s> written out at %s" % (lsf, lsf_fn))
print("\nDone!")



                                                                                          

