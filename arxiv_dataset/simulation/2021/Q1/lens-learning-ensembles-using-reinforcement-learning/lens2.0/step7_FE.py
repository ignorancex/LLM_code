from itertools import product
from os import environ, system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from common import load_properties

print("Starting . . .\n")

# ensure project directory exists
proj = argv[1].replace("/", "")
project_path = "path/RL/%s" % proj
code_dir     = dirname(abspath(argv[0]))
scripts_dir  = "%s/%s" % (dirname(abspath(proj)), proj)
assert exists(project_path)
SCRIPT_NAME  = "full_"

# load and parse project properties
p = load_properties(project_path)

ults_path       = p['classifierDir']
fold_count 	= int(p['foldCount'])
seeds 		= int(p['seeds'])
dirnames 	= sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))
sizes 		= range(2,len(dirnames)+1)
RULES 		= ['WA', 'L2']
epsilons 	= ['0.01', '0.10', '0.25', '0.50']
ages 		= [0]
conv_iters      = [2, 5, 10]
strategies      = ['greedy', 'pessimistic']
exits           = [0]
algos           = ['Q']
start_states    = ['0', 'best']
task            = 10

##################################################################################################################################

max_num_clsf = 180
#sizes = range(1, max_num_clsf+1)
#sizes = [10,20]
sizes = range(10, max_num_clsf+1, 10)

# Select parameters                                         
all_parameters = list(product([code_dir], [project_path], sizes, range(seeds), RULES))

SCRIPT_ID = "all"

##################################################################################################################################

SCRIPT_NAME = "%s%s" % (SCRIPT_NAME, SCRIPT_ID)
JOB_NAME = "7%s_%s" % (proj[:3], SCRIPT_ID)

total = 0

jobs_fn = "%s/%s.jobs" % (scripts_dir, SCRIPT_NAME)
with open(jobs_fn, "w+") as jobs_file:
    for parameters in all_parameters:
        code_dir, project_path, size, seed, RULE = parameters
        
        if not exists("%s/BASELINE/ORDER%s/FE_bp%s_seed%s_%s_rnd.fmax" % (project_path, seed, size, seed, RULE)):
            cmd = 'python -W ignore  %s/full_ens_rnd.py %s %s %s %s' % (code_dir, proj, size, seed, RULE) #flag -W ignore will ignore deprecation warnings
            total += 1
            jobs_file.write("%s\n" % cmd)
jobs_file.close()


lsf_fn = "%s/%s.lsf" % (scripts_dir, SCRIPT_NAME)
with open(lsf_fn, "w+") as lsf_file:
    lsf_file.write("#!/bin/bash\n"
                   "#BSUB -P account\n"
		   "#BSUB -q premium\n"
                   #"#BSUB -q expressalloc\n" ### Use expressalloc if < 2h
                   "#BSUB -W 00:55\n"

                   #"#BSUB -q low\n"
                   #"#BSUB -W 12:00\n"

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
                   "mpirun selfsched < %s\n\n" % (JOB_NAME, (total+1), project_path, jobs_fn)) #need one extra core to monitor the jobs

                   # # # # # # # # # #
                   # OR RUN THREADS: #
                   # # # # # # # # # #
                   #"python pathRL/code/doParallelJobs.py --ifile %s/%s --task %s\n"
                   #"\n" % (JOB_NAME, (math.ceil(len(all_parameters)/task)+1), project_path, jobs_fn, task))

    lsf_file.close()

print("Total jobs: %i/%i" % (total, len(all_parameters)))
print("<%s.jobs> written out at %s" % (SCRIPT_NAME, jobs_fn))
print("<%s.lsf>  written out at %s" % (SCRIPT_NAME, lsf_fn))
print("\nDone!")



                                                                                          

