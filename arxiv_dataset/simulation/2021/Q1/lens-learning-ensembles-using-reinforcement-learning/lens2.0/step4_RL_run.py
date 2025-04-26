from itertools import product
from os import environ, system, makedirs
from os.path import abspath, dirname, exists, isdir
from sys import argv
from glob import glob
from common import load_properties


print ("Starting . . .\n")

# ensure project directory exists
proj         = argv[1].replace("/", "")
project_path = "path/RL/%s" % proj
code_dir     = dirname(abspath(argv[0]))
scripts_dir  = "%s/%s" % (dirname(abspath(proj)), proj)
SCRIPT_NAME  = 'Rn_%s' % proj[:4]
assert exists(project_path)

reverse = False
if len(argv) == 3 and argv[2] == 'r':
    reverse = True

# load and parse project properties
p = load_properties(project_path)

results_path    = p['classifierDir']
fold_count 	= int(p['foldCount'])
seeds 		= int(p['seeds'])
dirnames 	= sorted(filter(isdir, glob('%s/weka.classifiers.*' % project_path)))
sizes 		= range(2,len(dirnames)+1)
max_num_clsf    = len(dirnames) * seeds

exits           = [0]
algos           = ['Q']
start_states    = ['0']
#task           = 10

sizes = range(10, max_num_clsf + 1, 10)
seeds = 10
#print(sizes)
if reverse:
    sizes.reverse()

SCRIPT_ID = sizes[len(sizes) - 1]
#if reverse:
#    SCRIPT_NAME = "%s%s_" % (SCRIPT_NAME, SCRIPT_ID)
#else:
#    SCRIPT_NAME = "%s%s" % (SCRIPT_NAME, SCRIPT_ID)


RULES 		= ['L2'] #'AVG', 'PROD', 'CON', 'WA']
epsilons        = [0.01] #, 0.1, 0.25, 0.5]

strategies     = ['greedyrnd'] #, 'diversityrnd', 'euclideanrnd', 'correlationrnd', 'yulernd', 'kapparnd', 'gsernd']
conv_iters     = [10]
ages           = [0]
all_parameters1 = list(product([code_dir], [project_path], sizes, ages, epsilons, range(fold_count), range(seeds), RULES, strategies, exits, algos, conv_iters, start_states))


strategies      = [ 'pessimeuclid', 'pessimcos', 'pessimcorr', 'pessimyule', 'pessimkappa', 'pessimisticrnd'] #, 'backtrackrnd'] #, 'pessimistic']
conv_iters      = [0]
ages            = [500000]
all_parameters2 = list(product([code_dir], [project_path], sizes, ages, epsilons, range(fold_count), range(seeds), RULES, strategies, exits, algos, conv_iters, start_states))


# Select parameters					    
all_parameters = all_parameters1 + all_parameters2


total_jobs      = 0
jobs_per_script = 0
script_number   = 0
num_of_jobs     = 5
str_cmd         = ""


if len(all_parameters) < num_of_jobs:
    num_of_jobs = len(all_parameters)

jobs     = ""
jobs_fn  = ""
lsf      = ""
lsf_fn   = "" 


jobs_to_run = 0
for parameters in all_parameters:
    code_dir, project_path, size, age, epsilon, fold, seed, RULE, strategy, exit, algo, conv, start = parameters
    if not exists("%s/RL_OUTPUT/ORDER%s/bp%s_fold%s_seed%s_epsilon%.2f_pre%i_conv%i_exit%i_%s_%s_%s_start-%s.fmax" % (results_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)):
        jobs_to_run += 1
final_script = jobs_to_run / num_of_jobs


for parameters in all_parameters:
    code_dir, project_path, size, age, epsilon, fold, seed, RULE, strategy, exit, algo, conv, start = parameters      
    if not exists("%s/RL_OUTPUT/ORDER%s/bp%s_fold%s_seed%s_epsilon%.2f_pre%i_conv%i_exit%i_%s_%s_%s_start-%s.fmax" % (results_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)):
        jobs_per_script += 1
        total_jobs      += 1
        cmd = 'python -W ignore %s/rl/run_rnd.py -i %s -o %s/RL_OUTPUT/ORDER%s/ -np %s -fold %s -m fmax -seed %i -epsilon %s -rule %s -strategy %s -exit %i -algo %s -age %i -conv %i -start %s' % (code_dir, project_path, results_path, seed, size, fold, seed, epsilon, RULE, strategy, exit, algo, age, conv, start)
        #print cmd
        str_cmd += "%s\n" % cmd       

        jobs     = "%s_J%i.jobs" % (SCRIPT_NAME, script_number)
        jobs_fn  = "%s/%s" % (scripts_dir, jobs)
        lsf      = "%s_J%i.lsf" % (SCRIPT_NAME, script_number)
        lsf_fn   = "%s/%s" % (scripts_dir, lsf)

        if total_jobs % num_of_jobs == 0 or (total_jobs == jobs_to_run): 
            with open(jobs_fn, "w+") as jobs_file:
                jobs_file.write("%s\n" % str_cmd)
            jobs_file.close()
            str_cmd = ""

            with open(lsf_fn, "w+") as lsf_file:
                lsf_file.write("#!/bin/bash\n"
			       "#BSUB -P account\n"

			       #"#BSUB -q low\n"
			       #"#BSUB -W 08:50\n"

			       #"#BSUB -q expressalloc\n" ### Use expressalloc if < 2h
			       #"#BSUB -W 04:59\n"

			       "#BSUB -q premium\n"
			       "#BSUB -W 48:00\n"

			       "#BSUB -J %sJ%i\n"
			       "#BSUB -n %i\n"

                               #"#BSUB -R rusage[mem=8000]\n"
			       #"#BSUB -R rusage[mem=16000]\n"
			       "#BSUB -R rusage[mem=32000]\n"

			       #"#BSUB -m manda\n"
			       #"#BSUB -m mothra\n"

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

			       "mpirun selfsched < %s\n" % (SCRIPT_NAME, script_number, (jobs_per_script+1), scripts_dir, jobs)) #need one extra core to monitor the job

			       # # # # # # # # # #
			       # OR RUN THREADS: #
			       # # # # # # # # # #
			       #"python pathRL/code/doParallelJobs.py --ifile %s/%s --task %i\n"
			       #"\n" % (SCRIPT_ID, (total_jobs)/task+1), scripts_dir, jobs, task)) #need one extra core to monitor the jobs
            lsf_file.close()
            jobs_per_script = 0
            script_number += 1
            print ("%s -- %s" % (jobs_fn, lsf_fn))
    
    #if exists("%s/RL_OUTPUT/ORDER%s/bp%s_fold%s_seed%s_epsilon%.2f_pre%i_conv%i_exit%i_%s_%s_%s_start-%s.fmax" % (results_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start)):
    #    print("%s/RL_OUTPUT/ORDER%s/bp%s_fold%s_seed%s_epsilon%.2f_pre%i_conv%i_exit%i_%s_%s_%s_start-%s.fmax" % (results_path, seed, size, fold, seed, epsilon, age, conv, exit, strategy, RULE, algo, start))




if not exists("%s/RL_OUTPUT/" % results_path):
    makedirs("%s/RL_OUTPUT/" % results_path)

for o in range(seeds):
    if not exists("%s/RL_OUTPUT/ORDER%i" % (results_path, o)):
        makedirs("%s/RL_OUTPUT/ORDER%i" % (results_path, o))


print ("Total jobs: %i/%i" % (total_jobs, len(all_parameters)))

print ("\nDone!")



