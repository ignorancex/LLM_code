from itertools import product
from os import environ, system
from os.path import abspath, dirname, exists
from sys import argv
from random import randrange
from common import load_arff_headers, load_properties
#from sklearn.externals.joblib import Parallel, delayed

print("Starting . . .")

# ensure project directory exists
project_path = "path/RL/%s" % argv[1].replace("/", "")
code_dir     = dirname(abspath(argv[0]))
scripts_dir  = "%s/%s" % (dirname(abspath(argv[1])), argv[1])
assert exists(project_path)
SCRIPT_NAME  = argv[1].replace("/", "") 


# load and parse project properties
p = load_properties(project_path)
classifierDir 	= p['classifierDir']
classifiers_fn 	= '%s' % (p['classifiersFilename'])
input_fn 	= '%s' % (p['inputFilename'])
assert exists(input_fn)

# generate cross validation values for leave-one-value-out or k-fold
assert ('foldAttribute' in p) or ('foldCount' in p)
if 'foldAttribute' in p:
    headers = load_arff_headers(input_fn)
    fold_values = headers[p['foldAttribute']]
else:
    fold_values = range(int(p['foldCount']))

# repetitions of the experiments (in terms of seeds used for randomizing the data)
seed_count = int(p['seeds'])
seeds 	   = range(seed_count) if seed_count > 1 else [0]

# bags of experiments (in terms of resampled training data to generate different versions of the same algorithm)
bag_count = int(p['bags'])
bags      = range(bag_count) if bag_count > 1 else [0]

#ensure java's classpath is set
#classpath = environ['CLASSPATH']

# command for cluster execution if enabled
use_cluster = False if 'useCluster' not in p else p['useCluster'] == 'true'
cluster_cmd = 'rc.py --cores 1 --walltime 06:00:00 --queue small --allocation acc_9'

# load classifiers from file, skip commented lines
classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
classifiers = [_.strip() for _ in classifiers]

all_parameters = list(product([code_dir], [project_path], classifiers, bags, seeds, fold_values))

total_jobs      = 0
jobs_per_script = 0
script_number   = 0
num_of_jobs     = 1
str_cmd         = ""

if len(all_parameters) < num_of_jobs:
    num_of_jobs = len(all_parameters)

jobs     = ""
jobs_fn  = ""
lsf      = ""
lsf_fn   = ""


jobs_to_run = 0
for parameters in all_parameters:
    code_dir, project_path, classifier, bag, seed, fold = parameters
    classifierName = classifier.split(" ")[0]
    shortClassifierName = classifierName.split(".")[-1]
    if not exists("%s/%s/valid-b%s-f%s-s%s.csv.gz" % (classifierDir, classifierName, bag, fold, seed)) or not exists("%s/%s/test-b%s-f%s-s%s.csv.gz"  % (classifierDir, classifierName, bag, fold, seed)) or not exists("%s/%s/%s-b%s-f%s-s%s.model.gz"  % (classifierDir, classifierName, shortClassifierName, bag, fold, seed)):
    #if not exists("%s/%s/valid-b%s-f%s-s%s.csv.gz" % (classifierDir, classifierName, bag, fold, seed)):
        cmd = 'TO-DO: groovy %s/generate.groovy %s %s %s %s %s' % (code_dir, project_path, bag, seed, fold, classifier)
        print(cmd)
        jobs_to_run += 1

print("\n")
cmd = ""
for parameters in all_parameters:
    code_dir, project_path, classifier, bag, seed, fold = parameters    
    classifierName = classifier.split(" ")[0]
    shortClassifierName = classifierName.split(".")[-1]
    code_dir, project_path, classifier, bag, seed, fold = parameters
    if not exists("%s/%s/valid-b%s-f%s-s%s.csv.gz" % (classifierDir, classifierName, bag, fold, seed)) or not exists("%s/%s/test-b%s-f%s-s%s.csv.gz"  % (classifierDir, classifierName, bag, fold, seed)) or not exists("%s/%s/%s-b%s-f%s-s%s.model.gz"  % (classifierDir, classifierName, shortClassifierName, bag, fold, seed)):
        jobs_per_script += 1
        total_jobs      += 1
        cmd = 'groovy %s/generate.groovy %s %s %s %s %s' % (code_dir, project_path, bag, seed, fold, classifier)
        #print(cmd)
        str_cmd += "%s\n" % cmd
        cmd = ""

        jobs     = "%s_J%--i.jobs" % (SCRIPT_NAME, script_number)
        jobs_fn  = "%s/%s" % (scripts_dir, jobs)
        lsf      = "%s_J%--i.lsf" % (SCRIPT_NAME, script_number)
        lsf_fn   = "%s/%s" % (scripts_dir, lsf)

        if total_jobs % num_of_jobs == 0 or (total_jobs == jobs_to_run):
            with open(jobs_fn, "w+") as jobs_file:
                jobs_file.write("%s\n" % str_cmd)
            #print str_cmd
            jobs_file.close()
            str_cmd = ""

            with open(lsf_fn, "w+") as lsf_file:
                lsf_file.write("#!/bin/bash\n"
                   "#BSUB -P account\n"
                   "#BSUB -q premium\n"
                   "#BSUB -n %i\n"
                   "#BSUB -R rusage[mem=16000]\n"
                   "#BSUB -W 48:00\n"
                   "#BSUB -J %sJ%i\n"
                   #"#BSUB -m manda\n"
                   "#BSUB -o %%J.stdout\n"
                   "#BSUB -eo %%J.stderr\n"
                   "#BSUB -L /bin/bash\n\n"
                   #"#export JAVA_HOME=/hpc/packages/minerva-common/java/1.8.0_66/jdk1.8.0_66\n"
                   #"#export CLASSPATH=/hpc/packages/minerva-common/java/1.8.0_66/jdk1.8.0_66/lib\n"
                   #"#export PATH:${JAVA_HOME}/bin\n"
                   "export JAVA_OPTS=\"-Xmx16G\"\n\n"
                   "module load java\n"
                   "module load groovy\n"
                   "module load weka\n"
                   "module load selfsched\n"
                   "cd %s\n"
                   "mpirun selfsched < %s\n\n" % ((jobs_per_script+1), SCRIPT_NAME, script_number, scripts_dir, jobs))
            lsf_file.close()
            jobs_per_script = 0
            script_number += 1
            print("%s -- %s" % (jobs_fn, lsf_fn))


print ("\nTotal jobs = %i/%i --(x3)--> %i files" % (total_jobs, len(all_parameters), total_jobs*3))
print ("Done!\n")



