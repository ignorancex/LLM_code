#This script submits jobs to ICS-ACI

import os
import os.path
import glob
import numpy as np
import sys

def submit_job(path, job_name, path_after=0):
    if path_after == 0:
        path_after = path
    os.system('mv %s %s'%(path, job_name))
    os.system('qsub %s'%job_name)
    os.system('mv %s %s'%(job_name, path_after))

###############################

actually_submit = True

if len(sys.argv) > 1:
    max_submissions = int(sys.argv[1])
else:
    max_submissions = 4096

Njobs_counter = 0

script_dir = os.path.dirname(os.path.realpath(__file__))
jobs_dir = "jobs/"
os.chdir(jobs_dir)
job_names = glob.glob("*.pbs")
os.chdir(script_dir)
for job_name in job_names:
    print(job_name)
    path = jobs_dir + job_name
    if actually_submit:
        submit_job(path, job_name, jobs_dir + "submitted/" + job_name)
    Njobs_counter += 1
    if Njobs_counter >= max_submissions:
        break


print('found and submitted %d jobs'%(Njobs_counter))