#!/usr/bin/env python

from pathlib import Path

from disbatchc.disBatch import DisBatcher

from params_Merge3D import *

import os

SRCDIR = Path(__file__).parent.resolve()
OUTDIR = Path("/mnt/home/tdharmawardena/ceph/FullMW3D/Merge4_10to2790_Sigmoid")

# Load the file which disbatcher is running
SCRIPT = "mergeChunks_MultiJob.py"

test = False #True #False #Set to True if we want to test the code by only creating the directories and not running the code

def get_task(l_min, l_max, b_min, b_max, i):

    region_name = str(l_min) + "_" +str(l_max) + "_" +str("%.5f" % b_min) + "_" +str("%.5f" % b_max)
    
    print(region_name)

    #Check if the folder exists because it ran in a previous job which only ran half way
    folder_exists = os.path.isdir("../"+region_name)

    if folder_exists: #seed == seeds[0]: #small modification to previous version - if the seed is not zero, we want to re-run the job
        print("Region already run in previous run before job broke")
        return None

    else:
        
        jobdir = OUTDIR / region_name

        prefix = f"mkdir -p {jobdir}; cd {jobdir}; cp ../*.py ."
    
        cmd =  f"{prefix}; python {SCRIPT} {l_min} {l_max} {b_min} {b_max} "
        cmd += f"--min_d_bounds_pred_Dchunk {str(min_d_bounds_pred_Dchunk).replace(',','').replace('[','').replace(']','')} "
        cmd += f"--dweight_lower_cutoff {str(dweight_lower_cutoff).replace(',','').replace('[','').replace(']','')} "
        cmd += f"--dweight_upper_cutoff {str(dweight_upper_cutoff).replace(',','').replace('[','').replace(']','')}  "
        cmd += f" &> slurm.log"

        task = dict(region_name=region_name, l_min=l_min, l_max=l_max, b_min=b_min, b_max=b_max, cmd=cmd, i=i)
    
    return task


# """.format(region_name, merge_l_min, merge_l_max, merge_b_min, merge_b_max,
#  str(min_d_bounds_pred_Dchunk).replace(',','').replace('[','').replace(']',''), 
#  str(dweight_lower_cutoff).replace(',','').replace('[','').replace(']',''), 
#  str(dweight_upper_cutoff).replace(',','').replace('[','').replace(']','')) 


def main(disbatcher):
    tasks = []

    if test:

        for i, l_min in enumerate(l_set[:2]): #We set the l_set=3 and b_set=3 to test the code. it can be more if we want
            l_max = l_set[i+1]
            for j, b_min in enumerate(b_set[:2]):  #We set the b_set=3 and b_set=3 to test the code. it can be more if we want
                b_max = b_set[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task

    else:
        for i, l_min in enumerate(l_set[:-1]):
            l_max = l_set[i+1]
            for j, b_min in enumerate(b_set[:-1]):
                b_max = b_set[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task



if __name__ == "__main__":
    disbatcher = DisBatcher(tasksname="dynamic-disBatch")
    try:
        main(disbatcher)
    finally:
        disbatcher.done()
