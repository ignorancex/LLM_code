#!/usr/bin/env python

from pathlib import Path

from disbatchc.disBatch import DisBatcher

from params_MW3D import *

import os

seeds = [0, 123456, 654321, 3333, 958666764, 635830919, 82, 50438]
# seeds = [2246128347, 48838948, 679166780, 42949, 496, 324618347, 654321, 3333]
# seeds = [8347, 534722612, 66780, 1679166, 55, 2461830000, 4302, 84932516]

SRCDIR = Path(__file__).parent.resolve()
OUTDIR = Path("/mnt/home/tdharmawardena/ceph/FullMW3D/MW3Dgp7_1800to2790_SL10")

# SCRIPT = Path(SRCDIR / "run_Gpytorch_Ext_andDens_Pred.py")
SCRIPT = "run_Gpytorch_Ext_andDens_Pred.py"

test=False #Set to True if we want to test the code by only creating the directories and not running the code

# def get_task(A, B, i):
#     """
#     Generate a task for parameters A and B, with seed index i.
#     """
#     if i >= len(seeds):
#         print(f"Warning: exceed max retry count for {A=} {B=}")
#         return None
    
#     jobdir = OUTDIR / f"{A}_{B}"
#     prefix = f"mkdir -p {jobdir}; cd {jobdir}"
#     cmd = f"{prefix}; python {SCRIPT} --seed={seeds[i]} {A} {B} &> {i}.log"
#     task = dict(A=A, B=B, i=i, cmd=cmd)
#     return task

def get_task(l_min, l_max, b_min, b_max, i):
    if i >= len(seeds):
        print(f"Warning: exceed max retry count for {l_min=} {l_max=} {b_min=} {b_max=}")
        return None
    
    if b_min >= -30.000000000000004 and b_max <= 30.000000000000004:
        min_iter = min_iter_GalPlane
    else:
        min_iter = min_iter_HighLowLat
    
    region_name = str(l_min) + "_" +str(l_max) + "_" +str("%.5f" % b_min) + "_" +str("%.5f" % b_max)
    
    print(region_name)

    #Check if the folder exists because it ran in a previous job which only ran half way
    folder_exists = os.path.isdir("../"+region_name)

    if folder_exists and i == 0: #seed == seeds[0]: #small modification to previous version - if the seed is not zero, we want to re-run the job
        print("Region already run in previous run before job broke")
        return None

    else:
        #Make directory with region name and write the two slurm files into that directory
        # os.mkdir(region_name)
        # os.mkdir(region_name+"/"+"slurm_stderr")
        # os.mkdir(region_name+"/"+"slurm_stdout")
        # instead of making the directory, we add it to the prefix for disBatch to do it:

        jobdir = OUTDIR / region_name

        if i == 0: #seed == seeds[0]:
            prefix = f"mkdir -p {jobdir}; cd {jobdir}; "
        #prefix += f"cp ../../LBol_Cat2022_CSVs/{region_name}_Cat.csv ./;"
            prefix += f"cp ../gpFiles/*py ./; "
            prefix += f'echo "seed={seeds[i]}" > global_seed.py'
        else:
            prefix = f'cd {jobdir}; echo "seed={seeds[i]}" > global_seed.py'
        #continue
        # if test: #Test 1: just make the directory and copy the files
        #     cmd = f'{prefix}; echo "test" &> {i}.log' 
        #else:
        cmd =  f"{prefix}; python {SCRIPT} ../../LBol_Cat2022_CSVs/{region_name}_Cat.csv {l_min} {l_max} {b_min} {b_max} {d_min} {d_max} "
        cmd += f"{first_d_chunk} {recalc_grid_train} {recalc_grid_pred} {recheck_sourcebounds} {retrain_gp} {repredict_gp} "
        cmd += f"{train_gpu} {pred_gpu} {plot_gpu} "
        cmd += f"{str(n_l_train)} {str(n_b_train)} {str(n_d_train)} "
        cmd += f"{str(n_l_pred)} {str(n_b_pred)} {str(n_d_pred)} "
        cmd += f"{str(scale_length_x)} {str(scale_length_y)} {str(scale_length_z)}  "
        cmd += f"{str(mean_ext_dens)} {str(exp_scalefac)} "
        cmd += f"{str(learning_rate)} {str(learning_eps)} {str(num_iter)} {str(num_particles)} {str(num_inducing)} "
        cmd += f"{str(min_iter)} {str(stop_prcnt)} {str(stop_iter)} {str(snapshot_iter)} "
        cmd += f"{str(pred_chunk_size)} {str(pred_sample_size)} &> {i}.log"

        task = dict(region_name=region_name, l_min=l_min, l_max=l_max, b_min=b_min, b_max=b_max, cmd=cmd, i=i)
    
    return task

def main(disbatcher):
    tasks = []
    # Submit all the tasks with the first seed
    if test: #Test 2: only run a few of the tasks
        for i, l_min in enumerate(l_set1[:2]):
            l_max = l_set1[i+1]
            for j, b_min in enumerate(b_set1[:2]):
                b_max = b_set1[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
            for j, b_min in enumerate(b_set2[:2]):
                b_max = b_set2[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
        for i, l_min in enumerate(l_set2[:2]):
            l_max = l_set2[i+1]
            for j, b_min in enumerate(b_set1[:2]):
                b_max = b_set1[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
            for j, b_min in enumerate(b_set2[:2]):
                b_max = b_set2[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
    else:       
        for i, l_min in enumerate(l_set1[:-1]):
            l_max = l_set1[i+1]
            for j, b_min in enumerate(b_set1[:-1]):
                b_max = b_set1[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
            for j, b_min in enumerate(b_set2[:-1]):
                b_max = b_set2[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
        for i, l_min in enumerate(l_set2[:-1]):
            l_max = l_set2[i+1]
            for j, b_min in enumerate(b_set1[:-1]):
                b_max = b_set1[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
            for j, b_min in enumerate(b_set2[:-1]):
                b_max = b_set2[j+1]
                task = get_task(l_min, l_max, b_min, b_max, 0)
                if task is None:
                    continue
                disbatcher.submit(task["cmd"])
                tasks += [task]
                print(f"Submitting task {task}")
                del task
    #Special case for l centered at l=0 so the left and right edges of the map which we need to overlap
    l_min_l0 = l_set2[-1]
    l_max_l0 = l_set2[0]

    l_min = l_min_l0
    l_max = l_max_l0

    for j, b_min in enumerate(b_set1[:-1]):
        b_max = b_set1[j+1]
        task = get_task(l_min, l_max, b_min, b_max, 0)
        if task is None:
            continue
        disbatcher.submit(task["cmd"])
        tasks += [task]
        print(f"Submitting task {task}")
        del task
    for j, b_min in enumerate(b_set2[:-1]):
        b_max = b_set2[j+1]
        task = get_task(l_min, l_max, b_min, b_max, 0)
        if task is None:
            continue
        disbatcher.submit(task["cmd"])
        tasks += [task]
        print(f"Submitting task {task}")
        del task


    # Wait for tasks to complete. Resubmit as necessary.
    njob = len(tasks)
    ndone = 0
    while ndone < njob:
        status = disbatcher.wait_one_task()
        oldtask = tasks[status["TaskId"]]
        if status["ReturnCode"] in (74, 84):
            # resubmit with new seed
            newi = oldtask["i"] + 1
            
            newtask = get_task(oldtask["l_min"], oldtask["l_max"], oldtask["b_min"], oldtask["b_max"], newi)
            if newtask is None:
                print(f"Task {newtask} has run out of valid seeds. Giving up.")
                ndone += 1
                continue
            disbatcher.submit(newtask["cmd"])
            tasks += [newtask]
            print(f"Resubmitting task {newtask}")
        else:
            # done task
            ndone += 1
            print(f"Finished task successfully: {oldtask}")


if __name__ == "__main__":
    disbatcher = DisBatcher(tasksname="dynamic-disBatch")
    try:
        main(disbatcher)
    finally:
        disbatcher.done()
