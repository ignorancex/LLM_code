##############################################
# created 27-08-2024
#
# Python script to run COMPAS using DisBatch
# I run 5 x 10^6 systems at a bunch of discrete metallicities (each divided into batches)
# Finally combines the output using h5copy
# 
##############################################
import numpy as np
import os
from subprocess import Popen, PIPE
import subprocess
import sys
import shutil
import h5py as h5
import re

from definitions import sim_flags_dict


#################################################################
## 
##    Step 0: set variables
##
#################################################################


#################################################################
##    Should be Changed by user ##
#################################################################
sim_name             = "NewWinds_RemFryer2012"# "OldWinds_RemFryer2012_noNSBHkick"#"OldWinds_RemFryer2012" # Note: the sim_name will determine which flags to run COMPAS with
compas_v             = "v03.01.02" #"v02.46.01/"#v02.35.02/"

root_out_dir         = f"/mnt/home/lvanson/ceph/CompasOutput/{compas_v}/{sim_name}"
file_name            = 'COMPAS_Output_wWeights.h5'
user_email           = "aac.van.son@gmail.com"
gid_filename         = "BSE_grid_N5e6_mass_sep_kick.txt"

### Different options for the metalicites:
# 0.0001, 0.0003, 0.001, 0.004, 0.01, 0.02, 0.03 # Hurley Z's
# [0.0001, 0.00017321, 0.0003, 0.00054772, 0.001, 0.002, 0.004, 0.00632456, 0.01, 0.01414214, 0.02, 0.03] # Hurley with extra steps
# np.logspace(-4, np.log10(0.03), 17)  # flat in log
metallicities = [0.0001, 0.00017321, 0.0003, 0.00054772, 0.001, 0.002, 0.004, 0.00632456, 0.01, 0.01414214, 0.02, 0.03] # Hurley with extra steps


############################################################################################################
### RUN ALL SIMULATIONS 
############################################################################################################
def setup_and_submit_sim(sim_name):

    # SHOULD BE REMOVED 
    if sim_name == 'NewWinds_RemFryer2012':
        print('Already ran NewWinds_RemFryer2012, skipping')
        return 

    print(f'Will setup and run {sim_name}')

    #################################################################
    ### What COMPAS flags to run with? 
    """
    #  Based on your sim_name, we will now construct a combination of COMPAS flags 
    #  that set the 'physics' we would like to run with.
    """
    # Check if sim_name exists in the dictionary
    if sim_name in sim_flags_dict:
        sim_variation_flags = sim_flags_dict[sim_name]
        print(sim_variation_flags)
    else:
        print(f"Unknown sim_name: {sim_name}")


    #################################################################
    ### Make root out dir and Copy your BSE_grid 
    """
    I am interested in rerunning the exact same ~1e6 binaries at different metallicities

    I am using masterfolder/BSE_grid_N5e6_mass_sep_kick.txt
    """
    ###############################################
    # Make the output directory if it doesn't exist
    if not os.path.isdir(root_out_dir):
        print('root_out_dir =  ', root_out_dir)
        os.makedirs(root_out_dir, exist_ok=True)

        # copy this python script to the ROOT out dir (for reference)
        shutil.copyfile('Grid_Call_allSims.py', f'{root_out_dir}/Grid_Call_allSims.py')  
        shutil.copyfile(f'{gid_filename}', f'{root_out_dir}/{gid_filename}')  
        shutil.copyfile('COMPAS_Output_Definitions.txt', f'{root_out_dir}/COMPAS_Output_Definitions.txt')  
    else:
        print(f'Nothing to do, {root_out_dir} already exists')

    def divide_with_remainder(numerator, denominator):
        batch_size = numerator // denominator
        n_jobs     = numerator/batch_size
        remainder  = numerator % denominator
        return batch_size, int(n_jobs), remainder

    # details for your run
    with open(f'{root_out_dir}/{gid_filename}', 'r') as f:
        # Read the file into a list of lines
        lines = f.readlines()
    num_lines = len(lines)
    print('num_lines',num_lines)

    N_binaries           = num_lines  # how many binaries to run in total
    N_chunks             = 80         # how many batches to run this in (N_binaries/N_chunks is not an int, you will run the remainder in an extra last batch)

    # Determine how many batches to run
    batch_size, n_jobs, remainder = divide_with_remainder(N_binaries, N_chunks)
    last_batch_size, extra_job    =  batch_size, 0
    if remainder != 0.:
        extra_job = 1
        print(r'N_binaries = %s can not be divided properly into N_chunks=%s'%(N_binaries, N_chunks))
        print('You will run 1 extra job with %s binaries'%(remainder))

    print('n_jobs',n_jobs, 'of batch_size', batch_size)


    #################################################################
    ## 
    ##    Step 1: Make a list of tasks to submit to [disBatch](https://github.com/flatironinstitute/disBatch)
    ##
    #################################################################
    """
    Now we are going to construct tasks. A task looks somthing like ( cd /path/to/workdir ; source SetupEnv ; myprog -a 0 -b 0 -c 0 ) &> task_0_0_0.log

    For my COMPAS batches, this consists of the follwing steps:
    * cd {rundir}
    * module load python gsl boost hdf5
    * $COMPAS_ROOT_DIR/src/COMPAS -flags  > COMPAS_batch_i.log
    """

    # open a file to write the tasks to 
    with open(f'{root_out_dir}/Tasks', 'w') as f:

        # Hurley metallicities + extra steps
        for metallicity in metallicities: #
            print('metallicity', metallicity)

            # Make a dir for this metallicity
            base_run_dir = root_out_dir+f'/logZ{np.round(np.log10(metallicity),2)}/'
            os.makedirs(base_run_dir, exist_ok=True)

            # Loop over every batch 
            for Njob in range(n_jobs + extra_job):
                # directory where you will copy the files to and run compas from
                run_dir = base_run_dir+'/batch'+'_%s'%(Njob) +'/'
                os.makedirs(run_dir, exist_ok=True)

                ############################################
                # Compile the flags you want for this task
                # if you are on the last job and the remainder is nonzero, run the remainder
                if np.logical_and(remainder !=0, Njob == n_jobs):
                    print('you are on the extra job, use remainder as batch_size ')
                    COMPAS_batch_flags = f"--metallicity {metallicity} {sim_variation_flags} --allow-touching-at-birth True --add-options-to-sysparms 'NEVER' --grid '{root_out_dir}/{gid_filename}' --logfile-definitions '{root_out_dir}/COMPAS_Output_Definitions.txt' --grid-start-line '{Njob*batch_size}' --grid-lines-to-process '{remainder}' --output-path '{run_dir}' "
                else:
                    COMPAS_batch_flags = f"--metallicity {metallicity} {sim_variation_flags} --allow-touching-at-birth True --add-options-to-sysparms 'NEVER' --grid '{root_out_dir}/{gid_filename}' --logfile-definitions '{root_out_dir}/COMPAS_Output_Definitions.txt' --grid-start-line '{Njob*batch_size}' --grid-lines-to-process '{batch_size}' --output-path '{run_dir}' "

                # print(COMPAS_batch_flags)
                # NOTE!!  --allow-touching-at-birth = True, otherwise as I increase Z, some systems will fail!

                task_line = f"cd {run_dir} ; module load python gsl boost hdf5 ; $COMPAS_ROOT_DIR/src/COMPAS {COMPAS_batch_flags}  > COMPAS_batch_{Njob}.log 2>&1 " 
                f.write(task_line + '\n')


    #################################################################
    ## 
    ##    Step 2: Execute the Tasks with DisBatch
    ##
    #################################################################
    """
    make sure to `module load disBatch` 

    Go to your root_out_dir and just run: 
    sbatch -n 50 disBatch Tasks
    """

    # disBatch Command
    command = f"module load disBatch && sbatch -p cca -n 100 disBatch {root_out_dir}/Tasks"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Extract the job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        disBatch_job_ids = match.group(1)

    print(disBatch_job_ids)


    #################################################################
    ## 
    ##    # Step 3: Combine the hdf5 files in post processing
    ##
    #################################################################

    ###############################################
    # post proces tasks 
    ###############################################
    print(10* "*" + ' You are Going to Run PostProcessing.py')

    with open(f'{root_out_dir}/PP_Tasks', 'w') as f:

        for metallicity in metallicities: 
            base_run_dir = root_out_dir+f'/logZ{np.round(np.log10(metallicity),2)}/'

            print(base_run_dir)

            # copy the h5copy to the root out dir
            shutil.copyfile('h5copy.py', f'{base_run_dir}/h5copy.py')  

            # task line
            task_line = f"cd {base_run_dir} ; module load python ; python h5copy.py {base_run_dir} -r 2 -o COMPAS_Output.h5  > COMPAS_PP.log 2>&1 " 
            f.write(task_line + '\n')


    ############################################
    # Submit the job! 
    # PP_job_ids = []

    # disBatch Command
    command = f"module load disBatch && sbatch --dependency=afterok:{disBatch_job_ids}  -p gen -n 20 disBatch {root_out_dir}/PP_Tasks" #
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Extract the job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        PP_job_ids = match.group(1)

    print(PP_job_ids)


    #################################################################
    ##
    ## # Step 4 combine individual Z sim into a big hdf5 file
    ### Finally combine each individual metallicity simulation
    ## 
    #################################################################

    # copy the h5copy to the root out dir
    shutil.copyfile('h5copy.py', f'{root_out_dir}/h5copy.py')  

    with open(f'{root_out_dir}/combineZ_Tasks', 'w') as f:

        # task line
        task_line = f"cd {root_out_dir} ; module load python ; python h5copy.py {root_out_dir} -r 1 -o COMPAS_Output_combinedZ.h5  > COMPAS_PP.log 2>&1 " 
        f.write(task_line + '\n')

    # disBatch Command
    command = f"module load disBatch && sbatch --dependency=afterok:{PP_job_ids} -p gen -n 2 disBatch {root_out_dir}/combineZ_Tasks" #{PP_job_ids}
    result = subprocess.run(command, shell=True, capture_output=True, text=True)



############################################################################################################
### MAIN 
############################################################################################################
if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(setup_and_submit_sim, sim_flags_dict.keys())