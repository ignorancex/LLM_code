import os
import subprocess
import time
from run_new_dset import run_next_gen_dset

# Preparing the initial data set
file_path = './Dataset/dset_0.mat'
if os.path.exists(file_path):
    print("The random data set exists.")
else:
    print("Generating the random data set")
    # Set the path of the script to run
    script_path = 'run_rand_dset.py'
    # Run the script and capture the output
    result = subprocess.run(['python3', script_path], capture_output=True)
    # Print the output
    print(result.stdout.decode())


# MONBO loop
for n_iter in range(1, 10):
    # Train the BNN
    sh_file_path = 'Forward_UANA_diverse_activation.sh'
    new_line = "n_iter=%d" %(n_iter)
    #edit the bash file with the new iteration number
    with open(sh_file_path, "r") as sh_file:
        lines = sh_file.readlines()
    lines[8] = new_line + "\n"
    with open(sh_file_path, "w") as sh_file:
        sh_file.writelines(lines)

    #make the path for the models
    new_dir = 'Models/iter_%d' %(n_iter)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print("Directory created:", new_dir)
    else:
        print("Directory already exists:", new_dir)
    # BNN model training
    print('Training the models ...')
    os.system('ssh slurm-submit.organization.de source /HPS/folder/work/NAGD/MO-NBO/Airfoil/MONBO_Automized/run_MONBO_slurm.sh')
    dir_path = 'Models/iter_%d' %(n_iter)
    # # Check if the training is completed
    while True:
        num_files = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))
        if num_files == 10:
            print('Training is completed')
            break
        time.sleep(1)
        
    # Run 4-D acquisition function
    sh_file_path = 'Acquisition.sh'
    new_line = "n_iter=%d" %(n_iter)
    with open(sh_file_path, "r") as sh_file:
        lines = sh_file.readlines()
    lines[8] = new_line + "\n"
    with open(sh_file_path, "w") as sh_file:
        sh_file.writelines(lines)
    print('Running acquisition ...')
    # make the path for the acquisition
    new_dir = 'Acquisition/iter_%d' % (n_iter)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print("Directory created:", new_dir)
    else:
        print("Directory already exists:", new_dir)
    os.system('ssh slurm-submit.organization.de source /HPS/folder/work/NAGD/MO-NBO/Airfoil/MONBO_Automized/run_acquisition_slurm.sh')
    dir_path_acquisition = 'Acquisition/iter_%d' %(n_iter)
    while True:
        num_files = (len([entry for entry in os.listdir(dir_path_acquisition) if os.path.isfile(os.path.join(dir_path_acquisition, entry))]))
        if num_files == 30:
            break
        time.sleep(1)
    #Evaluate the pareto set on the NFP
    print('Evaluating the pareto set on the NFP...')
    run_next_gen_dset(n_iter)
    