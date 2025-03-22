from Random_dataset_gen import random_dset_gen
# from HalftoneToSpec import net_train
# # from Halftone_AB_siren_gen import siren_halftone_AB_class
# # from Test_pareto_NN import pareto_run
from BO_surrogate_function_uncertainty import AB_BO_surrogate_uncertainty
from BO_surrogate_function import AB_BO_surrogate

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from query_oracle import Oracle_function
import scipy.io as sio
import numpy as np
import os
from New_dataset_gen import new_dset_gen
import subprocess
import time
from acquisition_organizeer import dset_organizer
#
# from Oracle_query import query

model_name = '../Oracle_Relu_ensemble/Models_44_inks'
dset = random_dset_gen(model_name)
sio.savemat('Dataset/dset_0.mat', {'dset': dset})

data_folder = 'surrogate_pareto/with_uncertainty/'


for n_iter in range(10):
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
    os.system('ssh slurm-submit.user.de source /HPS/userCAM/work/NAGD/MO-NBO/AB/40_ink_automized_not_random_slurm/run_MONBO_slurm.sh')
    dir_path = 'Models/iter_%d' %(n_iter)
    while True:
        # Get the number of files in the directory
        num_files = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))

        # If there are 10 files, break out of the loop
        if num_files == 10:
            break
        # If there are less than 10 files, wait for 1 second and try again
        time.sleep(1)


    # Acquisition
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
    os.system('ssh slurm-submit.user.de source /HPS/userCAM/work/NAGD/MO-NBO/AB/40_ink_automized_not_random_slurm/run_acquisition_slurm.sh')
    dir_path_acquisition = 'Acquisition/iter_%d' %(n_iter)
    while True:
        # Get the number of files in the directory
        num_files = (len([entry for entry in os.listdir(dir_path_acquisition) if os.path.isfile(os.path.join(dir_path_acquisition, entry))]))
        # If there are 10 files, break out of the loop
        if num_files == 20:
            break
        # If there are less than 10 files, wait for 1 second and try again
        time.sleep(1)

    dset_new = dset_organizer(n_iter)

    # # # query the oracle to generate new data set
    mat = sio.loadmat('Dataset/dset_%d.mat' %(n_iter))
    mat = mat['dset']
    dset = mat[0][0]


    dset['X'] = np.concatenate((dset['X'], dset_new['X']))
    dset['Y'] = np.concatenate((dset['Y'], dset_new['Y']))

    sio.savemat('Dataset/dset_%d.mat' %(n_iter+1), {'dset': dset})







