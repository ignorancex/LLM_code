import scipy.io as sio
import numpy as np
import subprocess
import os

# define three commands to run
# Note that before this step you need to make sure that all three containers' id are known 
command1 = "ssh user@puck docker exec 1bc7e17a221a ./home/airfoil_UANA/run_rand_dset_1.sh"
command2 = "ssh user@dione docker exec 9f5255033fd4 ./home/airfoil_UANA/run_rand_dset_2.sh"
command3 = "ssh user@carpo docker exec 3e64c5a9096b ./home/airfoil_UANA/run_rand_dset_3.sh"

# run commands in parallel
processes = [subprocess.Popen(cmd, shell=True) for cmd in [command1, command2, command3]]

# wait for all processes to finish
for process in processes:
    process.wait()

print('all 3 tasks done')

# Organizing the data
# Initialize empty arrays to store the data
latent_all = []
performance_all = []

# Loop through the three batches
for n_batch in range(1, 4):
    # Read the design dataset
    dset_design_path = f"./Dataset/temp/dset_design_{n_batch}.mat"
    dset_design =sio.loadmat(dset_design_path)
    latent = dset_design['latent']

    # Read the performance dataset
    dset_perform_path = f"./Dataset/temp/dset_perform_{n_batch}.mat"
    dset_perform = sio.loadmat(dset_perform_path)
    performance = dset_perform['performance']

    # Stack the arrays
    latent_all.append(latent)
    performance_all.append(performance)

# Concatenate the stacked arrays along the first axis
latent_all = np.concatenate(latent_all, axis=0)
performance_all = np.concatenate(performance_all, axis=0)
# clean the data of the failed cases
latent_all = latent_all[performance_all[:,0]!=-1000,:]
performance_all = performance_all[performance_all[:,0]!=-1000,:]
data = {'X':latent_all, 'Y':performance_all}
sio.savemat('Dataset/dset_0.mat', {'dset': data})


#Remove the temp files
folder_path = "./Dataset/temp"
# Get a list of all files in the folder
file_list = os.listdir(folder_path)
# Loop through the list and delete each file
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    os.remove(file_path)