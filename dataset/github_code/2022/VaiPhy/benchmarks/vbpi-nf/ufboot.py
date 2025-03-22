######################################################################################
#
# Authors :
#
#   Email :
#
# ufboot.py: Functions to create UFBoot for Vbpi-Nf.
#            User must specify the place where they installed IQTree.
#####################################################################################

import os
import time

print("Hello, world!")

# Some parameters
data_path = "../../data/"

n_rep = 10       # number of repetitions
n_trees = 10000  # number of bootstrap trees

print("n_rep: ", n_rep, "\tn_trees: ", n_trees)

'''
# For synthetic data
for data_seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for n_leaves in [10, 20, 40]:
        for n_pos in [300, 1000]:
            print(f"Creating UFBoot Trees for Taxa = {n_leaves}, \t Pos = {n_pos}, \t Data_seed = {data_seed}...")

            dataset_name = "data_seed_" + str(data_seed) + "_taxa_" + str(n_leaves) + "_pos_" + str(n_pos)
            data_folder = data_path + dataset_name + "/"

            # Create output folders
            ufboot_dir = data_folder + "ufboot/"
            if not os.path.exists(ufboot_dir):
                os.makedirs(ufboot_dir)

            ufboot_dir += dataset_name + "/"
            if not os.path.exists(ufboot_dir):
                os.makedirs(ufboot_dir)

            nex_fname = data_folder + dataset_name + ".nex"
            t_start = time.time()

            # Run UFBoot multiple times
            for i in range(n_rep):
                # Run UFBoot
                cmd = "iqtree-1.6.12-MacOSX/bin/iqtree -s " + nex_fname + " -bb " + str(n_trees) + " -wbt -m JC69 -quiet -redo"
                os.system(cmd)

                # Move results
                cmd = "cp " + nex_fname + ".ufboot " + ufboot_dir + dataset_name + "_ufboot_rep_" + str(i+1)
                os.system(cmd)

            print("\tTime: ", str(time.time() - t_start))

            # Clean files
            cmd = "rm " + nex_fname + ".*"
            os.system(cmd)
'''

# For real data
for data_seed in [5, 6, 7, 8, 9, 10, 11]:
    dataset_name = "DS" + str(data_seed)

    print(f"Creating UFBoot Trees for Dataset = {dataset_name}...")

    data_folder = data_path + dataset_name + "/"

    # Create output folders
    ufboot_dir = data_folder + "ufboot/"
    if not os.path.exists(ufboot_dir):
        os.makedirs(ufboot_dir)

    ufboot_dir += dataset_name + "/"
    if not os.path.exists(ufboot_dir):
        os.makedirs(ufboot_dir)

    fasta_fname = data_folder + dataset_name + ".fasta"
    t_start = time.time()

    # Run UFBoot multiple times
    for i in range(n_rep):
        # Run UFBoot
        cmd = "iqtree-1.6.12-MacOSX/bin/iqtree -s " + fasta_fname + " -bb " + str(n_trees) + " -wbt -m JC69 -quiet -redo"
        os.system(cmd)

        # Move results
        cmd = "cp " + fasta_fname + ".ufboot " + ufboot_dir + dataset_name + "_ufboot_rep_" + str(i+1)
        os.system(cmd)

    print("\tTime: ", str(time.time() - t_start))

    # Clean files
    cmd = "rm " + fasta_fname + ".*"
    os.system(cmd)
