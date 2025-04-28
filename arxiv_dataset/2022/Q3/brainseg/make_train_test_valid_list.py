import nilearn.image as ni
import glob
from tqdm import tqdm
from scipy.ndimage import zoom
import h5py
import numpy as np
import os

headreco_t1_dir = '/project/ajoshi_27/headreco_out_t1'
headreco_t1t2_dir = '/project/ajoshi_27/headreco_out'

sub_dirs = glob.glob(headreco_t1t2_dir + '/m2m*')

num_sub = 0
sub_lst = []

for subdir in sub_dirs:
    # Create a list of files
    subname = os.path.basename(subdir)[4:]
    t1 = os.path.join(subdir, 'T1fs_conform.nii.gz')
    t2 = os.path.join(subdir, 'T2_conform.nii.gz')
    seg = os.path.join(subdir, subname + '_masks_contr.nii.gz')
    t1only = os.path.join(headreco_t1_dir,'m2m_' + subname, 'T1fs_conform.nii.gz')
    seg_t1only = os.path.join(headreco_t1_dir,'m2m_' + subname, subname + '_masks_contr.nii.gz')

    if os.path.exists(t1) and os.path.exists(t2) and os.path.exists(seg) and os.path.exists(t1only) and os.path.exists(seg_t1only):
        print('All needed files exist for subject: ' + subname)
        sub_lst.append(subname)


num_sub = len(sub_lst)

print(num_sub)

num_train = np.uint16(np.round(num_sub*.7))

num_test = np.uint16(np.round(num_sub*.2))

num_valid = num_sub - num_test - num_train


sub_train = sub_lst[:num_train]
sub_test = sub_lst[num_train:num_train+num_test]
sub_valid = sub_lst[num_train+num_test:]


with open("train.txt", "w") as outfile:
    outfile.write("\n".join(sub_train))


with open("test.txt", "w") as outfile:
    outfile.write("\n".join(sub_test))

with open("valid.txt", "w") as outfile:
    outfile.write("\n".join(sub_valid))



# This is how to read these files
with open("train.txt",'r') as myfile:
    lst = myfile.read().splitlines()


