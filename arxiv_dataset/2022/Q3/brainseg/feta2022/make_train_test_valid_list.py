import nilearn.image as ni
import glob
from tqdm import tqdm
from scipy.ndimage import zoom
import h5py
import numpy as np
import os
import random

data_dir = '/project/ajoshi_27/feta_2022/feta_2.2'

sub_dirs = glob.glob(data_dir + '/sub*')

num_sub = 0
subbase_lst = []

for subdir in sub_dirs:
    # Create a list of files
    sub = os.path.basename(subdir)
    t2_irtk = os.path.join(subdir, 'anat', sub+'_rec-irtk_T2w.nii.gz')
    seg_irtk = os.path.join(subdir, 'anat', sub+'_rec-irtk_dseg.nii.gz')
    t2_mial = os.path.join(subdir, 'anat', sub+'_rec-mial_T2w.nii.gz')
    seg_mial = os.path.join(subdir, 'anat', sub+'_rec-mial_dseg.nii.gz')

    if os.path.exists(t2_irtk) and os.path.exists(seg_irtk):
        print('All needed files exist for subject: ' + sub)
        subbase_lst.append(os.path.join(subdir, 'anat', sub+'_rec-irtk'))
    elif os.path.exists(t2_mial) and os.path.exists(seg_mial):
        print('All needed files exist for subject: ' + sub)
        subbase_lst.append(os.path.join(subdir, 'anat', sub+'_rec-mial'))    
    else:

        print('Skipping!! All needed files fo not exist for subject: '+ sub)

random.seed(1231)
random.shuffle(subbase_lst)

num_sub = len(subbase_lst)

print(num_sub)

num_train = np.uint16(np.round(num_sub*.9))

num_test = np.uint16(np.round(num_sub*.05))

num_valid = num_sub - num_test - num_train


sub_train = subbase_lst[:num_train]
sub_test = subbase_lst[num_train:num_train+num_test]
sub_valid = subbase_lst[num_train+num_test:]


with open("train.txt", "w") as outfile:
    outfile.write("\n".join(sub_train))


with open("test.txt", "w") as outfile:
    outfile.write("\n".join(sub_test))

with open("valid.txt", "w") as outfile:
    outfile.write("\n".join(sub_valid))



# This is how to read these files
with open("train.txt",'r') as myfile:
    lst = myfile.read().splitlines()
    print(lst)


print(lst)
