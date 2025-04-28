import numpy as np
import nibabel as nib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_path',
    required=True,
    help='Root path of the data (default: None)',
    type=str
)
parser.add_argument(
    '--sub_id',
    required=True,
    help='Root path of the data (default: None)',
    type=str
)

def get_pot2(x):
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-16)
    theta = np.clip(np.abs(np.sum(x[:, None] * x[None, :], axis=-1)), 0, 1)
    distance = np.arccos(theta)
    potential = 1 / (distance + 1e-8)
    d = np.arange(len(x))
    potential[d, d] = 0
    potential = 2*potential.sum(axis=-1)
    return potential

args = parser.parse_args()
root_path = args.root_path
sub_id = args.sub_id

if root_path[-1] == '/':
    root_path = root_path[:-1]

data_path = f'{root_path}/{sub_id}'
save_path_masked_data = f'{root_path}_masked/{sub_id}'
bvals = np.loadtxt(f'{data_path}/bvals.bvals')
bvecs = np.loadtxt(f'{data_path}/bvecs.bvecs')
if bvecs.shape[1] != 3:
    bvecs = bvecs.T
data = nib.load(f'{data_path}/features.nii.gz')
affine = data.affine
header = data.header
data = data.get_fdata()

bvals = np.rint(bvals / 100) * 100
mask_bvals = np.ones(len(bvals))


b = 1000
v_ = 1e16
index_list = []
loss_list = []

index = []
x = bvecs[bvals==b]
x[x[:, 2]<0] = - x[x[:, 2]<0]
mask_b = np.zeros(len(bvals[bvals==b]))
index.append(np.argmax(x[:, 2]))
mask_b[index] = 1
sel_x = x[mask_b==1]
v_ = get_pot2(sel_x)
print(v_)
for i in range(28):
    ind_list = []
    v_list = []
    for ind in set(np.arange(len(bvals[bvals==b]))) - set(index):
        try_x = np.vstack((sel_x, x[ind]))
        v_ = get_pot2(try_x)
        ind_list.append(ind)
        v_list.append(v_.sum())
    index.append(ind_list[np.argmin(v_list)])
    mask_b[index] = 1
    sel_x = x[mask_b==1]
    v_ = get_pot2(sel_x)
    print(v_.sum())


mask_b = np.zeros(len(bvals[bvals==b]))
mask_b[index] = 1

mask = np.zeros(len(bvals))
mask[bvals==b] = mask_b
mask[bvals==0] = 1

np.savetxt(f'{data_path}/bvals_mask.txt', mask)

os.makedirs(save_path_masked_data, exist_ok=True)
img =  data[:, :, :, mask>0]
img = nib.Nifti1Image(img, affine, header)
nib.save(img, f"{save_path_masked_data}/features.nii.gz")
np.savetxt(f"{save_path_masked_data}/bvals.bvals", bvals[mask>0])
np.savetxt(f"{save_path_masked_data}/bvecs.bvecs", bvecs[mask>0])
os.system(f'cp {data_path}/mask.nii.gz {save_path_masked_data}/mask.nii.gz')
print(data_path)
print(save_path_masked_data)