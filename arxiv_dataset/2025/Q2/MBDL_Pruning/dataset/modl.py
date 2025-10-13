import sys
sys.path.append(".")

import os
import h5py
from tqdm import tqdm
import numpy as np
import torch
from dataset.torch_mri import generate_mask, addwgn
from torch_util.common import check_and_mkdir, to_tiff
from skimage.filters import gaussian
from torch.nn import functional as f
from torch.utils.data import DataLoader, Subset, Dataset
from torch_util.module import single_ftran, single_fmult, mul_ftran, mul_fmult
from dataset.pmri_fastmri_brain import RealMeasurement, uniformly_cartesian_mask, fmult, ftran

class MoDLDataset(Dataset):
    #def __init__(self, path='./dataset/dataset.hdf5'):
    def __init__(self, path="/project/cigserver4/export1/Dataset/MoDL/dataset.hdf5"):
        with h5py.File(path, 'r') as f:
            '''
            self.P = torch.from_numpy(f['trnMask'][:]).to(torch.float32)
            self.S = torch.from_numpy(f['trnCsm'][:])
            self.x = torch.from_numpy(f['trnOrg'][:])
            self.x: <class 'torch.Tensor'> torch.Size([524, 2, 256, 232]) torch.float32
            self.mul_x_init: <class 'torch.Tensor'> torch.Size([524, 2, 256, 232]) torch.float32
            self.<class 'torch.Tensor'> torch.Size([524, 256, 232]) torch.float32
            <class 'torch.Tensor'> torch.Size([524, 12, 256, 232, 2]) torch.float32
            <class 'torch.Tensor'> torch.Size([524, 12, 256, 232]) torch.complex64
            len(dataset):  1726
            <class 'numpy.ndarray'> (768, 396) complex64
            <class 'numpy.ndarray'> (20, 768, 396) complex64
            <class 'numpy.ndarray'> (20, 768, 396) complex64
            <class 'numpy.ndarray'> (768, 396) float32
            '''

            self.P = np.concatenate([f['trnMask'], f['tstMask']], 0)
            self.P = torch.from_numpy(self.P[:]).to(torch.float32)
            self.S = np.concatenate([f['trnCsm'], f['tstCsm']], 0)
            self.S = torch.from_numpy(self.S[:])
            self.x = np.concatenate([f['trnOrg'], f['tstOrg']], 0)
            self.x = torch.from_numpy(self.x[:])
            #print(f"Beginning\n[main.py - MoDLDataset] P.shape: {self.P.shape}") # torch.Size([524, 256, 232])
            #print(f"[main.py - MoDLDataset] S.shape: {self.S.shape}")  # torch.Size([524, 12, 256, 232])
            #print(f"[main.py - MoDLDataset] x.shape: {self.x.shape}")  # torch.Size([524, 256, 232])

            #####[mul_fmult] x.shape: torch.Size([524, 256, 232]) / x.dtype: torch.complex64
            #####[mul_fmult] S.shape: torch.Size([524, 12, 256, 232]) / S.dtype: torch.complex64
            #####[mul_fmult] P.shape: torch.Size([524, 256, 232]) / P.dtype: torch.float32
            #####[mul_fmult] y.shape: torch.Size([524, 12, 256, 232]) / y.dtype: torch.complex64

            self.single_y = single_fmult(self.x, self.S, self.P)
            self.single_x_init = single_ftran(self.single_y, self.S, self.P)
            self.mul_y = mul_fmult(self.x, self.S, self.P)
            self.mul_x_init = mul_ftran(self.mul_y, self.S, self.P)

            #print(f"Middle\n[main.py - MoDLDataset] single_y.shape: {self.single_y.shape}")    #torch.Size([524, 256, 232, 2])
            #print(f"[main.py - MoDLDataset] single_x_init.shape: {self.single_x_init.shape}")   #torch.Size([524, 256, 232, 2])
            #print(f"[main.py - MoDLDataset] mul_y.shape: {self.mul_y.shape}")  #torch.Size([524, 12, 256, 232, 2])
            #print(f"[main.py - MoDLDataset] mul_x_init.shape: {self.mul_x_init.shape}")  #torch.Size([524, 256, 232, 2])

            #print(f"[main.py - MoDLDataset] x.shape: {self.x.shape}") # torch.Size([524, 256, 232])
            #print(f"[main.py - MoDLDataset] single_x_init.shape: {self.single_x_init.shape}") torch.Size([524, 256, 232, 2])
            #print(f"[main.py - MoDLDataset] mul_x_init.shape: {self.mul_x_init.shape}") torch.Size([524, 256, 232, 2])
            self.x = torch.view_as_real(self.x).permute([0, 3, 1, 2]).contiguous()
            self.single_x_init = self.single_x_init.permute([0, 3, 1, 2]).contiguous()
            self.mul_x_init = self.mul_x_init.permute([0, 3, 1, 2]).contiguous()

            print("CHICAGO 0")
            # self.fast_x: self.fast_x = torch.view_as_real(self.fast_x).permute([0, 3, 1, 2]).contiguous()
            # self.fast_S: maintain
            # self.fast_P: maintain
            # self.fast_y: maintain
            '''
            self.fullysample_dataset = RealMeasurement(
                #idx_list=range(1360),
                idx_list=range(1360, 1375),
                acceleration_rate=1,
                is_return_y_smps_hat=True)
            print(len(DataLoader(self.fullysample_dataset)))

            print("CHICAGO 1")

            # Collect the data into separate lists
            self.fast_x, self.fast_S, self.fast_y, _ = [], [], [], []
            for idx in range(len(self.fullysample_dataset)):
                instant_x, instant_S, instant_y, _ = self.fullysample_dataset[idx]
                self.fast_x.append(instant_x)
                self.fast_S.append(instant_S)
                self.fast_y.append(instant_y)
                print(f"[{idx}/{len(self.fullysample_dataset)}]CHICAGO 2")
            # Convert the lists to arrays and reshape them
            self.fast_x = torch.as_tensor(np.array(self.fast_x))
            self.fast_S = torch.as_tensor(np.array(self.fast_S))
            self.fast_y = torch.as_tensor(np.array(self.fast_y))


            print("CHICAGO 3")
            _,_,n_x,n_y = self.fast_y.shape
            self.fast_P = torch.from_numpy(np.expand_dims(uniformly_cartesian_mask((n_x, n_y), acceleration_rate=4), 0))
            self.fast_y_init = fmult(self.fast_x, self.fast_S, self.fast_P)
            self.fast_x_init = ftran(self.fast_y_init, self.fast_S, self.fast_P)

            print("CHICAGO 4")

            #self.fast_x_init, _, self.fast_y_init, self.fast_P = DataLoader(self.undersample_dataset)
            self.fast_x = torch.view_as_real(self.fast_x).permute([0, 3, 1, 2]).contiguous()
            self.fast_x_init = torch.view_as_real(self.fast_x_init).permute([0, 3, 1, 2]).contiguous()

            print("CHICAGO 5")
            
            self.mul_x_init = np.concatenate([self.mul_x_init, self.fast_x_init], 0)
            self.mul_y = np.concatenate([self.mul_y, self.fast_y_init], 0)
            self.x = np.concatenate([self.x, self.fast_x], 0)
            self.P = np.concatenate([self.P, self.fast_P], 0)
            self.S = np.concatenate([self.S, self.fast_S], 0)
            '''
            #print("len(dataset): ", len(self.dataset))
            #x_hat, smps_hat, y, mask = self.dataset[10]
            # x_hat: 768, 396
            # smps_hat: 20, 768, 396
            # y: 20, 768, 396
            # mask:
            # x_hat: <class 'numpy.ndarray'> (768, 396) complex64
            # smps_hat: <class 'numpy.ndarray'> (20, 768, 396) complex64
            # y: <class 'numpy.ndarray'> (20, 768, 396) complex64
            # mask: <class 'numpy.ndarray'> (768, 396) float32

            #for ii in [x_hat, smps_hat, y, mask]:
            #    print(type(ii), ii.shape, ii.dtype)
            #self.x = np.concatenate([f['trnOrg'], f['tstOrg']], 0)


    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, item):

        return self.single_x_init[item],self.mul_x_init[item], self.single_y[item],self.mul_y[item], self.x[item], self.P[item], self.S[item]


import torch.utils.data as data


class CombinedDataset(Dataset):
    def __init__(self):
        self.modl_dataset = MoDLDataset()
        self.real_dataset = RealMeasurement(
            # idx_list=range(1360),
            idx_list=range(1375),
            acceleration_rate=1,
            is_return_y_smps_hat=True)

        # Compute the total number of samples in the combined dataset
        self.total_samples = len(self.modl_dataset) + len(self.real_dataset)

    def __getitem__(self, index):
        # If the index is within the range of the MoDLDataset, get a sample from that dataset
        if index < len(self.modl_dataset):
            return self.modl_dataset[index]
        # Otherwise, get a sample from the RealMeasurement dataset
        else:
            return self.real_dataset[index - len(self.modl_dataset)]

    def __len__(self):
        return self.total_samples


def generate_affine_grid(imgSize, translation=(0, 0), reflection=(1, 1), scale=1, rotate=0, shear=(0, 0)):
    T_translation = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]], dtype = np.float32)

    T_reflection = np.array([
        [reflection[0], 0, 0],
        [0, reflection[1], 0],
        [0, 0, 1]], dtype = np.float32)

    T_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]], dtype = np.float32)

    rotate = rotate / 180 * np.pi
    T_rotate = np.array([
        [np.cos(rotate), -np.sin(rotate), 0],
        [np.sin(rotate), np.cos(rotate),  0],
        [0, 0, 1]], dtype = np.float32)

    T_shear = np.array([
        [1, shear[0], 0],
        [shear[1], 1, 0],
        [0, 0, 1]], dtype = np.float32)

    rec = np.matmul(np.matmul(np.matmul(np.matmul(T_translation, T_reflection), T_scale), T_rotate), T_shear)
    rec = rec[:2, :]
    rec = torch.from_numpy(rec)
    theta = rec.unsqueeze_(0)

    return f.affine_grid(theta=theta, size=(1, ) + imgSize)

def generate_nonlinear_grid(imgSize, P, theta, sigma, mask):
    grid = generate_affine_grid(imgSize=imgSize).numpy()

    if P > 0:
        P_index = np.stack(np.meshgrid(range(imgSize[-2]), range(imgSize[-1])), -1)
        P_index = P_index.reshape([-1, 2])

        P_index_choice = np.arange(P_index.shape[0])
        np.random.shuffle(P_index_choice)

        P_index = P_index[P_index_choice[:P]]

        P_index_matrix = np.zeros([imgSize[-2], imgSize[-1]])
        for i in range(P_index.shape[0]):
            P_index_matrix[P_index[i, 0], P_index[i, 1]] = 1

        P_index_matrix = np.stack([P_index_matrix, P_index_matrix], -1)
        P_index_matrix = np.expand_dims(P_index_matrix, 0)

        mask = np.stack([mask, mask], -1)
        P_index_matrix = P_index_matrix * mask  # only do nonlinear transform in the brain region.

        theta = np.random.rand(grid.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]) * (theta * 2) - theta
        theta = theta * P_index_matrix

        for i in range(theta.shape[-1]):
            theta[0, ..., i] = gaussian(theta[0, ..., i], sigma=sigma)

        grid = grid + theta

    grid = torch.from_numpy(grid).to(torch.float32)

    return grid
'''
def fmult(ipt, type_, fold, noise_snr, sensitivity_map, mul_coil, pre_generated_mask=None, pre_generate_noise=None):
    comp, num_width, num_height = ipt.shape
    ipt = ipt.permute([1, 2, 0]).contiguous()

    ipt = torch.view_as_complex(ipt)

    # y_mul_coil: to return y_ considering multi coil (mul_coil, num_width, num_height)
    if mul_coil == False:
        ipt = torch.fft.fft2(ipt)
        ipt = torch.view_as_real(ipt)

        if pre_generated_mask is None:
            mask_ = generate_mask(type_=type_, fold=fold, imgSize=(num_width, num_height)).squeeze(0)

        else:
            mask_ = pre_generated_mask

        ipt = ipt * mask_

        if pre_generate_noise is not None:
            y_ = ipt + pre_generate_noise
            noise = pre_generate_noise
        else:
            y_, noise = addwgn(ipt, noise_snr)

    else:  # mul_coil == True
        # y = PFSx
        ipt = ipt.unsqueeze(0)

        # S
        ipt = ipt * sensitivity_map

        # F
        ipt = torch.fft.fft2(ipt)

        ipt = torch.view_as_real(ipt)

        if pre_generated_mask is None:
            mask_ = generate_mask(type_=type_, fold=fold, imgSize=(num_width, num_height)).squeeze(
                0)  # shape ([256, 232, 2])
        else:
            mask_ = pre_generated_mask  # shape ([256, 232, 2])

        # P
        ipt = ipt * mask_.unsqueeze(0)

        if pre_generate_noise is not None:
            y_ = ipt + pre_generate_noise
            noise = pre_generate_noise
        else:
            y_, noise = addwgn(ipt, noise_snr)

    return y_, mask_, noise

def ftran(y, S, P, mul_coil):
    # y, under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    # S, sensitivity maps, shape: batch, coils, width, height; dtype: complex
    # P, sampling mask, shape: batch, width, height; dtype: float/bool
    # compute adjoint of fast MRI, x = S^H F^H P^H x
    if mul_coil is True:
        # P^H
        y = torch.view_as_complex(y)
        P = torch.view_as_complex(P)
        P = P.unsqueeze(0)

        y = y * P
        # F^H
        x = torch.fft.ifft2(y)

        # S^H
        x = x * torch.conj(S)

        x = x.sum(0)

        x = torch.view_as_real(x)

    else:
        x = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(y)))
    return x
'''
def load_synthetic_MoDL_dataset(
        root_folder: str = './dataset/',
        translation: tuple = [0, 0],
        rotate: float = 0,
        scale: int = 0,
        nonlinear_P: int = 1000,
        nonlinear_theta: int = 5,
        nonlinear_sigma: int = 10,
        mask_type: str = 'cartesian',
        mask_fold: int = 4,
        input_snr: int = 40,
        mul_coil: bool = True
):
    #######################################
    # Source H5 File
    #######################################

    dataset_folder = root_folder

    source_folder_name = 'source_MoDL'

    source_h5_path = dataset_folder + '%s.h5' % source_folder_name

    source_path = dataset_folder + '%s/' % source_folder_name
    check_and_mkdir(source_path)

    source_qc   = dataset_folder + '%s_qc/' % source_folder_name
    check_and_mkdir(source_qc)

    if not os.path.exists(source_h5_path):
        print("Not Found Source H5 File. Start Generating it ...")

        with h5py.File(source_h5_path, 'w') as source_h5:

            with h5py.File(dataset_folder + 'dataset.hdf5') as f:
                for k in f.keys():
                    print(f[k].shape, k, f[k].dtype)

                x = np.concatenate([f['trnOrg'], f['tstOrg']], 0)
                x = np.stack([x.real, x.imag], 1)
                source_h5.create_dataset(name='x', data=x)

                s = np.concatenate([f['trnCsm'], f['tstCsm']], 0)
                source_h5.create_dataset(name='s', data=s)

                to_tiff(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), path=source_qc + 'fixed_x.tiff', is_normalized=False)

    else:
        print("Found Source H5 File.")

    #############################
    # Alignment H5 File
    #############################
    alignment_file_name = "alignment_translation=[%s]_rotate=[%s]_scale=[%s]_nonlinear_P=[%d]_theta=[%d]_sigma=[%d]" % (
        str(translation), str(rotate), str(scale), nonlinear_P, nonlinear_theta, nonlinear_sigma
    )

    alignment_h5_path = source_path + "%s.h5" % alignment_file_name

    alignment_path = source_path + "%s/" % alignment_file_name
    check_and_mkdir(alignment_path)

    alignment_qc = source_path + "%s_qc/" % alignment_file_name
    check_and_mkdir(alignment_qc)

    if not os.path.exists(alignment_h5_path):
        print("Not Found Alignment H5 File. Start Generating it ...")

        with h5py.File(source_h5_path, 'r') as source_h5:
            x = torch.from_numpy(source_h5['x'][:])
            print(x.shape, x.dtype)

            x_real, x_imag = x[:, 0], x[:, 1]

            moved_x = []

            n_slice = x.shape[0]
            for index_slice in tqdm(range(n_slice)):
                affine_grid = generate_affine_grid(
                    imgSize=x_real[index_slice].unsqueeze(0).shape,
                    translation=[
                        2 * translation[0] * np.random.rand(1) - translation[0],
                        2 * translation[1] * np.random.rand(1) - translation[1],
                    ],
                    rotate=2 * rotate * np.random.rand(1) - rotate,
                    scale=2 * scale * np.random.rand(1) - scale + 1,
                )

                non_linear_grid = generate_nonlinear_grid(
                    imgSize=x_real[index_slice].unsqueeze(0).shape,
                    P=nonlinear_P,
                    theta=nonlinear_theta,
                    sigma=nonlinear_sigma,
                    mask=1
                )

                moved_x_real = torch.nn.functional.grid_sample(x_real[index_slice].unsqueeze(0).unsqueeze(0), affine_grid, mode='bilinear')
                moved_x_real = torch.nn.functional.grid_sample(moved_x_real, non_linear_grid, mode='bilinear').squeeze(0).squeeze(0)

                moved_x_imag = torch.nn.functional.grid_sample(x_imag[index_slice].unsqueeze(0).unsqueeze(0), affine_grid, mode='bilinear')
                moved_x_imag = torch.nn.functional.grid_sample(moved_x_imag, non_linear_grid, mode='bilinear').squeeze(0).squeeze(0)

                moved_x.append(torch.stack([moved_x_real, moved_x_imag], 0))

            moved_x = torch.stack(moved_x, 0)

            with h5py.File(alignment_h5_path, 'w') as alignment_h5:
                alignment_h5.create_dataset(name='moved_x', data=moved_x)

            to_tiff(torch.sqrt(moved_x[:, 0] ** 2 + moved_x[:, 1] ** 2), path=alignment_qc + 'moved_x.tiff',
                    is_normalized=False)

    else:
        print("Found Alignment H5 File.")

    #############################
    # MRI H5 File
    #############################
    mri_file_name = 'mri_mask_type=[%s]_mask_fold=[%s]_input_snr=[%.2d]' % (
        mask_type, mask_fold, input_snr)

    mri_h5_path = alignment_path + "%s.h5" % mri_file_name

    mri_path = alignment_path + "%s/" % mri_file_name
    check_and_mkdir(mri_path)

    mri_qc = alignment_path + "%s_qc/" % mri_file_name
    check_and_mkdir(mri_qc)



    if not os.path.exists(mri_h5_path):
        print("Not Found MRI H5 File. Start Generating it ...")

        with h5py.File(source_h5_path, 'r') as source_h5:
            with h5py.File(alignment_h5_path, 'r') as alignment_h5:
                fixed_x = torch.from_numpy(source_h5['x'][:])
                moved_x = torch.from_numpy(alignment_h5['moved_x'][:])
                sensitivity_map = torch.from_numpy(source_h5['s'][:])

                num_shape = fixed_x.shape[0]
                fixed_y, fixed_mask, fixed_y_tran, moved_y, moved_mask, moved_y_tran, moved_y_warped_truth, moved_y_tran_warped_truth = [], [], [], [], [], [], [], []
                mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, mul_moved_y, mul_moved_mask, mul_moved_y_tran, mul_moved_y_warped_truth, mul_moved_y_tran_warped_truth = [], [], [], [], [], [], [], []
                for i_shape in tqdm(range(num_shape)):
                    # Single Coil
                    fixed_y_cur, fixed_mask_cur, fixed_noise_cur = fmult(fixed_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map = sensitivity_map[i_shape], mul_coil = False, noise_snr=input_snr)
                    moved_y_cur, moved_mask_cur, moved_noise_cur = fmult(moved_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map = sensitivity_map[i_shape], mul_coil= False, noise_snr=input_snr)
                    moved_y_warped_truth_cur, moved_mask_warped_truth_cur, _ = fmult(fixed_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map=sensitivity_map[i_shape], mul_coil = False, noise_snr=input_snr, pre_generate_noise=moved_noise_cur, pre_generated_mask=moved_mask_cur)

                    fixed_y_tran_cur = ftran(fixed_y_cur, sensitivity_map[i_shape], fixed_mask_cur, mul_coil=False)
                    moved_y_tran_cur = ftran(moved_y_cur, sensitivity_map[i_shape], moved_mask_cur, mul_coil=False)
                    moved_y_tran_warped_truth_cur = ftran(moved_y_warped_truth_cur, sensitivity_map[i_shape], moved_mask_warped_truth_cur, mul_coil=False)

                    fixed_y.append(fixed_y_cur)
                    fixed_mask.append(fixed_mask_cur)
                    fixed_y_tran.append(fixed_y_tran_cur)

                    moved_y.append(moved_y_cur)
                    moved_mask.append(moved_mask_cur)
                    moved_y_tran.append(moved_y_tran_cur)

                    moved_y_warped_truth.append(moved_y_warped_truth_cur)
                    moved_y_tran_warped_truth.append(moved_y_tran_warped_truth_cur)

                    # Multi Coil
                    mul_fixed_y_cur, mul_fixed_mask_cur, mul_fixed_noise_cur = fmult(fixed_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map = sensitivity_map[i_shape], mul_coil = True, noise_snr=input_snr)
                    mul_moved_y_cur, mul_moved_mask_cur, mul_moved_noise_cur = fmult(moved_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map = sensitivity_map[i_shape], mul_coil= True, noise_snr=input_snr)
                    mul_moved_y_warped_truth_cur, mul_moved_mask_warped_truth_cur, _ = fmult(fixed_x[i_shape], type_=mask_type, fold=mask_fold, sensitivity_map=sensitivity_map[i_shape], mul_coil = True, noise_snr=input_snr, pre_generate_noise=mul_moved_noise_cur, pre_generated_mask=mul_moved_mask_cur)

                    mul_fixed_y_tran_cur = ftran(mul_fixed_y_cur, sensitivity_map[i_shape], mul_fixed_mask_cur, mul_coil=True)
                    mul_moved_y_tran_cur = ftran(mul_moved_y_cur, sensitivity_map[i_shape], mul_moved_mask_cur, mul_coil=True)
                    mul_moved_y_tran_warped_truth_cur = ftran(mul_moved_y_warped_truth_cur, sensitivity_map[i_shape], mul_moved_mask_warped_truth_cur, mul_coil=True)

                    mul_fixed_y.append(mul_fixed_y_cur)
                    mul_fixed_mask.append(mul_fixed_mask_cur)
                    mul_fixed_y_tran.append(mul_fixed_y_tran_cur)

                    mul_moved_y.append(mul_moved_y_cur)
                    mul_moved_mask.append(mul_moved_mask_cur)
                    mul_moved_y_tran.append(mul_moved_y_tran_cur)

                    mul_moved_y_warped_truth.append(mul_moved_y_warped_truth_cur)
                    mul_moved_y_tran_warped_truth.append(mul_moved_y_tran_warped_truth_cur)

                    '''
                    [modl.py][fixed_y_cur.shape]: torch.Size([256, 232, 2])
                    [modl.py][fixed_mask_cur.shape]:  torch.Size([256, 232, 2])
                    [modl.py][fixed_noise_cur.shape]: torch.Size([256, 232, 2])
                    [modl.py][mul_fixed_y_cur.shape]:  torch.Size([12, 256, 232, 2])
                    [modl.py][mul_fixed_mask_cur.shape]: torch.Size([256, 232, 2])
                    [modl.py][mul_fixed_noise_cur.shape]:  torch.Size([12, 256, 232, 2])
                    [modl.py][fixed_y_cur.shape]: torch.Size([256, 232, 2])
                    [modl.py][fixed_mask_cur.shape]:  torch.Size([256, 232, 2])
                    [modl.py][fixed_noise_cur.shape]: torch.Size([256, 232, 2])
                    [modl.py][mul_fixed_y_cur.shape]:  torch.Size([12, 256, 232, 2])
                    [modl.py][mul_fixed_mask_cur.shape]: torch.Size([256, 232, 2])
                    [modl.py][mul_fixed_noise_cur.shape]:  torch.Size([12, 256, 232, 2])
                    '''

                fixed_y, fixed_mask, fixed_y_tran, moved_y, moved_mask, moved_y_tran, moved_y_warped_truth, moved_y_tran_warped_truth = [
                    torch.stack(i, 0) for i in [fixed_y, fixed_mask, fixed_y_tran, moved_y, moved_mask, moved_y_tran, moved_y_warped_truth, moved_y_tran_warped_truth]]

                mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, mul_moved_y, mul_moved_mask, mul_moved_y_tran, mul_moved_y_warped_truth, mul_moved_y_tran_warped_truth = [
                    torch.stack(i, 0) for i in [mul_fixed_y, mul_fixed_mask, mul_fixed_y_tran, mul_moved_y, mul_moved_mask, mul_moved_y_tran, mul_moved_y_warped_truth, mul_moved_y_tran_warped_truth]]

                with h5py.File(mri_h5_path, 'w') as mri_h5:
                    mri_h5.create_dataset(name='fixed_y', data=fixed_y)
                    mri_h5.create_dataset(name='fixed_mask', data=fixed_mask)
                    mri_h5.create_dataset(name='fixed_y_tran', data=fixed_y_tran)
                    mri_h5.create_dataset(name='moved_y', data=moved_y)
                    mri_h5.create_dataset(name='moved_mask', data=moved_mask)
                    mri_h5.create_dataset(name='moved_y_tran', data=moved_y_tran)
                    mri_h5.create_dataset(name='moved_y_warped_truth', data=moved_y_warped_truth)
                    mri_h5.create_dataset(name='moved_y_tran_warped_truth', data=moved_y_tran_warped_truth)

                    mri_h5.create_dataset(name='mul_fixed_y', data=mul_fixed_y)
                    mri_h5.create_dataset(name='mul_fixed_mask', data=mul_fixed_mask)
                    mri_h5.create_dataset(name='mul_fixed_y_tran', data=mul_fixed_y_tran)
                    mri_h5.create_dataset(name='mul_moved_y', data=mul_moved_y)
                    mri_h5.create_dataset(name='mul_moved_mask', data=mul_moved_mask)
                    mri_h5.create_dataset(name='mul_moved_y_tran', data=mul_moved_y_tran)
                    mri_h5.create_dataset(name='mul_moved_y_warped_truth', data=mul_moved_y_warped_truth)
                    mri_h5.create_dataset(name='mul_moved_y_tran_warped_truth', data=mul_moved_y_tran_warped_truth)
                '''
                if mul_coil is True:
                    #to_tiff(torch.sqrt(fixed_y[..., 0] ** 2 + fixed_y[..., 1] ** 2), path=mri_qc + 'fixed_y.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(fixed_mask[..., 0] ** 2 + fixed_mask[..., 1] ** 2), path=mri_qc + 'fixed_mask.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(fixed_y_tran[..., 0] ** 2 + fixed_y_tran[..., 1] ** 2), path=mri_qc + 'fixed_y_tran.tiff', is_normalized=False)
                    #to_tiff(torch.sqrt(moved_y[..., 0] ** 2 + moved_y[..., 1] ** 2), path=mri_qc + 'moved_y.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_mask[..., 0] ** 2 + moved_mask[..., 1] ** 2), path=mri_qc + 'moved_mask.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_tran[..., 0] ** 2 + moved_y_tran[..., 1] ** 2), path=mri_qc + 'moved_y_tran.tiff', is_normalized=False)
                    #to_tiff(torch.sqrt(moved_y_warped_truth[..., 0] ** 2 + moved_y_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_warped_truth.tiff', is_normalized=False)
                    to_tiff(torch.sqrt(moved_y_tran_warped_truth[..., 0] ** 2 + moved_y_tran_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_tran_warped_truth.tiff', is_normalized=False)
                '''
                to_tiff(torch.sqrt(fixed_y[..., 0] ** 2 + fixed_y[..., 1] ** 2), path=mri_qc + 'fixed_y.tiff', is_normalized=False)
                to_tiff(torch.sqrt(fixed_mask[..., 0] ** 2 + fixed_mask[..., 1] ** 2), path=mri_qc + 'fixed_mask.tiff', is_normalized=False)
                to_tiff(torch.sqrt(fixed_y_tran[..., 0] ** 2 + fixed_y_tran[..., 1] ** 2), path=mri_qc + 'fixed_y_tran.tiff', is_normalized=False)
                to_tiff(torch.sqrt(moved_y[..., 0] ** 2 + moved_y[..., 1] ** 2), path=mri_qc + 'moved_y.tiff', is_normalized=False)
                to_tiff(torch.sqrt(moved_mask[..., 0] ** 2 + moved_mask[..., 1] ** 2), path=mri_qc + 'moved_mask.tiff', is_normalized=False)
                to_tiff(torch.sqrt(moved_y_tran[..., 0] ** 2 + moved_y_tran[..., 1] ** 2), path=mri_qc + 'moved_y_tran.tiff', is_normalized=False)
                to_tiff(torch.sqrt(moved_y_warped_truth[..., 0] ** 2 + moved_y_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_warped_truth.tiff', is_normalized=False)
                to_tiff(torch.sqrt(moved_y_tran_warped_truth[..., 0] ** 2 + moved_y_tran_warped_truth[..., 1] ** 2), path=mri_qc + 'moved_y_tran_warped_truth.tiff', is_normalized=False)

                to_tiff(torch.sqrt(mul_fixed_y[..., 0] ** 2 + mul_fixed_y[..., 1] ** 2), path=mri_qc + 'mul_fixed_y.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_fixed_mask[..., 0] ** 2 + mul_fixed_mask[..., 1] ** 2), path=mri_qc + 'mul_fixed_mask.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_fixed_y_tran[..., 0] ** 2 + mul_fixed_y_tran[..., 1] ** 2), path=mri_qc + 'mul_fixed_y_tran.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_moved_y[..., 0] ** 2 + mul_moved_y[..., 1] ** 2), path=mri_qc + 'mul_moved_y.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_moved_mask[..., 0] ** 2 + mul_moved_mask[..., 1] ** 2), path=mri_qc + 'mul_moved_mask.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_moved_y_tran[..., 0] ** 2 + mul_moved_y_tran[..., 1] ** 2), path=mri_qc + 'mul_moved_y_tran.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_moved_y_warped_truth[..., 0] ** 2 + mul_moved_y_warped_truth[..., 1] ** 2), path=mri_qc + 'mul_moved_y_warped_truth.tiff', is_normalized=False)
                to_tiff(torch.sqrt(mul_moved_y_tran_warped_truth[..., 0] ** 2 + mul_moved_y_tran_warped_truth[..., 1] ** 2), path=mri_qc + 'mul_moved_y_tran_warped_truth.tiff', is_normalized=False)

    else:
        print("Found MRI H5 File.")

    #######################################
    # Read Images from H5 Files
    #######################################
    source_h5 = h5py.File(source_h5_path, 'r')
    alignment_h5 = h5py.File(alignment_h5_path, 'r')
    mri_h5 = h5py.File(mri_h5_path, 'r')


    ret_cur = {
        'fixed_x': source_h5['x'][:],
        'sensitivity_map': source_h5['s'][:],
        'moved_x': alignment_h5['moved_x'][:],

        'fixed_y': mri_h5['fixed_y'][:],
        'fixed_mask': mri_h5['fixed_mask'][:],
        'fixed_y_tran': np.transpose(mri_h5['fixed_y_tran'][:], [0, 3, 1, 2]),

        'moved_y': mri_h5['moved_y'][:],
        'moved_mask': mri_h5['moved_mask'][:],
        'moved_y_tran': np.transpose(mri_h5['moved_y_tran'][:], [0, 3, 1, 2]),
        'moved_y_warped_truth': mri_h5['moved_y_warped_truth'][:],
        'moved_y_tran_warped_truth': np.transpose(mri_h5['moved_y_tran_warped_truth'][:], [0, 3, 1, 2]),

        'mul_fixed_y': mri_h5['mul_fixed_y'][:],
        'mul_fixed_mask': mri_h5['mul_fixed_mask'][:],
        'mul_fixed_y_tran': np.transpose(mri_h5['mul_fixed_y_tran'][:], [0, 3, 1, 2]),

        'mul_moved_y': mri_h5['mul_moved_y'][:],
        'mul_moved_mask': mri_h5['mul_moved_mask'][:],
        'mul_moved_y_tran': np.transpose(mri_h5['mul_moved_y_tran'][:], [0, 3, 1, 2]),
        'mul_moved_y_warped_truth': mri_h5['mul_moved_y_warped_truth'][:],
        'mul_moved_y_tran_warped_truth': np.transpose(mri_h5['mul_moved_y_tran_warped_truth'][:], [0, 3, 1, 2])
    }


    ret = ret_cur

    return ret