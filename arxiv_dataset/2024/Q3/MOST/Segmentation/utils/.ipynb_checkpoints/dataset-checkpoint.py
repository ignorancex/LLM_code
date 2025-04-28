from os.path import splitext
from os import listdir
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import hdf5storage

class BasicDataset(Dataset):
    def __init__(self,path,acceleration,center_fraction,which_noise,noise_std):
        load = hdf5storage.loadmat(path)
        self.target = load['target']
        self.which_noise = which_noise
        self.noise_std = noise_std
        num_cols = self.target.shape[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
            num_low_freqs * acceleration - num_cols
        )
        
        offset = 0
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True
        
        # reshape the mask
        mask_shape = [1,320,1]
        mask = mask.reshape(*mask_shape)
        self.mask = np.fft.ifftshift(np.fft.ifftshift(mask,1),2)
        

        logging.info(f'Creating dataset with {self.target.shape[0]} examples')
        
    @classmethod
    def undersample(cls, x,mask):
        masked_kspace_sudo = mask*np.fft.ifft2(x) +0.0
        image_input_sudo = np.abs(np.fft.fft2(masked_kspace_sudo))
        return image_input_sudo, x

    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, i):
        img, target = self.undersample(self.target[i:i+1,:,:],self.mask)
        if self.which_noise == 'in_mask':
            noise = np.fft.fft2(np.where(self.mask!=0,self.noise_std*np.random.randn(self.mask.shape[0],self.mask.shape[1],self.mask.shape[2]),np.zeros_like(self.mask)))
        elif self.which_noise == 'out_mask':
            noise = np.fft.fft2(np.where(self.mask!=0,np.zeros_like(self.mask),self.noise_std*np.random.randn(self.mask.shape[0],self.mask.shape[1],self.mask.shape[2])))
        elif self.which_noise == 'every':
            noise = np.fft.fft2(self.noise_std*np.random.randn(self.mask.shape[0],self.mask.shape[1],self.mask.shape[2]))
        else:
            noise = np.zeros_like(self.mask)
        target = target+np.real(noise)
        assert img.size == target.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {target.size}'

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'target': torch.from_numpy(target).type(torch.FloatTensor)
        }


