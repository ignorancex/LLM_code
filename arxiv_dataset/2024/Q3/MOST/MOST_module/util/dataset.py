
from os.path import splitext
from os import listdir
import numpy as np
from numpy.compat.py3k import isfileobj
import scipy.io
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import matplotlib.pyplot as plt
import pickle

class BasicDataset(Dataset):
    def __init__(self,path,args,istrain = False):
        self.target = []
        self.args = args
        with open(path, "rb") as f:
            self.target += pickle.load(f)
        self.target = [np.transpose(image, [2, 0, 1]) for image in self.target]
        num_cols = self.target[0].shape[1]
        center_fraction = args.center_fraction
        acceleration = args.acceleration
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
        mask_shape = [1,num_cols,1]
        mask = mask.reshape(*mask_shape)
        self.mask = np.repeat(mask,num_cols,axis =2)
        
        self.img = [self.undersample(image,self.mask) for image in self.target]
        logging.info(f'Creating dataset with {len(self.target)} examples')
        
    @classmethod
    def undersample(cls, x,mask):
        x_k = fft2c(x)
        masked_kspace_sudo = mask*x_k +0.0
        image_input_sudo = np.abs(ifft2c(masked_kspace_sudo))
        return image_input_sudo

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, i):
        #img, target = self.undersample(self.target[i],self.mask)
        img= self.img[i]
        target =self.target[i]
        assert img.size == target.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {target.size}'
        return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(target).type(torch.FloatTensor)


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1))),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1))),axes = (-2,-1))
