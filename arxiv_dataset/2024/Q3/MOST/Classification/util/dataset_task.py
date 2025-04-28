#%%
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
import os

class BasicDataset(Dataset):
    def __init__(self,path,args,istrain = False):
       
        self.target = self.load_pkls(path)
        self.label = self.load_pkls(path.replace('.pklv4','_target.pklv4'))
        self.args = args
        self.istrain = istrain
        
        self.target = [np.transpose(image, [2, 0, 1]) for image in self.target]
        num_cols = self.target[0].shape[1]
        num_rows = self.target[0].shape[2]
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
        self.mask = np.repeat(mask,num_rows,axis =2)
        
        self.hr_images = [self.undersample(image,self.mask) for image in self.target]
        del self.target
        
    @classmethod
    def undersample(cls, x,mask):
        x_k = fft2c(x)
        masked_kspace_sudo = mask*x_k +0.0
        image_input_sudo = np.abs(ifft2c(masked_kspace_sudo))
        return image_input_sudo
    
    def load_pkls( self,path):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        #images = images[0:10]
        return images[0:50]
    
    @classmethod
    def norm_slice(self,img):
        crop = np.zeros_like(img)
        for j in range(img.shape[2]):
            slice = img[:,:,j:j+1]
            crop[:,:,j:j+1] = (slice-np.mean(slice))/np.std(slice)
        return crop
            
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, i):
        img= self.hr_images[i].astype('float32')
        label = 0 if self.label[i] == 'CN' else 1
        
        return torch.tensor(img).unsqueeze(0), torch.tensor(label)

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1))),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1))),axes = (-2,-1))