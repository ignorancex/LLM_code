# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import cv2
import math
import logging

class BasicDataset(data.Dataset):
    def __init__(self,path,args,istrain = False):
        self.len = args.len
        self.seg_images = self.load_pkls(path.replace('.pklv4','_target.pklv4'))
        
        self.args = args
        self.target = self.load_pkls(path)
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
        logging.info(f'Creating dataset with {len(self.target)} examples')
        
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
        return images[0:self.len*50]
    

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        img = np.abs(self.hr_images[item].astype('float32'))
        seg = self.seg_images[item] == 3

        #return img.transpose([2, 0, 1]),seg.astype('float32')
        return img,seg.astype('float32').transpose([2, 0, 1])
        #return gt, gt15
        
def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1))),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1))),axes = (-2,-1))
