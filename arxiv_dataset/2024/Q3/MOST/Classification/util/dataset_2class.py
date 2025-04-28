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

class BasicDataset(Dataset):
    def __init__(self,path,args,istrain = False):
       
        self.label = []
        if istrain:
            mode = 'train'
        else:
            mode = 'val'
        path_ad = f'{path}/AD_{mode}.pklv4'
        path_cn = f'{path}/CN_{mode}.pklv4'
        path_mci = f'{path}/MCI_{mode}.pklv4'
        path_emci = f'{path}/EMCI_{mode}.pklv4'
        path_lmci = f'{path}/LMCI_{mode}.pklv4'
        
        self.target = []
        target = []
        with open(path_ad, "rb") as f:
            target += pickle.load(f)
        for image in target:
            self.target.append(np.expand_dims(self.norm_slice(image),0))
            self.label.append(0)
        target = []
        with open(path_cn, "rb") as f:
            target += pickle.load(f)
        for image in target:
            self.target.append(np.expand_dims(self.norm_slice(image),0))
            self.label.append(1)
        target = []    
        with open(path_mci, "rb") as f:
            target += pickle.load(f)
        for image in target:
            self.target.append(np.expand_dims(self.norm_slice(image),0))
            self.label.append(1)
        target = []
        with open(path_emci, "rb") as f:
            target += pickle.load(f)
        for image in target:
            self.target.append(np.expand_dims(self.norm_slice(image),0))
            self.label.append(1)
        target = []    
        with open(path_lmci, "rb") as f:
            target += pickle.load(f)
        for image in target:
            self.target.append(np.expand_dims(self.norm_slice(image),0))
            self.label.append(1)
        del target
        self.args = args
        self.istrain = istrain

    @classmethod
    def norm_slice(self,img):
        crop = np.zeros_like(img)
        for j in range(img.shape[2]):
            slice = img[:,:,j:j+1]
            crop[:,:,j:j+1] = (slice-np.mean(slice))/np.std(slice)
        return crop
            
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, i):
        img= self.target[i]
        label =self.label[i]
        
        return torch.tensor(img), torch.tensor(label)
