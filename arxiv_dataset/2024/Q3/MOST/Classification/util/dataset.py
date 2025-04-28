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
from torchvision import transforms
class BasicDataset(Dataset):
    def __init__(self, args,mode):
        self.mode = mode
        self.dataset = args.dataset
        path_image = f'/fast_storage/hwihun/pkl_clcl/pkl_{args.dataset}_{mode}.pklv4'
        path_seg = f'/fast_storage/hwihun/pkl_clcl/pkl_{args.dataset}_{mode}_target.pklv4'
        self.target = self.load_pkls(path_image)
        self.label = self.load_pkls(path_seg)
        self.class_num = args.class_num
        
    def load_pkls( self,path):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        return images
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, i):
        img= self.target[i].astype('float32')
        if self.dataset ==  'Task4_IXI-T1_Sex_class':
            if self.class_num == 1:
                label = self.label[i] - 1
        elif self.dataset.startswith('Task5'):
            if self.class_num == 1:
                label = 0 if self.label[i] == 'CN' else 1
            if self.class_num == 2:
                if self.label[i] == 'CN':
                    label = 0
                elif self.label[i] == 'AD':
                    label = 1
                else:
                    label = 2
        if self.mode == 'train_task' and np.random.rand()>0.5:
            img_flip = np.flip(img.copy(),0)
        else:
            img_flip = img.copy()
        
        if self.mode == 'train_task':
            img_input = transforms.functional.rotate(torch.tensor(img_flip.copy()),np.random.uniform(-30,30)).unsqueeze(0)
        else:
            img_input = torch.tensor(img_flip.copy()).unsqueeze(0)
        return img_input, torch.tensor(label).unsqueeze(0).float()
