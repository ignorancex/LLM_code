import torch
from torch.utils.data import DataLoader, Dataset
import csv
import os
from PIL import Image
import numpy as np
from skimage.transform import resize
from torchvision.models import *
import pickle
import pandas as pd

class FairFace(Dataset):
    def __init__(self, file_path, split, attribute, resolution = 224, needBalance=False):
      
        # file_path = "/scratch/cw3437/Data/fairface/"
        self.resolution = resolution
        self.file_path = file_path  
      
        if split=='train':
            self.filename = "train/labels_train/train_gtruth_10000.csv"
            self.img_folder = "train/data/"
            self.num_data = 10000

        if split=='val':
            self.filename = "labels_val/val_gtruth_1000.csv"
            self.img_folder = "val/data/"
            self.num_data = 1000
            
        if split=='test':
            self.filename = "labels_test/test_gtruth_2000.csv"
            self.img_folder = "test/data/"
            self.num_data = 2000            
    
    def __getitem__(self, index):

        df = pd.read_csv(self.file_path+self.filename)
        line = df.iloc[index]

        imagePath = self.file_path + self.img_folder + str(line[1])+ '.jpg'
        imageData = Image.open(imagePath).convert('RGB')
        img_data = np.array(imageData)
        if img_data.shape[0] != self.resolution or img_data.shape[1] != self.resolution:
            img_data = resize(img_data, (self.resolution, self.resolution))
        img_data = np.transpose(img_data)
        img_data = img_data.astype(np.float32)  
        
        attr_label = []
        age_label = int(line[3])
        skinColor_label = int(line[4])
        gender_label = int(line[5])
        attr_label.append(age_label)
        attr_label.append(skinColor_label)
        attr_label.append(gender_label)

        label = int(line[8])

        return img_data, label, attr_label
    
    def __len__(self):
        return self.num_data 

