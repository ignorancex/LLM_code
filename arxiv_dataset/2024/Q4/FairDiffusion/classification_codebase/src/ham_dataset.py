import torch
from torch.utils.data import DataLoader, Dataset
import csv
import os
from PIL import Image
import numpy as np
from skimage.transform import resize
from torchvision.models import *
import pickle
import glob

class HAM10000(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224):
      
    # file_path = "../../Data/HAM10000/"
    self.resolution = resolution
      
    self.img_list = pickle.load(open(file_path + split +'_images.pkl','rb'))

    split = split
    list_file = "new_" + split + ".csv"
    
    gender_mapping = {'M':0, 
                'F':1}
    # img_list = []
    self.label_list = []
    self.attr_list = []
    
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        row_head = next(csvReader)
        for line in csvReader:
            
            label = int(float(line[13]))
            
            attr_label = []
            age_label = int(float(line[11]))
            gender_label = gender_mapping[line[12]]
            
            attr_label.append(gender_label)
            attr_label.append(age_label)
            
            self.label_list.append(label)
            self.attr_list.append(attr_label)

  def __getitem__(self, index):
    img_data = self.img_list[index]
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
    img_data = np.transpose(img_data)
    img_data = img_data.astype(np.float32)  
    return img_data, self.label_list[index], self.attr_list[index]

  def __len__(self):

    return len(self.img_list)


class HAM10000_Gen(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224):
      
    # file_path = "../../Data/HAM10000/"
    self.resolution = resolution
      
    self.img_list = sorted(glob.glob('/scratch/cw3437/Data/HAM10000/ham_gen/*.jpg'))

    split = split
    list_file = "new_" + split + ".csv"
    
    gender_mapping = {'M':0, 
                'F':1}
    # img_list = []
    self.label_list = []
    self.attr_list = []
    
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        row_head = next(csvReader)
        for line in csvReader:
            
            label = int(float(line[13]))
            
            attr_label = []
            age_label = int(float(line[11]))
            gender_label = gender_mapping[line[12]]
            
            attr_label.append(gender_label)
            attr_label.append(age_label)
            
            self.label_list.append(label)
            self.attr_list.append(attr_label)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    img_data = np.array(imageData)
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
    img_data = np.transpose(img_data)
    img_data = img_data.astype(np.float32)  
    return img_data, self.label_list[index], self.attr_list[index]

  def __len__(self):

    return len(self.img_list)



class HAM10000_FairGen(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224):
      
    # file_path = "../../Data/HAM10000/"
    self.resolution = resolution
      
    self.img_list = sorted(glob.glob('/scratch/cw3437/Data/HAM10000/fairham_gen/*.jpg'))

    split = split
    list_file = "new_" + split + ".csv"
    
    gender_mapping = {'M':0, 
                'F':1}
    # img_list = []
    self.label_list = []
    self.attr_list = []
    
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        row_head = next(csvReader)
        for line in csvReader:
            
            label = int(float(line[13]))
            
            attr_label = []
            age_label = int(float(line[11]))
            gender_label = gender_mapping[line[12]]
            
            attr_label.append(gender_label)
            attr_label.append(age_label)
            
            self.label_list.append(label)
            self.attr_list.append(attr_label)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    img_data = np.array(imageData)
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
    img_data = np.transpose(img_data)
    img_data = img_data.astype(np.float32)  
    return img_data, self.label_list[index], self.attr_list[index]

  def __len__(self):

    return len(self.img_list)