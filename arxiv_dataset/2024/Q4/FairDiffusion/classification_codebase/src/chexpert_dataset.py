import torch
from torch.utils.data import DataLoader, Dataset
import csv
import os
from PIL import Image
import numpy as np
from skimage.transform import resize
from torchvision.models import *
import glob

disease_list = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']


class CheXpertDataset(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224, disease = 'Pleural Effusion'):
      
    # file_path = "../Data/Chexpert/"
    if split == 'train':
        list_file = "chexpert_sample_train_6000.csv"
    elif split == 'val':
        list_file = "chexpert_sample_val_1000.csv"
    elif split == 'test':
        list_file = "chexpert_sample_test_3000.csv"
    # disease = 'Pleural Effusion'
    # attribute = 'gender'

    self.resolution = resolution
    
    gender_mapping = {'Male':0, 
                    'Female':1}
    
    race_mapping = {'Asian':0, 
                    'Black':1, 
                    'White':2}
    
    if disease:
        label_idx = disease_list.index(disease)
    
    if attribute == "gender":
        attr_idx = 3
    
    if attribute == "race":
        attr_idx = 5
    
    if attribute == "ethnicity":
        attr_idx = 6
    
    self.img_list = []
    self.label_list = []
    self.attr_list = []
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)
        for line in csvReader:
            attr_label_list = []
            img_filename = os.path.join(file_path, "Chexpert_sample_200", line[24])
            label = int(float(line[9+label_idx]))
        
            # if attribute == "gender":
            #     attr_idx = 3
            #     attr_label = gender_mapping[line[attr_idx]]
    
            # if attribute == "race":
            #     attr_idx = 5
            #     attr_label = race_mapping[line[attr_idx]]

            gender_attr_label = gender_mapping[line[3]]
            race_attr_label = race_mapping[line[5]]
            attr_label_list.append(gender_attr_label)
            attr_label_list.append(race_attr_label)
            
            self.img_list.append(img_filename)
            self.label_list.append(label)
            self.attr_list.append(attr_label_list)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    img_data = np.array(imageData)
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
    img_data = np.transpose(img_data)
    img_data = img_data.astype(np.float32)  

    label = self.label_list[index]  
    if label == -1:
        label = 0
    return img_data, label, self.attr_list[index]

  def __len__(self):

    return len(self.img_list)


class CheXpertDataset_Gen(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224, disease = 'Pleural Effusion'):
      
    # file_path = "../Data/Chexpert/"
    if split == 'train':
        list_file = "chexpert_sample_train_6000.csv"
    elif split == 'val':
        list_file = "chexpert_sample_val_1000.csv"
    elif split == 'test':
        list_file = "chexpert_sample_test_3000.csv"
    # disease = 'Pleural Effusion'
    # attribute = 'gender'

    self.resolution = resolution
    
    gender_mapping = {'Male':0, 
                    'Female':1}
    
    race_mapping = {'Asian':0, 
                    'Black':1, 
                    'White':2}
    
    if disease:
        label_idx = disease_list.index(disease)
    
    if attribute == "gender":
        attr_idx = 3
    
    if attribute == "race":
        attr_idx = 5
    
    if attribute == "ethnicity":
        attr_idx = 6
    
    self.img_list = sorted(glob.glob('/scratch/cw3437/Data/Chexpert/chexpert_gen/*.jpg'))
    self.label_list = []
    self.attr_list = []
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)
        for line in csvReader:
            attr_label_list = []
            label = int(float(line[9+label_idx]))
        
            # if attribute == "gender":
            #     attr_idx = 3
            #     attr_label = gender_mapping[line[attr_idx]]
    
            # if attribute == "race":
            #     attr_idx = 5
            #     attr_label = race_mapping[line[attr_idx]]

            gender_attr_label = gender_mapping[line[3]]
            race_attr_label = race_mapping[line[5]]
            attr_label_list.append(gender_attr_label)
            attr_label_list.append(race_attr_label)
            
            # self.img_list.append(img_filename)
            self.label_list.append(label)
            self.attr_list.append(attr_label_list)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    img_data = np.array(imageData)
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
    img_data = np.transpose(img_data)
    img_data = img_data.astype(np.float32)  

    label = self.label_list[index]  
    if label == -1:
        label = 0
    return img_data, label, self.attr_list[index]

  def __len__(self):

    return len(self.img_list)

class CheXpertDataset_FairGen(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224, disease = 'Pleural Effusion'):
      
    # file_path = "../Data/Chexpert/"
    if split == 'train':
        list_file = "chexpert_sample_train_6000.csv"
    elif split == 'val':
        list_file = "chexpert_sample_val_1000.csv"
    elif split == 'test':
        list_file = "chexpert_sample_test_3000.csv"
    # disease = 'Pleural Effusion'
    # attribute = 'gender'

    self.resolution = resolution
    
    gender_mapping = {'Male':0, 
                    'Female':1}
    
    race_mapping = {'Asian':0, 
                    'Black':1, 
                    'White':2}
    
    if disease:
        label_idx = disease_list.index(disease)
    
    if attribute == "gender":
        attr_idx = 3
    
    if attribute == "race":
        attr_idx = 5
    
    if attribute == "ethnicity":
        attr_idx = 6
    
    self.img_list = sorted(glob.glob('/scratch/cw3437/Data/Chexpert/fairchexpert_gen/*.jpg'))
    self.label_list = []
    self.attr_list = []
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)
        for line in csvReader:
            attr_label_list = []
            label = int(float(line[9+label_idx]))
        
            # if attribute == "gender":
            #     attr_idx = 3
            #     attr_label = gender_mapping[line[attr_idx]]
    
            # if attribute == "race":
            #     attr_idx = 5
            #     attr_label = race_mapping[line[attr_idx]]

            gender_attr_label = gender_mapping[line[3]]
            race_attr_label = race_mapping[line[5]]
            attr_label_list.append(gender_attr_label)
            attr_label_list.append(race_attr_label)
            
            # self.img_list.append(img_filename)
            self.label_list.append(label)
            self.attr_list.append(attr_label_list)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    img_data = np.array(imageData)
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
    img_data = np.transpose(img_data)
    img_data = img_data.astype(np.float32)  

    label = self.label_list[index]  
    if label == -1:
        label = 0
    return img_data, label, self.attr_list[index]

  def __len__(self):

    return len(self.img_list)