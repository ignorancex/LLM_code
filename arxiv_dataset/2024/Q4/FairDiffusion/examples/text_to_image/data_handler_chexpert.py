import torch
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
from glob import glob
import pandas as pd
import os
import cv2
import pickle

# import torch
# from torch.utils.data import DataLoader, Dataset
import csv      
# import os
from PIL import Image
# import numpy as np
# from skimage.transform import resize
# from torchvision.models import *
# import pickle
# import cv2

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

def generate_diffusion_prompt(label, attr):
    # 映射字典
    idx2gender = {0: 'Male', 1: 'Female'}
    idx2race = {0: 'Asian', 1: 'Black', 2: 'White'}
    idx2disease = {0: 'non Pleural Effusion', 1: 'Pleural Effusion'}

    patient_attr_list = []
    patient_attr_list.append(idx2gender[attr[0].item()]) 
    patient_attr_list.append(idx2race[attr[1].item()])    
    patient_attr = " ".join(patient_attr_list)  
    
    disease_category = idx2disease[label]

    prompt = f"Chest Radiography image of a {patient_attr} patient diagnosed with {disease_category}"

    return prompt


class CheXpertDataset(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224, disease = 'Pleural Effusion'):
      
    # file_path = "../Data/Chexpert/"
    if split == 'train':
        list_file = "chexpert_sample_" + split + "_6000.csv"
    elif split == 'val':
        list_file = "chexpert_sample_" + split + "_1000.csv"
    elif split == 'test':
        list_file = "chexpert_sample_" + split + "_3000.csv"
    
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
            img_filename = os.path.join(file_path, "Chexpert_sample_512", line[24])
            label = int(float(line[9+label_idx]))
        
            # if attribute == "gender":
            #     attr_idx = 3
            #     attr_label = gender_mapping[line[attr_idx]]
    
            # if attribute == "race":
            #     attr_idx = 5
            #     attr_label = race_mapping[line[attr_idx]]

            gender_attr_label = gender_mapping[line[3]]
            race_attr_label = race_mapping[line[5]]
            attr_label_list.append(torch.tensor(gender_attr_label))
            attr_label_list.append(torch.tensor(race_attr_label))
            
            self.img_list.append(img_filename)
            self.label_list.append(label)
            self.attr_list.append(attr_label_list)

  def __getitem__(self, index):
    print(index)
    imagePath = self.img_list[index]
    label = self.label_list[index]
    attr_label = self.attr_list[index]
      
    imageData = Image.open(imagePath).convert('RGB')
    img_data = np.array(imageData)
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
        img_data = img_data * 255
    # img_data = np.transpose(img_data)
    # print(img_data.dtype)
    # img_data = img_data.astype(np.uint8) 

    img_data = Image.fromarray((img_data).astype(np.uint8))
      
    path = imagePath.replace('Chexpert_sample_512', 'Chexpert_sample_200')
    img_data.save(path)
    diffusion_image = path

    
    label = self.label_list[index]  
    if label == -1:
        label = 0


    diffusion_text = generate_diffusion_prompt(label, attr_label)
      
    return {'image': diffusion_image, 'text': diffusion_text, 'attr': attr_label, 'labels': label}
    # return img_data, label, self.attr_list[index]

  def __len__(self):
    # return 10
    return len(self.img_list)
