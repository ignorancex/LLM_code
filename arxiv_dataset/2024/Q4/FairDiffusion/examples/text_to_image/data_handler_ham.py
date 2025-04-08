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
# from PIL import Image
# import numpy as np
# from skimage.transform import resize
# from torchvision.models import *
# import pickle
# import cv2

def generate_diffusion_prompt(label, attr):
    # 映射字典
    idx2gender = {0: 'Male', 1: 'Female'}
    idx2age = {0: 'Age under 60', 1: 'Age equal 60 or older'}
    idx2disease = {0: 'benign', 1: 'malignant'}

    # 生成患者属性列表
    patient_attr_list = []
    patient_attr_list.append(idx2gender[attr[0].item()])  # 性别
    patient_attr_list.append(idx2age[attr[1].item()])    # 年龄
    patient_attr = " ".join(patient_attr_list)

    # 根据 disease_mapping 构造诊断类别
    disease_category = idx2disease[label]

    # 生成描述性提示
    prompt = f"Dermoscopic image of a {patient_attr} patient with the following diagnostic category: {disease_category}"

    return prompt


class HAM10000(Dataset):

  def __init__(self, file_path, split, attribute, resolution = 224):
      
    self.file_path = file_path
    self.resolution = resolution
      
    self.img_list = pickle.load(open(file_path + split +'_images.pkl','rb'))

    split = 'train'
    list_file = "new_" + split + ".csv"
    
    gender_mapping = {'M':0, 
                'F':1}

    age_mapping = {'<60':0, 
                '>=6':1}

    disease_mapping = {'benign':0, 
                'malignant':1}
    # img_list = []
    self.label_list = []
    self.attr_list = []
    self.path_list = []
    
    with open(file_path+list_file, "r") as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        row_head = next(csvReader)
        for line in csvReader:
            
            label = int(float(line[13]))
            
            attr_label = []
            age_label = int(float(line[11]))
            gender_label = gender_mapping[line[12]]
            attr_label.append(torch.tensor(age_label))
            attr_label.append(torch.tensor(gender_label))
            
            self.label_list.append(label)
            self.attr_list.append(attr_label)

            path = line[9]
            self.path_list.append(path)

  def __getitem__(self, index):
    print("data_handler_ham_cc", index)
    img_data = self.img_list[index]
    label = self.label_list[index]
    attr_label = self.attr_list[index]
      
    if img_data.shape[0] != self.resolution:
        img_data = resize(img_data, (self.resolution, self.resolution))
        img_data = img_data * 255
    # print(img_data.shape)
    # img_data = np.transpose(img_data)
    # img_data = img_data.astype(np.float32) 

    path = self.path_list[index]
    # print(path, img_data.shape)

    # print(os.path.join(self.file_path, path))
    cv2.imwrite(os.path.join(self.file_path, path), img_data)
      
    diffusion_image = os.path.join(self.file_path, path)

    diffusion_text = generate_diffusion_prompt(label, attr_label)
      
    return {'image': diffusion_image, 'text': diffusion_text, 'attr': attr_label, 'labels': label}
      
    # return img_data, self.label_list[index], self.attr_list[index]

  def __len__(self):
    return 10
    # return len(self.img_list)
