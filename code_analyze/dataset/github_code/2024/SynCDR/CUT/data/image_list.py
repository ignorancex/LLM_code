import torch
import numpy as np
import random
import torch.utils.data as data
import os
import os.path
import PIL
from PIL import Image
import random
from data.base_dataset import BaseDataset, get_transform


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def m_hot(labels, num_class):
    vector = np.zeros(num_class)
    for label in labels:
        vector[label] = 1.0
    return vector

def make_dataset_from_list(image_list):
    image_paths = []
    label_list = []
    with open(image_list) as f:
        for r in f.read().splitlines():
            curr_path, curr_lbl = r.split(' ')
            image_paths.append(curr_path)
            label_list.append(int(curr_lbl))
    
    return image_paths, label_list