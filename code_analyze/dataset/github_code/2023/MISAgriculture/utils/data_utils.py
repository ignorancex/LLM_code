import yaml
import os
# from osgeo import gdal
# import rasterio
import numpy as np
import matplotlib.image as mpimg
import cv2
import tifffile
import torch
from PIL import Image



def normalize_batch_torch(data,ignore_pixel=0):
    if len(data.shape)<4:
        raise ValueError

    img_list = []
    for img in data:

        img = normalize_torch(img,ignore_pixel=ignore_pixel)
        img = torch.unsqueeze(img,dim=0)
        img_list.extend(img)
    img_stack = torch.stack(img_list,dim=0)
    return(img_stack)

def normalize_torch(data,ignore_pixel=0):
    
    if not torch.is_tensor(data):
        raise TypeError
    
    img_clone = data[data != ignore_pixel].clone()
    
    min_value  = torch.min(img_clone)
    max_value = torch.max(img_clone)

    img_norm = ((data - min_value))/(max_value - min_value)
    img_norm[img_norm<0]=0

    return(img_norm)
    #img_norm = (img_norm*255).astype('uint8')




def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
    if min is not None and max is not None:
        norm = (im - min) / (max-min)
    else:
        cv2.normalize(im, dst=norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm[norm<0.0] = 0.0
    norm[norm>1.0] = 1.0
    return norm

def standardization(im):
    for i in range(im.shape[2]):
        im[:,:,i] = (im[:,:,i] - np.mean(im[:,:,i])) / np.std(im[:,:,i])
    return(im)

def load_config(config_file = 'data_config.yaml'):
        if not os.path.isfile(config_file):
            raise NameError('File Does not Exist')
        conf_data = yaml.load(open(config_file), Loader=yaml.FullLoader)

        return(conf_data)

def get_files(folder_path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return(onlyfiles)


def load_file(file):
    file_type = file.split('.')[-1]
    if file_type == 'npy':
        im = np.load(file)
    elif file_type in ['JPG','PNG','jpg','png']:
        im = Image.open(file)
        im = np.array(im,dtype=np.uint8)
    else:
        im = tifffile.imread(file)
    return(im)