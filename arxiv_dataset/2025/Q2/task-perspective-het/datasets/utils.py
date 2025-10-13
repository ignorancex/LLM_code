import PIL
from PIL import Image
import PIL.PngImagePlugin
import cv2
import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    NEAREST = InterpolationMode.NEAREST
except ImportError:
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST

AVAILABLE_TASKS = ['rgb', 'reshading', 'keypoints3d', 'edge_texture', 'normal', 'segment_semantic', 'rgb_for_segmentation', 'class_scene', 'depth_euclidean', 'principal_curvature', 'principal_curvature_old']

# def process_img_path(img_path, channels, dest_task):
#     img = cv2.imread(img_path)
#     if channels == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
#     elif channels == 1:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         if dest_task != 'edge_texture':
#             img = torch.from_numpy(img).float() / np.max(img)
#         else:
#             img = torch.from_numpy(img).float() / 255
        
#     else:
#         raise ValueError("Unsupported image channels")
    
#     # img = read_image(img_path)
#     # img = decode_png(img)
#     # img /= 255.0
    
#     return img



def _convert_image_to_rgb(image):
    image = image.convert("RGB")
    return image

def scale_edge_image(image):
    if np.max(image) == 0:
        return image
    else:
        return image/np.max(image)

def _rescale_depth_img(image):
    # image = np.array(image).astype(np.float32)
    image[image==1.0] = np.max(image[image!=1.0]) * 1.01
    image /= np.max(image)

    return image

# We don't need this!
def scale_image(image):
    return np.array(image)/255.0

def fix_reshading_img(image:PIL.PngImagePlugin.PngImageFile):
    return image.convert("L")

def extend_segmentation(image):
    np_image = np.array(image)
    new_image = np.zeros((np_image.shape[0], np_image.shape[1], 18))
    for i in range(18):
        new_image[:, :, i] = (np_image == i).astype(int)

def _tensorize(image):
    image = torch.tensor(np.array(image))
    return image

def _resize_and_print(task):
    if task == 'segment_semantic':
        def transform_func(image):
            return Resize(224, interpolation=NEAREST, antialias=None)(image.unsqueeze(0)).squeeze(0)
        
        return transform_func
    
    else:
        def transform_func(image):
            return Resize(224, interpolation=BICUBIC, antialias=None)(image.unsqueeze(0)).squeeze(0)
        
        return transform_func

def drop_pc_channel(image):
    return np.array(image)[:,:,:2]


def return_preprocess(task):
    '''
    Sequentially add the transforms to an original empty list of transforms
    '''

    if task not in AVAILABLE_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    
    if task == "class_scene":
        return _tensorize
    
    transforms = []
    # Transforms to fix loaded image
    if task == 'reshading':
        transforms.append(fix_reshading_img)
    elif task in ['rgb', 'normal', 'principal_curvature_old']:
        transforms.append(_convert_image_to_rgb)
    elif task == 'principal_curvature':
        transforms.append(drop_pc_channel)
    
    # Value scaling transforms
    if task in ['edge_texture', 'keypoints3d', 'depth_euclidean']:
        transforms.append(scale_edge_image)
    if task == 'depth_euclidean':
        transforms.append(_rescale_depth_img)

    if task == 'segment_semantic':
        transforms.append(_tensorize)
    else:
        transforms.append(ToTensor())
    transforms.append(_resize_and_print(task))
    transforms.append(CenterCrop(224))

    return Compose(transforms)
