import os
import glob
from pathlib import Path

import cv2
import numpy as np

import torch
from monai.data import MetaTensor
from monai.metrics import DiceMetric

from tqdm import tqdm

def get_imgs_from_dir(img_dir: str):
    _IMG_EXTS_ = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']
    
    all_image_list = []
    
    for img_ext in _IMG_EXTS_:
        all_image_list += glob.glob(os.path.join(img_dir, f'**/*.{img_ext}'), recursive=True)
        
    return all_image_list

def calculate_dice(y_pred: np.ndarray, y_gt: np.ndarray):
    # pre-process data for DICE calculation
    y_gt = torch.from_numpy(y_gt/y_gt.max())
    y_gt = MetaTensor(y_gt)
    y_gt = [y_gt.unsqueeze(0)]

    y_pred = torch.from_numpy(y_pred)
    y_pred = MetaTensor(y_pred)
    y_pred = [y_pred.unsqueeze(0)]
    
    dice_metric = DiceMetric(include_background=False, get_not_nans=False)
    dice_metric(y_pred=y_pred, y=y_gt)

    return dice_metric.aggregate().item()

def get_image_info(img_name: str, crop_size=1024):
    """Get the WSI ID and coords from image filename
    E.g., 'normal_F1_14_14336_0_img' 
    --> wsi_id: normal_F1
    --> [x_min, y_min] = [14336, 0]
    Args:
        img_name: image filename without extension (e.g., normal_F1_14_14336_0_img)

    Returns: 
        wsi_id
        [x_min, y_min, x_max, y_max]
    """
    elements = img_name.split('_')

    coords = elements[-3:]
    wsi_id = elements[:-4]
    wsi_id = '_'.join(wsi_id)
    
    x_min, y_min = coords[:2]
    
    x_min, y_min = int(x_min), int(y_min)
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    
    return wsi_id, [x_min, y_min, x_max, y_max]

def get_wsi_data(img_paths: list, crop_size: int):
    """Get wsi data from image paths inside a directory
    """
    all_coords = []
    all_wsi_ids = []

    for img_path in tqdm(img_paths):
        # get filename without ext
        img_filename = Path(img_path).stem
        # get wsi_id, coord
        wsi_id, coord = get_image_info(img_filename, crop_size)
        
        all_coords.append(coord)
        all_wsi_ids.append(wsi_id)

    return all_wsi_ids, all_coords

def check_imgs_from_wsi(img_paths: list):
    """Check if all images are from the same WSI
    """
    all_parents = []
    for img_path in img_paths:
        if Path(img_path).parent.name in ['img', 'mask']:
            parent_name = Path(img_path).parent.parent.name
        else:
            parent_name = Path(img_path).parent.name

        all_parents.append(parent_name)
    
    if len(set(all_parents)) == 1:
        return True
    
    return False

def rescale_wsi(wsi_data, smallest_edge=3000):
    H, W, _ = wsi_data.shape
    # resize WSI
    if H > W:
        new_H = int(smallest_edge*H/W)
        new_W = smallest_edge
    else:
        new_W = int(smallest_edge*W/H)
        new_H = smallest_edge

    wsi_data_resized = cv2.resize(wsi_data, dsize=(new_W, new_H))
    return wsi_data_resized

def wsi_thresholding(img, gauss_kernel=51, dilate_kernel=7, iterations=5):
    # convert to grayscale complement image
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    
    img_c = cv2.GaussianBlur(img_c, (gauss_kernel, gauss_kernel), 0)
    _, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    thres_img = cv2.dilate(thres_img, kernel, iterations=iterations)
    
    return thres_img