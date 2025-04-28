
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel 
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.utils import shuffle
import glob
import torchio as tio
from einops import rearrange
import scipy
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu


def adaptive_binarize(arr):
    # Find the threshold using Otsu's method
    thresh = threshold_otsu(arr)
    
    # Binarize the array using the threshold
    binary = np.zeros_like(arr)
    binary[arr >= thresh] = 1
    
    return binary

def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume

def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 20:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume

# Load the image using torchio
image_path = '/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/model_outputs/AE_3D_cc_65_ls_512/1/residual/sub-2f67216f_polyps-r.nii.gz'
image = tio.ScalarImage(image_path)

# Define rescaling transform
rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)) 
# Define the RandomNoise augmentation


rescaled_image = rescale_transform(image)
rescaled_image   = rescaled_image.data.unsqueeze(0)
rescaled_image = torch.squeeze(rescaled_image).numpy()


x_mf_5  = apply_3d_median_filter(rescaled_image)
x_mf_3  = apply_3d_median_filter(rescaled_image,kernelsize=3)
x_mf_7  = apply_3d_median_filter(rescaled_image,kernelsize=7)

x_fc_5  = filter_3d_connected_components(x_mf_5)
x_fc_3  = filter_3d_connected_components(x_mf_3)
x_fc_7  = filter_3d_connected_components(x_mf_7)

x_otsu_5  = adaptive_binarize(x_mf_5)
x_otsu_3  = adaptive_binarize(x_mf_3)
x_otsu_7  = adaptive_binarize(x_mf_7)


x_mf_5 = torch.squeeze(torch.tensor(x_mf_5))
x_mf_5 = x_mf_5.unsqueeze(0)

x_mf_3 = torch.squeeze(torch.tensor(x_mf_3))
x_mf_3 = x_mf_3.unsqueeze(0)

x_mf_7 = torch.squeeze(torch.tensor(x_mf_7))
x_mf_7 = x_mf_7.unsqueeze(0)

x_fc_5 = torch.squeeze(torch.tensor(x_fc_5))
x_fc_5 = x_fc_5.unsqueeze(0)

x_fc_3 = torch.squeeze(torch.tensor(x_fc_3))
x_fc_3 = x_fc_3.unsqueeze(0)

x_fc_7 = torch.squeeze(torch.tensor(x_fc_7))
x_fc_7 = x_fc_7.unsqueeze(0)


x_otsu_5 = torch.squeeze(torch.tensor(x_otsu_5))
x_otsu_5 = x_otsu_5.unsqueeze(0)

x_otsu_3 = torch.squeeze(torch.tensor(x_otsu_3))
x_otsu_3 = x_otsu_3.unsqueeze(0)

x_otsu_7 = torch.squeeze(torch.tensor(x_otsu_7))
x_otsu_7 = x_otsu_7.unsqueeze(0)

rescaled_image   = tio.ScalarImage(tensor=x_mf_5)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_mf_5.nii.gz'
rescaled_image.save(path)


rescaled_image   = tio.ScalarImage(tensor=x_mf_3)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_mf_3.nii.gz'
rescaled_image.save(path)


rescaled_image   = tio.ScalarImage(tensor=x_mf_7)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_mf_7.nii.gz'
rescaled_image.save(path)



rescaled_image   = tio.ScalarImage(tensor=x_fc_5)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_fc_5.nii.gz'
rescaled_image.save(path)


rescaled_image   = tio.ScalarImage(tensor=x_fc_3)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_fc_3.nii.gz'
rescaled_image.save(path)


rescaled_image   = tio.ScalarImage(tensor=x_fc_7)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_fc_7.nii.gz'
rescaled_image.save(path)



rescaled_image   = tio.ScalarImage(tensor=x_otsu_5)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_otsu_5.nii.gz'
rescaled_image.save(path)


rescaled_image   = tio.ScalarImage(tensor=x_otsu_3)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_otsu_3.nii.gz'
rescaled_image.save(path)


rescaled_image   = tio.ScalarImage(tensor=x_otsu_7)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_otsu_7.nii.gz'
rescaled_image.save(path)


