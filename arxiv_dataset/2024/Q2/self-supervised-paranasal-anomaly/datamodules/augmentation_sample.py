
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


# Load the image using torchio
image_path = '/home/debayan/Desktop/MRI_HCHS/JAMA-unlabelled-1000-corrected/crop_size_65_std_factor_1/sub-0aab6c55/smax_l/residuals/AE_3D_cc_65_ls_512/MF_5/6_smax_l.nii.gz'

image = tio.ScalarImage(image_path)

# Define rescaling transform
rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)) 
# Define the Gaussian Blur transform
blur_transform = tio.RandomBlur(std=(0, 8), p=1)



#rescaled_image = rescale_transform(image)
blurred_image = blur_transform(image.data)
print(blurred_image.shape)

# Save the new image
image = tio.ScalarImage(tensor=blurred_image)

noisy_image_path = '/home/debayan/Desktop/MRI_HCHS/JAMA-unlabelled-1000-corrected/crop_size_65_std_factor_1/sub-0aab6c55/smax_l/residuals/AE_3D_cc_65_ls_512/MF_5/gaussian_6_smax_l.nii.gz'
image.save(noisy_image_path)