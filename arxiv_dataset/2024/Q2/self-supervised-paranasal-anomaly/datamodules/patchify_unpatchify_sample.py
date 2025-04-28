
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

def patchify_3D_spatiotemporal(imgs):

        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, D, H, W = imgs.shape
        p = 5
        assert H == W and H % p == 0 and D % p == 0
        h = w = d = H // p
        
        x = rearrange(imgs, 'b c (h l) (w m) (d n) -> b (h w d) (l m n c)', l=p,m=p,n=p)  
        #x = imgs.reshape(shape=(N, 1, d, p, h, p, w, p))
        #x = torch.einsum("ncduhpwq->ndhwupqc", x)
       
        #x = x.reshape(shape=(N, d * h * w, p**3 * 1))
        patch_info = (1, D, H, W, p, p, d, h, w)
        return x, patch_info
    

def unpatchify_3D_spatiotemporal( x, patch_info):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    N, T, H, W, p, u, t, h, w = patch_info
   
    x = rearrange(x, 'b (h w d) (l m n c) -> b c (h l) (w m) (d n) ', h=h,w=w,d=h,l=p,m=p,n=p,c=1)  
   
    #x = x.reshape(shape=(N,  1, H, W, T))
    
    #x = torch.einsum("nthwupqc->ncthuqpw", x)
    #imgs = x.reshape(shape=(N, 1, T, H, W))
    #print(x.shape)
    return x


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    print("ids",ids_keep.shape)


    mask = torch.zeros(N, L, D)
    mask[torch.arange(N).unsqueeze(1), ids_keep.unsqueeze(-1)] = 1 

    # generate the binary mask: 1 is keep, 0 is remove
    mask_remove = torch.zeros([N, L], device=x.device)
    mask_remove[:, :len_keep] = 1
    # unshuffle to get the binary mask
    mask_remove = torch.gather(mask_remove, dim=1, index=ids_restore)
    mask_remove = mask_remove.unsqueeze(2)

    x_masked_org_size = x * mask_remove
    
    return x_masked_org_size, mask, ids_restore


# Load the image using torchio
image_path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/crop_size_65_std_factor_1/sub-0a01b9d9/normal-r/0_smax_r.nii.gz'
image = tio.ScalarImage(image_path)

# Define rescaling transform
rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)) 
# Define the RandomNoise augmentation


rescaled_image = rescale_transform(image)
rescaled_image   = rescaled_image.data.unsqueeze(0)

x, patch_info  = patchify_3D_spatiotemporal(rescaled_image)

x, mask, ids_restore = random_masking(x, 0.75)

x  = unpatchify_3D_spatiotemporal(x,patch_info)



x = torch.squeeze(x)
x = x.unsqueeze(0)

rescaled_image   = tio.ScalarImage(tensor=x)
# Save the new image
path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-0a01b9d9_normal-r_0_smax_r_cc_65_unpatchified_masked_1.nii.gz'
rescaled_image.save(path)


