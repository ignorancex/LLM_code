"""
@author: hdsullivan
"""

"""
Various functions used for downsampling and upsampling images
"""

import numpy as np
import cv2
from skimage.measure import block_reduce
from skimage.transform import rescale
from astropy.nddata import block_replicate

def simulated_data_decimation(img, L):
    """
    Decimates the input image by a factor of L using rescaling with anit-aliasing.

    Args:
        img (np.ndarray):  M x N x C array containing image to be downsampled
        L (int): Downsampling factor

    Returns: 
        np.ndarray: M // L x N // L x C array containing the downsampled image
    """

    img_dec = rescale(img, 1 / L, anti_aliasing=True)

    return img_dec

def bicubic_upsample(img, scale_factor):
    """
    Upsamples the image using bicubic interpolation.

    Args:
        imagearr (np.ndarray): M x N x C array containing image to be upsampled
        L (int): Upsampling factor

    Returns: 
        np.ndarray: ML x NL X C array containing the upsampled image
    """
    height, width = img.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    upsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upsampled_img

def block_replication(imagearr, L):
    """
    Upsamples the image by pixel replication over L x L grids.

    Args:
        imagearr (np.ndarray): M x N x C array containing image to be upsampled
        L (int): Upsampling factor

    Returns: 
        np.ndarray: ML x NL X C array containing the upsampled image
    """
    if L == 1:
        return imagearr
    if imagearr.ndim == 3:
        [M, N, C] = imagearr.shape
        outarr = block_replicate(imagearr, (L, L, C), conserve_sum = False)
    elif imagearr.ndim == 2:
        [M, N] = imagearr.shape
        outarr = block_replicate(imagearr, (L, L), conserve_sum = False)
        C = 1
    else:
        print("Image array should have either 2 or 3 dimensions.")

    return outarr

def block_averaging(imagearr, L):
    """
    Decimates the input image by block averaging over L x L grids.

    Args:
        imagearr (np.ndarray): M x N x C array containing image to be downsampled
        L (int): Downsampling factor

    Returns: 
        np.ndarray: M // L x N // L x C array containing the downsampled image

    """
    if L == 1:
        return imagearr
    
    if imagearr.ndim == 3:
        [M, N, C] = imagearr.shape
        block_size = (L, L, C)
    elif imagearr.ndim == 2:
        [M, N] = imagearr.shape
        C = 1
        block_size = (L, L)

    img_dec = block_reduce(imagearr, block_size=block_size, func=np.mean)

    return img_dec

def A(imagearr, L):
    """
    Applies spatial downsampling operator for Sentinel-2 by block averaging in L x L grids.

    Args:
        img (np.ndarray):  M x N x C array containing image to be downsampled
        L (int): Downsampling factor

    Returns: 
        np.ndarray: M // L x N // L x C array containing the downsampled image

    """
    imagearr_output = block_averaging(imagearr, L)

    return imagearr_output

def A_transpose(imagearr, L):
    """
    Applies the transpose of the spatial decimation operator for Sentinel-2 by block replicating in L x L grids with a scaling factor.
    
    Args:
        img (np.ndarray):  M x N x C array containing image to be downsampled
        L (int): Downsampling factor

    Returns: 
        np.ndarray: M // L x N // L x C array containing the downsampled image
    """
    imagearr_block_rep = 1 / L**2 * block_replication(imagearr, L)

    return imagearr_block_rep
