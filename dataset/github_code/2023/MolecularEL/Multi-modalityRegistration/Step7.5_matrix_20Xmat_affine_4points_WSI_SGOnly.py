import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import glob
import openslide
import re

import matplotlib._png as png
from matplotlib.cbook import get_sample_data
from skimage.transform import warp
import nibabel as nib

import scipy.ndimage as ndi


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def affinebypoint(img, label, start_x, start_y, size_x, size_y, affine_matrix):
    src_point = np.array([[[start_x, start_y]], [[start_x, start_y+size_y]], [[start_x+size_x, start_y+size_y]], [[start_x+size_x, start_y]]],dtype="float32")
    dst_point = cv2.perspectiveTransform(src_point, affine_matrix)

    #src_patch = img[start_y:start_y + size_y, start_x:start_x + size_x, :]

    # src_patch = np.array(img.read_region((start_y * 2, start_x * 2), 0, (size_y * 2, size_x * 2)))
    src_patch = np.array(img.read_region((start_x * 2, start_y * 2), 0, (size_x * 2, size_y * 2)))
    src_patch = ndi.zoom(src_patch, (0.5, 0.5, 1), order=1)


    x,y,w,h  = cv2.boundingRect(dst_point)

    if x < 0 or y < 0 or y + h > label.level_dimensions[0][1] / 2 or x + w > label.level_dimensions[0][0] / 2 or src_patch.mean() >= 220:
        return src_patch, src_patch, 0
    else:
        # dst_patch = np.array(label.read_region((y * 2, x * 2), 0, (h * 2, w * 2)))
        dst_patch = np.array(label.read_region((x * 2, y * 2), 0, (w * 2, h * 2)))
        dst_patch = ndi.zoom(dst_patch, (0.5, 0.5, 1), order=1)

    if dst_patch.mean() < 20:
        return src_patch, src_patch, 0

    warp_point = np.array([[[0, 0]], [[0, size_y]], [[size_x, size_y]], [[size_x, 0]]],dtype="float32")
    dst_point[:,:,0] = dst_point[:,:,0] - x
    dst_point[:,:,1] = dst_point[:,:,1] - y


    M = cv2.getPerspectiveTransform(dst_point, warp_point)
    dst_patch = cv2.warpAffine(dst_patch, M[:2,:], (size_x, size_y))

    return src_patch, dst_patch, 1

if __name__ == "__main__":

    case_folder = '/Data/MolecularEL/PAS+Endo_png'
    pas_folder = '/Data/MolecularEL/IHC for CD31+PAS/PAS'
    ihc_folder = '/Data/MolecularEL/IHC for CD31+PAS/CD31_svs'

    cases = glob.glob(os.path.join(case_folder, '*'))
    now_middle_idx = 1

    down_x = 0.25
    down_y = 0.25

    upsample_x = 1. / down_x
    upsample_y = 1. / down_y

    for ci in range(0,len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)

        #image_input_dir = os.path.join(case, '5X')
        ANTs_root_dir = os.path.join(case, 'ANTs_affine')
        SG_root_dir = os.path.join(case, 'sg_affine', 'IHC-to-PAS')
        output_folder = os.path.join(case, 'patches_20X')
        #output_folder = os.path.join(case, 'final_image_5X_patches_WSI_overlap_bright_red')

        image_root = glob.glob(os.path.join(ihc_folder, '*%s*' % (now_case)))[0]

        IHC_image = openslide.open_slide(image_root)

        middle_image_root = glob.glob(os.path.join(pas_folder, '*%s*' % (now_case)))[0]

        PAS_image = openslide.open_slide(middle_image_root)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        affine_root = os.path.join(SG_root_dir, 'sg_affine_init.npy')

        # A = nib.load(
	        #'/Data/MolecularEL/PAS+Endo_png/77836/ANTs_affine/IHC-to-PAS/step2_run_ants_reg/output1Warp.nii.gz').get_fdata()

        M_new = np.zeros((3, 3))
        M_new[2, 2] = 1.
        M_new[:2, :3] = np.load(affine_root)

        M_down = np.array([[down_x, 0.], [0., down_y]])
        M_up = np.array([[upsample_x, 0.], [0., upsample_y]])

        M_new[:2, :2] = M_down.dot(M_new[:2, :2]).dot(M_up)

        M_new[1, 2] = M_new[1, 2] * upsample_y
        M_new[0, 2] = M_new[0, 2] * upsample_x

        affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
        M_new[:2, :] = affine_matrix_inv[:2, :]

        'start crop the image'

        patch_size = 2048
        stride_size = 2048

        stride_x = int((PAS_image.level_dimensions[0][0] / 2) / stride_size) + 1
        stride_y = int((PAS_image.level_dimensions[0][1] / 2)  / stride_size) + 1

        for xi in range(stride_x):
            for yi in range(stride_y):
                x_ind = int(xi * stride_size)
                y_ind = int(yi * stride_size)

                print(x_ind, y_ind)

                src_patch, img1_affine, yes = affinebypoint(PAS_image, IHC_image, x_ind, y_ind, patch_size, patch_size, M_new)

                if yes == 1:
                    new_root_src = os.path.join(output_folder, '%d_%d_img.png' % (x_ind, y_ind))
                    new_root_dst = os.path.join(output_folder, '%d_%d_lbl.png' % (x_ind, y_ind))
                    plt.imsave(new_root_src, src_patch)
                    plt.imsave(new_root_dst, img1_affine)
