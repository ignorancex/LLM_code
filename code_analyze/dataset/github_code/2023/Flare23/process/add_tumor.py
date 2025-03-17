import os
import random

from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read(img):
    img = sitk.ReadImage(img)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose(1, 2, 0)
    return img


def save_nii(img, path):
    img = img.transpose(2, 0, 1)
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out, path)


def run():
    old_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/labelsTr2200'
    gt_path = './nnUNet_preprocessed/Task022_FLARE22/gt_segmentations'

    save_path = './nnUNet_preprocessed/Task022_FLARE22/gt_segmentations_new'
    check_dir(save_path)

    ls = os.listdir(old_path)

    gt_ls = os.listdir(gt_path)
    for i in tqdm(ls):
        if i in gt_ls:
            old = read(os.path.join(old_path, i))
            gt = read(os.path.join(gt_path, i))

            mask = old > 0
            mask = mask.astype(np.uint8)
            new = gt * (1-mask) + mask * old

            print(np.unique(old), np.unique(gt), np.unique(new))
            save_nii(new, save_path + '/' + i)

        else:
            import shutil
            shutil.copy(os.path.join(gt_path, i), save_path + '/' + i)


if __name__=='__main__':
    run()

