import os
import random

from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read(img):
    img = sitk.ReadImage(img)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose(1, 2, 0)
    return img


def save_nii(img, path):
    img = img.transpose(2, 0, 1)
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out, path)


def main():
    BTCV_path = './a'

    ls = os.listdir(BTCV_path)
    print(ls)

    save_path = './eval_BTCV'
    check_dir(save_path)

    for i in tqdm(ls):
        BTCV = os.path.join(BTCV_path, i)
        BTCV = read(BTCV)

        res = BTCV.copy()
        res[BTCV == 6] = 1
        res[BTCV == 2] = 2
        res[BTCV == 1] = 3
        res[BTCV == 11] = 4
        res[BTCV == 8] = 5
        res[BTCV == 9] = 6
        res[BTCV == 12] = 7
        res[BTCV == 13] = 8
        res[BTCV == 4] = 9
        res[BTCV == 5] = 10
        res[BTCV == 7] = 11

        res[BTCV == 10] = 0

        res[BTCV == 3] = 13

        save_nii(res, save_path + '/' + i)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

        cmap[19] = np.array([0, 0, 0])
        cmap[255] = np.array([0, 0, 0])

    return cmap


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save(img, lab, pre, name):
    save_path = './view_res'
    check_dir(save_path)
    img.save(save_path + '/' + name + '.png')
    lab.save(save_path + '/' + name + '_lab.png')
    pre.save(save_path + '/' + name + '_pre.png')


def blend(img, lab):
    img = img.convert('RGB')
    lab = lab.convert('RGB')
    ble = Image.blend(img, lab, 0.6)
    return ble


if __name__ == "__main__":
    # view_train()
    main()



