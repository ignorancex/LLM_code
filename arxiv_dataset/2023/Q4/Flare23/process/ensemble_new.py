import os
import random

from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from threading import Thread
from multiprocessing import Process


def read(img):
    img = sitk.ReadImage(img)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose(1, 2, 0)
    return img.astype(np.uint8)


def save_nii(img, path):
    img = img.transpose(2, 0, 1)
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out, path)


def ensemble():
    pre14_path = './eval_0.4297'
    pre13_path = './eval_organ_0.8424'

    ls = os.listdir(pre13_path)
    print(ls)

    save_path = './ens'
    check_dir(save_path)

    for i in tqdm(ls):
        organ = os.path.join(pre13_path, i)
        tumor = os.path.join(pre14_path, i)

        organ = read(organ)
        tumor = read(tumor)

        ens = organ.copy()
        ens[tumor == 14] = 14
        save_nii(ens, save_path + '/' + i)


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


def ensemble_new():
    save_path = '/jhcnas1/wulinshan/nnUNet/nnUNet_preprocessed/Dataset101_Flare23/ens'
    check_dir(save_path)
    gt_path = '/jhcnas1/jiaxin/data/Flare23/nnUNet_raw/Dataset101_Flare23/labelsTr'
    gt_ls = os.listdir(gt_path)

    import multiprocessing
    with multiprocessing.Pool(80) as pool:
        pool.map(ensemble_eachV2, gt_ls, 1)


def ensemble_each(name):
    save_path = '/jhcnas1/wulinshan/nnUNet/nnUNet_preprocessed/Dataset101_Flare23/ens'
    #check_dir(save_path)

    pre1_path = '/jhcnas1/wulinshan/flare23/aladdin5-pseudo-labels-FLARE23/aladdin5-pseudo-labels-FLARE23/imagesTr2200-pseudo-labels'
    pre2_path = '/jhcnas1/wulinshan/flare23/pred_tr_0.84'
    pre3_path ='/jhcnas1/wulinshan/flare23/blackbean-pseudo-labels-FLARE23'
    gt_path = '/jhcnas1/jiaxin/data/Flare23/nnUNet_raw/Dataset101_Flare23/labelsTr'

    gt = read(os.path.join(gt_path, name))
    pre1 = read(os.path.join(pre1_path, name))
    pre2 = read(os.path.join(pre2_path, name))
    pre3 = read(os.path.join(pre2_path, name))

    abs1 = pre1 - pre2
    abs2 = pre1 - pre3
    pred = pre1.copy()
    pred[abs1 != 0] = 0
    pred[abs2 != 0] = 0
    
    ens = gt.copy()
    cls_ls = list(np.unique(gt))
    for cls in range(14):
        if cls not in cls_ls:
            ens[pred == cls] = cls
    
    save_nii(ens, save_path + '/' + name)
    print('save:', name)


def ensemble_eachV2(name):
    save_path = '/jhcnas1/wulinshan/nnUNet/nnUNet_preprocessed/Dataset101_Flare23/ens'

    pre1_path = '/jhcnas1/wulinshan/flare23/aladdin5-pseudo-labels-FLARE23/aladdin5-pseudo-labels-FLARE23/imagesTr2200-pseudo-labels'
    pre2_path ='/jhcnas1/wulinshan/flare23/blackbean-pseudo-labels-FLARE23'
    gt_path = '/jhcnas1/jiaxin/data/Flare23/nnUNet_raw/Dataset101_Flare23/labelsTr'

    gt = read(os.path.join(gt_path, name))
    pre1 = read(os.path.join(pre1_path, name))
    pre2 = read(os.path.join(pre2_path, name))

    pred = pre1.copy()
    pred[pre1 != pre2] = 0

    mask = gt > 0
    mask = mask.astype(np.uint8)
    new = pred * (1 - mask) + mask * gt
    new = new.astype(np.uint8)
    
    save_nii(new, save_path + '/' + name)
    print('save:', name)

def view():
    cmap = color_map()
    pre1_path = '/jhcnas1/wulinshan/flare23/aladdin5-pseudo-labels-FLARE23/aladdin5-pseudo-labels-FLARE23/imagesTr2200-pseudo-labels'
    pre2_path = '/jhcnas1/wulinshan/flare23/pred_tr_0.84'
    pre3_path ='/jhcnas1/wulinshan/flare23/blackbean-pseudo-labels-FLARE23'
    gt_path = '/jhcnas1/jiaxin/data/Flare23/nnUNet_raw/Dataset101_Flare23/labelsTr'
    ens_path = '/jhcnas1/wulinshan/nnUNet/nnUNet_preprocessed/Dataset101_Flare23/gt_segmentations'

    ls = os.listdir(gt_path)
    random.shuffle(ls)

    num = 0

    for i in tqdm(ls):
        pre1 = read(os.path.join(pre1_path, i))
        pre2 = read(os.path.join(pre2_path, i))
        pre3 = read(os.path.join(pre3_path, i))
        gt = read(os.path.join(gt_path, i))
        ens = read(os.path.join(ens_path, i))
        h,w,c = gt.shape

        if len(list(np.unique(pre1)))>4:
            for j in range(c):
                pre1_ = pre1[:, :, j]
                if len(list(np.unique(pre1_)))>4:
                    print(np.unique(pre1_))
                    pre2_ = pre2[:, :, j]
                    pre3_ = pre3[:, :, j]
                    gt_ = gt[:, :, j]
                    ens_ = ens[:, :, j]

                    pre1_ = Image.fromarray(pre1_.astype(np.uint8), mode='P')
                    pre1_.putpalette(cmap)

                    pre2_ = Image.fromarray(pre2_.astype(np.uint8), mode='P')
                    pre2_.putpalette(cmap)

                    pre3_ = Image.fromarray(pre3_.astype(np.uint8), mode='P')
                    pre3_.putpalette(cmap)

                    gt_ = Image.fromarray(gt_.astype(np.uint8), mode='P')
                    gt_.putpalette(cmap)

                    ens_ = Image.fromarray(ens_.astype(np.uint8), mode='P')
                    ens_.putpalette(cmap)

                    pre1_.save('1.png'), pre2_.save('2.png'), pre3_.save('3.png'), gt_.save('gt.png'), ens_.save('ens.png')

                    # fig, axs = plt.subplots(1, 4, figsize=(16, 5))
                    # axs[0].imshow(pre1_, cmap='gray')
                    # axs[0].axis("off")
                    # axs[1].imshow(pre1_)
                    # axs[1].axis("off")

                    # axs[2].imshow(pre3_)
                    # axs[2].axis("off")

                    # axs[3].imshow(gt_)
                    # axs[3].axis("off")

                    

                    # plt.tight_layout()
                    # plt.show()
                    # plt.close()


if __name__ == "__main__":

    ensemble_new()
    #view()



