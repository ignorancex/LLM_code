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


def view_train():
    cmap = color_map()
    train_img_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/imagesTr2200'
    train_lab_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/labelsTr2200'
    train_pre_path = './pred_tr'

    ls = os.listdir(train_lab_path)
    print(ls)
    random.shuffle(ls)

    num = 0

    for i in tqdm(ls):

        train_img = os.path.join(train_img_path, i[:-7]+'_0000.nii.gz')
        train_lab = os.path.join(train_lab_path, i)
        pre = os.path.join(train_pre_path, i)

        pre = read(pre)
        h, w, c = pre.shape

        if len(list(np.unique(pre))) > 10:
            for j in range(c):
                pre_ = pre[:, :, j]

                if len(list(np.unique(pre_))) > 10:
                    num += 1
                    # print(np.unique(pre_))

                    img = read(train_img)[:, :, j].astype(np.uint8)
                    img = Image.fromarray(img)

                    lab = read(train_lab)[:, :, j].astype(np.uint8)
                    print(np.unique(lab), np.unique(pre_))
                    lab = Image.fromarray(lab, mode='P')
                    lab.putpalette(cmap)

                    pre_ = pre_.astype(np.uint8)
                    pre_ = Image.fromarray(pre_, mode='P')
                    pre_.putpalette(cmap)

                    lab_ = blend(img, lab)
                    pre_ = blend(img, pre_)

                    save(img, lab_, pre_, i[:12]+'_'+str(num))

                    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
                    axs[0].imshow(img, cmap='gray')
                    axs[0].axis("off")
                    axs[1].imshow(lab_)
                    axs[1].axis("off")
                    axs[2].imshow(pre_)
                    axs[2].axis("off")

                    plt.tight_layout()
                    plt.show()
                    plt.close()


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
    view_train()
    # import pickle
    # with open('./nnUNet_preprocessed/Task022_FLARE22/nnUNetPlansv2.1_plans_3D.pkl', 'rb') as data:
    #     a = pickle.load(data)
    # print(a)
    # print(a.keys())
    # print(a['all_classes'], a['num_classes'])
    # print(a['plans_per_stage'][0]['batch_size'])
    #
    # b = a.copy()
    #
    # b['plans_per_stage'][0]['batch_size'] = 4
    # b['plans_per_stage'][1]['batch_size'] = 4
    # print(b)
    # pickle.dump(b, file=open('./nnUNetPlansv2.1_plans_3D.pkl', 'wb+'))


