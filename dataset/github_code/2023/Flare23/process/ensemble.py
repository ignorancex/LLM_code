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
    return img


def save_nii(img, path):
    img = img.transpose(2, 0, 1)
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out, path)


def view_train():
    cmap = color_map()
    img_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/imagesTs'
    pre14_path = './eval_0.4297'
    pre13_path = './ens'

    ls = os.listdir(pre13_path)
    print(ls)
    random.shuffle(ls)

    num = 0

    for i in tqdm(ls):
        pre13 = os.path.join(pre13_path, i)
        pre14 = os.path.join(pre14_path, i)

        img = os.path.join(img_path, i[:-7]+'_0000.nii.gz')

        pre14 = read(pre14)
        h, w, c = pre14.shape
        print(pre14.shape)

        if 14 in list(np.unique(pre14)):
            for j in range(c):
                pre14_ = pre14[:, :, j]
                if 14 in list(np.unique(pre14_)):
                    print(np.unique(pre14_))

                    img_ = read(img)[:, :, j].astype(np.uint8)
                    img_ = Image.fromarray(img_)

                    pre13_ = read(pre13)[:, :, j].astype(np.uint8)
                    pre13_ = pre13_.astype(np.uint8)

                    ens = pre13_.copy()
                    ens[pre14_ == 14] = 14

                    pre14_ = Image.fromarray(pre14_, mode='P')
                    pre14_.putpalette(cmap)
                    pre14_ = blend(img_, pre14_)

                    pre13_ = Image.fromarray(pre13_, mode='P')
                    pre13_.putpalette(cmap)
                    pre13_ = blend(img_, pre13_)

                    ens = Image.fromarray(ens, mode='P')
                    ens.putpalette(cmap)
                    ens = blend(img_, ens)

                    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
                    axs[0].imshow(img_, cmap='gray')
                    axs[0].axis("off")
                    axs[1].imshow(pre14_)
                    axs[1].axis("off")

                    axs[2].imshow(pre13_)
                    axs[2].axis("off")

                    axs[3].imshow(ens)
                    axs[3].axis("off")

                    plt.tight_layout()
                    plt.show()
                    plt.close()


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


def ensembleV2():
    pre1_path = './result_0.80'
    pre2_path = './result_0.84'

    ls = os.listdir(pre1_path)
    print(ls)

    save_path = './ens'
    check_dir(save_path)

    for i in tqdm(ls):
        pre1 = os.path.join(pre1_path, i)
        pre2 = os.path.join(pre2_path, i)

        pre1 = read(pre1)
        pre2 = read(pre2)

        ens = pre1.copy()

        ens[pre2 == 5] = 5
        ens[pre2 == 6] = 6
        ens[pre2 == 10] = 10
        ens[pre2 == 11] = 11
        ens[pre2 == 12] = 12
        ens[pre2 == 14] = 14
        save_nii(ens, save_path + '/' + i)


def ensembleV3():
    pre1_path = './pred_tr_0.80'
    pre2_path = './pred_tr_0.84'
    old_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/labelsTr2200'

    ls = os.listdir(pre2_path)
    gt_ls = os.listdir(old_path)
    print(ls)

    save_path = './pred_tr'
    # check_dir(save_path)
    import multiprocessing
    with multiprocessing.Pool(24) as pool:
        pool.map(ensemble_each, ls, 1)

    # for i in range(length, 32):
    #     name = ls[i]
    #     process_list = []
    #
    #     for _ in range(32):  # 开启5个子进程执行fun1函数
    #         p = Process(target=ensemble_each, args=(name,))  # 实例化进程对象
    #         p.start()
    #         process_list.append(p)
    #
    #     for _ in process_list:
    #         p.join()


def ensemble_each(name):
    save_path = './pred_tr'
    pre1_path = './pred_tr_0.80'
    pre2_path = './pred_tr_0.84'
    old_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/labelsTr2200'

    gt_ls = os.listdir(old_path)

    pre1 = os.path.join(pre1_path, name)
    pre2 = os.path.join(pre2_path, name)

    pre1 = read(pre1)
    pre2 = read(pre2)

    ens = pre1.copy()

    ens[pre2 == 5] = 5
    ens[pre2 == 6] = 6
    ens[pre2 == 10] = 10
    ens[pre2 == 11] = 11
    ens[pre2 == 12] = 12

    print('read:', name)
    if name in gt_ls:
        print('use old gt:', name)
        old = read(os.path.join(old_path, name))

        mask = old > 0
        mask = mask.astype(np.uint8)
        new = ens * (1 - mask) + mask * old

        print(np.unique(old), np.unique(new))
        save_nii(new, save_path + '/' + name)
    else:
        save_nii(ens, save_path + '/' + name)

    print('save:', name)
    # import time
    # time.sleep(5)


def check():
    ls = os.listdir('./pred_tr')
    all = os.listdir('./pred_tr_0.80')
    print(len(ls), len(all))

    for i in all:
        # print(i in all)
        if i not in ls:
            print(i)


if __name__ == "__main__":
    # python ensemble.py
    # view_train()

    # ensembleV3()
    check()



