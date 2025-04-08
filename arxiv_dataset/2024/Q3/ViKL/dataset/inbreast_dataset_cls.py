import random
import os
import numpy as np
import pandas as pd
from PIL import Image

import SimpleITK as sitk
import torch

# MAM_MAX_PIXEL, MAM_MIN_PIXEL = 4095, 0

"""
Following the paper <Automated Analysis of Unregistered Multi-View Mammograms With Deep Learning>(
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8032490), we also classify the Bi-Rads to 
three classes.
"""


class InBreastClsDataset:

    def __init__(self, db_root_path, dcm_dir, anno_file, stage="train", transform=None, binary_label=False,
                 pre_downsample=False):
        self.db_root_path = db_root_path
        self.dcm_dir = dcm_dir
        self.anno_file = anno_file
        self.binary_label = binary_label
        self.stage = stage
        self.pre_downsample = pre_downsample
        anno = pd.read_csv(os.path.join(db_root_path, anno_file))
        self.filename = anno["File Name"].to_list()
        self.laterality = anno.Laterality.to_list()
        self.view = anno.View.to_list()
        self.bi_rads = anno["Bi-Rads"].to_list()

        img_names = []
        img_lists_raw = os.listdir(os.path.join(db_root_path, dcm_dir)) #xinwei: this is for pre downsample
        img_lists = []
        for i in img_lists_raw:
            ext = os.path.splitext(i)[1]
            if ext != '.png':
                img_lists.append(i)

        for img in img_lists:
            if int(img.split("_")[0].strip()) in self.filename:
                img_names.append(img)
        self.img_names = img_names

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]

        file_name = img_name.split("_")[0].strip()
        file_name_idx = self.filename.index(int(file_name))
        if self.pre_downsample:
            img = Image.open(os.path.join(self.db_root_path, self.dcm_dir, f"{img_name}.a.png"))
        else:
            img = sitk.ReadImage(os.path.join(self.db_root_path, self.dcm_dir, img_name))
            img = sitk.GetArrayFromImage(img)

            # img = (img - MAM_MIN_PIXEL) / (MAM_MAX_PIXEL - MAM_MIN_PIXEL)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img[0]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        bi_rad = self.bi_rads[file_name_idx]
        if self.binary_label:
            if bi_rad in ["1", "2", "3"]:
                label = 0
            else:
                label = 1
        else:
            if bi_rad == "1":
                label = [1, 0, 0]
            elif bi_rad in ["2", "3"]:
                label = [0, 1, 0]
            elif bi_rad in ["4a", "4b", "4c", "5", "6"]:
                label = [0, 0, 1]
            else:
                label = [0, 0, 0]
        label = torch.tensor(label)

        return {"view1": img, "label": label}


def main():
    path = "/data2/zhai/InBreast/"

    print(len(os.listdir(path)))
    # img = sitk.ReadImage(path + "image/53587663_5fb370d4c1c71974_MG_R_CC_ANON.dcm")
    # img = sitk.GetArrayFromImage(img)
    # print(img.shape)
    # print(img.max(), img.min())
    # plt.imshow(img[0], cmap="gray")
    # plt.show()
    #
    # anno = pd.read_excel("/data2/zhai/InBreast/INbreast.xls")
    # print(anno)
    # print(anno.columns.to_list())
    #

    dataset = InBreastClsDataset(path, "image", "inbreast_valid.csv", stage="train", transform=None)
    print(len(dataset))
    img, label = dataset[1]
    print(label)

    import matplotlib.pyplot as plt
    plt.imshow(img, cmap="gray")
    # plt.show()

    # anno = pd.read_excel("/data2/zhai/InBreast/INbreast.xls")
    # anno.drop(anno.index[[410, 411]], inplace=True)
    # print(anno)
    # anno1 = anno.sample(frac=0.2)  # test set
    # print(anno1)
    #
    # anno_temp = anno[~anno.index.isin(anno1.index)]
    # print(anno_temp)
    #
    # anno2 = anno_temp.sample(frac=0.125)  # valid set
    # print(anno2)
    #
    # anno3 = anno_temp[~anno_temp.index.isin(anno2.index)]  # train set
    # print(anno3)
    #
    # anno1.to_csv(os.path.join(path, "inbreast_test.csv"), index=False)
    # anno2.to_csv(os.path.join(path, "inbreast_valid.csv"), index=False)
    # anno3.to_csv(os.path.join(path, "inbreast_train.csv"), index=False)


if __name__ == '__main__':
    main()
