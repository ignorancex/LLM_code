# -*- coding: utf-8 -*-

import re
import random
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform as ST

import SimpleITK as sitk
import nibabel as nib
import torch
from torchvision import transforms
# from dataset import pre_caption

VIEW_MAP = {"RCC": 1, "LCC": 2, "LMLO": 3, "LMO": 3, "RMLO": 4, "RMO": 4}
DEL_VIEWS = ["无", "无资料", "不会", np.nan]
MALIGNANT = {"恶性": 1, "良性": 0}

# MAM_MAX_PIXEL, MAM_MIN_PIXEL = 4095, 0

LABEL_MAP = {"calc": 0, "mass": 1}


# ====================== downstream segmentation task dataset =========================
class DDSMClsDataset:

    def __init__(self, db_root_path, dcm_dir, anno_file, stage="train", transform=None, binary_label=False,
                 pre_downsample=False, padding_square=False, partial_data=None):
        self.db_root_path = db_root_path
        self.dcm_dir = dcm_dir
        self.anno_file = anno_file
        self.padding_square = padding_square
        self.stage = stage
        self.binary_label = binary_label
        anno = pd.read_csv(os.path.join(db_root_path, f"annotation/{anno_file}"))

        img_file_path = anno["image file path"].to_list()
        img_file_path = [file.split("/")[0].strip() for file in img_file_path]
        self.img_file_path = img_file_path
        self.assessment = anno.assessment.to_list()
        self.pathology = anno.pathology.to_list()
        self.transform = transform
        self.pre_downsample = pre_downsample
        self.partial_data = partial_data
    def padding(self,image):
        # 获取图像尺寸
        width, height = image.size

        # 计算需要的填充尺寸
        max_dim = max(width, height)
        pad_width = max_dim - width
        pad_height = max_dim - height

        # 从这里我们可以得出需要在左侧和右侧填充多少（水平方向），以及在顶部和底部填充多少（垂直方向）
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        # 创建填充变换
        padding = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)

        # 应用填充变换
        image_padded = padding(image)
        return image_padded

    # def __len__(self):
    #
    #     return len(self.assessment)
    def __len__(self):
        if self.partial_data is not None:
            return int(len(self.assessment)*self.partial_data)
        else:
            return len(self.assessment)

    def __getitem__(self, item):
        img_file = self.img_file_path[item]
        img_name = ".dcm"
        for root, dirs, files in os.walk(os.path.join(self.db_root_path, self.dcm_dir, img_file)):
            for name in files:
                ext = os.path.splitext(name)[1]
                if ext == ".png": #xinwei: this is for pre downsample
                    continue
                img_name = os.path.join(root, name)

        if self.pre_downsample:
            img = Image.open(f"{img_name}.a.png")
        else:
            # view1 = sitk.ReadImage(os.path.join(view_dir, self.view_ids[item][0]))
            # view1 = sitk.GetArrayFromImage(view1)
            # # view1 = Image.fromarray(view1[0]).convert("RGB")
            # view1 = (view1 - MAM_MIN_PIXEL) / (MAM_MAX_PIXEL - MAM_MIN_PIXEL)
            # view1 = (view1 * 255).astype(np.uint8)
            # view1 = np.concatenate([view1, view1, view1], axis=0).transpose((1, 2, 0))
            # view1 = Image.fromarray(view1)

            img = sitk.ReadImage(img_name)
            img = sitk.GetArrayFromImage(img)

            # img = (img - MAM_MIN_PIXEL) / (MAM_MAX_PIXEL - MAM_MIN_PIXEL)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img[0]).convert("RGB")
        if self.padding_square:
            img = self.padding(img)
        if self.transform:
            img = self.transform(img)
        if self.binary_label:
            if "BENIGN" in self.pathology[item]:
                label = 0
            else:
                label = 1
        else:
            if self.assessment[item] in [0]:
                label = [1, 0, 0]
            elif self.assessment[item] in [1, 2]:
                label = [0, 1, 0]
            elif self.assessment[item] in [3, 4, 5]:
                label = [0, 0, 1]
            else:
                label = [0, 0, 0]
        label = torch.tensor(label)

        return {"view1": img, "label": label}  # align with mkvl_dataset


def main():
    db_root_path = "/data2/zhai/DDSM/"
    dcm_folder = "CBIS-DDSM/"
    anno_file = "calc_train_set.csv"  # None

    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    aug = T.Compose([
        T.Resize((384, 384)),
        # T.RandomResizedCrop((256, 256), scale=(0.2, 1.)),
        # T.RandomHorizontalFlip(),
        # T.RandomRotation((-10, 10)),
        # T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        normalize,
    ])
    # aug = None

    mask_aug = T.Compose([
        T.Resize((256, 256)),
        # T.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])

    ddsm_dataset = DDSMClsDataset(
        db_root_path, dcm_folder, anno_file, stage="test", transform=None
    )
    print(len(ddsm_dataset))
    # print(ddsm_dataset[0][0].size, ddsm_dataset[0][1])
    print(ddsm_dataset[0][0].max(), ddsm_dataset[0][0].min())

    # anno_calc = pd.read_csv("/data2/zhai/DDSM/annotation/calc_case_description_test_set.csv")
    # # anno_calc = pd.read_csv("/data2/zhai/DDSM/annotation/mass_case_description_train_set.csv")
    # print(anno_calc)
    # path1 = anno_calc.iloc[0, 11]
    # print(path1)
    # # print(os.listdir("/data2/zhai/DDSM/CBIS-DDSM/" + path1))
    # # img = sitk.ReadImage("/data2/zhai/DDSM/CBIS-DDSM/" + path1)
    # # img = sitk.GetArrayFromImage(img)
    #
    # path2 = path1.split("/")[0].strip()
    # print(path2)
    #
    # # for i in range(len(anno_calc)):
    # # # for i in range(1):
    # #     path1 = anno_calc.iloc[i, 11]
    # #     path2 = path1.split("/")[0].strip()
    # #
    # #     c = 0
    # #     for root, dirs, files in os.walk("/data2/zhai/DDSM/CBIS-DDSM/" + path2):
    # #         for name in files:
    # #             # print(os.path.join(root, name))
    # #             c += 1
    # #     if c == 2:
    # #         print(path2)
    #
    # # anno_calc_valid = anno_calc.sample(frac=0.1)  # valid set
    # # anno_calc_temp = anno_calc[~anno_calc.index.isin(anno_calc_valid.index)]  # train set
    # #
    # # anno_calc_valid.to_csv("/data2/zhai/DDSM/annotation/calc_valid_set.csv", index=False)
    # # anno_calc_temp.to_csv("/data2/zhai/DDSM/annotation/calc_train_set.csv", index=False)
    # # anno_calc.to_csv("/data2/zhai/DDSM/annotation/calc_test_set.csv", index=False)
    # #
    # # anno_mass = pd.read_csv("/data2/zhai/DDSM/annotation/mass_case_description_test_set.csv")
    # #
    # # anno_mass_valid = anno_mass.sample(frac=0.1)  # valid set
    # # anno_mass_temp = anno_mass[~anno_mass.index.isin(anno_mass_valid.index)]  # train set
    # #
    # # anno_mass_valid.to_csv("/data2/zhai/DDSM/annotation/mass_valid_set.csv", index=False)
    # # anno_mass_temp.to_csv("/data2/zhai/DDSM/annotation/mass_train_set.csv", index=False)
    # # anno_mass.to_csv("/data2/zhai/DDSM/annotation/mass_test_set.csv", index=False)


if __name__ == '__main__':
    main()
