import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from tools import *
from utils.metric import *


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, flag=None):
        self.root_dir = root_dir
        self.sample_list = os.listdir(os.path.join(self.root_dir, 'crop_image'))
        self.sample_list = [i for i in self.sample_list if i.endswith('.png') or i.endswith(".jpg")]
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample_name = self.sample_list[index]
        crop_image = cv2.imread(os.path.join(self.root_dir, "crop_image", sample_name))
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

        crop_mask = cv2.imread(os.path.join(self.root_dir, "crop_mask", sample_name), 0)
        crop_mask = cv2.threshold(crop_mask, 0, 255, cv2.THRESH_BINARY)[1]

        crop_predict = cv2.imread(os.path.join(self.root_dir, "crop_predict", sample_name), 0)
        crop_predict = cv2.threshold(crop_predict, 0, 255, cv2.THRESH_BINARY)[1]

        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        try:
            crop_image[:, :, 0] = 0.5*crop_image[:, :, 0]+0.5*crop_predict
            crop_image_trans = self.transform(crop_image)
        except:
            print(sample_name)

        if self.flag == 1:
            dice = calc_dice(crop_predict/255, crop_mask/255)
            return {"crop_image_trans": crop_image_trans, "dice": dice}
        elif self.flag == 2:
            dice = calc_dice(crop_predict/255, crop_mask/255)
            hd = calc_HD(crop_predict/255, crop_mask/255)
            return {"crop_image_trans": crop_image_trans, "dice": dice, "hd": hd}

