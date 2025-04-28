# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import cv2
import math

class BasicDataset(data.Dataset):
    def __init__(self, args,mode):
        self.dataset = args.dataset
        path_image = f'/fast_storage/hwihun/pkl_clcl/pkl_{args.dataset}_{mode}.pklv4'
        path_seg = f'/fast_storage/hwihun/pkl_clcl/pkl_{args.dataset}_{mode}_target.pklv4'
        self.hr_images = self.load_pkls(path_image)
        self.seg_images = self.load_pkls(path_seg)
        self.class_num = args.class_num
        
    def load_pkls( self,path):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        return images
    

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        img = np.abs(self.hr_images[item].astype('float32'))
        if self.dataset ==  'Task1_OASIS1_Tissue_seg':
            if self.class_num == 1:
                seg = np.where(self.seg_images[item] > 2, np.ones_like(self.seg_images[item]), np.zeros_like(self.seg_images[item]))
            elif self.class_num == 2:
                seg = np.where(self.seg_images[item] > 1, self.seg_images[item], np.zeros_like(self.seg_images[item]))
            else:
                seg = self.seg_images[item]
        elif self.dataset == 'Task2_ATLAS_2_Lesion_seg':
            if self.class_num == 1:
                seg = self.seg_images[item]
        elif self.dataset == 'Task3_BRATS_Tumor_seg':
            if self.class_num == 1:
                seg = np.where(self.seg_images[item] > 0, np.ones_like(self.seg_images[item]), np.zeros_like(self.seg_images[item]))
            elif self.class_num == 3:
                seg = np.where(self.seg_images[item] == 4, 3*np.ones_like(self.seg_images[item]) , self.seg_images[item])
                

        return img.transpose([2, 0, 1]),seg.astype('float32').transpose([2, 0, 1])
