import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from xpinyin import Pinyin
from PIL import Image
import copy
import pandas as pd
import csv
import random


class MyDataset(data_utils.Dataset):

    def __init__(self, dataset, pretrain=False, transform1=None, transform2=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform1 = transform1
        self.transform2 = transform2
        self.pretrain = pretrain




        

        if dataset == 'train':
            if fold == -1:
                data_num = [i for i in range(10)]
            else:
                data_num = [i for i in range(10) if i != fold]
        elif dataset == 'valid':

            data_num = [fold]

        
        for i in data_num:
            f = open('/.../DR_dataset/ten_fold/fold_{}.csv'.format(i), "r")
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.items.append(row[1:])

      

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img = item[0]
        label = int(item[1])
        img_path = '/.../DR_dataset/train/' + img + '.jpg'
        # label = int(item[-1])
        img = Image.open(img_path).convert('RGB')

        if self.pretrain:
            img1 = self.transform1(img)
            img2 = self.transform2(img)

            return [img1, img2], label
        
        else:

            img = self.transform1(img)

            if label < 2:
                label = 0
            else:
                label = 1
            
            
            return img, label  # l0, l1, l2#label

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(10):
        d1 = MyDataset(None,None,'valid',fold=fold)
        d2 = MyDataset(None,None,'train',fold=fold)

        print(len(d1), len(d2))
