
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

    def __init__(self, dataset, pretrain=False, transform=None, transform1=None, transform2=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2
        self.pretrain = pretrain

        if dataset == 'all':
            data_num = [i for i in range(5)]

        if dataset == 'train':
            data_num = [i for i in range(5) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]


        for i in range(1, 90):
            if i % 5 in data_num:
                if i<10:
                    self.items.append(['/.../amd/Training400/AMD/A000{}.jpg'.format(str(i)), 0] )
                elif i<100:
                    self.items.append(['/.../amd/Training400/AMD/A00{}.jpg'.format(str(i)), 0])

        for i in range(1, 312):
            if (i-1) % 5 in data_num:
                if i<10:
                    self.items.append(['/.../amd/Training400/Non-AMD/N000{}.jpg'.format(str(i)), 1])
                elif i<100:
                    self.items.append(['/.../amd/Training400/Non-AMD/N00{}.jpg'.format(str(i)), 1])
                elif i<1000:
                    self.items.append(['/.../amd/Training400/Non-AMD/N0{}.jpg'.format(str(i)), 1])

      

    def __getitem__(self, idx):
        item = copy.deepcopy(self.items[idx])
        img_path = item[0]
        label = int(item[1])
        img = Image.open(img_path).convert('RGB')

        if self.pretrain:
            img1 = self.transform1(img)
            img2 = self.transform2(img)

            return [img1, img2], label
        
        else:

            img = self.transform(img)
            return img, label # l0, l1, l2#label

        return img, label, img_path

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(5):
        d = MyDataset('valid',None,fold=fold)
        print(len(d))
    print(d[0])
