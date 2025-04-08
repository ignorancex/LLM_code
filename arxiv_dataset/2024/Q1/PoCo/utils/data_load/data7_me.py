
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
import xlrd

class MyDataset(data_utils.Dataset):

    def __init__(self, dataset, pretrain=False, transform=None, transform1=None, transform2=None, fold=3):
        self.data_list = []
        self.items = []
        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2
        self.pretrain = pretrain


        pr = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34']


        for i in pr:
         
            ex_path = '/.../Messidor/Annotation_Base{}.xls'.format(i)
            xls = xlrd.open_workbook(ex_path)                               
            sheet1 = xls.sheets()[0]
            rows = sheet1.nrows
            for j in range(1, rows):
                l = sheet1.row(j)
                c1 = sheet1.cell(j, 0).value
                c2 = sheet1.cell(j, 2).value

                if c2 > 0:
                    c2 = 1
                self.data_list.append(['/.../Messidor/image/'+c1, int(c2)])



        if dataset == 'all':
            data_num = [i for i in range(5)]

        if dataset == 'train':
            data_num = [i for i in range(5) if i != fold]
        elif dataset == 'valid':
            data_num = [fold]


        for i in range(1200):
            if i % 5 in data_num:
                self.items.append(self.data_list[i])
    
      

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

        return img, label  

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    for fold in range(5):
        d = MyDataset('valid',None,fold=fold)
        print(len(d))
    print(d[0])
