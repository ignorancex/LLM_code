from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        
        #print("fasdf")
        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            #print(if_exchange)
            return (input1_tensor, input2_tensor)
        else:
            #print(if_exchange)
            return (input2_tensor, input1_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])


class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()
        self.extensions = ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                full_img_list = []
                for ex in self.extensions:
                    full_img_list.extend(glob.glob(os.path.join(data, ex)))
               
                self.datas[data_name]['image'] = full_img_list
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        #input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        #input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0

        # # in case of OOM
        # max_range = 2000
        # if max(input1.shape[0],input1.shape[1]) > max_range:
        #     # print('the original image size is too big, scale the image max size in 2500 pixel range.')
        #     scale_width = int((max_range / max(input1.shape[1],input1.shape[0])) * input1.shape[0])
        #     scale_hight = int((max_range / max(input1.shape[1],input1.shape[0])) * input1.shape[1])
        #     input1 = cv2.resize(input1,(scale_hight,scale_width), interpolation=cv2.INTER_AREA)
        #     input2 = cv2.resize(input2,(scale_hight,scale_width), interpolation=cv2.INTER_AREA)

        if input1.shape != input2.shape:
            # print('input2:',input1.shape,input2.shape)
            input2 = cv2.resize(input2,(input1.shape[1],input1.shape[0]), interpolation=cv2.INTER_AREA)
            
        input1 = np.transpose(input1, [2, 0, 1])
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])





