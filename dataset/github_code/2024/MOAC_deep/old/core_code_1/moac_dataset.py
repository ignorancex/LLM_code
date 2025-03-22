import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from restore_module import pre_deconv,wiener_deconv
import random

class MOACDataset:
    def __init__(self, dataset_dir, train_or_test):
        '''图像对读取到列表'''
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.gt_dir = os.path.join(dataset_dir, 'train' if train_or_test == 'train' else 'test', 'GT')
        self.noised_dir = os.path.join(dataset_dir, 'train' if train_or_test == 'train' else 'test', 'noised')
        def file_to_list(folder_path):
            file_list = []
            files = os.listdir(folder_path) 
            for file in files:
                file_list.append(file)
            file_list.sort()    # 字典排序
            return file_list
        
        self.filename_GT = file_to_list(self.gt_dir)
        self.filename_noised = file_to_list(self.noised_dir)
#         print(self.filename_GT[0:10],self.filename_noised[0:10])# debug

    def __getitem__(self, index):
        # 获取图像对的文件名
        gt_path = os.path.join(self.gt_dir, self.filename_GT[index])
        noised_path = os.path.join(self.noised_dir, self.filename_noised[index])
        # print(self.filename_GT[index],self.filename_noised[index]) # debug
        # 读取图像
        gt_image = Image.open(gt_path).convert('L')
        noised_image = Image.open(noised_path).convert('RGB')   # 只有RG通道有效，作为实部和虚部

        # 归一化 GT 图像
        gt_array = np.array(gt_image) / 255.0
        gt_sum = np.sum(gt_array,axis=0)

        # 归一化噪声图像
        noised_array = np.array(noised_image) / 255.0
        noised_array = noised_array[:,:,0] + 1j*noised_array[:,:,1] # 转成复数矩阵
        # 文件名中隐含了信道信息（作为信道种子）
        hh_array_seed = self.filename_GT[index]
        hh_array_seed = int(hh_array_seed[0:5])
        np.random.seed(hh_array_seed+2)
        h_coff = np.exp(1j*np.random.uniform(0,1,(20,1)) * 4* np.pi /4).reshape(-1,1) # e^j0 ~ e^j pi
        # 文件名中隐含了SNR信息（作为种子）
        random.seed(hh_array_seed+1)
        SNR_db = random.randint(-4, 4)*5    # -20dB ~ 20dB
        noised_seq = noised_array.T.reshape(-1) # M x L -> ML x 1 seq
        noised_seq = noised_seq[:-1]  # 得到过采样有噪失真序列

        # 模型的第一阶段，由于采用numpy实现，故在getitem步骤直接执行，将第一步结果发给模型做第二步恢复
        _,noised_hinv_deconv_mat = pre_deconv(noised_seq,20,1000,h_coff,SNR_db)
        noised_hinv_deconv_mat = np.array([noised_hinv_deconv_mat.real,noised_hinv_deconv_mat.imag],\
                                          dtype=float)    # 2xHxW
        gt_array = np.expand_dims(gt_array,axis=0)  # 和1xHxW输出对齐
        # 返回 GT 和噪声图像的数组
        return torch.from_numpy(gt_array).float(),\
              torch.from_numpy(noised_array).float(), \
              torch.from_numpy(noised_hinv_deconv_mat).float(), \
            torch.from_numpy(gt_sum).float()

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.filename_GT)  # 假设数据集中有1000个样本

if __name__ == '__main__':
    # 使用示例
    dataset = MOACDataset('/kaggle/working/MOAC_deep/dataset', 'train')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for gt, noised, gt_sum in train_loader:  # 获取第0个样本的GT和噪声图像
        print('Size of gt, noised, gt_sum:{}{}{}'\
            .format(gt.size(), noised.size(), gt_sum.size()))
        break