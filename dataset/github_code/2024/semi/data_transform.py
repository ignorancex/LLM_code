import random,os,torch
import torch.utils.data as dataf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, STL10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transform
import numpy as np
import torch.nn.functional as F
from PIL import Image
from data_read import ImageFolder_L

BatchSize =100

class DeNormalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t,m,s in zip(tensor,self.mean,self.std):
            t.mul_(s).add_(m)
        return tensor

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)

################################# MNIST #################################
inver_transform_MNIST = transform.Compose([
    DeNormalize([0.5],[0.5]),
    lambda x: x.cpu().numpy()*255.,
])

data_transform = transform.Compose([
    #transform.Pad(padding=2,fill=0),
    transform.ToTensor(),
    #transform.Normalize(mean=[0.5],std=[0.5])
    transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# MNIST_TRAIN = r"./dataset_equality/train"
# MNIST_TEST = r"./dataset_equality/test"
#MNIST_TRAIN = r"./fashionMNIST_img/train"
#MNIST_TEST = r"./fashionMNIST_img1/test"
#
# MNIST_TRAIN = r"./CIFAR10/train"
# MNIST_TEST  = r"./CIFAR10/test"
# L_train_set_1 =  ImageFolder_L(MNIST_TRAIN,transform=data_transform)
# L_test_set_1  =  ImageFolder_L(MNIST_TEST,transform=data_transform)
# L_train_data_1 = DataLoader(L_train_set_1,batch_size=BatchSize,shuffle=True)
# L_test_data_1  =  DataLoader(L_test_set_1,batch_size=BatchSize,shuffle=True)


STL10_TRAIN = "./STL10/train"
STL10_TEST = "./STL10/train"

# 定义数据转换
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 这里假设你希望进行归一化处理
])

# 创建STL-10数据集的实例
STL10_train_set = STL10(root=STL10_TRAIN, split='train', transform=data_transform, download=True)
STL10_test_set = STL10(root=STL10_TEST, split='test', transform=data_transform, download=True)

# 创建数据加载器
BatchSize = 100  # 批量大小
STL10_train_loader = DataLoader(STL10_train_set, batch_size=BatchSize, shuffle=True)
STL10_test_loader = DataLoader(STL10_test_set, batch_size=BatchSize, shuffle=False)  # 测试集一般不需要打乱顺序

L_test_data_1 = STL10_test_loader
L_train_data_1 =STL10_train_loader

# # ################################# COIL20 #################################
# # # #
# #
# inver_transform_COL20 = transform.Compose([
#     DeNormalize([0.5],[0.5]),
#     lambda x: x.cpu().numpy()*255.,
# ])
#
# data_transform_COL20 = transform.Compose([
#     transform.ToTensor(),
#     transform.Normalize(mean=[0.5],std=[0.5])
# ])
#
# COL20_TRAIN = r"./COIL20/train"
# COL20_TEST = r"./COIL20/test"
#
# COL20_train_set_1 = ImageFolder_L(COL20_TRAIN,transform=data_transform_COL20)
# COL20_test_set_1 = ImageFolder_L(COL20_TEST,transform=data_transform_COL20)
# COL20_train_data_1 = DataLoader(COL20_train_set_1,batch_size=BatchSize,shuffle=True)
# COL20_test_data_1 = DataLoader(COL20_test_set_1,batch_size=BatchSize,shuffle=True)
