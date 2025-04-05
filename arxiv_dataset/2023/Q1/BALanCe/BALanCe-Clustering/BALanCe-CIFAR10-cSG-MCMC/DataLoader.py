import torch
import torch.utils.data
import numpy as np
import csv
import random
from Pool import Pool
import collections
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import datasets, transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pool, train=False):
        self.path_lt = pool.path_lt
        self.labels = pool.labels
        self.bitmap = pool.bitmap

        self.train_mode = train
        '''
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        '''
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        shared_transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

        if self.train_mode == True:
            self.transform = transforms.Compose([train_transform, shared_transform])
        else:
            self.transform = shared_transform


    def __len__(self):
        return len(self.bitmap)

    def size(self):
        return len(self.bitmap)
    
    def get_weight(self):
        label_lt = []
        for pos in self.bitmap:
            label = self.labels[pos]
            label_lt.append(label)

        count = collections.Counter(label_lt)
        num_class = len(count)
        weight_lt = []
        for i in range(num_class):
            weight = 1./count[i]
            weight_lt.append(weight)
        weight_lt = np.array(weight_lt)
        #print(f'weight: {weight_lt}')
        return weight_lt
        
    def get_weight4sample(self):
        label_lt = []
        for pos in self.bitmap:
            label = self.labels[pos]
            label_lt.append(label)
        count = collections.Counter(label_lt)
        num_class = len(count)
        weight_lt = []
        for i in range(num_class):
            weight = 1./count[i]
            weight_lt.append(weight)
        weight_lt = np.array(weight_lt)
        #print(f'weight: {weight_lt}')
        sample_weight_lt = []
        for pos in self.bitmap:
            label = self.labels[pos]
            weight = weight_lt[label]
            sample_weight_lt.append(weight)
        return sample_weight_lt
    
    def __getitem__(self, idx):
        #c_id = self.ids[idx]
        pos = self.bitmap[idx]
        image_path = self.path_lt[pos]
        image = Image.open(image_path).convert('RGB')
        feat = self.transform(image)
        label = torch.LongTensor([self.labels[pos]])
        #print(pos, feat.size, label.size())
        return (pos, feat, label)



