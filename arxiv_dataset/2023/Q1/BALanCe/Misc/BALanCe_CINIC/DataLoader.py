import torch
import torch.utils.data
import numpy as np
import csv
import random
from Buffer import Buffer
from Pool import Pool
import collections
from PIL import Image
from torchvision import datasets, transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, buff, transform_mode='val'):
        self.image_name_lt = list(np.squeeze(buff.image_name_lt))
        
        if len(buff.labels.shape) < 2:
            self.labels = np.expand_dims(buff.labels, axis=1)
        else:
            self.labels = buff.labels
        
        self.transform_mode = transform_mode
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        shared_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=cinic_mean, std=cinic_std)])
        if transform_mode == 'train':
            #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.transform = transforms.Compose([train_transform, shared_transform])
        else:
            self.transform = shared_transform

    def __len__(self):
        return self.labels.shape[0]

    def get_weight(self):
        if len(self.labels) == 1:
            count = collections.Counter(list(self.labels[0]))
        else:
            count = collections.Counter(np.squeeze(self.labels))
        
        #num_class = len(count)
        num_class = 10
        weight_lt = []
        for i in range(num_class):
            if i not in count or count[i] == 0:
                weight = 0.0
            else:
                weight = 1./count[i]
            weight_lt.append(weight)
        weight_lt = np.array(weight_lt)
        return weight_lt

    def __getitem__(self, idx):
        #c_id = self.ids[idx]
        image_name = self.image_name_lt[idx]
        #feat = torch.Tensor(feat)
        #print(image_name)
        image = Image.open(image_name)
        image = image.convert('RGB')
        feat = self.transform(image)
        label = torch.LongTensor(self.labels[idx])
        return (feat, label)



