import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models import *
from utils import *
from scipy.io import *
import shift

class split_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split_index, perc = 1.0, L = None):
        super(split_dataset, self).__init__()
        self.data = []
        self.targets = []
        for i in range(len(dataset)):
            # if idx[i] == True: 
            if dataset[i][1] in split_index:
                self.data.append(dataset[i][0]) # directly index type is tensor, range is 0-1; .data type is tensor, range is 0-255
                self.targets.append(dataset[i][1]) # directly index type is int
        if L == None: L = round(len(dataset) * perc)
        self.data = torch.stack(self.data)[0:L]
        self.targets = self.targets[0:L]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return len(self.targets)

class noise_generate(torch.utils.data.Dataset):
    def __init__(self, num, eps, targetlabel = 10): # eps range in [0, 255]
        super(noise_generate, self).__init__()
        self.data = torch.randint(low = 0, high = eps, size = (num, 28, 28))
        self.datalen = len(self.data)
        self.targets = torch.ones(num) * targetlabel

    def __getitem__(self, index):
        return self.data[index].float() / 255, int(self.targets[index])

    def __len__(self):
        return self.datalen

def mkdataset(args):
    if args.dataset == 'mnist':
        data_train = torchvision.datasets.MNIST(root = args.data_root, train = True, download = True, transform = transforms.ToTensor())
        data_test = torchvision.datasets.MNIST(root = args.data_root, train = False, download = True, transform = transforms.ToTensor())
        if args.shift: 
            data_train = shift.MNIST(root = args.data_root, train = True, download = True, transform = transforms.ToTensor())
        if args.num_tasks > 10: raise ValueError("Experiment 'SplitMNIST-10' cannot have more than 10 tasks!")
    else:
        raise ValueError("Only 'SplitMNIST-10' Experiment is supported!")

    classes_per_task = int(np.floor(10 / args.num_tasks))
    idx_list = []
    for i in range(args.num_tasks):
        idx_list.append(list(range(i * classes_per_task, (i+1) * classes_per_task)))

    data_train_list = []
    data_test_list = []
    for i in range(args.num_tasks):
        data_train_list.append(split_dataset(data_train, idx_list[i]))
        data_test_list.append(split_dataset(data_test, idx_list[i]))

    return data_train_list, data_test_list, data_train, data_test