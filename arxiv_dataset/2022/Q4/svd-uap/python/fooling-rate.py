'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from resnet import ResNet18
from utils import progress_bar
from torch.autograd import Variable
import numpy as np
import random
from deepfool import deepfool

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
n_epochs = args.n_epochs
bsize = 100

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

## Function to load the precalculated and stored SVD-UAP scaled as required
def test_pert_pred(scale_uap):
    global best_acc
    net.eval()

    tvh_grad = np.load("_tvh.npy") # Load the SVD-UAP calculated and stored by code svd-uap.py
    p = np.array([tvh_grad[0], tvh_grad[0], tvh_grad[0]]) # Assigns the top singular vector as the universal perturbation(SVD-UAP)
    pert = torch.from_numpy(p).type('torch.FloatTensor')
    pert = pert.view(3072)/torch.norm(pert.view(3072), p=2)

    test_loss = 0
    correct = 0
    total = 0
    total_pred = torch.LongTensor()
    total_target = torch.LongTensor()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        ## applying the SVD-UAP to an input batch
        for k in range(inputs.shape[0]):
            inputs[k] = inputs[k] + torch.FloatTensor(pert.view((3, 32, 32))*scale_uap)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total_pred = torch.cat((total_pred, predicted.cpu()), 0)
        total_target = torch.cat((total_target, targets.data.cpu()), 0)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    flr = 0
    for i in range(10000):
        if int(total_target[i]) != int(total_pred[i]):
            flr += 1
    print("Fooling rate for scale ", scale_uap, " is ", flr/10000)

net = torch.load("./saved_model/resnet18_cifar10_pth")

## Applying different scaling factor to SVD-UAP vector
scale = [0, 3, 6, 9, 18, 36, 72, 144, 288]
for s in scale:
    test_pert_pred(s)

