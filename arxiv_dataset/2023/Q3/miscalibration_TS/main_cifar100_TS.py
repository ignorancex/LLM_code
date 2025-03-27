import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import os
import argparse
import copy
import random
from PIL import Image
import numpy as np
import sys
import csv
import pandas as pd
from torch.distributions import Categorical
import tensorflow_probability as tfp
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# %matplotlib inline

def seed_everything(seed=12):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WarmUpLR(_LRScheduler):
    """
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def write_csv(filename, data):
    with open(filename, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

class CELossWith_ECELS(torch.nn.Module):
    
    def __init__(self, classes=100, smoothing=0.1, ignore_index=-1, ece_t_list=None, ece_t_weight=None):
        super(CELossWith_ECELS, self).__init__()
        self.smoothing = smoothing if ece_t_list is None else smoothing + ece_t_list/ece_t_weight
        self.complement = 1.0 - smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index

    def forward(self, logits, target, ):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).contiguous()
            smoothen_ohlabel = oh_labels * self.complement  + self.smoothing / self.cls
            smoothen_ohlabel = (smoothen_ohlabel/smoothen_ohlabel.sum(1)[:,None]) # to normalise the distr into sum 1
        
        logs = self.log_softmax(logits[target!=self.ignore_index])
        return -torch.sum(logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()



def train(model, trainloader, criterion, optimizer, epoch, warmup_scheduler, args):
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)    
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch <= args.warm:
            warmup_scheduler.step()
        
        
def valid(model, testloader, T=1, device=None):
    model.eval()
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            logits_list.append(outputs/T)
            labels_list.append(targets)

        logits = torch.cat(logits_list).cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()
        ece = tfp.stats.expected_calibration_error(10, logits=logits, labels_true=labels, labels_predicted=np.argmax(logits,1))
    return correct / total, logits, labels, ece


class CIFAR100_train_valid(torchvision.datasets.CIFAR100):

    def __init__(self, root, train=False, transform=None, download=False, is_train=False):
        super(CIFAR100_train_valid, self).__init__(root=root, train=train, transform=transform, download=download) 
        self.transform = transform
        self.train = train
        if self.train:
            if is_train:
                self.data = self.data[:40000]
                self.targets = self.targets[:40000]
            else:
                self.data = self.data[40000:]
                self.targets = self.targets[40000:]      

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target 

def main():
    seed_everything()

    #standard model 
    if args.resnet34:
        model = models.resnet34(pretrained=True).to(device)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=5e-4)

    if args.densenet121:
        model = models.densenet121(pretrained=True).to(device)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr_densenet, momentum=0.9, nesterov=False, weight_decay=5e-4)

    if args.vgg16:
        model = models.vgg16(pretrained=True).to(device)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr_vgg16, momentum=0.9, nesterov=False, weight_decay=5e-4)


    CIFAR100_TRAIN_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR100_TRAIN_STD = (0.2673, 0.2564, 0.2762)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    cifar100_training = CIFAR100_train_valid(root='data', train=True, download=True, transform=transform_train,is_train=True)
    cifar100_valid = CIFAR100_train_valid(root='data', train=True, download=True, transform=transform_train,is_train=False)
    cifar100_test = CIFAR100_train_valid(root='data', train=False, download=True, transform=transform_test)


    cifar100_training_loader = DataLoader(cifar100_training, shuffle=True, num_workers=2, batch_size=args.batch_size)
    cifar100_valid_loader = DataLoader(cifar100_valid, shuffle=False, num_workers=2, batch_size=args.test_batch_size)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=False, num_workers=2, batch_size=args.test_batch_size)
    print('sample size for main dataloader: train:{}, valid:{}, test:{}'.format (
            len(cifar100_training), len(cifar100_valid), len(cifar100_test)))
    
    #optimizer, criterion and scheduler
    
    # criterion = CELossWith_ECELS(classes=args.num_classes, smoothing=args.smoothing, ece_t_list=None, ece_t_weight=None)
    criterion = nn.CrossEntropyLoss().to(device)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm) 
    
    #training script
    best_epoch, best_acc, best_ece = 0.0, 0, np.inf
    
    for epoch in range(1, args.num_epoch + 1):
        if epoch > args.warm:
                train_scheduler.step()
        
        train(model, cifar100_training_loader, criterion, optimizer, epoch, warmup_scheduler, args)

        accuracy, _, _, ece = valid(model, cifar100_valid_loader, device = device)

        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            best_ece = ece
            best_model = copy.deepcopy(model)
            accuracy_test,  _, _, ece_test = valid(model, cifar100_test_loader, device = device)
            print('acc_test:{} ece:{}'.format(accuracy_test, ece_test))
            torch.save(best_model.state_dict(), 'ckpt_ts/best_cifar100_resnetmodel{}_desnetmodel{}_vggmodel{}_TS_smoothing{}.pth.tar'.format(
            int(args.resnet34),int(args.densenet121),int(args.vgg16), args.smoothing))
        print('epoch: {}  acc: {:.4f}  best epoch: {}  best acc: {:.4f} lr: {:.4f} best ece:{:.4f} smoothing:{:.4f}'.format(
                epoch, accuracy, best_epoch, best_acc, optimizer.param_groups[0]['lr'], best_ece, args.smoothing ))

    ckpt_name = 'best_cifar100_resnetmodel{}_desnetmodel{}_vggmodel{}_TS_smoothing{}.pth.tar'.format(
            int(args.resnet34),int(args.densenet121),int(args.vgg16), args.smoothing)
    write_csv("results_cifar100_TS.csv", [
                                    "best_epoch:"+ str(best_epoch),
                                    "best_acc:" + str(best_acc),
                                    "ece:" + str(np.array(best_ece)),
                                    "smoothing:" + str(args.smoothing),
                                    "ckpt_name:" + str(ckpt_name)
                                    ])
def get_args():
    parser = argparse.ArgumentParser(description='CIFAR100 TS Training')
    parser.add_argument('--root', default='data', type=str, help='root')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default=0, type=int, help='lr scheduler')
    parser.add_argument('--split', action='store_true', help='split or full')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=2048, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=200, type=int, help='epoch number')
    parser.add_argument('--num_classes', type=int, default=100, help='number classes')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--smoothing', default = 0, type=float, help='ls smoothing')
    parser.add_argument('--seed',default =12, type=int, help='seed')
    parser.add_argument('--resnet34', action='store_true', help='resnet34 or ls')
    parser.add_argument('--densenet121', action='store_true', help='densenet121 or ls')
    parser.add_argument('--lr_densenet', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--vgg16', action='store_true', help='vgg16 or ls')
    parser.add_argument('--lr_vgg16', default=0.001, type=float, help='learning rate')
    
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args
    

if __name__ == "__main__": 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_args()
    main()

