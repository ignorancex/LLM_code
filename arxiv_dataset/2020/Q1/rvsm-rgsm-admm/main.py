# -*- coding: utf-8 -*-
"""
main pgd resnet
Usage: python main_pgd_resNet.py
"""
import argparse
import os
import shutil
import time
import numpy as np

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

from prune_utils import *
from loss_func import *
from utils import *
from models import *




parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--dev', '--device', default=0, type=int, metavar='N', help='GPU to run on')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')

# Parameters for RVSM (beta, lamb) and RGSM (beta, lamb1, lamb2)
parser.add_argument('--beta', default=1e-2, type=float, help='beta value')
parser.add_argument('--lamb', default=1e-6, type=float, help='lambda value')
parser.add_argument('--beta1', default=1, type=float, help='beta value')
parser.add_argument('--lamb1', default=1e-2, type=float, help='lambda1 value')
parser.add_argument('--lamb2', default=1e-5, type=float, help='lambda2 value')

# Parameters for ADMM
parser.add_argument('--pcen', default=80, type=float, help='pruning percentage')
parser.add_argument('--sparsity', default='elem', type=str, metavar='M', help='type of sparsity pruning')

parser.add_argument('--method', default='default', type=str, help='type of training: default, rvsm, rgsm, admm')


parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')



if __name__ == '__main__':
    global best_acc
    best_acc = 0.0
    start_epoch = 0
    args = parser.parse_args()
    device = 'cuda:'+str(args.dev) if torch.cuda.is_available() else 'cpu'
    try:
        os.mkdir('weights')
    except:
        pass

    #--------------------------------------------------------------------------
    # Load Network type
    #--------------------------------------------------------------------------
    net = MobileNet()
    net = net.to(device)
    
    
    #--------------------------------------------------------------------------
    # Load Cifar data
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = './data'
    download = True
    
    #normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    train_set = torchvision.datasets.CIFAR10(root=root,
                                            train=True,
                                            download=download,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                #normalize,
                                            ]))
    
    test_set = torchvision.datasets.CIFAR10(root=root,
                                            train=False,
                                            download=download,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                #normalize,
                                            ]))    
    
    kwargs = {'num_workers':4, 'pin_memory':True}
    batchsize_test = int(len(test_set)/100)
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batchsize_test, shuffle=False, **kwargs)
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batchsize_train, shuffle=True, **kwargs)
 
    criterion = nn.CrossEntropyLoss()
    
    
    #--------------------------------------------------------------------------
    # Declare Hyper-parameteres 
    #--------------------------------------------------------------------------
    
    method = args.method
    
    # RVSM
    beta = args.beta
    lamb = args.lamb    
    gamma = np.sqrt(2*lamb/beta)
    
    # RGSM
    beta1 = args.beta1
    lamb1 = args.lamb1
    lamb2 = args.lamb2
    
    # ADMM
    rho = 1e-2
    pcen = args.pcen
    sparsity = args.sparsity

    #--------------------------------------------------------------------------
    # Training Process
    #--------------------------------------------------------------------------
    
    nepoch = 200
    for epoch in range(nepoch):
        print('Epoch ID', epoch)
        if epoch < 80:
            lr = 0.1
        elif epoch < 120:
            lr = 0.1/10
        elif epoch < 160:
            lr = 0.1/10/10
        else:
            lr = 0.1/10/10/10
        
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        correct = 0; total = 0; train_loss = 0
        net.train()
        
        if method != 'default':
            weights = [p for n,p in net.named_parameters() if 'weight' in n and len(p.size())==4]
        if method == 'rvsm':
            sp_wt = [thres(w,gamma) for w in weights]        
        if method == 'rgsm':
            sp_wt = rgsm_l0(weights, lamb1, beta1)        
        if method == 'admm':
            zs = [projection(w.data,pcen,sparsity) for w in weights]
            us = [torch.zeros_like(z) for z in zs]        
            
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = x.to(device), target.to(device)            
            score = net(x)            
            
            if method == 'rgsm':
                loss = criterion(score, target) + gpls(weights, lamb2)
            else:
                loss = criterion(score, target)
                
            if method == 'admm':
                for j in range(len(weights)):
                    zs[j] = weights[j].data + us[j]
                    zs[j] = projection(zs[j],pcen,sparsity)
                    us[j] += weights[j] - zs[j]
            
            loss.backward()
            
            if method != 'default':
                for j in range(len(weights)):
                    if method == 'rvsm':
                        weights[j].data -= lr*beta*(weights[j].data-sp_wt[j].data)
                    if method == 'rgsm':
                        weights[j].data -= lr*beta1*(weights[j].data-sp_wt[j].data)
                    if method == 'admm':
                        for j in range(len(weights)):
                            weights[j].data -= lr*rho*(weights[j].data-zs[j]+us[j])
                
            optimizer.step()
            if method == 'rvsm':
                sp_wt = [thres(w,gamma) for w in weights]
            if method == 'rgsm':
                sp_wt = rgsm_l0(weights, lamb1, beta1)
            
            train_loss += loss.item()
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total))
            
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------      
        
        correct = 0; total = 0; test_loss = 0
        net.eval()
        if method == 'rvsm':
            for wt in weights:
                replace_weight(wt,gamma)
        if method == 'rgsm':
            for j in range(len(sp_wt)):
                k_sp = sp_wt[j]
                k_wt = weights[j]
                k_sp.data, k_wt.data = k_wt.data, k_sp.data
        
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):            
                x, target = x.to(device), target.to(device)
                try:
                    score, pert_x = net(x, target)
                except:
                    score = net(x)
                
                loss = criterion(score, target)
                test_loss += loss.item()
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        
        if method == 'default':
            out = ''
        if method == 'rvsm':
            out = 'beta'+str(beta)+'lamb'+str(lamb)
        if method == 'rgsm':
            out = 'beta'+str(beta1)+'lamb1_'+str(lamb1)+'lamb2_'+str(lamb2)
        if method == 'admm':
            out = 'pcen'+str(pcen)+str(sparsity)
        
        acc = 100.*correct.numpy()/total
        if acc > best_acc:
            print('Saving model...')
            state = {
                'net': net, 
                'acc': acc,
                'epoch': epoch,
            }
            
            torch.save(state, './weights/'+str(method)+out+'.pth')
            best_acc = acc
    
    print('The best acc: ', best_acc)
