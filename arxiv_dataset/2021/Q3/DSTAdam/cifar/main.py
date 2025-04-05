'''
Train CIFAR10/100 with PyTorch.

highly base on https://github.com/kuangliu/pytorch-cifar
               https://github.com/Luolc/AdaBound
               
author: kun zeng
email: zki@163.com
data:2021/09/03        
'''

import os
import torch
import numpy as np
import random
from adabound import AdaBound
from dstadam import DSTAdam
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import datetime
import argparse
import math
from models import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fun(worker_id, seed):
    np.random.seed(seed + worker_id)


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--cifar',  default='cifar10',type=str, help='training CIFAR10/100',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='resnet18', type=str, help='model')
    parser.add_argument('--optimizer', default='DSTAdam', type=str, help='optimizer',
                        choices=['SGDM','Adam','AdaBound', 'DSTAdam'])
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--final_lr', default=0.1, type=float, help='final learning rate of AdaBound')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for optimizers, recommend: 5e-4')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--epochs',  default=200, type=int, help='iterations')       
    parser.add_argument('--up_lr', default=5, type=float, help='upper learning rate of DSTAdam')
    parser.add_argument('--low_lr', default=0.005, type=float, help='lower learning rate of DSTAdam')    
    parser.add_argument('--step_lr',  default=False, type=bool, help='learning rate scheduler StepLR')
    parser.add_argument('--seed',  default=1, type=float, help='random seed')
    parser.add_argument('--coeff',  default=1e-8, type=float, help='scaling coefficient')
    parser.add_argument('--batch_size',  default=128, type=int, help='batch_size')

    args = parser.parse_args() # parsering parameters
    
    return args


def create_data(worker_id, args, seed):
    if args.cifar == 'cifar10':
        print('==> Preparing data cifar10..')
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
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
    elif args.cifar == 'cifar100':
        print('==> Preparing data cifar100..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507,0.487,0.441), (0.267,0.256,0.276)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507,0.487,0.441), (0.267,0.256,0.276)),
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fun(worker_id, seed))
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2,worker_init_fn=worker_init_fun(worker_id, seed))  
    else:
          raise ValueError("Invalid cifar: {}".format(args.cifar))
          
    return train_loader, test_loader

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')


def create_model(args, device):
    if args.cifar == 'cifar10':
        num_classes = 10
    elif args.cifar == 'cifar100':
        num_classes = 100
        
    if args.model == 'resnet18':
        net = ResNet18(num_classes)
    
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        
    return net


def create_optimizer(args, net, iters):
    if args.optimizer == 'SGDM':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), 
                                weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optimizer == 'AdaBound':
        optimizer = AdaBound(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), 
                                weight_decay=args.weight_decay, amsgrad=args.amsgrad, 
                                final_lr=args.final_lr)        
    elif args.optimizer == 'DSTAdam':
        optimizer = DSTAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), 
                                weight_decay=args.weight_decay, amsgrad=args.amsgrad, iters=iters,
                                coeff=args.coeff, up_lr=args.up_lr, low_lr=args.low_lr)
        
    return optimizer


def train(epoch, net, args, train_loader, optimizer, criterion, device, seed):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
             
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
            
    train_error = train_loss/(batch_idx+1)
    train_acc = correct/total
    
    return train_acc, train_error


def test(epoch, net, test_loader, optimizer, criterion, device, seed): 
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
  
        test_error = test_loss / (batch_idx+1)
        test_acc = correct / total
        
    return  test_acc, test_error


def main():
    args = create_parser()
    seed = args.seed
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    worker_id = 0
    best_test_acc = 0
    all_train_error = []
    all_train_acc = []
    all_test_acc = []  
    all_test_error = []
    train_loader, test_loader = create_data(worker_id, args, seed)
    iters = math.ceil(len(train_loader.dataset) / args.batch_size) * args.epochs
    net = create_model(args, device)
    optimizer = create_optimizer(args, net, iters)
    criterion = nn.CrossEntropyLoss()
    root_path = os.path.abspath('./')
    info = args.cifar + '-' + args.optimizer + '-' + args.model + '-'

    print("\n")
    print("**********parameters**********")
    print("The dataset is: ", args.cifar)
    print("The model is: ", args.model)
    print("The optimizer is: ", args.optimizer)
    print("The epochs is: ", args.epochs)
    print("**********parameters**********")
    print("\n")

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_test_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('start_epoch is: ', start_epoch)

    start = datetime.datetime.now()
    for epoch in range(start_epoch, start_epoch+args.epochs):
        print('\nepoch:{}'.format(epoch))

        train_acc, train_error = train(epoch, net, args, train_loader, optimizer, criterion, device, seed)
        test_acc, test_error = test(epoch, net, test_loader, optimizer, criterion, device, seed)
        if args.step_lr:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1, last_epoch=-1)
            scheduler.step()
            
        # Save checkpoint.
        if test_acc > best_test_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_test_acc = test_acc  
            
        all_train_acc.append(train_acc)
        all_train_error.append(train_error)
        all_test_acc.append(test_acc)
        all_test_error.append(test_error)
        
        end = datetime.datetime.now()
        print('time%.3f:'%(((end-start).seconds)/60))
        print('train_acc: %.2f%%, train_error: %.4f'%(train_acc*100.0, train_error))
        print('test_acc: %.2f%%, test_error: %.4f'%(test_acc*100.0, test_error)) 
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    np.save(root_path + '/results/' + info + 'all_train_acc', all_train_acc)
    np.save(root_path + '/results/' + info + 'all_test_acc', all_test_acc)
    np.save(root_path + '/results/' + info + 'all_train_error', all_train_error)
    np.save(root_path + '/results/' + info + 'all_test_error', all_test_error)


if __name__ == '__main__':
    main()