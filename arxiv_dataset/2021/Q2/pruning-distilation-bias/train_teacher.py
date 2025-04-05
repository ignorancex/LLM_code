from __future__ import print_function

import os
os.environ['KMP_WARNINGS'] = 'off'
import sys
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.models as models

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_imbalanced
from dataset.imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate

##########################################################
import torch.nn.utils.prune as prune
from src.prune_scheduler import AgpPruningRate
from itertools import chain
import numpy as np

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--gpu', type=str, default='0', choices=['0', '1', '2', '3'], help='gpu to train on')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'pretrained_torch/resnet34'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument("--target_sparsity", default=0.45, type=float, choices=[0.30, 0.45, 0.60, 0.75, 0.90])
    parser.add_argument("--strat", default="struct", type=str, choices=["struct", "finegrain"])
    parser.add_argument("--bias", default=False, type=bool, choices=[True, False])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    # set training gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    elif opt.target_sparsity is not None:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)
    if opt.target_sparsity is not None:
        opt.model_name += '_ts:{}_strat:{}'.format(opt.target_sparsity, opt.strat)
        
    if opt.bias:
        opt.model_name += ':bias'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def get_pretrained_torch_model(model_name, num_classes):
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    return model
    
def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        
        seed = 0
        indices = [
                (83, 'shrew', 0.1), (17, 'can', 0.2), (86, 'oak_tree', 0.2), (87, 'palm_tree', 0.2), 
                (76, 'dinosaur', 0.5), (20, 'apple', 0.1), (75, 'crocodile', 0.1), (22, 'orange', 0.5), 
                (58, 'elephant', 0.5), (94, 'train', 0.2), (63, 'raccoon', 0.5), (85, 'maple_tree', 0.1), 
                (90, 'bicycle', 0.1), (37, 'butterfly', 0.2), (6, 'flatfish', 0.5)]

        percentages = np.ones((100, ))

        for sample in indices:
            percentages[sample[0]] = sample[2]
            
        percentages = 1 - percentages
        
        if not opt.bias:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        else:
            train_loader, val_loader, _ = get_cifar100_imbalanced(percentages, seed, 
                                                               batch_size=opt.batch_size,
                                                               num_workers=opt.num_workers)
            
        n_cls = 100
    elif opt.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 1000
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if 'pretrained_torch' not in opt.model:
        model = model_dict[opt.model](num_classes=n_cls)
    else:
        model_name = opt.model.split('/')[1]
        model = get_pretrained_torch_model(model_name, n_cls)
    
    if opt.path_t:
        model = load_teacher(opt.path_t, n_cls)
    elif 'pretrained_torch' in opt.model:
        opt.path_t = 'pretrained_torch'
    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    freq = 1
    prune_end = int(opt.epochs * 0.75)
    prune_sch = AgpPruningRate(.05, opt.target_sparsity, 1, prune_end, freq)
    prune_layers = [module for module in model.modules()][:-1]
    strat = opt.strat
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        
        if epoch % freq == 1 and epoch <= prune_end:
            target = prune_sch.step(epoch)
            print(f'pruning {target * 100}% sparsity')
            if epoch > 1 and epoch < prune_end:
               for i, layer in enumerate(prune_layers):
                   if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                       prune.remove(layer, "weight")
            for i, layer in enumerate(prune_layers):
                if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                    if "struct" in strat:
                        prune.ln_structured(layer, name="weight",
                                            amount=float(target), n=1, dim=0)
                    elif 'finegrain' in strat:
                        prune.l1_unstructured(layer, name='weight',
                                              amount=float(target))
                    layer_spar = float(torch.sum(layer.weight == 0))
                    layer_spar /= float(layer.weight.nelement())
                    print(f"Sparsity in layer {i}: {layer_spar: 3f}")
        elif epoch > prune_end:
            print("All done pruning")
            
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc and epoch > prune_end:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)

if __name__ == '__main__':
    main()
