import torch
import torch.nn as nn
import torch.utils.data as Data
from densenet import DenseNet3
import conf.config as conf
import utils
import argparse
from data.dataLoader import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(
    description='WideResnet Training With Pytorch')
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN'],
                    type=str, help='CIFAR10, CIFAR100 or SVHN')
parser.add_argument('--dataset_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='./models/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'CIFAR10':
        train_data = CIFAR10_train_data
        test_data = CIFAR10_test_data
        cfg = conf.CIFAR10
    if args.dataset == 'CIFAR100':

        train_data = CIFAR100_train_data
        test_data = CIFAR100_test_data
        cfg = conf.CIFAR100
    if args.dataset == 'SVHN':
        train_data = SVHN_train_data
        test_data = SVHN_test_data
        cfg = conf.SVHN


    else:
        print("dataset doesn't exist!")
        exit(0)


    utils.PEDCC_PATH = cfg['PEDCC_Type']
    cnn = DenseNet3(depth=100, num_classes=cfg['num_classes'], feature_size=cfg['feature_size'])
    # #PEDCC_loss
    #cifar10/SVHN:
    criterion = utils.AMSoftmax(5.5, 0.35, is_amp=False)
    #cifar100:
    #criterion = utils.AMSoftmax(10, 0.25, is_amp=False)

    criterion1 = nn.MSELoss()

    train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg['batch_size'], shuffle=False, num_workers=6, pin_memory=True)
    # start training
    utils.train(cnn, train_loader, test_loader, cfg, criterion, criterion1, args.save_folder)

if __name__ == '__main__':
    train()
