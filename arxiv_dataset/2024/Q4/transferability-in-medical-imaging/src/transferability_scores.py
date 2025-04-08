#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='either dataset (dataset transfer) or model (architecture transfer)')
parser.add_argument('--method', type=str, help='choose a tranferability method (logme, leep, nleep, sfda, parc, ncti, lp, fu)')
args = parser.parse_args()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from medmnist import INFO
import torch.nn.functional as F
from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, NCTI_Score, LP_Score, FU_Score
import pandas as pd
import math

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None, as_rgb=False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index].astype(int)
        x = Image.fromarray(x)

        if self.as_rgb:
            x = x.convert("RGB")
        
        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def load_data(source_flag, target_flag):
    info = INFO[target_flag]
    as_rgb = True if info['n_channels'] == 1 else False
    # preprocessing
    if source_flag in ['imagenet','densenet','efficientnet','googlenet','mnasnet','mobilenet','vgg','convnext','shufflenet','resnet']:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:    
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    target_data = np.load('/mnt/share/data/medmnist_similarity/' + target_flag +  '_fine-tune_224.npz')
    trainset = MyDataset(target_data["train_imgs_fold1"], target_data["train_labels_fold1"], transform=train_transform, as_rgb=as_rgb)

    return trainset

def get_fc(source_flag, target_flag):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_set = load_data(source_flag, target_flag)
    data_loader = DataLoader(dataset=train_set,
                                    batch_size=32,
                                    shuffle=False)   

    if source_flag == 'densenet':
        net = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        last = net.classifier
    elif source_flag == 'efficientnet':
        net = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        last = net.classifier[-1]
    elif source_flag == 'googlenet':
        net = torchvision.models.googlenet(weights='IMAGENET1K_V1')
        last = net.fc
    elif source_flag == 'mnasnet':
        net = torchvision.models.mnasnet1_0(weights='IMAGENET1K_V1')
        last = net.classifier[-1]
    elif source_flag == 'mobilenet':
        net = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        last = net.classifier
    elif source_flag == 'vgg':
        net = torchvision.models.vgg11(weights='IMAGENET1K_V1')
        last = net.classifier[-1]
    elif source_flag == 'convnext':
        net = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        last = net.classifier[-1]
    elif source_flag == 'shufflenet':
        net = torchvision.models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
        last = net.fc
    elif source_flag == 'resnet':
        net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        last = net.fc
    else:
        if source_flag == 'imagenet':
            net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        elif source_flag == 'radimagenet':
            model_path = '/mnt/share/models/doju_pre-trained_for/radimagenet.pt'
            source_classes = 165
            net = torchvision.models.resnet18(num_classes=source_classes)
            net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        elif source_flag == 'medmnist':
            model_path = '/mnt/share/models/doju_pre-trained_for/'+ target_flag +'.pt'
            info = INFO[target_flag]
            if target_flag == 'pneumoniamnist' or target_flag == 'organamnist' or target_flag == 'organcmnist' or target_flag == 'organsmnist':
                source_classes = 69
            elif target_flag == 'chestmnist':
                source_classes = 56
            else:
                source_classes = 69 - len(info['label'])
            net = torchvision.models.resnet18(num_classes=source_classes)
            net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        else:
            source_info = INFO[source_flag]
            source_classes = len(source_info['label'])
            model_path = '/mnt/share/models/doju_sim_pretrained/'+ source_flag +'/resnet18_224_1.pth'
            net = torchvision.models.resnet18(num_classes=source_classes)
            net.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=False)
        last = net.fc
        
    net.eval()

    all_feature_maps = []
    outputs = []
    all_labels = []
    # Define the hook function
    def hook_fn(module, input, output):
        all_feature_maps.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())

    last.register_forward_hook(hook_fn)

    # forward pass
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        all_labels.append(targets)
        _ = net(inputs)
    #penultimate layer
    all_feature_maps = torch.cat(all_feature_maps, dim=0)
    X = all_feature_maps.detach().numpy()

    outputs = torch.cat(outputs, dim=0)

    all_labels = torch.cat(all_labels, dim=0)
    y = all_labels.cpu().numpy()

    return X, outputs, y

def get_layers(source_flag, target_flag):

    target_info = INFO[target_flag]
    n_classes = len(target_info['label'])
    task = target_info['task']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    if source_flag == 'densenet':
        net = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        net.classifier = nn.Identity()
        first = net.features.conv0 #Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        second = net.features.denseblock1.denselayer1.conv1 #Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    elif source_flag == 'efficientnet':
        net = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        net.classifier[-1] = nn.Identity()
        first = net.features[0][0] #Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        second = net.features[1][0].block[0][0] #Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif source_flag == 'googlenet':
        net = torchvision.models.googlenet(weights='IMAGENET1K_V1')
        net.fc = nn.Identity()
        first = net.conv1.conv #Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        second = net.conv2.conv #Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    elif source_flag == 'mnasnet':
        net = torchvision.models.mnasnet1_0(weights='IMAGENET1K_V1')
        net.classifier[-1] = nn.Identity()
        first = net.layers[0] #Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        second = net.layers[3] #Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    elif source_flag == 'mobilenet':
        net = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        net.classifier = nn.Identity()
        first = net.features[0][0] #Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        second = net.features[1].block[0][0] #Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
    elif source_flag == 'vgg':
        net = torchvision.models.vgg11(weights='IMAGENET1K_V1')
        net.classifier[-1] = nn.Identity()
        first = net.features[0] #Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        second = net.features[3] #Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif source_flag == 'convnext':
        net = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        net.classifier[-1] = nn.Identity()
        first = net.features[0][0] #Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        second = net.features[1][0].block[0] #Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
    elif source_flag == 'shufflenet':
        net = torchvision.models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
        net.fc = nn.Identity()
        first = net.conv1[0] #Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        second = net.stage2[0].branch1[0] #Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
    elif source_flag == 'resnet':
        net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        net.fc = nn.Identity()
        first = net.conv1 #Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        second = net.layer1[0].conv1 #Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    else:
        if source_flag == 'imagenet':
            net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        elif source_flag == 'radimagenet':
            model_path = '/mnt/share/models/doju_pre-trained_for/radimagenet.pt'
            source_classes = 165
            net = torchvision.models.resnet18(num_classes=source_classes)
            net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        elif source_flag == 'medmnist':
            model_path = '/mnt/share/models/doju_pre-trained_for/'+ target_flag +'.pt'
            info = INFO[target_flag]
            if target_flag == 'pneumoniamnist' or target_flag == 'organamnist' or target_flag == 'organcmnist' or target_flag == 'organsmnist':
                source_classes = 69
            elif target_flag == 'chestmnist':
                source_classes = 56
            else:
                source_classes = 69 - len(info['label'])
            net = torchvision.models.resnet18(num_classes=source_classes)
            net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        else:
            source_info = INFO[source_flag]
            source_classes = len(source_info['label'])
            model_path = '/mnt/share/models/doju_sim_pretrained/'+ source_flag +'/resnet18_224_1.pth'
            net = torchvision.models.resnet18(num_classes=source_classes)
            net.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=False)
        net.fc = nn.Identity()
        first = net.conv1
        second = net.layer1[0].conv1

    train_set = load_data(source_flag, target_flag)
    data_loader = DataLoader(dataset=train_set,
                            batch_size= math.floor(len(train_set)/6), # if target_flag in ['pneumoniamnist', 'breastmnist'] else 256,
                            shuffle=False)

    return net, first, second, data_loader, device

if args.source == 'dataset':
    source_order=['imagenet','radimagenet','medmnist','bloodmnist','breastmnist', 'chestmnist', 'dermamnist','octmnist','organamnist','organcmnist','organsmnist','pathmnist','pneumoniamnist','retinamnist','tissuemnist']
elif args.source == 'model':
    source_order=['densenet','efficientnet','googlenet','mnasnet','mobilenet', 'vgg', 'convnext','shufflenet','resnet']
target_order=['bloodmnist','breastmnist', 'dermamnist','octmnist','organamnist','organcmnist','organsmnist','pathmnist','pneumoniamnist','retinamnist','tissuemnist']

score_df = pd.DataFrame({'source': source_order})
seli_df = pd.DataFrame({'source': source_order})
ncc_df = pd.DataFrame({'source': source_order})
vc_df = pd.DataFrame({'source': source_order})
for target in target_order:
    seli_scores = []
    ncc_scores = []
    vc_scores = []
    scores = []
    for source in source_order:
        if args.method == 'fu':
            model, first, second, data_loader, device = get_layers(source, target)
            score = FU_Score(model, first, second, data_loader, device)
            scores.append(score)
        else:
            X, output, y = get_fc(source, target)
            if args.method == 'logme':
                score = LogME_Score(X, y.reshape(-1))
            elif args.method == 'leep':
                score = LEEP(output, y.reshape(-1))
            elif args.method == 'nleep':
                score = NLEEP(X, y.reshape(-1), component_ratio=5)
            elif args.method == 'sfda':
                score = SFDA_Score(X, y.reshape(-1))
            elif args.method == 'parc':
                score = PARC_Score(X, y.reshape(-1), ratio=2)
            elif args.method == 'lp':
                score = LP_Score(X, y.reshape(-1))
            scores.append(score)
            if args.method == 'ncti':
                ncti = NCTI_Score(X, y.reshape(-1))
                if source == target:
                    seli_scores.append(np.nan)
                    ncc_scores.append(np.nan)
                    vc_scores.append(np.nan)
                else:
                    seli_scores.append(ncti[0])
                    ncc_scores.append(ncti[1])
                    vc_scores.append(ncti[2])
    if args.method == 'ncti':
        seli_df[target] = (seli_scores - np.nanmin(seli_scores)) / (np.nanmax(seli_scores) - np.nanmin(seli_scores))
        ncc_df[target] = (ncc_scores - np.nanmin(ncc_scores)) / (np.nanmax(ncc_scores) - np.nanmin(ncc_scores))
        vc_df[target] = (vc_scores - np.nanmin(vc_scores)) / (np.nanmax(vc_scores) - np.nanmin(vc_scores))
        score_df[target] = seli_df[target] + ncc_df[target] + vc_df[target]
    else:
        score_df[target] = scores

score_df.to_csv(f'./results/{args.method}_{args.source}.csv')


