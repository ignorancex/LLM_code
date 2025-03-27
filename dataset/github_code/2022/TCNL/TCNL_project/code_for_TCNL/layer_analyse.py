import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from model.XCSGCNN_exp import XCSGCNN
from utils.data_utils import get_train_loader, get_val_loader
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
from matplotlib import pyplot as plt

args = {
        'train_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/train.txt',
        'val_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/val.txt',
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/XCSGCNN_new_R_202205010',
        'train_batch_size': 1,
        'val_batch_size': 16,
        'device': 'cuda:1',
        'epoch': 1000,
        'num_classes': 20,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }

'''
loading data...
'''
print('===> loading data...')
train_loader = get_train_loader(args['train_txt_path'], args['train_batch_size'])
val_loader = get_val_loader(args['val_txt_path'], args['val_batch_size'])
print('train loader: %d, val loader %d' % (len(train_loader), len(val_loader)))
print('trainset size %d, valset size %d' % (len(train_loader)*args['train_batch_size'], len(val_loader)*args['val_batch_size']))

'''
model
'''
print('===> loading model...')
net = XCSGCNN(args)
ckpt_path = '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/XCSGCNN_new_R_202205010/ckpt_86_acc15.97_loss6.62.pt'
ckpt = torch.load(ckpt_path, map_location=args['device'])
net.load_state_dict(ckpt['net'])
net = net.to(args['device'])

for index, data in enumerate(train_loader):
    print(data['image_name'])
    image = data['image'].to(args['device'])
    print(image.shape)
    label = data['label'].to(args['device'])
    main_object = data['object'].to(args['device'])
    image_name = data['image_name']
     # target_dir = '/home/wangzhihao/XAI/XCSGCNN/train/visulization/reconstruct_image_in_training_20220510/' + str(epoch)
    pred_class, pred_fake, pred_real, reg, reconstruct_image, main_object, a, b = net(image, label, main_object)
    a = a.cpu().data.numpy()
    b = b.cpu().data.numpy()
    a = a[0]
    b = b[0]
    for i in range(a.shape[0]):
        tmp = a[i]
        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) * 255
        tmp = np.uint8(tmp)
        tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_CIVIDIS)
        output_path = '/home/wangzhihao/XAI/XCSGCNN/train/visulization/feature_map/first_conv/' + str(i) + '.png'
        cv2.imwrite(output_path, tmp)
    for i in range(b.shape[0]):
        tmp = b[i]
        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) * 255
        tmp = np.uint8(tmp)
        tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_CIVIDIS)
        output_path = '/home/wangzhihao/XAI/XCSGCNN/train/visulization/feature_map/layer2/' + str(i) + '.png'
        cv2.imwrite(output_path, tmp)
    break