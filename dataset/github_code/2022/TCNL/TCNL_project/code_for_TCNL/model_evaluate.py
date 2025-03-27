import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import shutil
import wandb
from model.new_vgg import *
from utils.data_utils import get_train_loader, get_val_loader
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
from utils.gpu_utils import MemTracker
from matplotlib import pyplot as plt
from tqdm import tqdm
plt.switch_backend('agg')


def predict(net: vgg_multi_concept_for_mammal, data_loader, args: dict):
    net.eval()
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            image = data['image'].to(args['device'])
            label = data['label'].to(args['device'])
            shape = data['outline'].to(args['device'])
            head = data['head'].to(args['device'])
            torso = data['torso'].to(args['device'])
            leg = data['leg'].to(args['device'])
            image_name = data['image_name']
            pred_class, reconstruct_result, loss_curr = net.validating(image, label, head, torso, leg, shape)

if __name__ == '__main__':
    args = {
        'train_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train.txt',
        'val_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val.txt',
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/exp_v4/vgg_multi_concept_for_mammal_1017/ckpt_280_acc80.91_loss2.35.pt',
        'visualization_path': '/home/wangzhihao/XAI/XCSGCNN/train/visualization/exp_v4/vgg_multi_concept_mammal_1017',
        'train_batch_size': 1,
        'val_batch_size': 8,
        'device': 'cuda:0',
        'epoch': 301,
        'num_classes': 5,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }

    val_loader = get_val_loader(args['val_txt_path'], args['val_batch_size'])
    train_loader = get_train_loader(args['train_txt_path'], args['train_batch_size'])
    ckpt = torch.load(args['ckpt_path'], map_location=args['device'])
    net = vgg_multi_concept_for_mammal(args)
    net.load_state_dict(ckpt['net'])
    net = net.to(args['device'])

    predict(net, train_loader, args)
