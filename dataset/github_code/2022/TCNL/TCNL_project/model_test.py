from model.new_resnet import *
from model.vgg_backbone import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# ckpt_path = '/root/XCSGCNN/train/model/resnet50-19c8e357.pth'
# ckpt_path = '/root/XCSGCNN/train/ckpt/exp_v3/resnet_multi_concept_for_mammal/ckpt_240_acc74.55_loss3.02.pt'
# ckpt = torch.load(ckpt_path, map_location='cuda:0')
# print(ckpt['net'].keys())
# model = vgg_shallow(num_classes=5)


# model = vgg_deep(num_classes=5)
# se = SE(in_chnls=512)
args = {
        'train_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train_multi_concept.txt',
        'val_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val_multi_concept.txt',
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/exp_v4/vgg_multi_concept_for_mammal',
        'visualization_path': '/home/wangzhihao/XAI/XCSGCNN/train/visualization/exp_v4/vgg_multi_concept_mammal',
        'train_batch_size': 8,
        'val_batch_size': 8,
        'device': 'cuda:0',
        'epoch': 301,
        'num_classes': 5,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }

model = resnet_multi_concept_for_mammal(args)
# ckpt = torch.load('/home/wangzhihao/XAI/XCSGCNN/train/ckpt/exp_v4/vgg_multi_concept_for_mammal_1017/ckpt_260_acc80.91_loss2.34.pt', map_location='cuda:0')
# model.load_state_dict(ckpt['net'])
# model.cuda()
a = torch.ones(size=(1, 3, 224, 224))
b = model.inference(a)


# b = model(a)
# print(b.shape)
# c = se(b)



# model = vgg_multi_concept_for_mammal(args)
# ckpt = torch.load('/home/wangzhihao/XAI/XCSGCNN/train/ckpt/exp_v4/vgg_multi_concept_for_mammal/ckpt_1_acc53.64_loss91.46.pt')
# model.load_state_dict(ckpt['net'])
# mask = model.torso_feature_extractor.layer_mask.mask.data.numpy()
# print(np.max(mask), np.min(mask))
